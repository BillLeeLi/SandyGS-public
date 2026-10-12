"""Derived-scene inpainting must mask unknown PGSR depth and preserve known pixels."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import yaml

from sandygs.existing_sand import e3_completion
from sandygs.existing_sand.e1_run import (_INFUSION_CONFIG, _INFUSION_WEIGHTS,
                                           _SDXL_CONFIG, _SDXL_WEIGHTS)
from sandygs.existing_sand.workspace import (ExistingSandError, initialize_run,
                                            record_stage, verify_run)

ROOT = Path(__file__).resolve().parents[1]
H, W = 64, 768


def _register(run: Path, kind: str, name: str, files: list[str], *,
              prompt=None, parameters=None, parents=None):
    stage = run / "stages" / kind / name
    (stage / "record.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "stage": kind, "stage_id": name,
        "camera_ids": ["ground.png"], "prompt": prompt or {}, "seed": None,
        "models": [], "parameters": parameters or {}, "commands": [],
        "parents": parents or [],
        "artifacts": [{"path": item, "role": "test"} for item in files],
    }), encoding="utf-8")
    record_stage(run, stage / "record.yaml", repo_root=ROOT)


def _scene(tmp_path: Path):
    ds = tmp_path / "dataset"
    ds.mkdir()
    (ds / "cameras.txt").write_text("camera", encoding="utf-8")
    source = tmp_path / "checkpoint"
    source.mkdir()
    ply = b"ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 2\n1 0 2\n"
    (source / "point_cloud.ply").write_bytes(ply)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "schema_version": 1, "scene_id": "synthetic",
        "source": {"dataset": str(ds), "checkpoint": str(source),
                   "point_cloud": "point_cloud.ply", "dataset_files": ["cameras.txt"]},
        "camera_ids": ["ground.png"],
        "coordinate": {"sim_to_gs": np.eye(4).tolist()},
    }), encoding="utf-8")
    run = tmp_path / "run_001"
    initialize_run(config, run, run_id="run_001", repo_root=ROOT)
    view = run / "stages/observations/observed/view_0000"
    view.mkdir(parents=True)
    rgb = np.full((H, W, 3), 0.25, np.float32)
    depth = np.full((H, W), 3.0, np.float32)
    np.save(view / "render_rgb.npy", rgb)
    np.save(view / "plane_depth.npy", depth)
    Image.fromarray(np.full((H, W, 3), 64, np.uint8)).save(view / "render_rgb.png")
    camera = {"camera_id": "ground.png", "render_size_wh": [W, H],
              "source_size_wh": [W, H],
              "colmap_intrinsics": {"fx": 500.0, "fy": 500.0},
              "pgsr_K": [[500, 0, 383.5], [0, 500, 31.5], [0, 0, 1]],
              "gs_w2c": np.eye(4).tolist()}
    (view / "camera.yaml").write_text(yaml.safe_dump(camera), encoding="utf-8")
    _register(run, "observations", "observed", [f"view_0000/{name}" for name in
        ("render_rgb.npy", "plane_depth.npy", "render_rgb.png", "camera.yaml")],
        parameters={"sh_degree": 0, "alpha_min": 0.1, "white_background": False})
    masks = run / "stages/segmentation/sam"
    (masks / "view_0000").mkdir(parents=True)
    mask = np.zeros((H, W), np.uint8)
    mask[16:48, 300:360] = 255
    Image.fromarray(mask).save(masks / "view_0000/mask.png")
    (masks / "quality.yaml").write_text(yaml.safe_dump({"views": [{
        "camera_id": "ground.png", "role": "holdout", "directory": "view_0000"}]}), encoding="utf-8")
    _register(run, "segmentation", "sam", ["view_0000/mask.png", "quality.yaml"],
              prompt={"type": "manual_sam_points_box"},
              parameters={"capture_id": "observed"},
              parents=[{"stage": "observations", "stage_id": "observed"}])
    selected = run / "stages/segmentation/selected"
    selected.mkdir()
    np.save(selected / "gaussian_labels.npy", np.array([0, 1], np.uint8))
    _register(run, "segmentation", "selected", ["gaussian_labels.npy"],
              prompt={"type": "multiview_sam_depth_seeds", "mask_id": "sam"},
              parents=[{"stage": "segmentation", "stage_id": "sam"}])
    derived = run / "stages/background/derived"
    (derived / "checkpoint").mkdir(parents=True)
    (derived / "checkpoint/point_cloud.ply").write_bytes(ply)
    _register(run, "background", "derived", ["checkpoint/point_cloud.ply"],
              prompt={"type": "approved_gaussian_deletion", "selection_id": "selected",
                      "check_id": "checked"},
              parameters={"point_cloud": "point_cloud.ply"},
              parents=[{"stage": "segmentation", "stage_id": "selected"},
                       {"stage": "segmentation", "stage_id": "sam"},
                       {"stage": "observations", "stage_id": "observed"}])
    return run, mask > 0


def test_derived_prepare_and_rgb_depth_completion(tmp_path, monkeypatch):
    run, mask = _scene(tmp_path)
    depth = np.full((H, W), 3.0, np.float32)
    depth[mask] = np.nan
    alpha = np.ones((H, W), np.float32)
    alpha[mask] = 0
    rgb = np.full((H, W, 3), 0.4, np.float32)
    monkeypatch.setattr(e3_completion, "_render_derived", lambda *_: (rgb, depth, alpha))
    result = e3_completion.prepare_background(
        run, "derived", "ground.png", "prepared", (0, 0, W, H), repo_root=ROOT)
    prepared = Path(result["stage_dir"])
    inverse = np.load(prepared / "input_inverse_depth.npy")
    assert np.isfinite(inverse).all()
    assert np.allclose(inverse, 1 / 3)
    assert result["derived_invalid_inside_edit"] == int(mask.sum())
    np.testing.assert_array_equal(np.load(prepared / "edit_mask.npy"), mask)
    assert {"stage": "segmentation", "stage_id": "prepared"} in verify_run(run)["stages"]

    code = tmp_path / "Infusion"
    for name in ("depth_inpainting/run/run_inference_inpainting.py",
                 "depth_inpainting/inference/depth_inpainting_pipeline_half.py",
                 "depth_inpainting/utils/image_util.py"):
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic\n", encoding="utf-8")
    infusion = tmp_path / "infusion_weights"
    sdxl = tmp_path / "sdxl_weights"
    for root, names in ((infusion, _INFUSION_WEIGHTS + _INFUSION_CONFIG),
                        (sdxl, _SDXL_WEIGHTS + _SDXL_CONFIG)):
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic")
    model_python = tmp_path / "venv_python"
    model_python.symlink_to(sys.executable)

    def fake_worker(argv, *, cwd, log):
        if "--output_dir" in argv:
            with Image.open(argv[argv.index("--input_rgb_path") + 1]) as image:
                assert image.size == (W, H)
            inverse_path = Path(argv[argv.index("--input_depth_path") + 1])
            assert np.isfinite(np.load(inverse_path)).all()
            output = Path(argv[argv.index("--output_dir") + 1])
            np.save(output / "completed_rgb_depth_dis.npy", np.full((H, W), 0.25, np.float32))
        else:
            output = Path(argv[argv.index("--output") + 1])
            Image.fromarray(np.full((H, W, 3), 180, np.uint8)).save(output)
        log.write_text("synthetic worker\n", encoding="utf-8")

    monkeypatch.setattr(e3_completion, "_run_logged", fake_worker)
    finished = e3_completion.run_background_completion(
        run, "prepared", "completed", repo_root=ROOT, model_python=model_python,
        infusion_code=code, infusion_model=infusion, sdxl_model=sdxl,
        prompt="empty ground", seed=37)
    stage = Path(finished["stage_dir"])
    completed_depth = np.load(stage / "completed_depth.npy")
    np.testing.assert_array_equal(completed_depth[~mask], depth[~mask])
    np.testing.assert_allclose(completed_depth[mask], 4.0)
    assert np.load(stage / "filled_points_world.npy").shape == (int(mask.sum()), 3)
    assert finished["quality"]["known_depth_exactly_preserved"]
    assert finished["quality"]["known_rgb_exactly_preserved"]
    assert (stage / "report/overview.png").is_file()
    assert {"stage": "background", "stage_id": "completed"} in verify_run(run)["stages"]


def test_prepare_rejects_unmasked_invalid_depth(tmp_path, monkeypatch):
    run, _ = _scene(tmp_path)
    depth = np.full((H, W), 3.0, np.float32)
    depth[0, 0] = np.nan
    monkeypatch.setattr(e3_completion, "_render_derived", lambda *_: (
        np.full((H, W, 3), 0.4, np.float32), depth, np.ones((H, W), np.float32)))
    with pytest.raises(ExistingSandError, match="invalid PGSR depth outside edit mask"):
        e3_completion.prepare_background(
            run, "derived", "ground.png", "invalid", (0, 0, W, H), repo_root=ROOT)
    assert not (run / "stages/segmentation/invalid").exists()


def test_prepare_expands_sam_and_respects_protected_foreground(tmp_path, monkeypatch):
    run, sam = _scene(tmp_path)
    protect = np.zeros((H, W), np.uint8)
    protect[20:28, 310:320] = 255
    protect_file = tmp_path / "protect.png"
    Image.fromarray(protect).save(protect_file)
    monkeypatch.setattr(e3_completion, "_render_derived", lambda *_: (
        np.full((H, W, 3), 0.4, np.float32),
        np.full((H, W), 3.0, np.float32),
        np.ones((H, W), np.float32)))
    result = e3_completion.prepare_background(
        run, "derived", "ground.png", "expanded", (0, 0, W, H),
        repo_root=ROOT, expand_pixels=2, protect_mask=protect_file)
    edit = np.load(Path(result["stage_dir"]) / "edit_mask.npy")
    assert not edit[20:28, 310:320].any()
    assert edit.sum() > sam.sum() - np.count_nonzero(protect)
    assert edit[14, 300]
    assert {"stage": "segmentation", "stage_id": "expanded"} in verify_run(run)["stages"]


def test_prepare_renders_previous_fit_and_authored_mask(tmp_path, monkeypatch):
    run, sam = _scene(tmp_path)
    fitted = run / 'stages/background/fitted'
    (fitted / 'checkpoint').mkdir(parents=True)
    (fitted / 'checkpoint/point_cloud.ply').write_bytes(b'synthetic fitted cloud')
    _register(run, 'background', 'fitted', ['checkpoint/point_cloud.ply'],
              prompt={'type': 'pgsr_local_background_optimization'},
              parameters={'derived_id': 'derived', 'checkpoint_point_cloud': 'point_cloud.ply',
                          'new_start_index': 1})
    authored = np.zeros_like(sam); authored[20:40, 305:340] = True
    authored_path = tmp_path / 'edit.png'
    Image.fromarray(authored.astype(np.uint8)*255).save(authored_path)
    calls = []
    def render(checkpoint, *args):
        calls.append(checkpoint)
        return (np.full((H, W, 3), .6, np.float32), np.full((H, W), 3., np.float32), np.ones((H, W), np.float32))
    monkeypatch.setattr(e3_completion, '_render_derived', render)
    result = e3_completion.prepare_background(run, 'derived', 'ground.png', 'from_fit', (0, 0, W, H),
                                             repo_root=ROOT, source_fit_id='fitted', edit_mask=authored_path)
    out = Path(result['stage_dir'])
    assert calls == [fitted / 'checkpoint']
    np.testing.assert_array_equal(np.load(out / 'edit_mask.npy'), authored)
    assert yaml.safe_load((out / 'crop.yaml').read_text())['source_fit_id'] == 'fitted'
    assert {'stage': 'background', 'stage_id': 'fitted'} in yaml.safe_load((out / 'record.yaml').read_text())['parents']
    verify_run(run)
