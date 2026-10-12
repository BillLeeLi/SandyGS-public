"""E1 leakage, geometry, scoring and stage integration without diffusion weights."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import yaml

from sandygs.existing_sand.e1 import prepare_hole
from sandygs.existing_sand.e1_evaluate import depth_metrics
from sandygs.existing_sand import e1_run
from sandygs.existing_sand.workspace import (
    ExistingSandError, initialize_run, record_stage, verify_run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path, *, invalid_pixel: bool = False) -> tuple[Path, np.ndarray, np.ndarray]:
    dataset = tmp_path / "dataset"
    (dataset / "sparse/0").mkdir(parents=True)
    (dataset / "sparse/0/cameras.bin").write_bytes(b"dummy")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "point_cloud.ply").write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"end_header\n0 0 3\n"
    )
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "schema_version": 1, "scene_id": "synthetic_ground",
        "source": {"dataset": str(dataset), "checkpoint": str(checkpoint),
                   "point_cloud": "point_cloud.ply",
                   "dataset_files": ["sparse/0/cameras.bin"]},
        "camera_ids": ["ground.png"],
        "coordinate": {"sim_to_gs": np.eye(4).tolist()},
    }))
    run = tmp_path / "run_001"
    initialize_run(config, run, run_id="run_001", repo_root=REPO_ROOT)
    view = run / "stages/observations/capture_001/view_0000"
    view.mkdir(parents=True)
    v, u = np.indices((64, 768))
    depth = (3 + 0.001 * u + 0.002 * v).astype(np.float32)
    rgb = np.stack((u / 767, v / 63, np.full_like(u, 0.4)), axis=-1).astype(np.float32)
    mask = np.zeros(depth.shape, dtype=np.uint8)
    mask[15:45, 310:370] = 255
    np.save(view / "plane_depth.npy", depth)
    valid = np.ones_like(depth, dtype=np.bool_)
    if invalid_pixel:
        valid[0, 0] = False
    np.save(view / "depth_valid.npy", valid)
    np.save(view / "render_rgb.npy", rgb)
    camera = {
        "camera_id": "ground.png", "render_size_wh": [768, 64],
        "pgsr_K": [[500, 0, 383.5], [0, 500, 31.5], [0, 0, 1]],
        "gs_w2c": np.eye(4).tolist(),
    }
    (view / "camera.yaml").write_text(yaml.safe_dump(camera))
    observation_stage = view.parent
    observation_record = {
        "schema_version": 1, "stage": "observations", "stage_id": "capture_001",
        "camera_ids": ["ground.png"], "prompt": {}, "seed": None,
        "models": [], "parameters": {}, "commands": [], "parents": [],
        "artifacts": [
            {"path": path.relative_to(observation_stage).as_posix(), "role": path.stem}
            for path in sorted(view.iterdir())
        ],
    }
    (observation_stage / "record.yaml").write_text(yaml.safe_dump(observation_record))
    record_stage(run, observation_stage / "record.yaml", repo_root=REPO_ROOT)
    return run, depth, mask


def test_prepare_masks_infusion_initial_latent_without_reference_leak(tmp_path: Path) -> None:
    run, depth, mask = _fixture(tmp_path)
    annotation = tmp_path / "hole.png"
    Image.fromarray(mask).save(annotation)
    result = prepare_hole(run, "capture_001", "ground.png", annotation,
                          "front_hole", (0, 0, 768, 64), repo_root=REPO_ROOT)
    stage = Path(result["stage_dir"])
    generated_mask = np.load(stage / "edit_mask.npy")
    model_input = np.load(stage / "input_inverse_depth.npy")
    assert result["hole_pixels"] == 1800
    assert generated_mask.dtype == np.bool_
    np.testing.assert_array_equal(generated_mask, mask > 0)
    np.testing.assert_allclose(model_input[~generated_mask], 1 / depth[~generated_mask])
    assert np.all(model_input[generated_mask] == model_input[generated_mask][0])
    assert not np.allclose(model_input[generated_mask], 1 / depth[generated_mask])
    assert yaml.safe_load((stage / "normalization.yaml").read_text())["infusion_resize_scale"] == 1
    assert len(verify_run(run)["stages"]) == 2
    with pytest.raises(ExistingSandError, match="overwrite"):
        prepare_hole(run, "capture_001", "ground.png", annotation,
                     "front_hole", (0, 0, 768, 64), repo_root=REPO_ROOT)


def test_prepare_rejects_invalid_known_depth(tmp_path: Path) -> None:
    run, _, mask = _fixture(tmp_path, invalid_pixel=True)
    annotation = tmp_path / "hole.png"
    Image.fromarray(mask).save(annotation)
    with pytest.raises(ExistingSandError, match="invalid PGSR depth"):
        prepare_hole(run, "capture_001", "ground.png", annotation,
                     "invalid", (0, 0, 768, 64), repo_root=REPO_ROOT)


def test_depth_metrics_use_camera_z_and_report_invalid_predictions() -> None:
    reference = np.full((16, 16), 4, dtype=np.float32)
    prediction = reference.copy()
    mask = np.zeros((16, 16), dtype=np.bool_)
    mask[4:12, 4:12] = True
    prediction[mask] = 5
    prediction[4, 4] = np.nan
    metrics, angles = depth_metrics(reference, prediction, mask,
                                    np.array([[30, 0, 8], [0, 30, 8], [0, 0, 1]]))
    assert metrics["hole_pixels"] == 64
    assert metrics["valid_prediction_pixels"] == 63
    assert metrics["depth_mae"] == 1
    assert metrics["depth_rmse"] == 1
    assert metrics["valid_prediction_fraction"] == pytest.approx(63 / 64)
    assert angles.shape == reference.shape


def test_completion_registers_model_and_keeps_known_pgsr_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, depth, mask = _fixture(tmp_path)
    annotation = tmp_path / "hole.png"
    Image.fromarray(mask).save(annotation)
    prepare_hole(run, "capture_001", "ground.png", annotation,
                 "front_hole", (0, 0, 768, 64), repo_root=REPO_ROOT)
    code = tmp_path / "Infusion"
    for rel in ("depth_inpainting/run/run_inference_inpainting.py",
                "depth_inpainting/inference/depth_inpainting_pipeline_half.py",
                "depth_inpainting/utils/image_util.py"):
        path = code / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    model = tmp_path / "model"
    for rel in e1_run._INFUSION_WEIGHTS + e1_run._INFUSION_CONFIG:
        path = model / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("dummy")

    def fake_worker(argv: list[str], *, cwd: Path, log: Path) -> None:
        output = Path(argv[argv.index("--output_dir") + 1])
        input_depth = Path(argv[argv.index("--input_depth_path") + 1])
        rgb = Path(argv[argv.index("--input_rgb_path") + 1])
        assert argv[0] == str(tmp_path / "venv_python")
        inverse = np.load(input_depth)
        assert np.isfinite(inverse).all()
        np.save(output / (rgb.stem + "_depth_dis.npy"), np.full_like(inverse, 0.25))
        (output / (rgb.stem + "_mask.ply")).write_text(
            "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n"
        )
        log.write_text("synthetic worker\n")

    monkeypatch.setattr(e1_run, "_run_logged", fake_worker)
    python_link = tmp_path / "venv_python"
    python_link.symlink_to(sys.executable)
    result = e1_run.run_completion(
        run, "front_hole", "front_seed_7", repo_root=REPO_ROOT,
        model_python=python_link, infusion_code=code,
        infusion_model=model, seed=7,
    )
    stage = Path(result["stage_dir"])
    completed = np.load(stage / "completed_depth.npy")
    np.testing.assert_array_equal(completed[mask == 0], depth[mask == 0])
    np.testing.assert_allclose(completed[mask > 0], 4)
    assert (stage / "report/overview.png").is_file()
    assert result["metrics"]["valid_prediction_fraction"] == 1
    assert len(verify_run(run)["stages"]) == 3


def test_model_worker_does_not_inherit_repository_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    log = tmp_path / "worker.log"
    e1_run._run_logged(
        [sys.executable, "-c", "import os; print(os.getenv('PYTHONPATH'))"],
        cwd=tmp_path, log=log,
    )
    assert log.read_text().strip() == "None"
