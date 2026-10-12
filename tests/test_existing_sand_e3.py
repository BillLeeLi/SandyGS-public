"""E3 must preserve PGSR vertex bytes, index order, resources, and review binding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import pytest
import yaml

from sandygs.existing_sand.e3 import derive_scene, plan_derivation
from sandygs.existing_sand.workspace import (ExistingSandError, initialize_run,
                                            record_stage, verify_run)

ROOT = Path(__file__).resolve().parents[1]


def _register(run: Path, kind: str, name: str, files: list[str], *,
              prompt: dict | None = None, parents: list[dict] | None = None,
              parameters: dict | None = None):
    stage = run / "stages" / kind / name
    (stage / "record.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "stage": kind, "stage_id": name,
        "camera_ids": ["a.png"], "prompt": prompt or {}, "seed": None,
        "models": [], "parameters": parameters or {}, "commands": [],
        "parents": parents or [],
        "artifacts": [{"path": file, "role": "test"} for file in files],
    }), encoding="utf-8")
    record_stage(run, stage / "record.yaml", repo_root=ROOT)


def _fixture(tmp_path: Path, *, selection_type: str = "multiview_sam_depth_seeds"):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "cameras.txt").write_text("camera", encoding="utf-8")
    source = tmp_path / "original"
    cloud = source / "point_cloud/iteration_30000"
    cloud.mkdir(parents=True)
    fields = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0",
              "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1",
              "rot_2", "rot_3")
    vertices = np.empty(5, dtype=[(key, "<f4") for key in fields])
    for j, key in enumerate(fields):
        vertices[key] = np.arange(5, dtype=np.float32) * 0.125 + j
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(
        str(cloud / "point_cloud.ply"))
    (source / "cfg_args").write_text("Namespace(sh_degree=0, source_path='/old/data', model_path='/old/model', images='images')", encoding="utf-8")
    (source / "env.png").write_bytes(b"environment data")
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump({
        "schema_version": 1, "scene_id": "sand_scene",
        "source": {"dataset": str(dataset), "checkpoint": str(source),
                   "point_cloud": "point_cloud/iteration_30000/point_cloud.ply",
                   "dataset_files": ["cameras.txt"]},
        "camera_ids": ["a.png"], "coordinate": {"sim_to_gs": np.eye(4).tolist()},
    }), encoding="utf-8")
    run = tmp_path / "run_001"
    initialize_run(config, run, run_id="run_001", repo_root=ROOT)
    obs = run / "stages/observations/observed/view_0000"
    obs.mkdir(parents=True)
    (obs / "camera.yaml").write_text(yaml.safe_dump({"camera_id": "a.png",
        "render_size_wh": [4, 3]}), encoding="utf-8")
    for name in ("source_rgb.png", "render_rgb.png"):
        Image.fromarray(np.full((3, 4, 3), 80, np.uint8)).save(obs / name)
    np.save(obs / "plane_depth.npy", np.full((3, 4), 2.0, np.float32))
    np.save(obs / "depth_valid.npy", np.ones((3, 4), bool))
    _register(run, "observations", "observed", [
        f"view_0000/{name}" for name in ("camera.yaml", "source_rgb.png",
        "render_rgb.png", "plane_depth.npy", "depth_valid.npy")])
    mask = run / "stages/segmentation/masks"
    (mask / "view_0000").mkdir(parents=True)
    arr = np.zeros((3, 4), np.uint8)
    arr[0, 0] = 255
    Image.fromarray(arr).save(mask / "view_0000/mask.png")
    (mask / "quality.yaml").write_text(yaml.safe_dump({"views": [
        {"camera_id": "a.png", "role": "holdout", "directory": "view_0000"}]}), encoding="utf-8")
    _register(run, "segmentation", "masks", ["view_0000/mask.png", "quality.yaml"],
              prompt={"type": "manual_sam_points_box"},
              parameters={"capture_id": "observed"},
              parents=[{"stage": "observations", "stage_id": "observed"}])
    selection = run / "stages/segmentation/selected"
    selection.mkdir()
    np.save(selection / "gaussian_labels.npy", np.array([0, 1, 2, 1, 0], np.uint8))
    _register(run, "segmentation", "selected", ["gaussian_labels.npy"],
              prompt={"type": selection_type, "mask_id": "masks"},
              parents=[{"stage": "segmentation", "stage_id": "masks"}])
    check = run / "stages/segmentation/checked"
    check.mkdir()
    (check / "quality.yaml").write_text(yaml.safe_dump({"holdout_views": [
        {"camera_id": "a.png", "mask_iou_vs_sam": 0.9}]}), encoding="utf-8")
    _register(run, "segmentation", "checked", ["quality.yaml"],
              prompt={"type": "pgsr_rasterized_holdout", "seed_id": "selected"},
              parents=[{"stage": "segmentation", "stage_id": "selected"}])
    return run, source, dataset, vertices


def test_derived_scene_exact_partition_and_review(tmp_path):
    run, source, dataset, vertices = _fixture(tmp_path)
    original = (source / "point_cloud/iteration_30000/point_cloud.ply").read_bytes()
    review_path = tmp_path / "review.yaml"
    plan = plan_derivation(run, "selected", "checked", review_path)
    assert plan["counts"] == {"positive": 2, "negative": 2, "unknown": 1}
    with pytest.raises(ExistingSandError, match="not approved"):
        derive_scene(run, "selected", "checked", review_path, "derived", repo_root=ROOT)
    decision = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    decision["approved"] = True
    decision["selected_indices_sha256"] = "bad"
    review_path.write_text(yaml.safe_dump(decision), encoding="utf-8")
    with pytest.raises(ExistingSandError, match="binding changed"):
        derive_scene(run, "selected", "checked", review_path, "derived", repo_root=ROOT)
    decision = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    plan_derivation(run, "selected", "checked", tmp_path / "review2.yaml")
    decision["selected_indices_sha256"] = yaml.safe_load(
        (tmp_path / "review2.yaml").read_text(encoding="utf-8"))["selected_indices_sha256"]
    review_path.write_text(yaml.safe_dump(decision), encoding="utf-8")
    result = derive_scene(run, "selected", "checked", review_path, "derived", repo_root=ROOT, chunk_size=2)
    stage = Path(result["stage_dir"])
    kept = PlyData.read(str(stage / "checkpoint/point_cloud/iteration_30000/point_cloud.ply"))["vertex"].data
    selected = PlyData.read(str(stage / "selected_sand.ply"))["vertex"].data
    np.testing.assert_array_equal(kept, vertices[[0, 2, 4]])
    np.testing.assert_array_equal(selected, vertices[[1, 3]])
    np.testing.assert_array_equal(np.load(stage / "retained_source_indices.npy"), [0, 2, 4])
    np.testing.assert_array_equal(np.load(stage / "selected_source_indices.npy"), [1, 3])
    assert (source / "point_cloud/iteration_30000/point_cloud.ply").read_bytes() == original
    assert (stage / "checkpoint/env.png").read_bytes() == (source / "env.png").read_bytes()
    assert f"source_path='{dataset}'" in (stage / "checkpoint/cfg_args").read_text()
    assert str(stage / "checkpoint") in (stage / "checkpoint/cfg_args").read_text()
    assert "model_path='/old/model'" in (stage / "original_cfg_args.txt").read_text()
    depth = np.load(stage / "references/view_0000/sam_region_pgsr_depth.npy")
    assert depth[0, 0] == 2 and np.isnan(depth[1, 1])
    assert {"stage": "background", "stage_id": "derived"} in verify_run(run)["stages"]
    with pytest.raises(ExistingSandError, match="overwrite"):
        derive_scene(run, "selected", "checked", review_path, "derived", repo_root=ROOT)


def test_failed_derivation_does_not_register_partial_stage(tmp_path, monkeypatch):
    import sandygs.existing_sand.e3 as e3

    run, source, _dataset, _vertices = _fixture(tmp_path)
    review_path = tmp_path / "review.yaml"
    plan_derivation(run, "selected", "checked", review_path)
    review = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    review["approved"] = True
    review_path.write_text(yaml.safe_dump(review), encoding="utf-8")

    def fail_references(*_args):
        raise ExistingSandError("reference copy failed")

    monkeypatch.setattr(e3, "_references", fail_references)
    with pytest.raises(ExistingSandError, match="reference copy failed"):
        derive_scene(run, "selected", "checked", review_path, "failed", repo_root=ROOT)
    assert not (run / "stages/background/failed").exists()
    assert not list(run.glob(".e3_pending_*"))
    assert verify_run(run)["run_id"] == "run_001"
    with pytest.raises(ExistingSandError, match="outside source checkpoint"):
        plan_derivation(run, "selected", "checked", source / "review.yaml")


def test_e3_accepts_frequency_selection_after_holdout_review(tmp_path):
    run, _source, _dataset, _vertices = _fixture(
        tmp_path, selection_type="multiview_sam_frequency")
    review_path = tmp_path / "frequency_review.yaml"
    plan = plan_derivation(run, "selected", "checked", review_path)
    assert plan["counts"] == {"positive": 2, "negative": 2, "unknown": 1}
