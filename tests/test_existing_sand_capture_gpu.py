"""Tiny real PGSR rasterization check for the read-only observation exporter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import struct

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import pytest
import yaml

from sandygs.existing_sand.capture import capture_source_views
from sandygs.existing_sand.workspace import initialize_run, record_stage, verify_run
from sandygs.existing_sand.e2_render import render_holdout


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_one_synthetic_pgsr_view_is_sealed_without_source_writes(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or importlib.util.find_spec("diff_plane_rasterization") is None:
        pytest.skip("CUDA PGSR rasterizer is unavailable")
    dataset = tmp_path / "dataset"
    sparse = dataset / "sparse/0"
    sparse.mkdir(parents=True)
    (dataset / "images").mkdir()
    Image.fromarray(np.full((24, 32, 3), 127, np.uint8)).save(dataset / "images/frame.png")
    with (sparse / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 1))
        stream.write(struct.pack("<iiQQ", 1, 1, 32, 24))
        stream.write(struct.pack("<dddd", 32, 32, 16, 12))
    with (sparse / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 1))
        stream.write(struct.pack("<idddddddi", 1, 1, 0, 0, 0, 0, 0, 0, 1))
        stream.write(b"frame.png\x00")
        stream.write(struct.pack("<Q", 0))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    fields = [(name, "f4") for name in (
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    )]
    vertex = np.zeros(1, dtype=fields)
    vertex["z"] = 3
    vertex["opacity"] = np.log(9)
    vertex["scale_0"] = vertex["scale_1"] = np.log(0.3)
    vertex["scale_2"] = np.log(0.05)
    vertex["rot_0"] = 1
    PlyData([PlyElement.describe(vertex, "vertex")]).write(checkpoint / "point_cloud.ply")
    source_bytes = (checkpoint / "point_cloud.ply").read_bytes()
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump({
        "schema_version": 1, "scene_id": "smoke",
        "source": {"dataset": str(dataset), "checkpoint": str(checkpoint),
                   "point_cloud": "point_cloud.ply",
                   "dataset_files": ["sparse/0/cameras.bin", "sparse/0/images.bin",
                                     "images/frame.png"]},
        "camera_ids": ["frame.png"],
        "coordinate": {"sim_to_gs": np.eye(4).tolist()},
    }), encoding="utf-8")
    run = tmp_path / "run"
    initialize_run(config, run, run_id="smoke", repo_root=REPO_ROOT)
    result = capture_source_views(
        run, "tiny", ["frame.png"], dataset_name="garden", width=32, height=24,
        sh_degree=0, alpha_min=0.1, white_background=False, repo_root=REPO_ROOT,
        command=["python", "synthetic_gpu_test.py"],
    )
    assert result["cameras"][0]["valid_pixels"] > 0
    stage = run / "stages/observations/tiny"
    depth = np.load(stage / "view_0000/plane_depth.npy")
    valid = np.load(stage / "view_0000/depth_valid.npy")
    assert depth.shape == valid.shape == (24, 32)
    np.testing.assert_allclose(depth[valid], 3, atol=1e-5)
    assert (checkpoint / "point_cloud.ply").read_bytes() == source_bytes
    assert verify_run(run)["stages"] == [{"stage": "observations", "stage_id": "tiny"}]


def test_one_gaussian_e2_holdout_rasterization(tmp_path: Path) -> None:
    """Render E2 labels through the original PGSR geometry and opacity."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or importlib.util.find_spec("diff_plane_rasterization") is None:
        pytest.skip("CUDA PGSR rasterizer is unavailable")
    # Reuse the tiny captured source fixture via a dedicated local invocation.
    # The first test sets up its own temporary run, so create the same one by
    # invoking its body with a separate temp root and then extend that run.
    run_root = tmp_path / "capture"
    run_root.mkdir()
    test_one_synthetic_pgsr_view_is_sealed_without_source_writes(run_root)
    run = run_root / "run"
    masks = run / "stages/segmentation/masks"
    (masks / "view_0000").mkdir(parents=True)
    expected = np.zeros((24, 32), dtype=np.uint8)
    expected[7:17, 10:22] = 255
    Image.fromarray(expected).save(masks / "view_0000/mask.png")
    (masks / "quality.yaml").write_text(yaml.safe_dump({"views": [
        {"camera_id": "frame.png", "directory": "view_0000", "role": "holdout"}]}))
    (masks / "record.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "stage": "segmentation", "stage_id": "masks",
        "camera_ids": ["frame.png"], "prompt": {"type": "manual_sam_points_box"},
        "parameters": {"capture_id": "tiny"}, "seed": None, "models": [], "commands": [],
        "parents": [{"stage": "observations", "stage_id": "tiny"}],
        "artifacts": [{"path": file, "role": "fixture"} for file in
                      ("view_0000/mask.png", "quality.yaml")]}))
    record_stage(run, masks / "record.yaml", repo_root=REPO_ROOT)
    seeds = run / "stages/segmentation/seeds"
    seeds.mkdir()
    np.save(seeds / "gaussian_labels.npy", np.array([1], dtype=np.uint8))
    (seeds / "record.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "stage": "segmentation", "stage_id": "seeds",
        "camera_ids": ["frame.png"], "prompt": {"type": "multiview_sam_depth_seeds",
                                              "mask_id": "masks"},
        "parameters": {}, "seed": None, "models": [], "commands": [],
        "parents": [{"stage": "segmentation", "stage_id": "masks"}],
        "artifacts": [{"path": "gaussian_labels.npy", "role": "fixture"}]}))
    record_stage(run, seeds / "record.yaml", repo_root=REPO_ROOT)
    result = render_holdout(run, "seeds", "check", repo_root=REPO_ROOT)
    score = result["holdout_views"][0]
    assert score["predicted_pixels"] > 0
    assert score["mask_iou_vs_sam"] is not None
    assert (run / "stages/segmentation/check/view_0000/overlay.png").is_file()
    assert len(verify_run(run)["stages"]) == 4
