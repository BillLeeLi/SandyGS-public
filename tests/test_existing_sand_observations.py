"""Camera-Z and source-observation contracts without a dataset or model download."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

from sandygs.existing_sand.capture import _selected_colmap_cameras
from sandygs.existing_sand.geometry import (
    backproject_camera_z, crop_resize_intrinsics, depth_products,
    project_world, transform_world, validate_pinhole,
)
from sandygs.existing_sand.workspace import ExistingSandError


def test_camera_z_roundtrip_and_coordinate_transform() -> None:
    K = np.array([[100, 0, 2], [0, 80, 1], [0, 0, 1]], dtype=np.float64)
    w2c = np.eye(4)
    w2c[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    w2c[:3, 3] = [3, -2, 1]
    depth = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.float32)
    points = backproject_camera_z(depth, K, w2c)
    pixels, recovered_z = project_world(points, K, w2c)
    v, u = np.indices(depth.shape)
    np.testing.assert_allclose(pixels, np.stack([u, v], axis=-1), atol=1e-12)
    np.testing.assert_allclose(recovered_z, depth, atol=1e-12)
    gs_to_sim = np.eye(4)
    gs_to_sim[:3, :3] *= 2
    gs_to_sim[:3, 3] = [1, 2, 3]
    np.testing.assert_allclose(transform_world(points, gs_to_sim), 2 * points + [1, 2, 3])
    with pytest.raises(ExistingSandError, match="rigid"):
        validate_pinhole(K, gs_to_sim)


def test_depth_validity_preserves_raw_values_and_crop_intrinsics() -> None:
    raw = np.array([[2, np.nan, 0], [-4, 8, 12]], dtype=np.float32)
    alpha = np.array([[1, 1, 1], [1, 0.05, 0.6]], dtype=np.float32)
    depth, valid, inverse = depth_products(raw, alpha, alpha_min=0.1)
    assert np.isnan(depth[0, 1])
    np.testing.assert_array_equal(valid, [[True, False, False], [False, False, True]])
    np.testing.assert_allclose(inverse, [[0.5, 0, 0], [0, 0, 1/12]])
    K = np.array([[120, 0, 8], [0, 100, 6], [0, 0, 1]], dtype=np.float64)
    updated = crop_resize_intrinsics(K, (2, 1, 10, 8), (20, 16))
    np.testing.assert_allclose(updated, [[240, 0, 12.5], [0, 200, 10.5], [0, 0, 1]])
    np.testing.assert_allclose(K, [[120, 0, 8], [0, 100, 6], [0, 0, 1]])
    with pytest.raises(ExistingSandError, match="inverse depth"):
        depth_products(np.array([[1e-45]], dtype=np.float32),
                       np.array([[1]], dtype=np.float32), alpha_min=0.1)


def _write_colmap_minimal(dataset: Path) -> None:
    sparse = dataset / "sparse/0"
    sparse.mkdir(parents=True)
    with (sparse / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 1))
        stream.write(struct.pack("<iiQQ", 1, 1, 8, 6))
        stream.write(struct.pack("<dddd", 12, 10, 4, 3))
    with (sparse / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 1))
        stream.write(struct.pack("<idddddddi", 1, 1, 0, 0, 0, 1, 2, 3, 1))
        stream.write(b"frame_0001.png\x00")
        stream.write(struct.pack("<Q", 0))


def test_colmap_selection_uses_exact_camera_identity_without_writes(tmp_path: Path) -> None:
    _write_colmap_minimal(tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    selected = _selected_colmap_cameras(tmp_path, ["frame_0001.png"], "garden")
    assert len(selected) == 1
    assert selected[0]["pose_mode"] == "colmap_original"
    assert selected[0]["colmap_intrinsics"] == {
        "fx": 12, "fy": 10, "cx": 4, "cy": 3,
    }
    np.testing.assert_allclose(selected[0]["w2c"][:3, 3], [1, 2, 3])
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    aligned = _selected_colmap_cameras(tmp_path, ["frame_0001.png"], "sand_scene")
    assert aligned[0]["pose_mode"] == "pgsr_pca"
    original_c2w = np.linalg.inv(selected[0]["w2c"])
    aligned_c2w = np.linalg.inv(aligned[0]["w2c"])
    np.testing.assert_allclose(
        aligned_c2w, aligned[0]["colmap_to_gs"] @ original_c2w, atol=1e-12
    )
    with pytest.raises(ExistingSandError, match="same COLMAP image"):
        _selected_colmap_cameras(tmp_path, ["frame_0001.png", "frame_0001"], "garden")
    with pytest.raises(ExistingSandError, match="matches 0"):
        _selected_colmap_cameras(tmp_path, ["missing.png"], "garden")
