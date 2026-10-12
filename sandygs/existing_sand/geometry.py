"""CPU geometry and depth contracts for PGSR source observations.

Pixels use integer (u, v) centers with u right and v down. ``plane_depth`` is
positive camera-Z, not Euclidean ray distance. Matrices act on column vectors;
``w2c`` maps the PGSR Gaussian world into OpenCV-style camera coordinates.
"""

from __future__ import annotations

import numpy as np

from .workspace import ExistingSandError


def validate_pinhole(K: np.ndarray, w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Check the exact pinhole and rigid camera convention used by this adapter."""
    K = np.asarray(K, dtype=np.float64)
    w2c = np.asarray(w2c, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ExistingSandError("camera K must be a finite 3x3 matrix")
    if not np.allclose(K[2], [0, 0, 1], rtol=0, atol=1e-12) or (
        abs(K[0, 1]) > 1e-12 or abs(K[1, 0]) > 1e-12
    ) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ExistingSandError("camera K must have positive fx/fy and no skew")
    if w2c.shape != (4, 4) or not np.isfinite(w2c).all():
        raise ExistingSandError("camera w2c must be a finite 4x4 matrix")
    rotation = w2c[:3, :3]
    if not np.allclose(w2c[3], [0, 0, 0, 1], rtol=0, atol=1e-12) or (
        not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-5)
    ) or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-5):
        raise ExistingSandError("camera w2c must be a rigid right-handed transform")
    return K, w2c


def depth_products(plane_depth: np.ndarray, rendered_alpha: np.ndarray,
                   *, alpha_min: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw float32 depth, numeric validity and zero-filled inverse depth."""
    depth = np.asarray(plane_depth)
    alpha = np.asarray(rendered_alpha)
    if depth.ndim != 2 or alpha.shape != depth.shape or depth.size == 0:
        raise ExistingSandError("plane_depth and rendered_alpha must be matching HxW arrays")
    if depth.dtype.kind not in "fiu" or alpha.dtype.kind not in "fiu":
        raise ExistingSandError("plane_depth and rendered_alpha must be numeric")
    if not np.isfinite(alpha_min) or not 0 < alpha_min <= 1:
        raise ExistingSandError("alpha_min must be in (0, 1]")
    if not np.isfinite(alpha).all() or np.any((alpha < -1e-5) | (alpha > 1 + 1e-5)):
        raise ExistingSandError("rendered_alpha must be finite and within [0, 1]")
    depth = depth.astype(np.float32, copy=True)
    alpha = alpha.astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > 0) & (alpha >= alpha_min)
    inverse = np.zeros_like(depth, dtype=np.float32)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        inverse[valid] = 1.0 / depth[valid]
    if not np.isfinite(inverse).all():
        raise ExistingSandError("valid depth produces non-finite inverse depth")
    return depth, valid, inverse


def backproject_camera_z(depth: np.ndarray, K: np.ndarray, w2c: np.ndarray,
                         valid: np.ndarray | None = None) -> np.ndarray:
    """Backproject depth to PGSR world; invalid pixels become NaN in HxWx3."""
    K, w2c = validate_pinhole(K, w2c)
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2 or depth.size == 0:
        raise ExistingSandError("depth must be a non-empty HxW array")
    good = np.isfinite(depth) & (depth > 0)
    if valid is not None:
        valid = np.asarray(valid)
        if valid.shape != depth.shape or valid.dtype != np.bool_:
            raise ExistingSandError("valid must be a bool HxW array")
        good &= valid
    height, width = depth.shape
    v, u = np.indices((height, width), dtype=np.float64)
    camera = np.stack(((u - K[0, 2]) / K[0, 0] * depth,
                       (v - K[1, 2]) / K[1, 1] * depth, depth), axis=-1)
    c2w = np.linalg.inv(w2c)
    world = camera @ c2w[:3, :3].T + c2w[:3, 3]
    world[~good] = np.nan
    return world


def project_world(points: np.ndarray, K: np.ndarray,
                  w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project PGSR world points into image pixels and positive camera-Z."""
    K, w2c = validate_pinhole(K, w2c)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ExistingSandError("points must have a final dimension of three")
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    depth = camera[..., 2]
    pixels = np.full(points.shape[:-1] + (2,), np.nan, dtype=np.float64)
    good = np.isfinite(camera).all(axis=-1) & (depth > 0)
    pixels[..., 0][good] = K[0, 0] * camera[..., 0][good] / depth[good] + K[0, 2]
    pixels[..., 1][good] = K[1, 1] * camera[..., 1][good] / depth[good] + K[1, 2]
    return pixels, depth


def transform_world(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine coordinate transform to points with final axis 3."""
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if points.shape[-1] != 3 or transform.shape != (4, 4) or (
        not np.isfinite(transform).all()
    ) or not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ExistingSandError("points need xyz and transform must be a finite affine 4x4")
    return points @ transform[:3, :3].T + transform[:3, 3]


def crop_resize_intrinsics(K: np.ndarray, crop_xywh: tuple[int, int, int, int],
                           output_wh: tuple[int, int]) -> np.ndarray:
    """Update K for a crop and center-aligned resize without changing source K."""
    K, _ = validate_pinhole(K, np.eye(4))
    x, y, width, height = crop_xywh
    out_width, out_height = output_wh
    if any(not isinstance(item, (int, np.integer)) for item in
           (x, y, width, height, out_width, out_height)) or (
        x < 0 or y < 0 or min(width, height, out_width, out_height) <= 0
    ):
        raise ExistingSandError("crop and output size must contain valid integer pixels")
    sx, sy = out_width / width, out_height / height
    result = K.copy()
    result[0, 0] *= sx
    result[1, 1] *= sy
    result[0, 2] = (K[0, 2] - x + 0.5) * sx - 0.5
    result[1, 2] = (K[1, 2] - y + 0.5) * sy - 0.5
    return result
