"""Conservative, index-preserving PGSR Gaussian visibility votes for E2.

These votes use projected centers, a scale-dependent border exclusion, and
observed PGSR camera-Z. They are seeds, not a substitute for rasterized
Gaussian coverage: wide splats, depth discontinuities and occlusions stay
unknown.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter

from .workspace import ExistingSandError


UNKNOWN = np.uint8(2)
NEGATIVE = np.uint8(0)


class VoteMaps(NamedTuple):
    high: np.ndarray
    low: np.ndarray
    nearby_valid: np.ndarray
    inside: np.ndarray
    outside: np.ndarray


def prepare_vote_maps(mask: np.ndarray, depth: np.ndarray,
                      valid: np.ndarray) -> VoteMaps:
    numeric = valid & np.isfinite(depth) & (depth > 0)
    return VoteMaps(
        maximum_filter(np.where(numeric, depth, -np.inf), size=3),
        minimum_filter(np.where(numeric, depth, np.inf), size=3),
        minimum_filter(numeric.astype(np.uint8), size=3).astype(bool),
        distance_transform_edt(mask), distance_transform_edt(~mask),
    )
POSITIVE = np.uint8(1)


def source_gaussians(path: Path, count: int):
    """Memory-map the PLY vertex table in its original, immutable order."""
    from plyfile import PlyData

    ply = PlyData.read(str(path), mmap="r")
    vertex = ply["vertex"].data
    required = {"x", "y", "z", "scale_0", "scale_1"}
    if len(vertex) != count or not required.issubset(vertex.dtype.names or ()):
        raise ExistingSandError("source PLY count or PGSR xyz/scale fields changed")
    return vertex


def view_votes(xyz: np.ndarray, scales: np.ndarray, mask: np.ndarray,
               depth: np.ndarray, valid: np.ndarray, K: np.ndarray,
               w2c: np.ndarray, *, depth_absolute: float = 0.025,
               depth_relative: float = 0.015, footprint_sigma: float = 2.0,
               max_footprint_pixels: float = 12.0,
               border_pixels: float = 1.5,
               maps: VoteMaps | None = None) -> np.ndarray:
    """Return 0/1/2 for one view, only voting on clearly visible interiors.

    ``xyz`` and ``scales`` are small *chunks*, never the entire source PLY.
    The support bound is intentionally pessimistic for the flat PGSR splats.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.bool_)
    valid = np.asarray(valid, dtype=np.bool_)
    depth = np.asarray(depth)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or scales.shape != (len(xyz), 2):
        raise ExistingSandError("Gaussian xyz/scales must have N×3 and N×2 shapes")
    if mask.shape != depth.shape or valid.shape != mask.shape:
        raise ExistingSandError("E2 mask, depth and validity grids must match")
    if any(value < 0 or not np.isfinite(value) for value in
           (depth_absolute, depth_relative, footprint_sigma,
            max_footprint_pixels, border_pixels)) or max_footprint_pixels == 0:
        raise ExistingSandError("E2 visibility thresholds must be finite and nonnegative")
    K = np.asarray(K, dtype=np.float64)
    w2c = np.asarray(w2c, dtype=np.float64)
    if K.shape != (3, 3) or w2c.shape != (4, 4) or not np.isfinite(K).all() or not np.isfinite(w2c).all():
        raise ExistingSandError("E2 camera matrices must be finite 3x3 and 4x4")
    result = np.full(len(xyz), UNKNOWN, dtype=np.uint8)
    if not len(xyz):
        return result
    camera = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = camera[:, 2]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        u = K[0, 0] * camera[:, 0] / z + K[0, 2]
        v = K[1, 1] * camera[:, 1] / z + K[1, 2]
        support = np.exp(np.clip(scales.max(axis=1), -30, 30)) * footprint_sigma
        radius = support * max(K[0, 0], K[1, 1]) / z
    height, width = mask.shape
    candidate = (np.isfinite(camera).all(axis=1) & np.isfinite(u) & np.isfinite(v)
                 & np.isfinite(radius) & (z > 0) & (radius <= max_footprint_pixels)
                 & (u >= 1) & (v >= 1) & (u < width - 1) & (v < height - 1))
    indices = np.flatnonzero(candidate)
    if not indices.size:
        return result
    x = np.rint(u[indices]).astype(np.int64)
    y = np.rint(v[indices]).astype(np.int64)
    observed = depth[y, x]
    # A 3×3 depth jump signals silhouettes or a different surface under
    # the center; prevent a foreground/background collision from voting.
    if maps is None:
        maps = prepare_vote_maps(mask, depth, valid)
    margin = depth_absolute + depth_relative * observed
    agreement = (valid[y, x] & np.isfinite(observed) & (observed > 0)
                 & (np.abs(z[indices] - observed) <= margin)
                 & maps.nearby_valid[y, x]
                 & ((maps.high[y, x] - maps.low[y, x]) <= 2 * margin))
    clear = np.where(mask[y, x], maps.inside[y, x], maps.outside[y, x])
    agreed = agreement & (clear > np.maximum(radius[indices], border_pixels))
    result[indices[agreed]] = mask[y[agreed], x[agreed]].astype(np.uint8)
    return result


def combine_votes(positive: np.ndarray, negative: np.ndarray,
                  min_views: int) -> np.ndarray:
    if positive.shape != negative.shape or min_views < 1:
        raise ExistingSandError("E2 vote arrays must match and min_views >= 1")
    labels = np.full(positive.shape, UNKNOWN, dtype=np.uint8)
    labels[(positive >= min_views) & (negative == 0)] = POSITIVE
    labels[(negative >= min_views) & (positive == 0)] = NEGATIVE
    return labels


def view_votes_frequency(xyz: np.ndarray, mask: np.ndarray, depth: np.ndarray,
                         valid: np.ndarray, K: np.ndarray, w2c: np.ndarray, *,
                         depth_absolute: float = 0.025,
                         depth_relative: float = 0.03) -> np.ndarray:
    """Vote at projected centers unless PGSR depth places them behind a surface.

    The captured ``valid`` map already includes the rendered-alpha threshold.
    A one-sided camera-Z test rejects occluded Gaussians without excluding
    foreground centers whose PGSR plane-depth estimate lies farther away.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.bool_)
    depth = np.asarray(depth)
    valid = np.asarray(valid, dtype=np.bool_)
    K = np.asarray(K, dtype=np.float64)
    w2c = np.asarray(w2c, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ExistingSandError("Gaussian xyz must have N×3 shape")
    if mask.ndim != 2 or mask.shape != depth.shape or valid.shape != mask.shape:
        raise ExistingSandError("E2 mask, depth and validity grids must match")
    if (K.shape != (3, 3) or w2c.shape != (4, 4)
            or not np.isfinite(K).all() or not np.isfinite(w2c).all()):
        raise ExistingSandError("E2 camera matrices must be finite 3x3 and 4x4")
    if (not np.isfinite(depth_absolute) or depth_absolute < 0
            or not np.isfinite(depth_relative) or depth_relative < 0):
        raise ExistingSandError("E2 depth thresholds must be finite and nonnegative")
    result = np.full(len(xyz), UNKNOWN, dtype=np.uint8)
    if not len(xyz):
        return result
    camera = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = camera[:, 2]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        u = K[0, 0] * camera[:, 0] / z + K[0, 2]
        v = K[1, 1] * camera[:, 1] / z + K[1, 2]
    height, width = mask.shape
    candidate = (np.isfinite(camera).all(axis=1) & np.isfinite(u) & np.isfinite(v)
                 & (z > 0) & (u >= 0) & (v >= 0)
                 & (u < width - 0.5) & (v < height - 0.5))
    indices = np.flatnonzero(candidate)
    if not indices.size:
        return result
    x = np.rint(u[indices]).astype(np.int64)
    y = np.rint(v[indices]).astype(np.int64)
    observed = depth[y, x]
    visible = (valid[y, x] & np.isfinite(observed) & (observed > 0)
               & (z[indices] <= observed + depth_absolute + depth_relative * observed))
    result[indices[visible]] = mask[y[visible], x[visible]].astype(np.uint8)
    return result


def frequency_labels(positive: np.ndarray, visible: np.ndarray,
                     threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Classify every observed Gaussian; zero-view Gaussians remain unknown."""
    positive = np.asarray(positive)
    visible = np.asarray(visible)
    if (positive.shape != visible.shape or positive.ndim != 1
            or positive.dtype.kind not in "iu" or visible.dtype.kind not in "iu"
            or np.any(positive > visible) or not np.isfinite(threshold)
            or not 0 < threshold <= 1):
        raise ExistingSandError("invalid E2 frequency votes or sand-fraction threshold")
    fraction = np.full(positive.shape, np.nan, dtype=np.float32)
    np.divide(positive, visible, out=fraction, where=visible > 0)
    labels = np.full(positive.shape, UNKNOWN, dtype=np.uint8)
    labels[visible > 0] = NEGATIVE
    labels[(visible > 0) & (fraction >= threshold)] = POSITIVE
    return labels, fraction
