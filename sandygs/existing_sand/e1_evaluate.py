"""CPU metrics and visual reports for controlled E1 depth/RGB completion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion

from .workspace import ExistingSandError


def _finite_stats(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "median": None, "p90": None}
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90))}


def _normals(depth: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normals from camera-Z and the cropped PGSR pinhole model."""
    height, width = depth.shape
    v, u = np.indices((height, width), dtype=np.float64)
    z = np.asarray(depth, dtype=np.float64)
    xyz = np.stack(((u - K[0, 2]) * z / K[0, 0],
                    (v - K[1, 2]) * z / K[1, 1], z), axis=-1)
    dv = np.gradient(xyz, axis=0)
    du = np.gradient(xyz, axis=1)
    normal = np.cross(du, dv)
    norm = np.linalg.norm(normal, axis=-1)
    good = np.isfinite(normal).all(axis=-1) & np.isfinite(norm) & (norm > 1e-12)
    normal[good] /= norm[good, None]
    normal[~good] = np.nan
    good[[0, -1], :] = False
    good[:, [0, -1]] = False
    return normal, good


def depth_metrics(reference: np.ndarray, predicted: np.ndarray, mask: np.ndarray,
                  K: np.ndarray, *, boundary_pixels: int = 8,
                  raw_predicted: np.ndarray | None = None) -> tuple[dict[str, Any], np.ndarray]:
    """Score predicted PGSR camera-Z against the unmasked PGSR reconstruction."""
    reference = np.asarray(reference, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.bool_)
    K = np.asarray(K, dtype=np.float64)
    if (reference.ndim != 2 or reference.shape != predicted.shape or
            reference.shape != mask.shape or K.shape != (3, 3) or
            not mask.any() or not np.isfinite(reference).all() or
            np.any(reference <= 0)):
        raise ExistingSandError("depth evaluation needs matching valid PGSR arrays and a nonempty hole")
    valid_prediction = np.isfinite(predicted) & (predicted > 0)
    comparison = mask & valid_prediction
    error = np.full(reference.shape, np.nan, dtype=np.float32)
    error[comparison] = np.abs(predicted[comparison] - reference[comparison])
    inner = mask & ~binary_erosion(mask, iterations=boundary_pixels)
    outer = binary_dilation(mask, iterations=boundary_pixels) & ~mask
    diff = predicted[comparison] - reference[comparison]
    normal_ref, valid_ref_normal = _normals(reference, K)
    normal_pred, valid_pred_normal = _normals(predicted, K)
    normal_valid = comparison & valid_ref_normal & valid_pred_normal
    angle = np.full(reference.shape, np.nan, dtype=np.float32)
    if normal_valid.any():
        dot = np.sum(normal_ref[normal_valid] * normal_pred[normal_valid], axis=-1)
        angle[normal_valid] = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    hole_abs = np.abs(diff)
    metrics: dict[str, Any] = {
        "reference": "unmasked_pgsr_plane_depth_camera_z_not_sensor_truth",
        "hole_pixels": int(mask.sum()),
        "valid_prediction_pixels": int(comparison.sum()),
        "valid_prediction_fraction": float(comparison.sum() / mask.sum()),
        "depth_mae": float(hole_abs.mean()) if diff.size else None,
        "depth_rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else None,
        "depth_abs_relative_mean": float(np.mean(hole_abs / reference[comparison])) if diff.size else None,
        "depth_error_p90": float(np.percentile(hole_abs, 90)) if diff.size else None,
        "inner_boundary_pixels": int(inner.sum()),
        "inner_boundary_mae": float(np.mean(error[inner & comparison]))
                              if (inner & comparison).any() else None,
        "outer_boundary_pixels": int(outer.sum()),
        "normal_angle_deg": _finite_stats(angle[mask]),
        "normal_inner_angle_deg": _finite_stats(angle[inner]),
    }
    if raw_predicted is not None:
        raw = np.asarray(raw_predicted, dtype=np.float64)
        if raw.shape != reference.shape:
            raise ExistingSandError("raw predicted depth has the wrong shape")
        raw_known = outer & np.isfinite(raw) & (raw > 0)
        metrics["raw_known_outer_mae"] = (
            float(np.mean(np.abs(raw[raw_known] - reference[raw_known])))
            if raw_known.any() else None
        )
    return metrics, angle


def rgb_metrics(reference: np.ndarray, generated: np.ndarray,
                mask: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if (reference.ndim != 3 or reference.shape != generated.shape or
            reference.shape[:2] != mask.shape or reference.shape[2] != 3 or
            not np.isfinite(reference).all() or not np.isfinite(generated).all()):
        raise ExistingSandError("RGB evaluation needs matching finite HxWx3 arrays")
    diff = generated[mask] - reference[mask]
    mse = float(np.mean(diff * diff))
    result: dict[str, Any] = {
        "rgb_hole_mae": float(np.mean(np.abs(diff))),
        "rgb_hole_psnr_db": float(-10 * np.log10(mse)) if mse > 0 else None,
        "rgb_hole_psnr_infinite": mse == 0,
    }
    from skimage.metrics import structural_similarity
    _, ssim_map = structural_similarity(
        reference.astype(np.float32), generated.astype(np.float32),
        channel_axis=-1, data_range=1.0, full=True,
    )
    result["rgb_hole_ssim_local_mean"] = float(np.mean(ssim_map[mask]))
    return result


def _depth_preview(depth: np.ndarray, reference: np.ndarray) -> Image.Image:
    valid_ref = reference[np.isfinite(reference) & (reference > 0)]
    low, high = np.percentile(valid_ref, [2, 98])
    if high <= low:
        high = low + 1e-6
    normalized = np.clip((depth - low) / (high - low), 0, 1)
    # Fixed color mapping for reference and prediction. Blue is near, yellow far.
    rgb = np.stack((normalized, np.sin(np.pi * normalized) * 0.85,
                    1 - normalized), axis=-1)
    rgb[~(np.isfinite(depth) & (depth > 0))] = 0
    return Image.fromarray(np.round(rgb * 255).astype(np.uint8))


def _error_preview(error: np.ndarray, mask: np.ndarray,
                   scale: float) -> Image.Image:
    normalized = np.clip(np.nan_to_num(error, nan=0, posinf=1e9) / max(scale, 1e-9), 0, 1)
    rgb = np.zeros(error.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = np.round(255 * normalized).astype(np.uint8)
    rgb[..., 1] = np.round(200 * (1 - normalized)).astype(np.uint8)
    rgb[~mask] = [30, 30, 30]
    return Image.fromarray(rgb)


def save_depth_report(output_dir: Path, reference: np.ndarray, predicted: np.ndarray,
                      completed: np.ndarray, mask: np.ndarray, K: np.ndarray,
                      rgb: np.ndarray, *, raw_predicted: np.ndarray | None = None,
                      completed_rgb: np.ndarray | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics, normal_angle = depth_metrics(reference, completed, mask, K,
                                          raw_predicted=raw_predicted)
    if completed_rgb is not None:
        metrics["rgb"] = rgb_metrics(rgb, completed_rgb, mask)
    import yaml
    with (output_dir / "metrics.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(metrics, stream, sort_keys=False, allow_unicode=True)
    depth_error = np.full(reference.shape, np.nan, dtype=np.float32)
    valid = mask & np.isfinite(completed) & (completed > 0)
    depth_error[valid] = np.abs(completed[valid] - reference[valid])
    np.save(output_dir / "depth_abs_error.npy", depth_error, allow_pickle=False)
    np.save(output_dir / "normal_angle_deg.npy", normal_angle, allow_pickle=False)
    finite_error = depth_error[np.isfinite(depth_error)]
    scale = float(np.percentile(finite_error, 95)) if finite_error.size else 1.0
    images = {
        "reference_depth.png": _depth_preview(reference, reference),
        "predicted_depth.png": _depth_preview(predicted, reference),
        "completed_depth.png": _depth_preview(completed, reference),
        "depth_error.png": _error_preview(depth_error, mask, scale),
        "normal_error.png": _error_preview(normal_angle, mask, 30.0),
    }
    for name, image in images.items():
        image.save(output_dir / name)
    rgb_byte = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    top = Image.fromarray(rgb_byte)
    if completed_rgb is not None:
        rgb_pred = Image.fromarray(np.round(np.clip(completed_rgb, 0, 1) * 255).astype(np.uint8))
    else:
        rgb_pred = top.copy()
    panels = [top, rgb_pred, images["reference_depth.png"],
              images["completed_depth.png"], images["depth_error.png"],
              images["normal_error.png"]]
    width, height = top.size
    canvas = Image.new("RGB", (width * 3, height * 2 + 44), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["PGSR RGB", "Condition / completed RGB", "PGSR reference Z",
              "Completed Z", "Absolute Z error", "Normal angle error"]
    for index, (panel, label) in enumerate(zip(panels, labels)):
        column, row = index % 3, index // 3
        y = row * (height + 22)
        canvas.paste(panel, (column * width, y + 22))
        draw.text((column * width + 5, y + 3), label, fill="black")
    canvas.save(output_dir / "overview.png")
    return metrics
