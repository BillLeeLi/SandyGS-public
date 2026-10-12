"""Synthetic visible-ground holes for the E1 PGSR depth experiment.

The source of truth is PGSR camera-Z.  InFusion only receives a derived,
masked inverse-depth image.  This module deliberately has no diffusion or
CUDA imports, so preparation and evaluation work in the normal SandyGS env.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
import yaml

from .geometry import crop_resize_intrinsics, validate_pinhole
from .workspace import (
    ExistingSandError, _load_yaml, _stage_location, record_stage, verify_run,
)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)


def _save_array(directory: Path, name: str, array: np.ndarray,
                artifacts: list[dict[str, str]], role: str) -> None:
    np.save(directory / name, array, allow_pickle=False)
    artifacts.append({"path": name, "role": role})


def _save_image(directory: Path, name: str, array: np.ndarray,
                artifacts: list[dict[str, str]], role: str) -> None:
    Image.fromarray(array).save(directory / name)
    artifacts.append({"path": name, "role": role})


def _read_binary_mask(path: Path, size_wh: tuple[int, int], label: str,
                      threshold: int) -> tuple[np.ndarray, int]:
    if path.suffix.lower() != ".png":
        raise ExistingSandError(f"{label} must be a lossless PNG")
    with Image.open(path) as image:
        if image.size != size_wh or image.mode not in {"1", "L"}:
            raise ExistingSandError(
                f"{label} must be a grayscale PNG of exactly {size_wh[0]}x{size_wh[1]}"
            )
        pixels = np.asarray(image.convert("L"), dtype=np.uint8)
    ambiguous = int(np.count_nonzero((pixels != 0) & (pixels != 255)))
    return pixels >= threshold, ambiguous


def _observation_view(run_root: Path, capture_id: str,
                      camera_id: str) -> tuple[Path, dict[str, Any]]:
    stage = _stage_location(run_root, "observations", capture_id)
    if not (stage / "manifest.yaml").is_file():
        raise ExistingSandError(f"unregistered observations stage: {stage}")
    matches = []
    for path in sorted(stage.glob("view_*/camera.yaml")):
        camera = _load_yaml(path, "observation camera")
        if camera.get("camera_id") == camera_id:
            matches.append((path.parent, camera))
    if len(matches) != 1:
        raise ExistingSandError(
            f"camera_id {camera_id!r} matches {len(matches)} observation views"
        )
    return matches[0]


def _crop(crop_xywh: tuple[int, int, int, int], size_wh: tuple[int, int]) -> tuple[slice, slice]:
    x, y, width, height = crop_xywh
    source_width, source_height = size_wh
    if any(isinstance(v, bool) or not isinstance(v, int) for v in crop_xywh) or (
        x < 0 or y < 0 or width <= 0 or height <= 0 or
        x + width > source_width or y + height > source_height
    ):
        raise ExistingSandError("crop must be an in-bounds integer x,y,width,height rectangle")
    # InFusion's depth helper independently resizes disparity to max edge 768.
    # The first controlled E1 experiment keeps its factor exactly one.
    if max(width, height) != 768 or width % 8 or height % 8:
        raise ExistingSandError("E1 crop must have max edge 768 and dimensions divisible by 8")
    return slice(y, y + height), slice(x, x + width)


def _model_inverse_depth(depth: np.ndarray, mask: np.ndarray,
                         *, ring_pixels: int = 16) -> tuple[np.ndarray, dict[str, float]]:
    """Build a leak-free InFusion input from a valid PGSR camera-Z crop."""
    if depth.shape != mask.shape or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ExistingSandError("E1 crop needs finite positive PGSR depth at every pixel")
    known = ~mask
    ring = binary_dilation(mask, iterations=ring_pixels) & known
    if not ring.any():
        raise ExistingSandError("the hole needs a known-depth context ring")
    inverse = np.reciprocal(depth, dtype=np.float64).astype(np.float32)
    fill = float(np.median(inverse[ring]))
    low = float(inverse[known].min())
    high = float(inverse[known].max())
    inverse[mask] = fill
    if not np.isfinite(inverse).all() or np.any(inverse <= 0):
        raise ExistingSandError("inverse-depth preparation produced invalid values")
    return inverse, {
        "known_inverse_min": low, "known_inverse_max": high,
        "hole_fill_inverse": fill, "fill_ring_pixels": ring_pixels,
        "infusion_resize_scale": 1.0,
        "infusion_normalization_epsilon": 1e-5,
    }


def prepare_hole(run_root: Path, capture_id: str, camera_id: str,
                 annotation: Path, stage_id: str,
                 crop_xywh: tuple[int, int, int, int], *, repo_root: Path,
                 protect_mask: Path | None = None, threshold: int = 128,
                 command: list[str] | None = None) -> dict[str, Any]:
    """Register a hand-painted synthetic hole and its leakage-free model inputs."""
    run_root = Path(run_root).resolve()
    repo_root = Path(repo_root).resolve()
    verify_run(run_root)
    if not isinstance(threshold, int) or not 1 <= threshold <= 255:
        raise ExistingSandError("mask threshold must be an integer in [1, 255]")
    view_dir, camera = _observation_view(run_root, capture_id, camera_id)
    size_wh = tuple(camera["render_size_wh"])
    rows, cols = _crop(crop_xywh, size_wh)
    authored, ambiguous = _read_binary_mask(Path(annotation), size_wh, "annotation", threshold)
    if protect_mask is not None:
        protected, protect_ambiguous = _read_binary_mask(
            Path(protect_mask), size_wh, "protect mask", threshold
        )
    else:
        protected = np.zeros_like(authored)
        protect_ambiguous = 0
    edit_full = authored & ~protected
    edit = edit_full[rows, cols]
    if not edit.any() or edit.all():
        raise ExistingSandError("hole must contain pixels and leave known ground in the crop")
    if int(edit_full.sum()) != int(edit.sum()):
        raise ExistingSandError("the crop must contain the entire unprotected annotation")
    depth = np.load(view_dir / "plane_depth.npy", allow_pickle=False)[rows, cols]
    valid = np.load(view_dir / "depth_valid.npy", allow_pickle=False)[rows, cols]
    rgb = np.load(view_dir / "render_rgb.npy", allow_pickle=False)[rows, cols]
    if valid.shape != edit.shape or valid.dtype != np.bool_ or not valid.all():
        raise ExistingSandError(
            "E1 crop includes invalid PGSR depth; choose a fully valid ground crop"
        )
    if rgb.shape != edit.shape + (3,) or not np.isfinite(rgb).all():
        raise ExistingSandError("PGSR RGB and depth do not match the selected crop")
    K = crop_resize_intrinsics(np.asarray(camera["pgsr_K"]), crop_xywh,
                               (crop_xywh[2], crop_xywh[3]))
    w2c = np.asarray(camera["gs_w2c"], dtype=np.float64)
    validate_pinhole(K, w2c)
    inverse, normalization = _model_inverse_depth(depth, edit)
    stage_dir = _stage_location(run_root, "segmentation", stage_id)
    if stage_dir.exists():
        raise ExistingSandError(f"refusing to overwrite E1 hole: {stage_dir}")
    stage_dir.mkdir(parents=True, exist_ok=False)
    artifacts: list[dict[str, str]] = []
    shutil.copy2(annotation, stage_dir / "annotation.png")
    artifacts.append({"path": "annotation.png", "role": "authored_hole_mask"})
    if protect_mask is not None:
        shutil.copy2(protect_mask, stage_dir / "protect_mask.png")
        artifacts.append({"path": "protect_mask.png", "role": "authored_protect_mask"})
    _save_array(stage_dir, "edit_mask.npy", edit.astype(np.bool_), artifacts, "edit_mask")
    _save_image(stage_dir, "edit_mask.png", edit.astype(np.uint8) * 255,
                artifacts, "edit_mask_preview")
    _save_array(stage_dir, "reference_depth.npy", depth.astype(np.float32),
                artifacts, "pgsr_camera_z_reference")
    _save_array(stage_dir, "reference_rgb.npy", rgb.astype(np.float32),
                artifacts, "pgsr_rgb_reference")
    _save_array(stage_dir, "input_inverse_depth.npy", inverse,
                artifacts, "masked_inverse_camera_z_model_input")
    _save_array(stage_dir, "camera_K.npy", K, artifacts, "cropped_pgsr_intrinsics")
    _save_array(stage_dir, "camera_c2w.npy", np.linalg.inv(w2c),
                artifacts, "pgsr_camera_to_world")
    rgb_byte = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    _save_image(stage_dir, "condition_rgb.png", rgb_byte, artifacts, "fixed_rgb_condition")
    overlay = rgb_byte.copy()
    overlay[edit] = np.round(0.55 * overlay[edit] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
    _save_image(stage_dir, "mask_overlay.png", overlay, artifacts, "hole_overlay_preview")
    crop_payload = {
        "source_capture_id": capture_id, "source_view": view_dir.name,
        "camera_id": camera_id, "source_size_wh": list(size_wh),
        "crop_xywh": list(crop_xywh), "model_size_wh": [crop_xywh[2], crop_xywh[3]],
        "pixel_centers": "integer_u_right_v_down", "depth": "positive_pgsr_camera_z",
        "annotation_threshold": threshold,
        "annotation_ambiguous_pixels": ambiguous,
        "protect_ambiguous_pixels": protect_ambiguous,
        "hole_pixels": int(edit.sum()),
    }
    _write_yaml(stage_dir / "crop.yaml", crop_payload)
    artifacts.append({"path": "crop.yaml", "role": "crop_and_mask_contract"})
    _write_yaml(stage_dir / "normalization.yaml", normalization)
    artifacts.append({"path": "normalization.yaml", "role": "inverse_depth_normalization"})
    record = {
        "schema_version": 1, "stage": "segmentation", "stage_id": stage_id,
        "camera_ids": [camera_id],
        "prompt": {"type": "manual_synthetic_ground_hole", "annotation": "annotation.png"},
        "seed": None, "models": [],
        "parameters": crop_payload,
        "commands": [command] if command else [],
        "parents": [{"stage": "observations", "stage_id": capture_id}],
        "artifacts": artifacts,
        "code_files": ["sandygs/existing_sand/e1.py",
                       "scripts/experiments/existing_sand_e1.py"],
    }
    _write_yaml(stage_dir / "record.yaml", record)
    record_stage(run_root, stage_dir / "record.yaml", repo_root=repo_root)
    return {"stage_dir": str(stage_dir), "hole_pixels": int(edit.sum()),
            "model_size_wh": crop_payload["model_size_wh"]}
