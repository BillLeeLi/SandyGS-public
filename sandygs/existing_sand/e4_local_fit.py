"""CPU-only, reviewable local tabletop fits from an E4 orthographic capture."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, median_filter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import yaml

from .e4_height import image_yx_to_exchange_xy, orthographic_grid
from .workspace import ExistingSandError, _load_yaml, _stage_location, record_stage


_INPUTS = (
    "camera.yaml", "source_surface_height_yx.npy", "source_valid_yx.npy", "source_alpha.npy",
    "sand_surface_height_yx.npy", "sand_valid_yx.npy", "base_surface_height_yx.npy",
    "base_valid_yx.npy", "base_alpha.npy", "source_rgb.png",
)


def _verified_input(directory: Path, name: str, artifacts: dict[str, dict]) -> Path:
    item = artifacts.get(name)
    if item is None:
        raise ExistingSandError(f"E4 source stage does not register {name}")
    path = directory / name
    if not path.is_file() or path.stat().st_size != item["size_bytes"]:
        raise ExistingSandError(f"missing or changed E4 input: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item["sha256"]:
        raise ExistingSandError(f"E4 input hash mismatch: {path}")
    return path


def _fit_visible_plane(x: np.ndarray, y: np.ndarray, height: np.ndarray,
                       outer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if int(outer.sum()) < 100:
        raise ExistingSandError("E4 local fit needs at least 100 visible outer tabletop pixels")
    design = np.column_stack((x[outer], y[outer], np.ones(int(outer.sum()))))
    values = height[outer].astype(np.float64)
    keep = np.ones(values.size, dtype=bool)
    for _ in range(4):
        if int(keep.sum()) < 100:
            raise ExistingSandError("too few visible tabletop inliers to fit a plane")
        coefficients = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
        residual = values - design @ coefficients
        new_keep = np.abs(residual - np.median(residual[keep])) <= .01
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    coefficients = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
    plane = coefficients[0] * x + coefficients[1] * y + coefficients[2]
    return plane, coefficients


def _harmonic_residual(sand_mask: np.ndarray, anchor: np.ndarray,
                       anchor_residual: np.ndarray) -> np.ndarray:
    """Interpolate only untrusted sand pixels; plane residual is zero outside."""
    unknown = sand_mask & ~anchor
    result = np.zeros(sand_mask.shape, dtype=np.float64)
    result[anchor] = anchor_residual[anchor]
    if not unknown.any():
        return result
    index = np.full(sand_mask.shape, -1, dtype=np.int32)
    index[unknown] = np.arange(int(unknown.sum()), dtype=np.int32)
    rows, cols = np.nonzero(unknown)
    rr, cc, values = [], [], []
    rhs = np.zeros(rows.size, dtype=np.float64)
    height, width = sand_mask.shape
    for k, (row, col) in enumerate(zip(rows, cols)):
        rr.append(k); cc.append(k); values.append(4.0)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue  # zero plane residual at the image border
            neighbor = int(index[nr, nc])
            if neighbor >= 0:
                rr.append(k); cc.append(neighbor); values.append(-1.0)
            elif anchor[nr, nc]:
                rhs[k] += anchor_residual[nr, nc]
            # Outside the selected sand footprint the residual is zero.
    solution = spsolve(csr_matrix((values, (rr, cc)), shape=(rows.size, rows.size)), rhs)
    if not np.isfinite(solution).all():
        raise ExistingSandError("E4 local harmonic fit produced nonfinite heights")
    result[unknown] = solution
    return result


def local_tabletop_candidates(source: np.ndarray, source_valid: np.ndarray,
                              source_alpha: np.ndarray, sand: np.ndarray, sand_valid: np.ndarray,
                              base: np.ndarray, base_valid: np.ndarray, base_alpha: np.ndarray,
                              bounds: tuple[float, float, float, float], *, outer_min: int = 8,
                              outer_max: int = 32, inner_min: int = 32,
                              alpha_min: float = .98, residual_max: float = .02,
                              local_deviation_max: float = .005) -> dict[str, np.ndarray]:
    """Fit visible tabletop and preserve only stable completed-background anchors.

    The sand surface is used only for after-fit evaluation, never as a target
    for the hidden tabletop. No candidate is silently accepted as terrain.
    """
    arrays = (source, source_valid, source_alpha, sand, sand_valid,
              base, base_valid, base_alpha)
    if any(item.shape != source.shape or item.ndim != 2 for item in arrays):
        raise ExistingSandError("E4 local-fit arrays must share one 2D grid")
    if not 0 <= outer_min < outer_max or inner_min <= 0:
        raise ExistingSandError("outer/inner distances must be positive ordered pixels")
    if not 0 < alpha_min < 1 or residual_max <= 0 or local_deviation_max <= 0:
        raise ExistingSandError("invalid E4 local-fit thresholds")
    height, width = source.shape
    (x, y), _ = orthographic_grid(bounds, width, height)
    outer_distance = distance_transform_edt(~sand_valid)
    inner_distance = distance_transform_edt(sand_valid)
    outer = (~sand_valid & source_valid & np.isfinite(source) & (source_alpha >= alpha_min)
             & (outer_distance > outer_min) & (outer_distance <= outer_max))
    plane, coefficients = _fit_visible_plane(x, y, source, outer)
    plane_error = source[outer] - plane[outer]
    if np.quantile(np.abs(plane_error), .95) > residual_max:
        raise ExistingSandError("visible outer tabletop is not sufficiently planar for this comparison")
    residual = base.astype(np.float64) - plane
    local_median = median_filter(np.nan_to_num(residual, nan=0), size=5, mode="nearest")
    anchor = (sand_valid & base_valid & np.isfinite(base) & (base_alpha >= alpha_min)
              & (inner_distance > inner_min) & (np.abs(residual) <= residual_max)
              & (np.abs(residual - local_median) <= local_deviation_max))
    if int(anchor.sum()) < 100:
        raise ExistingSandError("E4 local fit needs at least 100 stable interior pixels")
    correction = _harmonic_residual(sand_valid, anchor, residual)
    exterior = np.where(source_valid & np.isfinite(source), source, base)
    raw = np.where(sand_valid, base, exterior)
    planar = np.where(sand_valid, plane, exterior)
    local = np.where(sand_valid, plane + correction, exterior)
    return {
        "outer": outer, "inner": anchor, "unknown": sand_valid & ~anchor,
        "plane": plane.astype(np.float32), "raw": raw.astype(np.float32),
        "planar": planar.astype(np.float32), "local": local.astype(np.float32),
        "coefficients": coefficients, "plane_error": plane_error,
    }


def _quantiles(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [.01, .5, .99])]


def _comparison_metrics(candidate: np.ndarray, sand: np.ndarray,
                        valid: np.ndarray, cell_spacing: float) -> dict:
    delta = sand[valid].astype(np.float64) - candidate[valid].astype(np.float64)
    negative = delta < 0
    return {
        "valid_pixels": int(valid.sum()), "negative_pixels": int(negative.sum()),
        "negative_fraction": float(negative.mean()),
        "thickness_p01_p50_p99": _quantiles(delta),
        "signed_sand_volume_simulation_units_cubed": float(delta.sum() * cell_spacing ** 2),
        "negative_volume_simulation_units_cubed": float(delta[negative].sum() * cell_spacing ** 2),
    }


def _color_field(values: np.ndarray, valid: np.ndarray, limit: float) -> Image.Image:
    scaled = np.clip(np.nan_to_num(values, nan=0) / limit, -1, 1)
    rgb = np.empty(values.shape + (3,), dtype=np.float32)
    rgb[..., 0] = 235 - 175 * np.maximum(-scaled, 0)
    rgb[..., 1] = 235 - 155 * np.abs(scaled)
    rgb[..., 2] = 235 - 175 * np.maximum(scaled, 0)
    rgb[~valid] = [30, 30, 30]
    return Image.fromarray(np.round(rgb).astype(np.uint8))


def _overview(directory: Path, source_rgb: Image.Image, candidates: dict,
              sand: np.ndarray, sand_valid: np.ndarray) -> None:
    height, width = sand.shape
    overlay = np.asarray(source_rgb.convert("RGB"), dtype=np.float32).copy()
    for mask, color in ((candidates["unknown"], [230, 75, 65]),
                        (candidates["outer"], [255, 195, 35]),
                        (candidates["inner"], [55, 215, 100])):
        overlay[mask] = .48 * overlay[mask] + .52 * np.asarray(color)
    panels = [
        ("anchors: yellow outer, green inner, red fit", Image.fromarray(overlay.astype(np.uint8))),
        ("raw base minus visible-table plane", _color_field(candidates["raw"] - candidates["plane"], sand_valid, .08)),
        ("local fit minus visible-table plane", _color_field(candidates["local"] - candidates["plane"], sand_valid, .08)),
        ("sand minus raw base", _color_field(sand - candidates["raw"], sand_valid, .20)),
        ("sand minus plane base", _color_field(sand - candidates["planar"], sand_valid, .20)),
        ("sand minus local-fit base", _color_field(sand - candidates["local"], sand_valid, .20)),
    ]
    out = Image.new("RGB", (3 * width, 2 * (height + 22)), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    for i, (label, tile) in enumerate(panels):
        px, py = (i % 3) * width, (i // 3) * (height + 22)
        draw.text((px + 3, py + 3), label, fill=(0, 0, 0))
        out.paste(tile, (px, py + 22))
    out.save(directory / "overview.png")


def run_local_tabletop_fit(run_root: Path, source_id: str, stage_id: str, *, repo_root: Path,
                           outer_min: int = 8, outer_max: int = 32, inner_min: int = 32,
                           alpha_min: float = .98, residual_max: float = .02,
                           local_deviation_max: float = .005,
                           command: list[str] | None = None) -> dict:
    run_root = Path(run_root).resolve()
    source_dir = _stage_location(run_root, "modeling", source_id)
    target = _stage_location(run_root, "modeling", stage_id)
    if target.exists():
        raise ExistingSandError(f"refusing to overwrite: {target}")
    manifest = _load_yaml(source_dir / "manifest.yaml", "E4 source manifest")
    if manifest.get("stage") != "modeling" or manifest.get("stage_id") != source_id:
        raise ExistingSandError("E4 source manifest does not match requested stage")
    record = _load_yaml(source_dir / "record.yaml", "E4 source record")
    if record.get("prompt", {}).get("type") != "same_orthographic_sand_background_review":
        raise ExistingSandError("E4 local fit requires an orthographic source stage")
    artifacts = {item["path"]: item for item in manifest["artifacts"]}
    paths = {name: _verified_input(source_dir, name, artifacts) for name in _INPUTS}
    camera = _load_yaml(paths["camera.yaml"], "E4 camera")
    if camera.get("type") != "orthographic":
        raise ExistingSandError("E4 source camera must be orthographic")
    bounds = tuple(float(value) for value in camera["bounds_simulation_xy"])
    arrays = {name: np.load(path, allow_pickle=False) for name, path in paths.items()
              if name.endswith(".npy")}
    candidates = local_tabletop_candidates(
        arrays["source_surface_height_yx.npy"], arrays["source_valid_yx.npy"],
        arrays["source_alpha.npy"], arrays["sand_surface_height_yx.npy"],
        arrays["sand_valid_yx.npy"], arrays["base_surface_height_yx.npy"],
        arrays["base_valid_yx.npy"], arrays["base_alpha.npy"], bounds,
        outer_min=outer_min, outer_max=outer_max, inner_min=inner_min,
        alpha_min=alpha_min, residual_max=residual_max,
        local_deviation_max=local_deviation_max,
    )
    sand = arrays["sand_surface_height_yx.npy"]
    valid = (arrays["sand_valid_yx.npy"] & arrays["base_valid_yx.npy"]
             & np.isfinite(sand) & np.isfinite(arrays["base_surface_height_yx.npy"]))
    spacing = float(camera["cell_spacing_simulation"])
    metrics = {name: _comparison_metrics(candidates[name], sand, valid, spacing)
               for name in ("raw", "planar", "local")}
    metrics["outer_pixels"] = int(candidates["outer"].sum())
    metrics["inner_anchor_pixels"] = int(candidates["inner"].sum())
    metrics["interpolated_pixels"] = int(candidates["unknown"].sum())
    metrics["visible_plane_z_ax_by_c"] = [float(value) for value in candidates["coefficients"]]
    metrics["outer_source_height_mean"] = float(arrays["source_surface_height_yx.npy"][candidates["outer"]].mean())
    metrics["inner_base_height_mean"] = float(arrays["base_surface_height_yx.npy"][candidates["inner"]].mean())
    metrics["outer_plane_abs_residual_p95"] = float(np.quantile(np.abs(candidates["plane_error"]), .95))
    metrics["inner_base_minus_plane_mean"] = float(np.mean(
        (arrays["base_surface_height_yx.npy"] - candidates["plane"])[candidates["inner"]]))
    metrics["status"] = "geometry comparison only; no exchange or checkpoint modification"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = Path(tempfile.mkdtemp(prefix=f".e4_local_{stage_id}_", dir=run_root))
    try:
        for name in ("outer", "inner", "unknown", "plane", "raw", "planar", "local"):
            array = candidates[name]
            np.save(payload / f"{name}_yx.npy", array, allow_pickle=False)
            if name in ("plane", "raw", "planar", "local"):
                np.save(payload / f"{name}_xy.npy", image_yx_to_exchange_xy(array), allow_pickle=False)
        with (payload / "metrics.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(metrics, stream, sort_keys=False, allow_unicode=True)
        _overview(payload, Image.open(paths["source_rgb.png"]), candidates, sand, valid)
        stage_record = {
            "schema_version": 1, "stage": "modeling", "stage_id": stage_id,
            "camera_ids": [], "prompt": {"type": "local_visible_tabletop_geometry_comparison"},
            "seed": None, "models": [],
            "parameters": {"source_stage_id": source_id, "outer_min_pixels": outer_min,
                           "outer_max_pixels": outer_max, "inner_min_pixels": inner_min,
                           "alpha_min": alpha_min, "residual_max": residual_max,
                           "local_deviation_max": local_deviation_max,
                           "axis_conversion": "PGSR raster [row,col] -> exchange [x,y] by transpose"},
            "commands": [command] if command else [], "command_cwd": str(Path(repo_root).resolve()),
            "parents": [{"stage": "modeling", "stage_id": source_id}],
            "artifacts": [{"path": item.name, "role": "e4_local_tabletop_comparison"}
                          for item in sorted(payload.iterdir()) if item.is_file()],
            "code_files": ["sandygs/existing_sand/e4_height.py",
                           "sandygs/existing_sand/e4_local_fit.py",
                           "scripts/experiments/existing_sand_e4_local_fit.py"],
        }
        with (payload / "record.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(stage_record, stream, sort_keys=False, allow_unicode=True)
        payload.rename(target)
        try:
            record_stage(run_root, target / "record.yaml", repo_root=repo_root)
        except BaseException:
            shutil.rmtree(target)
            raise
        return {"stage_dir": str(target), **metrics}
    finally:
        if payload.exists():
            shutil.rmtree(payload)
