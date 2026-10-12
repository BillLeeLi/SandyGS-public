"""Compare selected-only and full-scene PGSR sand-top heights on one E4 grid."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, median_filter
import yaml

from .e4_height import image_yx_to_exchange_xy
from .e4_local_fit import _verified_input
from .workspace import ExistingSandError, _load_yaml, _stage_location, record_stage


_ORTHO_FILES = (
    "source_surface_height_yx.npy", "source_valid_yx.npy", "sand_surface_height_yx.npy",
    "sand_valid_yx.npy", "sand_alpha.npy", "base_valid_yx.npy", "source_rgb.png", "camera.yaml",
)
_BASE_FILES = ("local_yx.npy",)


def compare_sand_surfaces(source: np.ndarray, selected: np.ndarray, selected_alpha: np.ndarray,
                          support: np.ndarray, *, alpha_min: float = .98,
                          agreement_max: float = .02, core_distance: float = 6.0,
                          spike_max: float = .05) -> dict[str, np.ndarray]:
    """Build a diagnostic hybrid; retain the original signed heights for review.

    The selected-only mask defines semantic support. The full-scene top is used
    where selected-only rendering is unreliable. A median replaces isolated
    full-scene spikes, but no negative thickness is clamped here.
    """
    if any(array.shape != source.shape or array.ndim != 2
           for array in (selected, selected_alpha, support)):
        raise ExistingSandError("E4 sand-surface arrays must share one 2D grid")
    if not 0 < alpha_min < 1 or agreement_max <= 0 or core_distance < 0 or spike_max <= 0:
        raise ExistingSandError("invalid E4 sand-surface confidence thresholds")
    if not np.isfinite(source[support]).all() or not np.isfinite(selected[support]).all():
        raise ExistingSandError("E4 sand support includes nonfinite source or selected height")
    distance = distance_transform_edt(support)
    median_source = median_filter(source, size=5, mode="nearest")
    source_spike = support & (np.abs(source - median_source) > spike_max)
    cleaned_source = source.copy()
    cleaned_source[source_spike] = median_source[source_spike]
    agreement = np.abs(source - selected)
    trusted_selected = (support & (selected_alpha >= alpha_min)
                        & (distance > core_distance) & (agreement <= agreement_max)
                        & ~source_spike)
    hybrid = np.where(trusted_selected, selected, cleaned_source)
    return {
        "selected": selected.astype(np.float32), "full": source.astype(np.float32),
        "cleaned_full": cleaned_source.astype(np.float32), "hybrid": hybrid.astype(np.float32),
        "trusted_selected": trusted_selected, "source_spike": source_spike,
        "distance": distance.astype(np.float32), "agreement": agreement.astype(np.float32),
    }


def _metrics(top: np.ndarray, base: np.ndarray, valid: np.ndarray, spacing: float) -> dict:
    thickness = top[valid].astype(np.float64) - base[valid].astype(np.float64)
    negative = thickness[thickness < 0]
    return {
        "valid_pixels": int(valid.sum()),
        "negative_pixels": int(negative.size),
        "negative_below_minus_002_pixels": int((thickness < -.02).sum()),
        "negative_p01_p50_p99": ([float(x) for x in np.quantile(negative, [.01, .5, .99])]
                                 if negative.size else None),
        "minimum_thickness": float(np.min(thickness)),
        "maximum_thickness": float(np.max(thickness)),
        "thickness_p01_p50_p99": [float(x) for x in np.quantile(thickness, [.01, .5, .99])],
        "signed_volume_simulation_units_cubed": float(thickness.sum() * spacing ** 2),
        "positive_volume_simulation_units_cubed": float(np.maximum(thickness, 0).sum() * spacing ** 2),
    }


def _signed_image(values: np.ndarray, valid: np.ndarray, limit: float) -> Image.Image:
    scaled = np.clip(np.nan_to_num(values, nan=0) / limit, -1, 1)
    rgb = np.empty(values.shape + (3,), dtype=np.float32)
    rgb[..., 0] = 235 - 175 * np.maximum(-scaled, 0)
    rgb[..., 1] = 235 - 155 * np.abs(scaled)
    rgb[..., 2] = 235 - 175 * np.maximum(scaled, 0)
    rgb[~valid] = [28, 28, 28]
    return Image.fromarray(np.round(rgb).astype(np.uint8))


def _overview(destination: Path, source_rgb: Path, candidates: dict,
              base: np.ndarray, support: np.ndarray, valid: np.ndarray) -> None:
    rgb = np.asarray(Image.open(source_rgb).convert("RGB"), dtype=np.float32).copy()
    confidence = rgb.copy()
    low = support & ~candidates["trusted_selected"]
    confidence[low] = .5 * confidence[low] + .5 * np.array([255, 185, 30])
    confidence[candidates["trusted_selected"]] = (
        .5 * confidence[candidates["trusted_selected"]] + .5 * np.array([50, 220, 90]))
    confidence[candidates["source_spike"]] = [235, 30, 180]
    panels = [
        ("original scene RGB", Image.fromarray(rgb.astype(np.uint8))),
        ("selected-only sand minus local base", _signed_image(candidates["selected"] - base, valid, .20)),
        ("full-scene top minus local base", _signed_image(candidates["full"] - base, valid, .20)),
        ("full minus selected top", _signed_image(candidates["full"] - candidates["selected"], valid, .08)),
        ("hybrid sand minus local base", _signed_image(candidates["hybrid"] - base, valid, .20)),
        ("green selected, yellow full, pink spike", Image.fromarray(confidence.astype(np.uint8))),
    ]
    height, width = base.shape
    output = Image.new("RGB", (3 * width, 2 * (height + 22)), (255, 255, 255))
    draw = ImageDraw.Draw(output)
    for index, (name, tile) in enumerate(panels):
        x, y = (index % 3) * width, (index // 3) * (height + 22)
        draw.text((x + 3, y + 3), name, fill=(0, 0, 0))
        output.paste(tile, (x, y + 22))
    output.save(destination / "overview.png")


def run_sand_surface_comparison(run_root: Path, ortho_id: str, base_id: str,
                                stage_id: str, *, repo_root: Path,
                                alpha_min: float = .98, agreement_max: float = .02,
                                core_distance: float = 6.0, spike_max: float = .05,
                                command: list[str] | None = None) -> dict:
    run_root = Path(run_root).resolve()
    ortho_dir = _stage_location(run_root, "modeling", ortho_id)
    base_dir = _stage_location(run_root, "modeling", base_id)
    target = _stage_location(run_root, "modeling", stage_id)
    if target.exists():
        raise ExistingSandError(f"refusing to overwrite: {target}")
    ortho_manifest = _load_yaml(ortho_dir / "manifest.yaml", "E4 orthographic manifest")
    base_manifest = _load_yaml(base_dir / "manifest.yaml", "E4 local base manifest")
    if ortho_manifest.get("stage_id") != ortho_id or base_manifest.get("stage_id") != base_id:
        raise ExistingSandError("E4 input manifest ID mismatch")
    ortho_record = _load_yaml(ortho_dir / "record.yaml", "E4 orthographic record")
    base_record = _load_yaml(base_dir / "record.yaml", "E4 local base record")
    if ortho_record.get("prompt", {}).get("type") != "same_orthographic_sand_background_review":
        raise ExistingSandError("E4 sand-surface comparison requires an orthographic stage")
    if (base_record.get("prompt", {}).get("type") != "local_visible_tabletop_geometry_comparison"
            or base_record.get("parameters", {}).get("source_stage_id") != ortho_id):
        raise ExistingSandError("E4 local base must derive from the requested orthographic stage")
    ortho_artifacts = {item["path"]: item for item in ortho_manifest["artifacts"]}
    base_artifacts = {item["path"]: item for item in base_manifest["artifacts"]}
    ortho_paths = {name: _verified_input(ortho_dir, name, ortho_artifacts) for name in _ORTHO_FILES}
    base_paths = {name: _verified_input(base_dir, name, base_artifacts) for name in _BASE_FILES}
    arrays = {name: np.load(path, allow_pickle=False) for name, path in ortho_paths.items()
              if name.endswith(".npy")}
    base = np.load(base_paths["local_yx.npy"], allow_pickle=False)
    support = arrays["sand_valid_yx.npy"].astype(bool)
    valid = (support & arrays["source_valid_yx.npy"] & arrays["base_valid_yx.npy"]
             & np.isfinite(base) & np.isfinite(arrays["sand_surface_height_yx.npy"])
             & np.isfinite(arrays["source_surface_height_yx.npy"]))
    if int(valid.sum()) < 100 or base.shape != support.shape:
        raise ExistingSandError("E4 sand-surface inputs have no common valid grid")
    camera = _load_yaml(ortho_paths["camera.yaml"], "E4 orthographic camera")
    spacing = float(camera["cell_spacing_simulation"])
    candidates = compare_sand_surfaces(
        arrays["source_surface_height_yx.npy"], arrays["sand_surface_height_yx.npy"],
        arrays["sand_alpha.npy"], support,
        alpha_min=alpha_min, agreement_max=agreement_max,
        core_distance=core_distance, spike_max=spike_max,
    )
    metrics = {name: _metrics(candidates[name], base, valid, spacing)
               for name in ("selected", "full", "cleaned_full", "hybrid")}
    metrics["selected_support_pixels"] = int(support.sum())
    metrics["trusted_selected_pixels"] = int((valid & candidates["trusted_selected"]).sum())
    metrics["full_scene_fallback_pixels"] = int((valid & ~candidates["trusted_selected"]).sum())
    metrics["full_scene_spike_pixels"] = int((valid & candidates["source_spike"]).sum())
    metrics["agreement_p50_p95_p99"] = [float(x) for x in np.quantile(
        candidates["agreement"][valid], [.5, .95, .99])]
    metrics["status"] = "sand-top comparison only; no terrain, exchange, or checkpoint modification"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = Path(tempfile.mkdtemp(prefix=f".e4_sand_{stage_id}_", dir=run_root))
    try:
        for name in ("selected", "full", "cleaned_full", "hybrid"):
            top = candidates[name]
            thickness = np.where(valid, top - base, np.nan).astype(np.float32)
            np.save(payload / f"{name}_top_yx.npy", top, allow_pickle=False)
            np.save(payload / f"{name}_thickness_yx.npy", thickness, allow_pickle=False)
            np.save(payload / f"{name}_thickness_xy.npy", image_yx_to_exchange_xy(thickness), allow_pickle=False)
        for name in ("trusted_selected", "source_spike", "distance", "agreement"):
            np.save(payload / f"{name}_yx.npy", candidates[name], allow_pickle=False)
        with (payload / "metrics.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(metrics, stream, sort_keys=False, allow_unicode=True)
        _overview(payload, ortho_paths["source_rgb.png"], candidates, base, support, valid)
        record = {
            "schema_version": 1, "stage": "modeling", "stage_id": stage_id,
            "camera_ids": [], "prompt": {"type": "selected_vs_full_pg_sand_surface_comparison"},
            "seed": None, "models": [],
            "parameters": {"orthographic_stage_id": ortho_id, "local_base_stage_id": base_id,
                           "alpha_min": alpha_min, "agreement_max": agreement_max,
                           "core_distance_pixels": core_distance, "source_spike_max": spike_max,
                           "median_window_pixels": 5,
                           "axis_conversion": "PGSR raster [row,col] -> exchange [x,y] by transpose"},
            "commands": [command] if command else [], "command_cwd": str(Path(repo_root).resolve()),
            "parents": [{"stage": "modeling", "stage_id": ortho_id},
                        {"stage": "modeling", "stage_id": base_id}],
            "artifacts": [{"path": path.name, "role": "e4_sand_surface_comparison"}
                          for path in sorted(payload.iterdir()) if path.is_file()],
            "code_files": ["sandygs/existing_sand/e4_height.py",
                           "sandygs/existing_sand/e4_local_fit.py",
                           "sandygs/existing_sand/e4_sand_surface.py",
                           "scripts/experiments/existing_sand_e4_sand_surface.py"],
        }
        with (payload / "record.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(record, stream, sort_keys=False, allow_unicode=True)
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
