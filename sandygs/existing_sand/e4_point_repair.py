"""Audit and fill one PGSR-invalid background pixel in an E4 sand field."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageDraw
import yaml

from .e4_height import image_yx_to_exchange_xy, orthographic_grid
from .e4_local_fit import _verified_input
from .workspace import ExistingSandError, _load_yaml, _stage_location, record_stage


def repair_one_background_invalid(thickness: np.ndarray, top: np.ndarray,
                                  local_base: np.ndarray, support: np.ndarray,
                                  original_base_valid: np.ndarray,
                                  row: int, column: int, *,
                                  max_neighbor_difference: float = .01) -> tuple[np.ndarray, dict]:
    """Repair one reviewed interior hole; leave every other thickness unchanged."""
    arrays = (top, local_base, support, original_base_valid)
    if thickness.ndim != 2 or any(item.shape != thickness.shape for item in arrays):
        raise ExistingSandError("E4 point repair inputs must share one 2D grid")
    if not (0 <= row < thickness.shape[0] and 0 <= column < thickness.shape[1]):
        raise ExistingSandError("expected invalid pixel is outside the E4 grid")
    if not np.isfinite(max_neighbor_difference) or max_neighbor_difference <= 0:
        raise ExistingSandError("neighbor agreement tolerance must be positive and finite")
    missing = support & ~np.isfinite(thickness)
    if int(missing.sum()) != 1 or not missing[row, column]:
        raise ExistingSandError("expected exactly the reviewed internal invalid pixel")
    if original_base_valid[row, column] or not np.isfinite(top[row, column]) or not np.isfinite(local_base[row, column]):
        raise ExistingSandError("repair requires invalid original base and finite top/local-fit base")
    if (row in (0, thickness.shape[0] - 1) or column in (0, thickness.shape[1] - 1)
            or not np.all(support[row-1:row+2, column-1:column+2])):
        raise ExistingSandError("repair pixel needs an eight-pixel interior neighborhood")
    neighborhood = thickness[row-1:row+2, column-1:column+2].reshape(-1)
    neighbors = np.delete(neighborhood, 4)
    if not np.isfinite(neighbors).all() or np.any(neighbors < 0):
        raise ExistingSandError("repair neighbors must have finite nonnegative thickness")
    value = float(top[row, column] - local_base[row, column])
    median = float(np.median(neighbors))
    if value < 0 or abs(value - median) > max_neighbor_difference:
        raise ExistingSandError("local-fit repair disagrees with neighboring sand thickness")
    result = thickness.copy()
    result[row, column] = value
    if not np.isfinite(result[support]).all() or np.any(result[support] < 0):
        raise ExistingSandError("repaired sand thickness is not finite and nonnegative on support")
    audit = {
        "pixel_row_column": [int(row), int(column)],
        "hybrid_top_height": float(top[row, column]),
        "local_fitted_base_height": float(local_base[row, column]),
        "filled_thickness": float(result[row, column]),
        "eight_neighbor_thickness_p01_p50_p99": [
            float(x) for x in np.quantile(neighbors, [.01, .5, .99])],
        "difference_from_neighbor_median": float(value - median),
        "other_supported_pixels_bitwise_unchanged": bool(np.array_equal(
            result[support & ~missing], thickness[support & ~missing])),
    }
    return result, audit


def _repair_preview(destination: Path, source_rgb: Path, top: np.ndarray,
                    base: np.ndarray, before: np.ndarray, after: np.ndarray,
                    row: int, column: int) -> None:
    rgb = Image.open(source_rgb).convert("RGB")
    marked = rgb.copy()
    draw = ImageDraw.Draw(marked)
    draw.ellipse((column-4, row-4, column+4, row+4), outline=(255, 0, 180), width=2)
    marked.save(destination / "location.png")
    radius = 8
    box = (column-radius, row-radius, column+radius+1, row+radius+1)
    low, high = -2.65, -2.45
    panels = []
    for name, field, value_range in (
        ("hybrid top", top, (low, high)),
        ("local base", base, (low, high)),
        ("thickness before", before, (0, .20)),
        ("thickness after", after, (0, .20)),
    ):
        patch = field[box[1]:box[3], box[0]:box[2]]
        scale = np.clip((np.nan_to_num(patch, nan=value_range[0]) - value_range[0]) /
                        (value_range[1] - value_range[0]), 0, 1)
        color = np.stack((255 * scale, 255 * (1 - np.abs(2 * scale - 1)),
                          255 * (1 - scale)), axis=-1).astype(np.uint8)
        color[~np.isfinite(patch)] = [25, 25, 25]
        tile = Image.fromarray(color).resize((204, 204), resample=Image.Resampling.NEAREST)
        panels.append((name, tile))
    output = Image.new("RGB", (408, 2 * (204 + 24)), "white")
    draw = ImageDraw.Draw(output)
    for index, (name, tile) in enumerate(panels):
        x, y = (index % 2) * 204, (index // 2) * 228
        draw.text((x + 4, y + 4), name, fill="black")
        output.paste(tile, (x, y + 24))
    output.save(destination / "repair_patch.png")


def run_point_repair(run_root: Path, edge_id: str, surface_id: str, base_id: str,
                     ortho_id: str, stage_id: str, *, row: int, column: int,
                     repo_root: Path, width_pixels: int = 5,
                     max_neighbor_difference: float = .01,
                     command: list[str] | None = None) -> dict:
    run_root = Path(run_root).resolve()
    ids = {"edge": edge_id, "surface": surface_id, "base": base_id, "ortho": ortho_id}
    stages = {name: _stage_location(run_root, "modeling", stage_id)
              for name, stage_id in ids.items()}
    target = _stage_location(run_root, "modeling", stage_id)
    if target.exists():
        raise ExistingSandError(f"refusing to overwrite: {target}")
    manifests = {name: _load_yaml(path / "manifest.yaml", f"E4 {name} manifest")
                 for name, path in stages.items()}
    records = {name: _load_yaml(path / "record.yaml", f"E4 {name} record")
               for name, path in stages.items()}
    for name, identifier in ids.items():
        if manifests[name].get("stage_id") != identifier:
            raise ExistingSandError(f"E4 {name} stage manifest ID mismatch")
    if (records["edge"].get("prompt", {}).get("type") != "harmonic_sand_edge_transition_comparison"
            or records["edge"].get("parameters", {}).get("sand_surface_stage_id") != surface_id
            or records["edge"].get("parameters", {}).get("orthographic_stage_id") != ortho_id
            or width_pixels not in records["edge"].get("parameters", {}).get("widths_pixels", [])):
        raise ExistingSandError("edge stage is not the requested sand-surface comparison")
    if (records["surface"].get("parameters", {}).get("local_base_stage_id") != base_id
            or records["base"].get("parameters", {}).get("source_stage_id") != ortho_id):
        raise ExistingSandError("E4 surface/base/orthographic stages do not share a source")
    artifacts = {name: {item["path"]: item for item in manifest["artifacts"]}
                 for name, manifest in manifests.items()}
    def verified(stage: str, filename: str) -> Path:
        return _verified_input(stages[stage], filename, artifacts[stage])
    before = np.load(verified("edge", f"taper_{width_pixels}_thickness_yx.npy"), allow_pickle=False)
    top = np.load(verified("surface", "hybrid_top_yx.npy"), allow_pickle=False)
    local_base = np.load(verified("base", "local_yx.npy"), allow_pickle=False)
    support = np.load(verified("ortho", "sand_valid_yx.npy"), allow_pickle=False)
    original_base_valid = np.load(verified("ortho", "base_valid_yx.npy"), allow_pickle=False)
    original_source_valid = np.load(verified("ortho", "source_valid_yx.npy"), allow_pickle=False)
    original_base_alpha = np.load(verified("ortho", "base_alpha.npy"), allow_pickle=False)
    original_base_depth = np.load(verified("ortho", "base_depth_camera_z.npy"), allow_pickle=False)
    camera = _load_yaml(verified("ortho", "camera.yaml"), "E4 orthographic camera")
    source_rgb = verified("ortho", "source_rgb.png")
    if not original_source_valid[row, column] or original_base_alpha[row, column] < .5:
        raise ExistingSandError("reviewed invalid pixel must have source depth and high base coverage")
    if np.isfinite(original_base_depth[row, column]):
        raise ExistingSandError("original background depth should be invalid at the reviewed pixel")
    after, audit = repair_one_background_invalid(
        before, top, local_base, support, original_base_valid, row, column,
        max_neighbor_difference=max_neighbor_difference,
    )
    terrain = local_base.astype(np.float32)
    sand_height = np.where(support, after, 0).astype(np.float32)
    if not np.isfinite(terrain).all() or not np.isfinite(sand_height).all() or np.any(sand_height < 0):
        raise ExistingSandError("candidate finite terrain/sand fields failed validation")
    bounds = tuple(float(x) for x in camera["bounds_simulation_xy"])
    (x_grid, y_grid), spacing = orthographic_grid(bounds, before.shape[1], before.shape[0])
    audit.update({
        "simulation_xy": [float(x_grid[row, column]), float(y_grid[row, column])],
        "original_base_alpha": float(original_base_alpha[row, column]),
        "original_base_depth_saved": "invalid_nan",
        "original_base_valid": bool(original_base_valid[row, column]),
        "original_source_valid": bool(original_source_valid[row, column]),
        "width_pixels": int(width_pixels),
        "cell_spacing_simulation": float(spacing),
        "support_pixels": int(support.sum()),
        "previous_invalid_support_pixels": int((support & ~np.isfinite(before)).sum()),
        "remaining_invalid_support_pixels": int((support & ~np.isfinite(after)).sum()),
        "remaining_negative_support_pixels": int((after[support] < 0).sum()),
        "positive_volume_before_repair_simulation_units_cubed": float(
            np.nansum(np.maximum(before[support], 0)) * spacing ** 2),
        "positive_volume_after_repair_simulation_units_cubed": float(
            sand_height.sum() * spacing ** 2),
        "status": "finite E4 height-field candidate; solver exchange not generated",
    })
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = Path(tempfile.mkdtemp(prefix=f".e4_point_{stage_id}_", dir=run_root))
    try:
        repair_mask = np.zeros(before.shape, dtype=bool)
        repair_mask[row, column] = True
        for name, array in (
            ("repair_mask_yx.npy", repair_mask), ("filled_thickness_yx.npy", after),
            ("candidate_terrain_yx.npy", terrain), ("candidate_sand_height_yx.npy", sand_height),
        ):
            np.save(payload / name, array, allow_pickle=False)
            np.save(payload / name.replace("_yx.npy", "_xy.npy"),
                    image_yx_to_exchange_xy(array), allow_pickle=False)
        with (payload / "audit.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(audit, stream, sort_keys=False, allow_unicode=True)
        _repair_preview(payload, source_rgb, top, local_base, before, after, row, column)
        record = {
            "schema_version": 1, "stage": "modeling", "stage_id": stage_id,
            "camera_ids": [], "prompt": {"type": "reviewed_single_background_invalid_repair"},
            "seed": None, "models": [],
            "parameters": {"edge_stage_id": edge_id, "sand_surface_stage_id": surface_id,
                           "local_base_stage_id": base_id, "orthographic_stage_id": ortho_id,
                           "width_pixels": width_pixels, "reviewed_row_column": [row, column],
                           "max_neighbor_difference": max_neighbor_difference,
                           "axis_conversion": "PGSR raster [row,col] -> exchange [x,y] by transpose"},
            "commands": [command] if command else [], "command_cwd": str(Path(repo_root).resolve()),
            "parents": [{"stage": "modeling", "stage_id": identifier}
                        for identifier in (edge_id, surface_id, base_id, ortho_id)],
            "artifacts": [{"path": path.name, "role": "e4_reviewed_point_repair"}
                          for path in sorted(payload.iterdir()) if path.is_file()],
            "code_files": ["sandygs/existing_sand/e4_height.py",
                           "sandygs/existing_sand/e4_point_repair.py",
                           "scripts/experiments/existing_sand_e4_point_repair.py"],
        }
        with (payload / "record.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(record, stream, sort_keys=False, allow_unicode=True)
        payload.rename(target)
        try:
            record_stage(run_root, target / "record.yaml", repo_root=repo_root)
        except BaseException:
            shutil.rmtree(target)
            raise
        return {"stage_dir": str(target), **audit}
    finally:
        if payload.exists():
            shutil.rmtree(payload)
