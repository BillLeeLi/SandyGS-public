"""CPU comparison of narrow nonnegative sand-thickness edge transitions."""
from __future__ import annotations

import csv
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import yaml

from .e4_height import image_yx_to_exchange_xy
from .e4_local_fit import _verified_input
from .e4_sand_surface import _signed_image
from .workspace import ExistingSandError, _load_yaml, _stage_location, record_stage


def harmonic_edge_transition(thickness: np.ndarray, support: np.ndarray,
                             width_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep the positive core and interpolate its narrow rim to zero outside.

    A four-neighbor discrete harmonic field is solved only within the band.
    Out-of-support neighbors have zero thickness. No raw negative value is
    used as an interpolation target, and nonfinite interior values stay NaN.
    """
    if thickness.ndim != 2 or thickness.shape != support.shape or support.dtype != np.bool_:
        raise ExistingSandError("E4 edge transition needs matching 2D thickness and boolean support")
    if width_pixels < 1:
        raise ExistingSandError("edge width must be a positive pixel count")
    distance = distance_transform_edt(support)
    finite = np.isfinite(thickness)
    band = support & (distance <= width_pixels)
    if np.any(band & ~finite):
        raise ExistingSandError("edge band contains invalid thickness; repair validity first")
    core = support & (distance > width_pixels) & finite
    if not core.any() or np.any(thickness[core] < 0):
        raise ExistingSandError("edge width leaves negative or empty core; review the source geometry")
    index = np.full(thickness.shape, -1, dtype=np.int32)
    rows, cols = np.nonzero(band)
    index[band] = np.arange(rows.size, dtype=np.int32)
    row_index: list[int] = []
    col_index: list[int] = []
    coefficients: list[float] = []
    rhs = np.zeros(rows.size, dtype=np.float64)
    height, width = thickness.shape
    for i, (row, col) in enumerate(zip(rows, cols)):
        row_index.append(i)
        col_index.append(i)
        coefficients.append(4.0)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor_row, neighbor_col = row + dr, col + dc
            if not (0 <= neighbor_row < height and 0 <= neighbor_col < width):
                continue  # zero boundary outside the image
            j = int(index[neighbor_row, neighbor_col])
            if j >= 0:
                row_index.append(i)
                col_index.append(j)
                coefficients.append(-1.0)
            elif core[neighbor_row, neighbor_col]:
                rhs[i] += float(thickness[neighbor_row, neighbor_col])
            elif support[neighbor_row, neighbor_col]:
                raise ExistingSandError("invalid core pixel touches the edge interpolation band")
            # Outside the selected sand support, thickness is zero.
    matrix = csr_matrix((coefficients, (row_index, col_index)), shape=(rows.size, rows.size))
    solved = spsolve(matrix, rhs)
    if not np.isfinite(solved).all() or np.min(solved) < -1e-8:
        raise ExistingSandError("edge interpolation produced invalid or negative thickness")
    candidate = thickness.copy()
    candidate[band] = solved.astype(candidate.dtype)
    return candidate, band


def _quantiles(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [.01, .5, .99])]


def _metrics(thickness: np.ndarray, valid: np.ndarray, spacing: float) -> dict:
    values = thickness[valid].astype(np.float64)
    negative = values[values < 0]
    return {
        "valid_pixels": int(valid.sum()),
        "negative_pixels": int(negative.size),
        "minimum_thickness": float(values.min()),
        "thickness_p01_p50_p99": _quantiles(values),
        "signed_volume_simulation_units_cubed": float(values.sum() * spacing ** 2),
        "positive_volume_simulation_units_cubed": float(np.maximum(values, 0).sum() * spacing ** 2),
        "negative_volume_simulation_units_cubed": float(negative.sum() * spacing ** 2),
    }


def _difference_image(values: np.ndarray, valid: np.ndarray) -> Image.Image:
    return _signed_image(values, valid, .03)


def _overview(destination: Path, source_rgb: Path, raw: np.ndarray,
              candidates: dict[int, np.ndarray], bands: dict[int, np.ndarray],
              support: np.ndarray, valid: np.ndarray) -> None:
    widths = sorted(candidates)
    rgb = np.asarray(Image.open(source_rgb).convert("RGB"), dtype=np.float32).copy()
    overlay = rgb.copy()
    overlay[bands[widths[-1]]] = .5 * overlay[bands[widths[-1]]] + .5 * np.array([255, 190, 35])
    overlay[bands[widths[0]]] = .5 * rgb[bands[widths[0]]] + .5 * np.array([225, 65, 55])
    overlay[support & ~bands[widths[-1]]] = (
        .5 * rgb[support & ~bands[widths[-1]]] + .5 * np.array([55, 205, 95]))
    panels = [
        ("red inner band, yellow extra band, green core", Image.fromarray(overlay.astype(np.uint8))),
        ("raw hybrid thickness", _signed_image(raw, valid, .20)),
        (f"harmonic edge width {widths[0]} pixels", _signed_image(candidates[widths[0]], valid, .20)),
        (f"harmonic edge width {widths[1]} pixels", _signed_image(candidates[widths[1]], valid, .20)),
        (f"width {widths[0]} minus raw thickness", _difference_image(candidates[widths[0]] - raw, valid)),
        (f"width {widths[1]} minus raw thickness", _difference_image(candidates[widths[1]] - raw, valid)),
    ]
    height, width = raw.shape
    output = Image.new("RGB", (3 * width, 2 * (height + 22)), (255, 255, 255))
    draw = ImageDraw.Draw(output)
    for i, (label, tile) in enumerate(panels):
        x, y = (i % 3) * width, (i // 3) * (height + 22)
        draw.text((x + 3, y + 3), label, fill=(0, 0, 0))
        output.paste(tile, (x, y + 22))
    output.save(destination / "overview.png")


def _write_distance_profile(path: Path, raw: np.ndarray, candidates: dict[int, np.ndarray],
                            distance: np.ndarray, valid: np.ndarray) -> None:
    widths = sorted(candidates)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["distance_gt_pixels", "distance_le_pixels", "pixels",
                         "raw_mean", f"width_{widths[0]}_mean", f"width_{widths[1]}_mean",
                         "raw_negative_pixels", f"width_{widths[0]}_negative_pixels",
                         f"width_{widths[1]}_negative_pixels"])
        for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 16), (16, np.inf)):
            mask = valid & (distance > lo) & (distance <= hi)
            if not np.any(mask):
                continue
            fields = [raw[mask], candidates[widths[0]][mask], candidates[widths[1]][mask]]
            writer.writerow([lo, hi, int(mask.sum()), *[float(values.mean()) for values in fields],
                             *[int((values < 0).sum()) for values in fields]])


def run_edge_transition_comparison(run_root: Path, surface_id: str, ortho_id: str,
                                   stage_id: str, widths: tuple[int, int], *, repo_root: Path,
                                   command: list[str] | None = None) -> dict:
    run_root = Path(run_root).resolve()
    surface_dir = _stage_location(run_root, "modeling", surface_id)
    ortho_dir = _stage_location(run_root, "modeling", ortho_id)
    target = _stage_location(run_root, "modeling", stage_id)
    if target.exists():
        raise ExistingSandError(f"refusing to overwrite: {target}")
    if len(widths) != 2 or widths[0] < 1 or widths[1] <= widths[0]:
        raise ExistingSandError("two increasing positive edge widths are required")
    surface_manifest = _load_yaml(surface_dir / "manifest.yaml", "E4 surface manifest")
    ortho_manifest = _load_yaml(ortho_dir / "manifest.yaml", "E4 orthographic manifest")
    surface_record = _load_yaml(surface_dir / "record.yaml", "E4 surface record")
    if (surface_manifest.get("stage_id") != surface_id
            or ortho_manifest.get("stage_id") != ortho_id
            or surface_record.get("prompt", {}).get("type") != "selected_vs_full_pg_sand_surface_comparison"
            or surface_record.get("parameters", {}).get("orthographic_stage_id") != ortho_id):
        raise ExistingSandError("E4 edge comparison inputs have incompatible provenance")
    surface_artifacts = {item["path"]: item for item in surface_manifest["artifacts"]}
    ortho_artifacts = {item["path"]: item for item in ortho_manifest["artifacts"]}
    raw = np.load(_verified_input(surface_dir, "hybrid_thickness_yx.npy", surface_artifacts),
                  allow_pickle=False)
    recorded_distance = np.load(_verified_input(surface_dir, "distance_yx.npy", surface_artifacts),
                                allow_pickle=False)
    support = np.load(_verified_input(ortho_dir, "sand_valid_yx.npy", ortho_artifacts),
                      allow_pickle=False)
    camera = _load_yaml(_verified_input(ortho_dir, "camera.yaml", ortho_artifacts), "E4 camera")
    source_rgb = _verified_input(ortho_dir, "source_rgb.png", ortho_artifacts)
    if raw.shape != support.shape or raw.shape != recorded_distance.shape:
        raise ExistingSandError("E4 edge comparison inputs have different grids")
    distance = distance_transform_edt(support)
    if not np.allclose(distance, recorded_distance, atol=1e-5):
        raise ExistingSandError("recorded sand boundary distance disagrees with source support")
    valid = support & np.isfinite(raw)
    if not np.any(valid):
        raise ExistingSandError("E4 edge comparison has no valid sand pixels")
    spacing = float(camera["cell_spacing_simulation"])
    candidates: dict[int, np.ndarray] = {}
    bands: dict[int, np.ndarray] = {}
    for width in widths:
        candidates[width], bands[width] = harmonic_edge_transition(raw, support, width)
    positive_reference = float(np.maximum(raw[valid], 0).sum() * spacing ** 2)
    metrics = {
        "source_surface_id": surface_id, "source_orthographic_id": ortho_id,
        "widths_pixels": list(widths), "cell_spacing_simulation": spacing,
        "source_support_pixels": int(support.sum()),
        "common_valid_pixels": int(valid.sum()),
        "unresolved_invalid_core_pixels": int((support & ~valid).sum()),
        "raw": _metrics(raw, valid, spacing),
    }
    for width in widths:
        candidate = candidates[width]
        band = bands[width]
        if not np.array_equal(candidate[valid & ~band], raw[valid & ~band]):
            raise ExistingSandError("E4 edge transition changed the fixed core")
        result = _metrics(candidate, valid, spacing)
        result["modified_band_pixels"] = int(band.sum())
        result["positive_volume_change_vs_raw_positive_percent"] = float(
            100 * (result["positive_volume_simulation_units_cubed"] / positive_reference - 1))
        result["band_abs_adjustment_mean_max"] = [
            float(np.mean(np.abs(candidate[band] - raw[band]))),
            float(np.max(np.abs(candidate[band] - raw[band]))),
        ]
        metrics[f"width_{width}"] = result
    metrics["status"] = "edge geometry comparison only; no exchange or checkpoint modification"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = Path(tempfile.mkdtemp(prefix=f".e4_edge_{stage_id}_", dir=run_root))
    try:
        np.save(payload / "support_yx.npy", support, allow_pickle=False)
        np.save(payload / "raw_thickness_yx.npy", raw, allow_pickle=False)
        for width in widths:
            np.save(payload / f"band_{width}_yx.npy", bands[width], allow_pickle=False)
            np.save(payload / f"taper_{width}_thickness_yx.npy", candidates[width], allow_pickle=False)
            np.save(payload / f"taper_{width}_thickness_xy.npy",
                    image_yx_to_exchange_xy(candidates[width]), allow_pickle=False)
        with (payload / "metrics.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(metrics, stream, sort_keys=False, allow_unicode=True)
        _write_distance_profile(payload / "distance_profile.csv", raw, candidates, distance, valid)
        _overview(payload, source_rgb, raw, candidates, bands, support, valid)
        record = {
            "schema_version": 1, "stage": "modeling", "stage_id": stage_id,
            "camera_ids": [], "prompt": {"type": "harmonic_sand_edge_transition_comparison"},
            "seed": None, "models": [],
            "parameters": {"sand_surface_stage_id": surface_id, "orthographic_stage_id": ortho_id,
                           "widths_pixels": list(widths),
                           "boundary_rule": "zero thickness outside selected support; fixed positive inner core",
                           "solver": "four-neighbor discrete harmonic interpolation inside each edge band",
                           "axis_conversion": "PGSR raster [row,col] -> exchange [x,y] by transpose"},
            "commands": [command] if command else [], "command_cwd": str(Path(repo_root).resolve()),
            "parents": [{"stage": "modeling", "stage_id": surface_id},
                        {"stage": "modeling", "stage_id": ortho_id}],
            "artifacts": [{"path": path.name, "role": "e4_sand_edge_transition_comparison"}
                          for path in sorted(payload.iterdir()) if path.is_file()],
            "code_files": ["sandygs/existing_sand/e4_height.py",
                           "sandygs/existing_sand/e4_local_fit.py",
                           "sandygs/existing_sand/e4_sand_surface.py",
                           "sandygs/existing_sand/e4_edge_transition.py",
                           "scripts/experiments/existing_sand_e4_edge_transition.py"],
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
