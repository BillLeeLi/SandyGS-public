"""Small initialization previews that do not require CUDA or matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import yaml


def _normalize(values: np.ndarray, *, nonzero: bool = False) -> np.ndarray:
    selected = values[np.isfinite(values)]
    if nonzero:
        selected = selected[selected > 0.0]
    if selected.size == 0:
        return np.zeros_like(values, dtype=np.float64)
    low, high = np.percentile(selected, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _rgb_field(values: np.ndarray, color: tuple[int, int, int], *, nonzero: bool = False) -> np.ndarray:
    weight = _normalize(values, nonzero=nonzero)[..., None]
    background = np.full((*values.shape, 3), 24.0)
    foreground = np.asarray(color, dtype=np.float64)
    return np.asarray(background * (1.0 - weight) + foreground * weight, dtype=np.uint8)


def _to_image(rgb_xy: np.ndarray, size: int = 512) -> Image.Image:
    # Solver arrays are [x,y]; transpose to the conventional image [row=y,col=x]
    # so +x points right and +y points down in every initialization audit.
    image = Image.fromarray(np.transpose(rgb_xy, (1, 0, 2)))
    return image.resize((size, size), resample=Image.Resampling.BILINEAR)


def _signed_rgb_field(values: np.ndarray) -> np.ndarray:
    """Map negative/positive field changes to red/cyan with a shared zero."""

    finite = np.abs(values[np.isfinite(values)])
    scale = float(np.percentile(finite, 99.0)) if finite.size else 0.0
    if scale <= 0.0:
        return np.full((*values.shape, 3), 24, dtype=np.uint8)
    weight = np.clip(np.abs(values) / scale, 0.0, 1.0)[..., None]
    positive = np.asarray((55, 188, 210), dtype=np.float64)
    negative = np.asarray((218, 82, 72), dtype=np.float64)
    color = np.where((values >= 0.0)[..., None], positive, negative)
    return np.asarray(24.0 * (1.0 - weight) + color * weight, dtype=np.uint8)


def write_initialization_preview(case_directory: Path) -> dict[str, Any]:
    """Write individual fields and a labeled overview beside a prepared case."""

    case_directory = case_directory.resolve()
    terrain = np.load(case_directory / "terrain.npy", allow_pickle=False)
    sand = np.load(case_directory / "sand.npy", allow_pickle=False)
    water = np.load(case_directory / "total_water.npy", allow_pickle=False)
    if terrain.shape != sand.shape or terrain.shape != water.shape:
        raise ValueError("terrain, sand and total_water preview fields must align")

    preview = case_directory / "preview"
    preview.mkdir(parents=True, exist_ok=False)
    terrain_rgb = _rgb_field(terrain, (224, 224, 224))
    sand_rgb = _rgb_field(sand, (222, 154, 72), nonzero=True)
    water_rgb = _rgb_field(water, (66, 158, 224), nonzero=True)
    overlay = np.full((*terrain.shape, 3), (34, 38, 34), dtype=np.float64)
    sand_weight = _normalize(sand, nonzero=True)[..., None]
    water_weight = _normalize(water, nonzero=True)[..., None]
    overlay = overlay * (1.0 - 0.85 * sand_weight) + np.asarray(
        (222, 154, 72)
    ) * (0.85 * sand_weight)
    overlay = overlay * (1.0 - 0.65 * water_weight) + np.asarray(
        (66, 158, 224)
    ) * (0.65 * water_weight)
    overlay_rgb = np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    panels = {
        "terrain": _to_image(terrain_rgb),
        "sand_thickness": _to_image(sand_rgb),
        "total_water": _to_image(water_rgb),
        "material_overlay": _to_image(overlay_rgb),
    }
    for name, image in panels.items():
        image.save(preview / f"{name}.png")

    panel_size = 512
    title_height = 28
    overview = Image.new(
        "RGB", (2 * panel_size, 2 * (panel_size + title_height)), "white"
    )
    draw = ImageDraw.Draw(overview)
    for index, (name, image) in enumerate(panels.items()):
        column = index % 2
        row = index // 2
        x = column * panel_size
        y = row * (panel_size + title_height)
        draw.text((x + 8, y + 7), name, fill="black")
        overview.paste(image, (x, y + title_height))
    overview.save(preview / "initialization_overview.png")

    cell_count = int(terrain.size)
    report = {
        "schema_version": 1,
        "array_axis_order": ["x", "y"],
        "preview_image_axes": {"horizontal": "+x", "vertical": "+y"},
        "shape": list(terrain.shape),
        "sand_cells": int(np.count_nonzero(sand > 0.0)),
        "sand_fraction": float(np.count_nonzero(sand > 0.0) / cell_count),
        "sand_height_max": float(np.max(sand)),
        "water_cells": int(np.count_nonzero(water > 0.0)),
        "water_fraction": float(np.count_nonzero(water > 0.0) / cell_count),
        "water_height_max": float(np.max(water)),
        "overlap_cells": int(np.count_nonzero((sand > 0.0) & (water > 0.0))),
        "overview": str(preview / "initialization_overview.png"),
    }
    with (preview / "preview.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False, allow_unicode=True)
    return report


def write_solver_evolution_preview(
    run_directory: Path, output_directory: Path
) -> dict[str, Any]:
    """Write a CPU-only first/final material audit for a solver run.

    Render snapshots deliberately omit diagnostic fields such as saturation.
    This preview therefore consumes only the five arrays guaranteed by the
    SandyGS renderer and remains available on CPU-only login/workstation nodes.
    """

    run_directory = run_directory.resolve()
    output_directory = output_directory.resolve()
    with (run_directory / "metadata.yaml").open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    boundary_width = int(metadata["grid"]["boundary_width"])
    spacing = float(metadata["grid"]["spacing"])
    frames = sorted(
        path
        for path in (run_directory / "frames").iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if not frames:
        raise ValueError("solver run has no numeric frame directories")

    def field(frame: Path, name: str) -> np.ndarray:
        values = np.load(frame / f"{name}.npy", allow_pickle=False)
        if boundary_width:
            values = values[
                boundary_width:-boundary_width,
                boundary_width:-boundary_width,
            ]
        return np.asarray(values, dtype=np.float64)

    first, final = frames[0], frames[-1]
    sand_first = field(first, "sand_height")
    sand_final = field(final, "sand_height")
    water_first = field(first, "total_water")
    water_final = field(final, "total_water")
    pure_final = field(final, "pure_water_height")
    mixed_final = field(final, "mixed_water_height")
    fields = {
        "sand_initial": _rgb_field(sand_first, (222, 154, 72), nonzero=True),
        "sand_final": _rgb_field(sand_final, (222, 154, 72), nonzero=True),
        "sand_delta": _signed_rgb_field(sand_final - sand_first),
        "mixed_water_final": _rgb_field(mixed_final, (118, 92, 176), nonzero=True),
        "water_initial": _rgb_field(water_first, (66, 158, 224), nonzero=True),
        "water_final": _rgb_field(water_final, (66, 158, 224), nonzero=True),
        "water_delta": _signed_rgb_field(water_final - water_first),
        "pure_water_final": _rgb_field(pure_final, (70, 188, 224), nonzero=True),
    }

    output_directory.mkdir(parents=True, exist_ok=False)
    panel_size = 320
    title_height = 28
    panels = {name: _to_image(rgb, panel_size) for name, rgb in fields.items()}
    for name, panel in panels.items():
        panel.save(output_directory / f"{name}.png")
    overview = Image.new(
        "RGB", (4 * panel_size, 2 * (panel_size + title_height)), "white"
    )
    draw = ImageDraw.Draw(overview)
    for index, (name, panel) in enumerate(panels.items()):
        column, row = index % 4, index // 4
        x, y = column * panel_size, row * (panel_size + title_height)
        draw.text((x + 8, y + 7), name, fill="black")
        overview.paste(panel, (x, y + title_height))
    overview_path = output_directory / "evolution_overview.png"
    overview.save(overview_path)

    cell_area = spacing * spacing
    report = {
        "schema_version": 1,
        "first_step": int(first.name),
        "final_step": int(final.name),
        "shape_after_ghost_crop": list(sand_first.shape),
        "sand_volume_initial": float(sand_first.sum() * cell_area),
        "sand_volume_final": float(sand_final.sum() * cell_area),
        "water_volume_initial": float(water_first.sum() * cell_area),
        "water_volume_final": float(water_final.sum() * cell_area),
        "pure_water_volume_final": float(pure_final.sum() * cell_area),
        "mixed_water_volume_final": float(mixed_final.sum() * cell_area),
        "active_sand_cells_initial": int(np.count_nonzero(sand_first > 0.0)),
        "active_sand_cells_final": int(np.count_nonzero(sand_final > 0.0)),
        "active_water_cells_initial": int(np.count_nonzero(water_first > 0.0)),
        "active_water_cells_final": int(np.count_nonzero(water_final > 0.0)),
        "overview": str(overview_path),
    }
    with (output_directory / "preview.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False, allow_unicode=True)
    return report
