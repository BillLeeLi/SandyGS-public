"""Utilities for the flat-cylinder sand saturation benchmark.

This module deliberately renders the solver's height field directly.  It does
not load a PGSR scene and it does not pass through the dynamic-Gaussian sand
geometry path, because the benchmark is intended to isolate physical behavior.
"""

from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

# WSL/Codex sessions may expose ~/.config read-only. A task-specific cache also
# avoids rebuilding Matplotlib's font index for every headless benchmark run.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sandygs-pile-benchmark-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import yaml

from .exchange import GridSpec, cell_center_coordinates
from .prepare import sample_height_spec


@dataclass(frozen=True)
class PileView:
    """One deterministic orthographic camera used by the benchmark."""

    name: str
    elevation: float
    azimuth: float


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Load and validate the small benchmark-specific YAML contract."""

    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return load_experiment_config_value(config)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def select_experiment_phase(
    config: dict[str, Any], phase: str
) -> dict[str, Any]:
    """Apply one tracked phase override while preserving the common baseline."""

    phases = config.get("phases")
    if not isinstance(phases, dict) or not isinstance(phases.get(phase), dict):
        raise ValueError(f"pile benchmark config has no phase {phase!r}")
    selected = _deep_merge(config, phases[phase])
    selected["selected_phase"] = phase
    return load_experiment_config_value(selected)


def load_experiment_config_value(config: dict[str, Any]) -> dict[str, Any]:
    """Validate an already parsed or phase-resolved benchmark mapping."""

    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("pile benchmark config must be a schema_version 1 mapping")
    for key in ("solver_scene", "experiment", "render"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"pile benchmark config requires a {key} mapping")
    initial = config["solver_scene"].get("initial")
    if not isinstance(initial, dict) or "sand" not in initial:
        raise ValueError("solver_scene.initial must define sand")
    phi = float(initial.get("saturation", -1.0))
    if not np.isfinite(phi) or not 0.0 <= phi < 1.0:
        raise ValueError("solver_scene.initial.saturation must lie in [0,1)")
    if "total_water" in initial:
        raise ValueError("use initial.saturation instead of total_water in this benchmark")
    views = config["render"].get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("render.views must be a non-empty sequence")
    return config


def _scale_height_spec(spec: Any, scale: float) -> Any:
    """Scale a solver height primitive without changing its spatial support."""

    if isinstance(spec, (int, float)):
        return float(spec) * scale
    if not isinstance(spec, dict):
        raise ValueError("sand height specification must be numeric or a mapping")
    result = deepcopy(spec)
    if "type" in result:
        if "height" not in result:
            raise ValueError("sand primitive must define height for saturation expansion")
        result["height"] = float(result["height"]) * scale
        return result
    shapes = result.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        raise ValueError("composite sand height requires a non-empty shapes sequence")
    result["shapes"] = [_scale_height_spec(shape, scale) for shape in shapes]
    return result


def resolve_solver_scene(config: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Expand the compact saturation initializer into baseline-solver YAML.

    The unmodified ``sand-water-mixtures`` scene loader accepts total-water
    height.  For a requested column saturation phi, the exact corresponding
    height is h_water = phi/(1-phi) * h_sand.  Copying and scaling the sand
    primitive guarantees identical support, including its smooth edge.
    """

    scene = deepcopy(config["solver_scene"])
    initial = scene["initial"]
    phi = float(initial.pop("saturation"))
    initial["total_water"] = _scale_height_spec(
        initial["sand"], phi / (1.0 - phi)
    )
    return scene, phi


def materialize_initial_perturbation(
    scene: dict[str, Any],
    phi: float,
    perturbation: dict[str, Any],
    output_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write a controlled sinusoidal initializer without changing the solver.

    All three experiment members, including the unperturbed control, use the
    same double-precision NPY input path. Height perturbations scale with the
    original box coverage and keep saturation exact; saturation perturbations
    keep sand geometry exact and modify only the water/sand column ratio.
    """

    if not isinstance(perturbation, dict):
        raise ValueError("initial_perturbation must be a mapping")
    kind = str(perturbation.get("kind", "")).strip()
    if kind not in {"control", "saturation_sine", "sand_height_sine"}:
        raise ValueError(
            "initial_perturbation.kind must be control, saturation_sine, "
            "or sand_height_sine"
        )
    grid_value = scene.get("grid")
    if not isinstance(grid_value, dict):
        raise ValueError("solver scene must define grid for perturbation materialization")
    resolution_value = grid_value.get("resolution")
    center_value = grid_value.get("center", [0.0, 0.0])
    if (
        not isinstance(resolution_value, list)
        or len(resolution_value) != 2
        or not isinstance(center_value, list)
        or len(center_value) != 2
    ):
        raise ValueError("solver grid resolution and center must contain two values")
    resolution = (int(resolution_value[0]), int(resolution_value[1]))
    spacing = float(grid_value["spacing"])
    center = np.asarray(center_value, dtype=np.float64)
    if (
        any(value < 2 for value in resolution)
        or not np.isfinite(spacing)
        or spacing <= 0.0
        or not np.isfinite(center).all()
    ):
        raise ValueError("invalid solver grid for perturbation materialization")
    half_extent = 0.5 * spacing * np.asarray(resolution, dtype=np.float64)
    grid = GridSpec(
        resolution,
        tuple(center - half_extent),
        tuple(center + half_extent),
    )
    initial = scene.get("initial")
    if not isinstance(initial, dict) or "sand" not in initial:
        raise ValueError("solver scene initial mapping must define sand")
    base_sand = sample_height_spec(
        initial["sand"], grid, name="solver_scene.initial.sand"
    )
    active = base_sand > 0.0
    if not np.any(active):
        raise ValueError("perturbation initializer contains no sand")

    x, y = cell_center_coordinates(grid)
    axis = str(perturbation.get("axis", "x")).strip().lower()
    if axis not in {"x", "y"}:
        raise ValueError("initial_perturbation.axis must be x or y")
    coordinates = x[:, None] if axis == "x" else y[None, :]
    amplitude = float(perturbation.get("amplitude", 0.0))
    wavelength = float(perturbation.get("wavelength", 1.0))
    phase_degrees = float(perturbation.get("phase_degrees", 0.0))
    if (
        not np.isfinite(amplitude)
        or amplitude < 0.0
        or not np.isfinite(wavelength)
        or wavelength <= 0.0
        or not np.isfinite(phase_degrees)
    ):
        raise ValueError("perturbation amplitude/phase must be finite and wavelength positive")
    if kind == "control" and amplitude != 0.0:
        raise ValueError("control perturbation amplitude must be zero")
    wave = np.sin(
        2.0 * np.pi * coordinates / wavelength + np.deg2rad(phase_degrees)
    )
    wave = np.broadcast_to(wave, resolution)

    if kind == "sand_height_sine":
        base_height = float(np.max(base_sand))
        if amplitude >= base_height:
            raise ValueError("sand-height perturbation amplitude must be below dam height")
        # Multiplication by normalized base coverage preserves the authored
        # smooth boundary instead of adding sand outside the dam support.
        sand = base_sand * (1.0 + amplitude / base_height * wave)
        local_phi = np.full(resolution, phi, dtype=np.float64)
    else:
        sand = base_sand.copy()
        local_phi = np.full(resolution, phi, dtype=np.float64)
        if kind == "saturation_sine":
            local_phi += amplitude * wave
            active_phi = local_phi[active]
            if active_phi.min() < 0.0 or active_phi.max() >= 1.0:
                raise ValueError("saturation perturbation leaves the valid [0,1) range")
    total_water = np.zeros_like(sand)
    total_water[active] = (
        local_phi[active] / (1.0 - local_phi[active]) * sand[active]
    )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    sand_path = output_directory / "sand.npy"
    water_path = output_directory / "total_water.npy"
    np.save(sand_path, np.ascontiguousarray(sand, dtype=np.float64), allow_pickle=False)
    np.save(
        water_path,
        np.ascontiguousarray(total_water, dtype=np.float64),
        allow_pickle=False,
    )
    origin = [float(value) for value in grid.cell_center_origin]
    initial["sand"] = {
        "type": "npy",
        "file": str(sand_path.relative_to(output_directory.parent)),
        "spacing": spacing,
        "origin": origin,
    }
    initial["total_water"] = {
        "type": "npy",
        "file": str(water_path.relative_to(output_directory.parent)),
        "spacing": spacing,
        "origin": origin,
    }
    realized_phi = np.divide(
        total_water,
        total_water + sand,
        out=np.zeros_like(sand),
        where=(total_water + sand) > 0.0,
    )
    report = {
        "kind": kind,
        "axis": axis,
        "requested_amplitude": amplitude,
        "wavelength": wavelength,
        "wavelength_in_cells": wavelength / spacing,
        "phase_degrees": phase_degrees,
        "active_sand_cell_count": int(np.count_nonzero(active)),
        "sand_height_min_active": float(sand[active].min()),
        "sand_height_max": float(sand.max()),
        "sand_delta_min_active": float((sand - base_sand)[active].min()),
        "sand_delta_max_active": float((sand - base_sand)[active].max()),
        "saturation_min_active": float(realized_phi[active].min()),
        "saturation_max_active": float(realized_phi[active].max()),
        "sand_file": str(sand_path),
        "total_water_file": str(water_path),
    }
    return scene, report


def _load_frame_fields(frame: Path) -> dict[str, np.ndarray]:
    frame = Path(frame)
    required = (
        "terrain",
        "sand_height",
        "pure_water_height",
        "mixed_water_height",
    )
    fields: dict[str, np.ndarray] = {}
    for name in required:
        path = frame / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing solver field: {path}")
        field = np.load(path, allow_pickle=False)
        if field.ndim != 2:
            raise ValueError(f"{path} must contain a 2D height field")
        fields[name] = np.asarray(field, dtype=np.float64)
    shapes = {field.shape for field in fields.values()}
    if len(shapes) != 1:
        raise ValueError("solver height fields do not have matching shapes")
    return fields


def _interior_coordinates(
    metadata: dict[str, Any], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
    grid = metadata.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("solver metadata is missing grid")
    spacing = float(grid["spacing"])
    boundary = int(grid["boundary_width"])
    center = np.asarray(grid.get("center", [0.0, 0.0]), dtype=np.float64)
    if spacing <= 0.0 or boundary < 0 or center.shape != (2,):
        raise ValueError("invalid grid metadata")
    if shape[0] <= 2 * boundary or shape[1] <= 2 * boundary:
        raise ValueError("grid has no interior after removing boundary cells")

    # StaggeredGrid stores cell centers half a spacing from the full-domain
    # origin. Reconstruct those exact positions before dropping ghost cells.
    origin = center - 0.5 * np.asarray(shape, dtype=np.float64) * spacing
    x_full = origin[0] + (np.arange(shape[0], dtype=np.float64) + 0.5) * spacing
    y_full = origin[1] + (np.arange(shape[1], dtype=np.float64) + 0.5) * spacing
    interior = (
        slice(boundary, shape[0] - boundary if boundary else None),
        slice(boundary, shape[1] - boundary if boundary else None),
    )
    return x_full[interior[0]], y_full[interior[1]], interior


def _sand_triangles(
    x: np.ndarray,
    y: np.ndarray,
    surface: np.ndarray,
    sand: np.ndarray,
    epsilon: float,
    *,
    allow_empty: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build supported surface triangles and smooth normals for shading.

    The vertices remain the exact solver samples. Normals are averaged from the
    finite-difference height field only to stop per-triangle flat shading from
    exaggerating the fixed diagonal pattern of the regular grid. ``sand`` is
    the support field for historical API compatibility; the optional free-water
    pass supplies pure-water thickness instead.
    """

    triangles: list[np.ndarray] = []
    shading_normals: list[np.ndarray] = []
    dz_dx, dz_dy = np.gradient(surface, x, y, edge_order=1)
    grid_normals = np.stack((-dz_dx, -dz_dy, np.ones_like(surface)), axis=-1)
    grid_normals /= np.maximum(
        np.linalg.norm(grid_normals, axis=-1, keepdims=True), 1.0e-12
    )
    for i in range(surface.shape[0] - 1):
        for j in range(surface.shape[1] - 1):
            candidates = (
                ((i, j), (i + 1, j), (i + 1, j + 1)),
                ((i, j), (i + 1, j + 1), (i, j + 1)),
            )
            for indices in candidates:
                if max(sand[index] for index in indices) <= epsilon:
                    continue
                triangles.append(
                    np.asarray(
                        [(x[a], y[b], surface[a, b]) for a, b in indices],
                        dtype=np.float64,
                    )
                )
                normal = np.mean([grid_normals[index] for index in indices], axis=0)
                shading_normals.append(normal / max(np.linalg.norm(normal), 1.0e-12))
    if not triangles:
        if allow_empty:
            return (
                np.empty((0, 3, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )
        raise ValueError("initial frame contains no visible sand triangles")
    triangle_array = np.stack(triangles)
    normals = np.stack(shading_normals)
    normals[normals[:, 2] < 0.0] *= -1.0
    return triangle_array, normals


def _views_from_config(render: dict[str, Any]) -> list[PileView]:
    return [
        PileView(
            name=str(value["name"]),
            elevation=float(value["elevation"]),
            azimuth=float(value["azimuth"]),
        )
        for value in render["views"]
    ]


def _subject_label(render: dict[str, Any]) -> str:
    """Return an optional human-readable subject without changing old plots."""

    label = str(render.get("subject_label", "pile")).strip()
    return label or "pile"


def _draw_pile_axis(
    axis: plt.Axes,
    *,
    triangles: np.ndarray,
    normals: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    terrain: np.ndarray,
    title_prefix: str,
    render: dict[str, Any],
    view: PileView,
    z_limits: tuple[float, float] | None = None,
    water_triangles: np.ndarray | None = None,
    water_normals: np.ndarray | None = None,
) -> None:
    base_color = np.asarray(render["sand_color"], dtype=np.float64)
    ground_color = np.asarray(render["ground_color"], dtype=np.float64)
    light = np.asarray(render["light_direction"], dtype=np.float64)
    light /= np.linalg.norm(light)
    brightness = float(render["ambient_strength"]) + float(
        render["diffuse_strength"]
    ) * np.maximum(normals @ light, 0.0)
    face_colors = np.clip(base_color[None, :] * brightness[:, None], 0.0, 1.0)

    extent = float(render["crop_half_extent"])
    ground_height = float(np.min(terrain)) - max(1.0e-4, 0.002 * extent)
    ground_corners = np.asarray(
        [
            [-extent, -extent, ground_height],
            [extent, -extent, ground_height],
            [extent, extent, ground_height],
            [-extent, extent, ground_height],
        ],
        dtype=np.float64,
    )
    ground = Poly3DCollection(
        [[ground_corners[0], ground_corners[1], ground_corners[2]],
         [ground_corners[0], ground_corners[2], ground_corners[3]]],
        facecolors=[ground_color, ground_color],
        edgecolors="none",
        antialiased=False,
        zorder=0,
    )
    axis.add_collection3d(ground)
    collection = Poly3DCollection(
        triangles,
        facecolors=face_colors,
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
        zorder=10,
    )
    axis.add_collection3d(collection)

    # Mixed water occupies pore volume and therefore has no independent
    # geometric surface. Only the solver's positive pure-water layer is drawn
    # here, after opaque sand, as a lightweight diagnostic approximation.
    if (
        water_triangles is not None
        and water_normals is not None
        and water_triangles.shape[0] > 0
    ):
        water_color = np.asarray(render["water_color"], dtype=np.float64)
        water_brightness = float(render.get("water_ambient_strength", 0.55)) + float(
            render.get("water_diffuse_strength", 0.45)
        ) * np.maximum(water_normals @ light, 0.0)
        water_rgb = np.clip(
            water_color[None, :] * water_brightness[:, None], 0.0, 1.0
        )
        water_alpha = float(render.get("water_alpha", 0.45))
        water_rgba = np.concatenate(
            (
                water_rgb,
                np.full((water_rgb.shape[0], 1), water_alpha, dtype=np.float64),
            ),
            axis=1,
        )
        water_collection = Poly3DCollection(
            water_triangles,
            facecolors=water_rgba,
            edgecolors="none",
            linewidths=0.0,
            antialiased=False,
            zorder=20,
        )
        axis.add_collection3d(water_collection)

    if z_limits is None:
        z_min = ground_height
        visible_surface_max = float(triangles[:, :, 2].max())
        if water_triangles is not None and water_triangles.shape[0] > 0:
            visible_surface_max = max(
                visible_surface_max,
                float(water_triangles[:, :, 2].max()),
            )
        z_max = max(visible_surface_max * 1.15, z_min + 0.05)
    else:
        z_min, z_max = z_limits
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_zlim(z_min, z_max)
    axis.set_box_aspect((2.0 * extent, 2.0 * extent, z_max - z_min))
    axis.set_proj_type("ortho")
    # A single broad ground polygon can otherwise receive one collection-level
    # depth and cover the smaller pile collection in oblique views. The two
    # surfaces are opaque and the pile is always above this lowered ground, so
    # an explicit ground-then-sand order is both correct and deterministic.
    axis.computed_zorder = False
    axis.view_init(elev=view.elevation, azim=view.azimuth)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("height")
    axis.zaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.set_title(
        f"{title_prefix}  |  {view.name}\n"
        f"elev={view.elevation:g}°, azim={view.azimuth:g}°"
    )
    axis.grid(False)
    axis.xaxis.pane.set_alpha(0.0)
    axis.yaxis.pane.set_alpha(0.0)
    axis.zaxis.pane.set_alpha(0.0)


def render_initial_pile(
    run_directory: Path,
    config: dict[str, Any],
    phi: float,
) -> dict[str, Any]:
    """Render individual fixed views and a compact review sheet."""

    run_directory = Path(run_directory)
    with (run_directory / "metadata.yaml").open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    fields = _load_frame_fields(run_directory / "frames" / "000000")
    shape = fields["sand_height"].shape
    x, y, interior = _interior_coordinates(metadata, shape)
    terrain = fields["terrain"][interior]
    sand = fields["sand_height"][interior]
    pure = fields["pure_water_height"][interior]
    mixed = fields["mixed_water_height"][interior]
    surface = terrain + sand
    render = config["render"]
    subject = _subject_label(render)
    epsilon = float(render["sand_epsilon"])
    triangles, normals = _sand_triangles(x, y, surface, sand, epsilon)
    water_triangles: np.ndarray | None = None
    water_normals: np.ndarray | None = None
    if bool(render.get("render_free_water", False)):
        water_surface = surface + pure
        water_triangles, water_normals = _sand_triangles(
            x,
            y,
            water_surface,
            pure,
            float(render.get("water_epsilon", epsilon)),
            allow_empty=True,
        )
    views = _views_from_config(render)
    previews = run_directory / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    background = render["background_color"]
    dpi = int(render["dpi"])

    outputs: list[str] = []
    for view in views:
        figure = plt.figure(figsize=(7.2, 6.2), facecolor=background)
        axis = figure.add_subplot(111, projection="3d", facecolor=background)
        _draw_pile_axis(
            axis,
            triangles=triangles,
            normals=normals,
            x=x,
            y=y,
            terrain=terrain,
            title_prefix=f"initial {subject}, $\\phi$={phi:.2f}",
            render=render,
            view=view,
            water_triangles=water_triangles,
            water_normals=water_normals,
        )
        output = previews / f"initial_{view.name}.png"
        figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=background)
        plt.close(figure)
        outputs.append(str(output))

    figure = plt.figure(
        figsize=(7.0 * len(views), 6.1), facecolor=background, constrained_layout=True
    )
    for index, view in enumerate(views, start=1):
        axis = figure.add_subplot(1, len(views), index, projection="3d", facecolor=background)
        _draw_pile_axis(
            axis,
            triangles=triangles,
            normals=normals,
            x=x,
            y=y,
            terrain=terrain,
            title_prefix=f"initial {subject}, $\\phi$={phi:.2f}",
            render=render,
            view=view,
            water_triangles=water_triangles,
            water_normals=water_normals,
        )
    sheet = previews / "initial_view_comparison.png"
    figure.savefig(sheet, dpi=dpi, facecolor=background)
    plt.close(figure)
    outputs.append(str(sheet))

    active = sand > epsilon
    total_water = pure + mixed
    saturation = np.divide(
        total_water,
        total_water + sand,
        out=np.zeros_like(sand),
        where=(total_water + sand) > epsilon,
    )
    spacing = float(metadata["grid"]["spacing"])
    return {
        "requested_saturation": phi,
        "active_sand_cell_count": int(active.sum()),
        "sand_height_max": float(sand.max()),
        "sand_volume": float(sand.sum() * spacing * spacing),
        "water_volume": float(total_water.sum() * spacing * spacing),
        "pure_water_height_max": float(pure.max()),
        "active_sand_saturation_min": float(saturation[active].min()),
        "active_sand_saturation_max": float(saturation[active].max()),
        "triangle_count": int(triangles.shape[0]),
        "outputs": outputs,
    }


def _frame_times(run_directory: Path) -> dict[int, float]:
    """Map saved step indices to physical time from the all-step metrics log."""

    with (Path(run_directory) / "metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("solver metrics.csv contains no rows")
    return {int(row["step"]): float(row["time"]) for row in rows}


def render_pile_range(
    run_directory: Path,
    config: dict[str, Any],
    phi: float,
) -> dict[str, Any]:
    """Render sparse 3D keyframes and measure the short-run pile envelope."""

    run_directory = Path(run_directory)
    with (run_directory / "metadata.yaml").open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    frame_paths = sorted(path for path in (run_directory / "frames").iterdir() if path.is_dir())
    if len(frame_paths) < 2:
        raise ValueError("range preflight requires at least two saved frames")
    times_by_step = _frame_times(run_directory)
    render = config["render"]
    subject = _subject_label(render)
    epsilon = float(render["sand_epsilon"])
    visible_threshold = float(render["range_height_threshold"])
    if visible_threshold < epsilon:
        raise ValueError("range_height_threshold must be >= sand_epsilon")

    first_fields = _load_frame_fields(frame_paths[0])
    shape = first_fields["sand_height"].shape
    x, y, interior = _interior_coordinates(metadata, shape)
    spacing = float(metadata["grid"]["spacing"])
    records: list[dict[str, Any]] = []
    geometries: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    maximum_surface = -np.inf
    for frame_path in frame_paths:
        fields = _load_frame_fields(frame_path)
        sand = fields["sand_height"][interior]
        terrain = fields["terrain"][interior]
        surface = terrain + sand
        triangles, normals = _sand_triangles(x, y, surface, sand, epsilon)
        pure = fields["pure_water_height"][interior]
        if bool(render.get("render_free_water", False)):
            water_surface = surface + pure
            water_triangles, water_normals = _sand_triangles(
                x,
                y,
                water_surface,
                pure,
                float(render.get("water_epsilon", epsilon)),
                allow_empty=True,
            )
        else:
            water_surface = surface
            water_triangles = np.empty((0, 3, 3), dtype=np.float64)
            water_normals = np.empty((0, 3), dtype=np.float64)
        geometries.append(
            (triangles, normals, terrain, water_triangles, water_normals)
        )
        maximum_surface = max(maximum_surface, float(water_surface.max()))

        def extent(mask: np.ndarray) -> dict[str, float]:
            indices = np.nonzero(mask)
            if indices[0].size == 0:
                return {"max_abs_x": 0.0, "max_abs_y": 0.0, "max_radius": 0.0}
            active_x = x[indices[0]]
            active_y = y[indices[1]]
            return {
                "max_abs_x": float(np.max(np.abs(active_x))),
                "max_abs_y": float(np.max(np.abs(active_y))),
                "max_radius": float(np.max(np.hypot(active_x, active_y))),
            }

        step = int(frame_path.name)
        records.append(
            {
                "frame": frame_path.name,
                "step": step,
                "time": times_by_step[step],
                "sand_height_max": float(sand.max()),
                "sand_volume": float(sand.sum() * spacing * spacing),
                "support": extent(sand > epsilon),
                "visible": extent(sand > visible_threshold),
            }
        )

    visible_half_extent = max(
        max(record["visible"]["max_abs_x"], record["visible"]["max_abs_y"])
        for record in records
    )
    support_half_extent = max(
        max(record["support"]["max_abs_x"], record["support"]["max_abs_y"])
        for record in records
    )
    interior_half_extent = min(float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    margin = max(3.0 * spacing, 0.10 * visible_half_extent)
    recommended = min(interior_half_extent, visible_half_extent + margin)
    recommended = min(interior_half_extent, np.ceil(recommended / 0.05) * 0.05)

    views = _views_from_config(render)
    background = render["background_color"]
    dpi = int(render["dpi"])
    previews = run_directory / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    ground_height = min(float(geometry[2].min()) for geometry in geometries) - max(
        1.0e-4, 0.002 * float(render["crop_half_extent"])
    )
    z_limits = (
        ground_height,
        max(maximum_surface * 1.15, ground_height + 0.05),
    )

    target_times = [records[0]["time"], 0.5 * records[-1]["time"], records[-1]["time"]]
    key_indices: list[int] = []
    for target in target_times:
        index = min(range(len(records)), key=lambda i: abs(records[i]["time"] - target))
        if index not in key_indices:
            key_indices.append(index)
    figure = plt.figure(
        figsize=(7.0 * len(views), 5.4 * len(key_indices)),
        facecolor=background,
        constrained_layout=True,
    )
    for row, frame_index in enumerate(key_indices):
        triangles, normals, terrain, water_triangles, water_normals = geometries[
            frame_index
        ]
        for column, view in enumerate(views):
            axis = figure.add_subplot(
                len(key_indices), len(views), row * len(views) + column + 1,
                projection="3d", facecolor=background,
            )
            _draw_pile_axis(
                axis,
                triangles=triangles,
                normals=normals,
                x=x,
                y=y,
                terrain=terrain,
                title_prefix=(
                    f"{subject}, $\\phi$={phi:.2f}, "
                    f"t={records[frame_index]['time']:.3f} s"
                ),
                render=render,
                view=view,
                z_limits=z_limits,
                water_triangles=water_triangles,
                water_normals=water_normals,
            )
    keyframe_sheet = previews / "range_keyframes.png"
    figure.savefig(keyframe_sheet, dpi=dpi, facecolor=background)
    plt.close(figure)

    final_outputs: list[str] = []
    (
        final_triangles,
        final_normals,
        final_terrain,
        final_water_triangles,
        final_water_normals,
    ) = geometries[-1]
    for view in views:
        figure = plt.figure(figsize=(7.2, 6.2), facecolor=background)
        axis = figure.add_subplot(111, projection="3d", facecolor=background)
        _draw_pile_axis(
            axis,
            triangles=final_triangles,
            normals=final_normals,
            x=x,
            y=y,
            terrain=final_terrain,
            title_prefix=(
                f"{subject}, $\\phi$={phi:.2f}, "
                f"t={records[-1]['time']:.3f} s"
            ),
            render=render,
            view=view,
            z_limits=z_limits,
            water_triangles=final_water_triangles,
            water_normals=final_water_normals,
        )
        output = previews / f"range_final_{view.name}.png"
        figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=background)
        plt.close(figure)
        final_outputs.append(str(output))

    times = np.asarray([record["time"] for record in records])
    support_x = np.asarray([record["support"]["max_abs_x"] for record in records])
    support_y = np.asarray([record["support"]["max_abs_y"] for record in records])
    visible_x = np.asarray([record["visible"]["max_abs_x"] for record in records])
    visible_y = np.asarray([record["visible"]["max_abs_y"] for record in records])
    figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    axis.plot(times, support_x, "--", label=f"support |x| (h>{epsilon:g})")
    axis.plot(times, support_y, "--", label=f"support |y| (h>{epsilon:g})")
    axis.plot(times, visible_x, label=f"visible |x| (h>{visible_threshold:g})")
    axis.plot(times, visible_y, label=f"visible |y| (h>{visible_threshold:g})")
    axis.axhline(
        float(render["crop_half_extent"]), color="black", linestyle=":",
        label="range-preview crop",
    )
    axis.axhline(
        recommended, color="tab:red", linestyle="-.", label="recommended sweep crop"
    )
    axis.set_xlabel("physical time (s)")
    axis.set_ylabel("half extent from origin")
    axis.set_title(f"{subject} footprint range, $\\phi$={phi:.2f}")
    axis.grid(alpha=0.25)
    axis.legend()
    extent_plot = previews / "range_extent_over_time.png"
    figure.savefig(extent_plot, dpi=dpi)
    plt.close(figure)

    return {
        "requested_saturation": phi,
        "duration_seconds": records[-1]["time"],
        "saved_frame_count": len(records),
        "range_height_threshold": visible_threshold,
        "maximum_support_half_extent": support_half_extent,
        "maximum_visible_half_extent": visible_half_extent,
        "interior_half_extent": interior_half_extent,
        "range_preview_crop_half_extent": float(render["crop_half_extent"]),
        "recommended_sweep_crop_half_extent": float(recommended),
        "support_reached_domain_boundary": bool(
            support_half_extent >= interior_half_extent - 2.0 * spacing
        ),
        "frames": records,
        "outputs": [str(keyframe_sheet), str(extent_plot), *final_outputs],
    }
