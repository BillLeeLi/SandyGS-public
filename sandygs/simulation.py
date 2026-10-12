"""Read sand-water solver snapshots without disguising them as SimRain files.

The solver owns ghost cells and uses ``float64 [x,y]`` arrays.  The renderer
must see only physical cells in the P1 exchange convention.  This module is the
single boundary where cropping, time lookup and dtype conversion are allowed.
"""

from __future__ import annotations

from bisect import bisect_left
import csv
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .exchange import (
    ExchangeValidationError,
    GridSpec,
    apply_transform,
    load_metadata,
    validate_bundle,
)


class SimulationRunError(ValueError):
    """Raised when a solver run cannot satisfy the P2 renderer contract."""


@dataclass(frozen=True)
class WaterSurfaceFrame:
    """One renderer-facing sand/water snapshot in exchange ``[x,y]`` order.

    The historical class name is retained because P2 callers already import
    it.  P3 extends the same immutable sample with sand fields instead of
    introducing a second, subtly different time-interpolation path.
    """

    requested_time_seconds: float
    lower_step: int
    upper_step: int
    interpolation_weight: float
    terrain: np.ndarray
    sand_height: np.ndarray
    pure_water_height: np.ndarray
    mixed_water_height: np.ndarray
    sand_surface: np.ndarray
    absolute_surface: np.ndarray
    saturation: np.ndarray
    mixed_water_wetness: np.ndarray
    sand_mask: np.ndarray
    free_water_mask: np.ndarray


@dataclass(frozen=True)
class WaterSurfaceGeometry:
    """Active water points and normals after transformation to 3DGS world space."""

    xyz_gs: np.ndarray
    normals_gs: np.ndarray
    pure_water_thickness: np.ndarray
    active_cell_indices: np.ndarray


@dataclass(frozen=True)
class SandSurfaceGeometry:
    """Renderer-independent attributes for one dynamic sand representation.

    ``scales_gs`` stores the three *activated* Gaussian standard deviations,
    not their logarithms.  ``rotations_wxyz`` follows the quaternion convention
    consumed by the 3DGS rasterizer. ``active_cell_indices`` is retained for
    compatibility: for ``cell_tangent`` it contains ``[i,j]`` cells; for
    ``triangle_games`` level 0 contains
    ``[base_triangle_id, clipped_piece_id]``; level 1 appends a deterministic
    ``child_triangle_id`` in ``[0,3]``. Keeping these values in NumPy makes the
    geometry contract testable without CUDA.
    """

    xyz_gs: np.ndarray
    normals_gs: np.ndarray
    rotations_wxyz: np.ndarray
    scales_gs: np.ndarray
    coverage: np.ndarray
    saturation: np.ndarray
    mixed_water_wetness: np.ndarray
    active_cell_indices: np.ndarray
    geometry_backend: str
    boundary_class: np.ndarray
    # Final clipped/subdivided triangle vertices in 3DGS world space.  This is
    # empty for the legacy cell backend and is deliberately kept separate from
    # Gaussian centers: T2.1 rasterizes it as an independent hard-coverage
    # reference, never as part of the production image.
    triangle_vertices_gs: np.ndarray
    # Sand thickness at the corresponding triangle vertices. T2.2
    # perspective-interpolates this physical field to define a soft target
    # boundary independently of Gaussian opacity and footprint.
    triangle_vertex_sand_height: np.ndarray
    # Magnitude of the final triangle plane's height gradient in simulation
    # coordinates. T2.3 thresholds this view-independent scalar to distinguish
    # steep surface primitives from ordinary top-surface triangles.
    triangle_slope: np.ndarray
    sand_epsilon: float
    full_coverage_height: float
    base_triangle_count: int
    candidate_triangle_count: int
    rejected_triangle_count: int
    rejected_degenerate_count: int
    rejected_terrain_cliff_count: int
    rejected_high_terrain_contact_count: int
    accepted_buried_step_count: int
    steep_sand_triangle_count: int
    subdivision_level: int
    pre_subdivision_triangle_count: int


@lru_cache(maxsize=8)
def _fixed_checkerboard_triangles(nx: int, ny: int) -> np.ndarray:
    """Return stable CCW cell-center triangles for an ``[x,y]`` grid.

    Choosing a diagonal from the current 3D edge lengths would make a slightly
    moving height field flip topology between frames. The parity-only rule is
    independent of physical state and therefore stable for an entire video.
    """

    if nx < 2 or ny < 2:
        return np.empty((0, 3), dtype=np.int32)
    ii, jj = np.meshgrid(
        np.arange(nx - 1, dtype=np.int32),
        np.arange(ny - 1, dtype=np.int32),
        indexing="ij",
    )
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)
    p00 = ii * ny + jj
    p10 = (ii + 1) * ny + jj
    p01 = ii * ny + (jj + 1)
    p11 = (ii + 1) * ny + (jj + 1)
    even = ((ii + jj) & 1) == 0
    first = np.where(
        even[:, None],
        np.stack((p00, p10, p11), axis=-1),
        np.stack((p00, p10, p01), axis=-1),
    )
    second = np.where(
        even[:, None],
        np.stack((p00, p11, p01), axis=-1),
        np.stack((p10, p11, p01), axis=-1),
    )
    triangles = np.stack((first, second), axis=1).reshape(-1, 3)
    triangles = np.ascontiguousarray(triangles, dtype=np.int32)
    triangles.setflags(write=False)
    return triangles


def _edge_intersection(
    inside_value: np.ndarray,
    outside_value: np.ndarray,
    inside_phi: np.ndarray,
    outside_phi: np.ndarray,
) -> np.ndarray:
    """Linearly intersect batched material edges with ``phi=0``."""

    weight = inside_phi / (inside_phi - outside_phi)
    return inside_value + weight[:, None] * (outside_value - inside_value)


def _clip_triangle_values(
    values: np.ndarray,
    phi: np.ndarray,
    source_ids: np.ndarray,
    *,
    boundary_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clip CCW triangles against positive sand thickness.

    ``values`` may carry any linearly interpolated per-vertex attributes after
    xyz. A triangle with two inside vertices becomes a CCW quad and is split
    deterministically into two pieces. No Python loop scales with triangle
    count; the six inside/outside patterns are handled in vectorized batches.
    """

    if boundary_mode not in {"clipped", "fully_active"}:
        raise SimulationRunError(
            "sand triangle boundary_mode must be clipped or fully_active"
        )
    inside = phi > 0.0
    inside_count = inside.sum(axis=1)
    output_values: list[np.ndarray] = []
    output_sources: list[np.ndarray] = []
    output_pieces: list[np.ndarray] = []
    output_boundary: list[np.ndarray] = []

    def append(
        triangles: np.ndarray,
        selected_sources: np.ndarray,
        pieces: int | np.ndarray,
        boundary_class: int,
    ) -> None:
        if triangles.shape[0] == 0:
            return
        output_values.append(triangles)
        output_sources.append(selected_sources.astype(np.int32, copy=False))
        output_pieces.append(
            np.full(triangles.shape[0], pieces, dtype=np.int32)
            if np.isscalar(pieces)
            else np.asarray(pieces, dtype=np.int32)
        )
        output_boundary.append(
            np.full(triangles.shape[0], boundary_class, dtype=np.uint8)
        )

    full = inside_count == 3
    append(values[full], source_ids[full], 0, 0)
    if boundary_mode == "fully_active":
        if not output_values:
            empty_shape = (0, 3, values.shape[-1])
            return (
                np.empty(empty_shape, dtype=values.dtype),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.uint8),
            )
        return (
            np.concatenate(output_values),
            np.concatenate(output_sources),
            np.concatenate(output_pieces),
            np.concatenate(output_boundary),
        )

    # One inside vertex: preserve cyclic CCW order [inside,next,previous].
    for inside_index in range(3):
        selected = (inside_count == 1) & inside[:, inside_index]
        if not np.any(selected):
            continue
        next_index = (inside_index + 1) % 3
        previous_index = (inside_index + 2) % 3
        selected_values = values[selected]
        selected_phi = phi[selected]
        center = selected_values[:, inside_index]
        to_next = _edge_intersection(
            center,
            selected_values[:, next_index],
            selected_phi[:, inside_index],
            selected_phi[:, next_index],
        )
        to_previous = _edge_intersection(
            center,
            selected_values[:, previous_index],
            selected_phi[:, inside_index],
            selected_phi[:, previous_index],
        )
        append(
            np.stack((center, to_next, to_previous), axis=1),
            source_ids[selected],
            0,
            1,
        )

    # Two inside vertices form [crossing,next,previous,crossing] in CCW order.
    for outside_index in range(3):
        selected = (inside_count == 2) & ~inside[:, outside_index]
        if not np.any(selected):
            continue
        next_index = (outside_index + 1) % 3
        previous_index = (outside_index + 2) % 3
        selected_values = values[selected]
        selected_phi = phi[selected]
        outside = selected_values[:, outside_index]
        next_value = selected_values[:, next_index]
        previous_value = selected_values[:, previous_index]
        cross_to_next = _edge_intersection(
            next_value,
            outside,
            selected_phi[:, next_index],
            selected_phi[:, outside_index],
        )
        cross_from_previous = _edge_intersection(
            previous_value,
            outside,
            selected_phi[:, previous_index],
            selected_phi[:, outside_index],
        )
        first = np.stack((cross_to_next, next_value, previous_value), axis=1)
        second = np.stack(
            (cross_to_next, previous_value, cross_from_previous), axis=1
        )
        append(
            np.concatenate((first, second), axis=0),
            np.concatenate((source_ids[selected], source_ids[selected])),
            np.concatenate(
                (
                    np.zeros(first.shape[0], dtype=np.int32),
                    np.ones(second.shape[0], dtype=np.int32),
                )
            ),
            1,
        )

    if not output_values:
        empty_shape = (0, 3, values.shape[-1])
        return (
            np.empty(empty_shape, dtype=values.dtype),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.uint8),
        )
    return (
        np.concatenate(output_values),
        np.concatenate(output_sources),
        np.concatenate(output_pieces),
        np.concatenate(output_boundary),
    )


def _subdivide_triangle_values_once(
    values: np.ndarray,
    source_ids: np.ndarray,
    piece_ids: np.ndarray,
    boundary_class: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split every CCW triangle into four deterministic CCW children.

    Midpoints interpolate every carried attribute, including sand height,
    saturation and mixed-water wetness. Parent-major output ordering and fixed
    child ids make repeated frames comparable without consulting geometry.
    """

    if values.shape[0] == 0:
        return (
            np.empty((0, 3, values.shape[-1]), dtype=values.dtype),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.int32),
        )

    v0 = values[:, 0]
    v1 = values[:, 1]
    v2 = values[:, 2]
    midpoint_01 = 0.5 * (v0 + v1)
    midpoint_12 = 0.5 * (v1 + v2)
    midpoint_20 = 0.5 * (v2 + v0)
    children = np.stack(
        (
            np.stack((v0, midpoint_01, midpoint_20), axis=1),
            np.stack((midpoint_01, v1, midpoint_12), axis=1),
            np.stack((midpoint_20, midpoint_12, v2), axis=1),
            np.stack((midpoint_01, midpoint_12, midpoint_20), axis=1),
        ),
        axis=1,
    ).reshape(-1, 3, values.shape[-1])
    parent_count = values.shape[0]
    return (
        children,
        np.repeat(source_ids, 4).astype(np.int32, copy=False),
        np.repeat(piece_ids, 4).astype(np.int32, copy=False),
        np.repeat(boundary_class, 4).astype(np.uint8, copy=False),
        np.tile(np.arange(4, dtype=np.int32), parent_count),
    )


def _free_water_gradient(
    surface: np.ndarray,
    free_water_mask: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate a height field without crossing the free-water boundary.

    A dry neighboring cell may contain a reconstructed obstacle such as a table
    leg.  Reading that height in a centered difference would turn the obstacle
    into an artificial near-vertical water normal.  Use centered differences
    only when both neighbors contain free water, one-sided differences at the
    shoreline, and a zero derivative when no same-axis water neighbor exists.
    """

    if surface.ndim != 2 or surface.shape != free_water_mask.shape:
        raise ValueError("surface and free_water_mask must be matching 2D arrays")

    surface = np.asarray(surface, dtype=np.float64)
    active = np.asarray(free_water_mask, dtype=np.bool_)

    def accumulate_axis_gradient(axis: int, spacing: float) -> np.ndarray:
        # Each valid wet-wet edge contributes its one-sided slope to both end
        # cells.  Averaging two contributions gives the centered difference;
        # a shoreline cell receives one contribution; an isolated cell none.
        gradient = np.zeros_like(surface)
        contribution_count = np.zeros(surface.shape, dtype=np.uint8)
        lower = [slice(None), slice(None)]
        upper = [slice(None), slice(None)]
        lower[axis] = slice(None, -1)
        upper[axis] = slice(1, None)
        lower_index = tuple(lower)
        upper_index = tuple(upper)
        edge_is_wet = active[lower_index] & active[upper_index]
        edge_slope = np.diff(surface, axis=axis) / spacing
        edge_slope[~edge_is_wet] = 0.0

        gradient[lower_index] += edge_slope
        gradient[upper_index] += edge_slope
        contribution_count[lower_index] += edge_is_wet
        contribution_count[upper_index] += edge_is_wet
        np.divide(
            gradient,
            contribution_count,
            out=gradient,
            where=contribution_count > 0,
        )
        return gradient

    # Axis 0 is simulation x and axis 1 is simulation y; spelling that out
    # prevents image-style row/column assumptions at this boundary.
    dz_dx = accumulate_axis_gradient(0, dx)
    dz_dy = accumulate_axis_gradient(1, dy)

    # Inactive entries are not consumed by geometry(), but keeping them zero
    # makes this helper's domain contract explicit and safer to reuse.
    dz_dx[~active] = 0.0
    dz_dy[~active] = 0.0
    return dz_dx, dz_dy


def _load_yaml_mapping(path: Path, name: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise SimulationRunError(f"cannot load {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SimulationRunError(f"{name} must contain a YAML mapping: {path}")
    return value


def _number(mapping: dict[str, Any], key: str, context: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise SimulationRunError(f"{context}.{key} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SimulationRunError(
            f"{context}.{key} must be a finite number"
        ) from exc
    if not math.isfinite(number):
        raise SimulationRunError(f"{context}.{key} must be a finite number")
    # yaml-cpp emits values such as ``1e-08``. PyYAML follows YAML 1.1 here
    # and may parse that spelling as a string, so conversion is explicit.
    return number


def _rotation_matrices_to_quaternions(matrices: np.ndarray) -> np.ndarray:
    """Convert proper rotation matrices to deterministic ``[w,x,y,z]`` quaternions.

    The square-root form is vectorized for a million-cell height field and is
    sufficient here because P3 constructs orthonormal, right-handed bases
    explicitly.  Normalization removes the last few ulps of round-off.
    """

    if matrices.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must have shape [...,3,3]")
    if matrices.size == 0:
        return np.empty(matrices.shape[:-2] + (4,), dtype=np.float64)
    m00 = matrices[..., 0, 0]
    m11 = matrices[..., 1, 1]
    m22 = matrices[..., 2, 2]
    quaternions = np.stack(
        (
            0.5 * np.sqrt(np.maximum(0.0, 1.0 + m00 + m11 + m22)),
            0.5
            * np.copysign(
                np.sqrt(np.maximum(0.0, 1.0 + m00 - m11 - m22)),
                matrices[..., 2, 1] - matrices[..., 1, 2],
            ),
            0.5
            * np.copysign(
                np.sqrt(np.maximum(0.0, 1.0 - m00 + m11 - m22)),
                matrices[..., 0, 2] - matrices[..., 2, 0],
            ),
            0.5
            * np.copysign(
                np.sqrt(np.maximum(0.0, 1.0 - m00 - m11 + m22)),
                matrices[..., 1, 0] - matrices[..., 0, 1],
            ),
        ),
        axis=-1,
    )
    norm = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(np.float64).eps):
        raise SimulationRunError("cannot construct a quaternion from sand tangent basis")
    quaternions /= norm
    return quaternions


class SandWaterRunAdapter:
    """Validate and time-sample a ``sand-water-mixtures`` run directory.

    P2 intentionally requires a no-sand run.  Supporting non-zero sand here
    would make water visible before P3 has a corresponding opaque sand surface,
    which produces an incorrect layer order.
    """

    REQUIRED_FIELDS = (
        "terrain",
        "sand_height",
        "pure_water_height",
        "mixed_water_height",
    )

    def __init__(
        self,
        exchange_bundle: Path,
        run_directory: Path,
        *,
        require_pure_water: bool = True,
        free_water_epsilon: float | None = None,
        sand_epsilon: float | None = None,
        water_normal_gradient_mode: str = "legacy_full",
    ) -> None:
        self.exchange_bundle = Path(exchange_bundle).resolve()
        self.run_directory = Path(run_directory).resolve()
        try:
            self.exchange_metadata = validate_bundle(self.exchange_bundle)
        except ExchangeValidationError as exc:
            raise SimulationRunError(
                f"invalid exchange bundle {self.exchange_bundle}: {exc}"
            ) from exc

        grid = self.exchange_metadata["grid"]
        self.grid = GridSpec(
            tuple(grid["resolution"]),
            tuple(grid["domain_min"]),
            tuple(grid["domain_max"]),
        )
        self.sim_to_gs = np.asarray(
            self.exchange_metadata["coordinates"]["sim_to_gs"], dtype=np.float64
        )
        self.run_metadata = _load_yaml_mapping(
            self.run_directory / "metadata.yaml", "solver metadata"
        )
        self.boundary_width = self._validate_run_grid()
        time_metadata = self.run_metadata.get("time")
        if not isinstance(time_metadata, dict):
            raise SimulationRunError("solver metadata.time must be a mapping")
        self.height_epsilon = _number(
            time_metadata, "height_epsilon", "solver metadata.time"
        )
        if self.height_epsilon < 0.0:
            raise SimulationRunError("solver height_epsilon must be non-negative")
        physics_metadata = self.run_metadata.get("physics", {})
        if not isinstance(physics_metadata, dict):
            raise SimulationRunError("solver metadata.physics must be a mapping")
        contact_tolerance = physics_metadata.get(
            "sand_terrain_contact_tolerance", 0.0
        )
        try:
            contact_tolerance = float(contact_tolerance)
        except (TypeError, ValueError) as exc:
            raise SimulationRunError(
                "solver physics.sand_terrain_contact_tolerance must be numeric"
            ) from exc
        if not math.isfinite(contact_tolerance) or contact_tolerance < 0.0:
            raise SimulationRunError(
                "solver physics.sand_terrain_contact_tolerance must be finite "
                "and non-negative"
            )
        # Match the physics-side high-terrain contact comparison. Otherwise a
        # face accepted by the solver could be cut by a stricter renderer test.
        self.sand_terrain_contact_tolerance = contact_tolerance
        if free_water_epsilon is not None and (
            not math.isfinite(free_water_epsilon) or free_water_epsilon < 0.0
        ):
            raise SimulationRunError("free_water_epsilon must be finite and non-negative")
        # Never render a layer that the solver itself classifies as dry.  A
        # larger rendering threshold may intentionally suppress numerical films.
        self.free_water_epsilon = max(
            self.height_epsilon,
            0.0 if free_water_epsilon is None else free_water_epsilon,
        )
        if sand_epsilon is not None and (
            not math.isfinite(sand_epsilon) or sand_epsilon < 0.0
        ):
            raise SimulationRunError("sand_epsilon must be finite and non-negative")
        # The physics solver and renderer must agree that a numerically dry
        # material cell is absent. A larger threshold may intentionally hide
        # sub-pixel films, but a smaller one may not resurrect solver noise.
        self.sand_epsilon = max(
            self.height_epsilon,
            0.0 if sand_epsilon is None else sand_epsilon,
        )
        if (
            not isinstance(water_normal_gradient_mode, str)
            or water_normal_gradient_mode
            not in {"legacy_full", "free_water_masked"}
        ):
            raise SimulationRunError(
                "water_normal_gradient_mode must be legacy_full or "
                "free_water_masked"
            )
        # Keep the pre-debug behavior as the adapter default. Experiments may
        # opt into shoreline-aware differences without silently changing old
        # P2/P3/P4 render configurations when this repository is updated.
        self.water_normal_gradient_mode = water_normal_gradient_mode
        self.require_pure_water = require_pure_water

        self.step_times = self._load_step_times()
        self.frame_steps = self._discover_frames()
        self.frame_times = tuple(self.step_times[step] for step in self.frame_steps)
        self._exchange_terrain = self._load_exchange_terrain()
        self._validate_static_terrain()

    def _validate_run_grid(self) -> int:
        run_grid = self.run_metadata.get("grid")
        if not isinstance(run_grid, dict):
            raise SimulationRunError("solver metadata.grid must be a mapping")
        resolution = run_grid.get("resolution")
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in resolution
            )
        ):
            raise SimulationRunError(
                "solver metadata.grid.resolution must contain two integers"
            )
        boundary = run_grid.get("boundary_width")
        if isinstance(boundary, bool) or not isinstance(boundary, int) or boundary < 1:
            raise SimulationRunError(
                "solver metadata.grid.boundary_width must be a positive integer"
            )
        expected = [
            self.grid.resolution[0] + 2 * boundary,
            self.grid.resolution[1] + 2 * boundary,
        ]
        if resolution != expected:
            raise SimulationRunError(
                f"solver resolution {resolution} does not match exchange "
                f"interior {list(self.grid.resolution)} plus boundary {boundary}"
            )
        spacing = _number(run_grid, "spacing", "solver metadata.grid")
        dx, dy = self.grid.spacing
        if not (
            np.isclose(spacing, dx, rtol=0.0, atol=1e-12)
            and np.isclose(spacing, dy, rtol=0.0, atol=1e-12)
        ):
            raise SimulationRunError(
                "solver spacing does not match the exchange cell spacing"
            )
        return boundary

    def _load_step_times(self) -> dict[int, float]:
        metrics_path = self.run_directory / "metrics.csv"
        try:
            with metrics_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        except OSError as exc:
            raise SimulationRunError(f"cannot load {metrics_path}: {exc}") from exc
        if not rows or "step" not in rows[0] or "time" not in rows[0]:
            raise SimulationRunError(
                "metrics.csv must contain at least one row with step and time"
            )
        result: dict[int, float] = {}
        previous_time = -math.inf
        for row_number, row in enumerate(rows, start=2):
            try:
                step = int(row["step"])
                time_seconds = float(row["time"])
            except (TypeError, ValueError) as exc:
                raise SimulationRunError(
                    f"invalid step/time at metrics.csv row {row_number}"
                ) from exc
            if step < 0 or not math.isfinite(time_seconds) or time_seconds < previous_time:
                raise SimulationRunError(
                    f"non-monotonic or invalid step/time at metrics.csv row {row_number}"
                )
            if step in result:
                raise SimulationRunError(f"duplicate solver step {step} in metrics.csv")
            result[step] = time_seconds
            previous_time = time_seconds
        return result

    def _discover_frames(self) -> tuple[int, ...]:
        frames_directory = self.run_directory / "frames"
        try:
            directories = sorted(path for path in frames_directory.iterdir() if path.is_dir())
        except OSError as exc:
            raise SimulationRunError(f"cannot inspect {frames_directory}: {exc}") from exc
        if not directories:
            raise SimulationRunError(f"no solver snapshots in {frames_directory}")
        steps: list[int] = []
        for directory in directories:
            try:
                step = int(directory.name)
            except ValueError as exc:
                raise SimulationRunError(
                    f"solver frame directory is not a step index: {directory.name}"
                ) from exc
            if step not in self.step_times:
                raise SimulationRunError(
                    f"solver frame step {step} has no time in metrics.csv"
                )
            for field in self.REQUIRED_FIELDS:
                if not (directory / f"{field}.npy").is_file():
                    raise SimulationRunError(
                        f"solver frame {step} is missing {field}.npy"
                    )
            steps.append(step)
        if len(set(steps)) != len(steps):
            raise SimulationRunError("solver frame steps must be unique")
        return tuple(steps)

    def _load_exchange_terrain(self) -> np.ndarray:
        frames = self.exchange_metadata["frames"]
        terrain_descriptor = frames[0]["fields"].get("terrain")
        if not isinstance(terrain_descriptor, dict):
            raise SimulationRunError("exchange frame 0 must contain terrain")
        terrain = np.load(
            self.exchange_bundle / terrain_descriptor["path"], allow_pickle=False
        )
        return np.ascontiguousarray(terrain, dtype=np.float32)

    def _crop_interior(self, field: np.ndarray) -> np.ndarray:
        width = self.boundary_width
        return np.ascontiguousarray(field[width:-width, width:-width])

    @lru_cache(maxsize=4)
    def _load_snapshot(self, step: int) -> dict[str, np.ndarray]:
        """Load and validate one saved step; the small cache serves interpolation."""

        frame_directory = self.run_directory / "frames" / f"{step:06d}"
        fields: dict[str, np.ndarray] = {}
        full_shape = (
            self.grid.resolution[0] + 2 * self.boundary_width,
            self.grid.resolution[1] + 2 * self.boundary_width,
        )
        for name in self.REQUIRED_FIELDS:
            path = frame_directory / f"{name}.npy"
            array = np.load(path, allow_pickle=False)
            if array.shape != full_shape or array.dtype != np.float64:
                raise SimulationRunError(
                    f"{path} must be float64 with shape {full_shape}, "
                    f"got {array.dtype} {array.shape}"
                )
            if not np.isfinite(array).all():
                raise SimulationRunError(f"{path} contains non-finite values")
            fields[name] = self._crop_interior(array)

        # Thickness fields should be non-negative.  A tiny negative tolerance
        # avoids rejecting harmless floating-point noise, but larger violations
        # are reported rather than silently hidden from the renderer.
        for name in (
            "sand_height",
            "pure_water_height",
            "mixed_water_height",
        ):
            minimum = float(fields[name].min())
            if minimum < -1e-10:
                raise SimulationRunError(
                    f"solver frame {step} has negative {name}: {minimum}"
                )
            fields[name] = np.maximum(fields[name], 0.0)
        if self.require_pure_water and np.any(
            fields["sand_height"] > self.height_epsilon
        ):
            raise SimulationRunError(
                "P2 pure-water rendering received non-zero sand; use P3's "
                "layered renderer once sand support is implemented"
            )
        if self.require_pure_water and np.any(
            fields["mixed_water_height"] > self.height_epsilon
        ):
            raise SimulationRunError(
                "P2 pure-water rendering received mixed water without its "
                "sand layer"
            )
        return fields

    def _validate_static_terrain(self) -> None:
        first = self._load_snapshot(self.frame_steps[0])["terrain"]
        difference = float(
            np.max(np.abs(first - self._exchange_terrain.astype(np.float64)))
        )
        if difference > 1e-6:
            raise SimulationRunError(
                "solver terrain does not match exchange terrain "
                f"(maximum absolute difference {difference})"
            )

    @property
    def time_range(self) -> tuple[float, float]:
        return self.frame_times[0], self.frame_times[-1]

    def sample(self, time_seconds: float) -> WaterSurfaceFrame:
        """Linearly sample scalar water fields at a physical time.

        Rendering outside the saved interval is an error.  Clamping would turn
        a too-short simulation into a deceptively static tail of video.
        """

        if not math.isfinite(time_seconds):
            raise SimulationRunError("requested render time must be finite")
        first_time, last_time = self.time_range
        tolerance = 1e-12 * max(1.0, abs(first_time), abs(last_time))
        if time_seconds < first_time - tolerance or time_seconds > last_time + tolerance:
            raise SimulationRunError(
                f"requested time {time_seconds} is outside saved solver range "
                f"[{first_time}, {last_time}]"
            )
        time_seconds = min(max(time_seconds, first_time), last_time)
        upper_position = bisect_left(self.frame_times, time_seconds)
        if upper_position == 0:
            lower_position = upper_position = 0
        elif upper_position == len(self.frame_times):
            lower_position = upper_position = len(self.frame_times) - 1
        elif self.frame_times[upper_position] == time_seconds:
            lower_position = upper_position
        else:
            lower_position = upper_position - 1

        lower_step = self.frame_steps[lower_position]
        upper_step = self.frame_steps[upper_position]
        lower = self._load_snapshot(lower_step)
        upper = self._load_snapshot(upper_step)
        if lower_position == upper_position:
            weight = 0.0
        else:
            lower_time = self.frame_times[lower_position]
            upper_time = self.frame_times[upper_position]
            weight = (time_seconds - lower_time) / (upper_time - lower_time)

        def interpolate(name: str) -> np.ndarray:
            values = (1.0 - weight) * lower[name] + weight * upper[name]
            return np.ascontiguousarray(values, dtype=np.float32)

        terrain = interpolate("terrain")
        sand = interpolate("sand_height")
        pure = interpolate("pure_water_height")
        mixed = interpolate("mixed_water_height")
        sand_surface = np.ascontiguousarray(terrain + sand, dtype=np.float32)
        # Mixed water contributes to the hydraulic level, but only a non-zero
        # pure layer creates a visible transparent free surface.
        absolute_surface = np.ascontiguousarray(
            terrain + mixed + pure, dtype=np.float32
        )
        total_material = sand + mixed + pure
        saturation = np.divide(
            mixed + pure,
            total_material,
            out=np.zeros_like(total_material, dtype=np.float32),
            where=total_material > self.height_epsilon,
        )
        saturation = np.ascontiguousarray(
            np.clip(saturation, 0.0, 1.0), dtype=np.float32
        )
        # P4 distinguishes pore water from the transparent free-water layer.
        # The old saturation included ``pure`` water and therefore made sand
        # change material simply because water happened to lie above it.
        mixed_water_wetness = np.divide(
            mixed,
            np.maximum(sand, self.height_epsilon),
            out=np.zeros_like(mixed, dtype=np.float32),
            where=sand > self.sand_epsilon,
        )
        mixed_water_wetness = np.ascontiguousarray(
            np.clip(mixed_water_wetness, 0.0, 1.0), dtype=np.float32
        )
        sand_mask = np.ascontiguousarray(
            sand > self.sand_epsilon, dtype=np.bool_
        )
        free_water_mask = np.ascontiguousarray(
            pure > self.free_water_epsilon, dtype=np.bool_
        )
        return WaterSurfaceFrame(
            requested_time_seconds=float(time_seconds),
            lower_step=lower_step,
            upper_step=upper_step,
            interpolation_weight=float(weight),
            terrain=terrain,
            sand_height=sand,
            pure_water_height=pure,
            mixed_water_height=mixed,
            sand_surface=sand_surface,
            absolute_surface=absolute_surface,
            saturation=saturation,
            mixed_water_wetness=mixed_water_wetness,
            sand_mask=sand_mask,
            free_water_mask=free_water_mask,
        )

    def geometry(self, frame: WaterSurfaceFrame) -> WaterSurfaceGeometry:
        """Build active cell-center points and correctly transformed normals."""

        nx, ny = self.grid.resolution
        dx, dy = self.grid.spacing
        x = self.grid.domain_min[0] + (np.arange(nx) + 0.5) * dx
        y = self.grid.domain_min[1] + (np.arange(ny) + 0.5) * dy
        xx, yy = np.meshgrid(x, y, indexing="ij")

        sim_points = np.stack((xx, yy, frame.absolute_surface), axis=-1)
        active = np.argwhere(frame.free_water_mask)
        xyz_gs = apply_transform(
            sim_points[frame.free_water_mask], self.sim_to_gs
        )

        # z=f(x,y) has normal (-dz/dx,-dz/dy,1). ``legacy_full`` reproduces
        # the original full-height-field gradient exactly. The experimental
        # masked mode prevents dry table/ball heights from entering shoreline
        # derivatives, while retaining centered differences inside wet areas.
        if self.water_normal_gradient_mode == "legacy_full":
            dz_dx, dz_dy = np.gradient(
                frame.absolute_surface.astype(np.float64),
                dx,
                dy,
                edge_order=1,
            )
        else:
            dz_dx, dz_dy = _free_water_gradient(
                frame.absolute_surface, frame.free_water_mask, dx, dy
            )
        normals_sim = np.stack(
            (-dz_dx, -dz_dy, np.ones_like(dz_dx)), axis=-1
        )
        normals_sim /= np.linalg.norm(normals_sim, axis=-1, keepdims=True)

        # For p_gs=A*p_sim+t, normals transform with A^{-T}.  With row vectors
        # this is n_gs_row=n_sim_row*A^{-1}; it also supports future scale.
        linear = self.sim_to_gs[:3, :3]
        normals_gs = normals_sim[frame.free_water_mask] @ np.linalg.inv(linear)
        normals_gs /= np.linalg.norm(normals_gs, axis=-1, keepdims=True)
        return WaterSurfaceGeometry(
            xyz_gs=np.ascontiguousarray(xyz_gs, dtype=np.float32),
            normals_gs=np.ascontiguousarray(normals_gs, dtype=np.float32),
            # This remains in simulation-height units.  P4's absorption and
            # image-space refraction expose explicit calibration factors rather
            # than silently pretending the Garden reconstruction is metric.
            pure_water_thickness=np.ascontiguousarray(
                frame.pure_water_height[frame.free_water_mask],
                dtype=np.float32,
            ),
            active_cell_indices=np.ascontiguousarray(active, dtype=np.int32),
        )

    def terrain_reference_triangles(
        self, frame: WaterSurfaceFrame
    ) -> np.ndarray:
        """Return the complete physical terrain as fixed cell-center triangles.

        This mesh is an independent depth reference for experimental water
        visibility.  It uses the same grid, terrain sample and sim-to-GS
        transform as the solver-facing water geometry, and never enters the
        Gaussian attribute arrays.
        """

        nx, ny = self.grid.resolution
        if frame.terrain.shape != (nx, ny):
            raise SimulationRunError(
                "terrain reference shape differs from the simulation grid"
            )
        dx, dy = self.grid.spacing
        x = self.grid.domain_min[0] + (np.arange(nx) + 0.5) * dx
        y = self.grid.domain_min[1] + (np.arange(ny) + 0.5) * dy
        xx, yy = np.meshgrid(x, y, indexing="ij")
        points_sim = np.stack((xx, yy, frame.terrain), axis=-1)
        points_gs = apply_transform(
            points_sim.reshape(-1, 3), self.sim_to_gs
        )
        indices = _fixed_checkerboard_triangles(nx, ny)
        return np.ascontiguousarray(points_gs[indices], dtype=np.float32)

    def water_reference_triangles(
        self, frame: WaterSurfaceFrame
    ) -> np.ndarray:
        """Return a conservative physical free-water surface mesh.

        The mesh shares the solver cell centres and fixed checkerboard
        topology used by other physical references.  A triangle is retained
        only when all three vertices contain free water, so it cannot bridge
        a dry shoreline.  This reference is intended for camera-Z visibility
        tests; the production water appearance remains Gaussian based.
        """

        nx, ny = self.grid.resolution
        expected_shape = (nx, ny)
        if frame.absolute_surface.shape != expected_shape:
            raise SimulationRunError(
                "water reference surface shape differs from the simulation grid"
            )
        if frame.free_water_mask.shape != expected_shape:
            raise SimulationRunError(
                "water reference mask shape differs from the simulation grid"
            )
        dx, dy = self.grid.spacing
        x = self.grid.domain_min[0] + (np.arange(nx) + 0.5) * dx
        y = self.grid.domain_min[1] + (np.arange(ny) + 0.5) * dy
        xx, yy = np.meshgrid(x, y, indexing="ij")
        points_sim = np.stack((xx, yy, frame.absolute_surface), axis=-1)
        points_gs = apply_transform(
            points_sim.reshape(-1, 3), self.sim_to_gs
        )
        indices = _fixed_checkerboard_triangles(nx, ny)
        wet_vertices = frame.free_water_mask.reshape(-1)[indices]
        indices = indices[np.all(wet_vertices, axis=1)]
        return np.ascontiguousarray(points_gs[indices], dtype=np.float32)

    def sand_geometry(
        self,
        frame: WaterSurfaceFrame,
        *,
        coverage_factor: float = 0.55,
        normal_thickness_ratio: float = 0.10,
        max_tangent_scale_ratio: float = 4.0,
        full_coverage_height: float | None = None,
    ) -> SandSurfaceGeometry:
        """Build one tangent-aligned, flat Gaussian surfel per active sand cell.

        Positions stay on the physical surface ``terrain + sand_height``.
        Rendering overlap is expressed only through covariance and opacity,
        rather than moving the surface upward and introducing a visible float.
        """

        if not math.isfinite(coverage_factor) or coverage_factor <= 0.0:
            raise SimulationRunError("sand coverage_factor must be positive")
        if (
            not math.isfinite(normal_thickness_ratio)
            or normal_thickness_ratio <= 0.0
        ):
            raise SimulationRunError(
                "sand normal_thickness_ratio must be positive"
            )
        if (
            not math.isfinite(max_tangent_scale_ratio)
            or max_tangent_scale_ratio < 1.0
        ):
            raise SimulationRunError(
                "sand max_tangent_scale_ratio must be at least one"
            )
        nx, ny = self.grid.resolution
        dx, dy = self.grid.spacing
        if full_coverage_height is None:
            # A one-cell-high layer is fully opaque by default. Thinner leading
            # edges fade in smoothly so topology changes do not visibly pop.
            full_coverage_height = self.sand_epsilon + min(dx, dy)
        if (
            not math.isfinite(full_coverage_height)
            or full_coverage_height <= self.sand_epsilon
        ):
            raise SimulationRunError(
                "sand full_coverage_height must exceed sand_epsilon"
            )

        x = self.grid.domain_min[0] + (np.arange(nx) + 0.5) * dx
        y = self.grid.domain_min[1] + (np.arange(ny) + 0.5) * dy
        xx, yy = np.meshgrid(x, y, indexing="ij")
        surface = frame.sand_surface.astype(np.float64)
        dz_dx, dz_dy = np.gradient(surface, dx, dy, edge_order=1)

        sim_points = np.stack((xx, yy, surface), axis=-1)
        active_mask = frame.sand_mask
        active = np.argwhere(active_mask)
        xyz_gs = apply_transform(sim_points[active_mask], self.sim_to_gs)

        # The two derivatives span the physical height-field patch. Transform
        # them as vectors, then orthonormalize to obtain a stable local frame.
        tangent_x_sim = np.stack(
            (np.ones_like(dz_dx), np.zeros_like(dz_dx), dz_dx), axis=-1
        )
        tangent_y_sim = np.stack(
            (np.zeros_like(dz_dy), np.ones_like(dz_dy), dz_dy), axis=-1
        )
        normal_sim = np.stack(
            (-dz_dx, -dz_dy, np.ones_like(dz_dx)), axis=-1
        )
        linear = self.sim_to_gs[:3, :3]
        tangent_x_gs = tangent_x_sim[active_mask] @ linear.T
        tangent_y_gs = tangent_y_sim[active_mask] @ linear.T
        tangent_x_length = np.linalg.norm(tangent_x_gs * dx, axis=-1)
        tangent_y_length = np.linalg.norm(tangent_y_gs * dy, axis=-1)
        # PGSR-derived terrain may contain near-discontinuous height jumps at
        # foreground silhouettes. Their analytic slope is not a real surface
        # patch and would create a giant surfel spanning unrelated geometry.
        base_x_length = np.linalg.norm(
            np.asarray([dx, 0.0, 0.0]) @ linear.T
        )
        base_y_length = np.linalg.norm(
            np.asarray([0.0, dy, 0.0]) @ linear.T
        )
        tangent_x_length = np.minimum(
            tangent_x_length, max_tangent_scale_ratio * base_x_length
        )
        tangent_y_length = np.minimum(
            tangent_y_length, max_tangent_scale_ratio * base_y_length
        )
        normals_gs = normal_sim[active_mask] @ np.linalg.inv(linear)
        normals_gs /= np.linalg.norm(normals_gs, axis=-1, keepdims=True)

        tangent_x_gs -= (
            np.sum(tangent_x_gs * normals_gs, axis=-1, keepdims=True)
            * normals_gs
        )
        tangent_x_gs /= np.linalg.norm(tangent_x_gs, axis=-1, keepdims=True)
        tangent_y_orthogonal = np.cross(normals_gs, tangent_x_gs)
        tangent_y_orthogonal /= np.linalg.norm(
            tangent_y_orthogonal, axis=-1, keepdims=True
        )
        rotations = np.stack(
            (tangent_x_gs, tangent_y_orthogonal, normals_gs), axis=-1
        )
        quaternions = _rotation_matrices_to_quaternions(rotations)

        # A scale near half the center spacing makes adjacent Gaussian kernels
        # overlap without turning the surface into spherical blobs. Surface
        # derivatives naturally enlarge a cell measured along a steep slope.
        scale_x = coverage_factor * tangent_x_length
        scale_y = coverage_factor * tangent_y_length
        scale_normal = normal_thickness_ratio * np.minimum(scale_x, scale_y)
        scales = np.stack((scale_x, scale_y, scale_normal), axis=-1)

        normalized_height = np.clip(
            (frame.sand_height[active_mask] - self.sand_epsilon)
            / (full_coverage_height - self.sand_epsilon),
            0.0,
            1.0,
        )
        coverage = normalized_height * normalized_height * (
            3.0 - 2.0 * normalized_height
        )
        return SandSurfaceGeometry(
            xyz_gs=np.ascontiguousarray(xyz_gs, dtype=np.float32),
            normals_gs=np.ascontiguousarray(normals_gs, dtype=np.float32),
            rotations_wxyz=np.ascontiguousarray(quaternions, dtype=np.float32),
            scales_gs=np.ascontiguousarray(scales, dtype=np.float32),
            coverage=np.ascontiguousarray(coverage, dtype=np.float32),
            saturation=np.ascontiguousarray(
                frame.saturation[active_mask], dtype=np.float32
            ),
            mixed_water_wetness=np.ascontiguousarray(
                frame.mixed_water_wetness[active_mask], dtype=np.float32
            ),
            active_cell_indices=np.ascontiguousarray(active, dtype=np.int32),
            geometry_backend="cell_tangent",
            # Cell surfels have no explicit mesh-boundary classification.
            boundary_class=np.full(active.shape[0], 255, dtype=np.uint8),
            triangle_vertices_gs=np.empty((0, 3, 3), dtype=np.float32),
            triangle_vertex_sand_height=np.empty((0, 3), dtype=np.float32),
            triangle_slope=np.empty(0, dtype=np.float32),
            sand_epsilon=float(self.sand_epsilon),
            full_coverage_height=float(full_coverage_height),
            base_triangle_count=0,
            candidate_triangle_count=int(active.shape[0]),
            rejected_triangle_count=0,
            rejected_degenerate_count=0,
            rejected_terrain_cliff_count=0,
            rejected_high_terrain_contact_count=0,
            accepted_buried_step_count=0,
            steep_sand_triangle_count=0,
            subdivision_level=0,
            pre_subdivision_triangle_count=0,
        )

    def triangle_sand_geometry(
        self,
        frame: WaterSurfaceFrame,
        *,
        coverage_gamma: float = 1.0,
        normal_thickness_ratio: float = 0.10,
        max_tangent_scale_ratio: float = 4.0,
        terrain_max_edge_ratio: float | None = None,
        contact_tolerance: float | None = None,
        full_coverage_height: float | None = None,
        diagonal_mode: str = "checkerboard",
        boundary_mode: str = "clipped",
        subdivision_level: int = 0,
    ) -> SandSurfaceGeometry:
        """Build one analytic Gaussian per clipped height-field triangle.

        This is the untrained T1 representation. Shared cell-center vertices
        define a piecewise-linear surface; a fixed checkerboard diagonal keeps
        topology stable in time. Material boundaries are clipped at
        ``sand_height=sand_epsilon`` before Gaussian parameters are derived.
        Connectivity uses terrain geometry rather than total sand-surface
        slope, so a steep sand pile remains valid while an exposed terrain
        cliff is rejected unless sand physically buries the step.
        """

        if diagonal_mode != "checkerboard":
            raise SimulationRunError(
                "sand triangle diagonal_mode must be checkerboard"
            )
        if not math.isfinite(coverage_gamma) or coverage_gamma <= 0.0:
            raise SimulationRunError(
                "sand triangle coverage_gamma must be positive"
            )
        if (
            not math.isfinite(normal_thickness_ratio)
            or normal_thickness_ratio <= 0.0
        ):
            raise SimulationRunError(
                "sand normal_thickness_ratio must be positive"
            )
        if (
            not math.isfinite(max_tangent_scale_ratio)
            or max_tangent_scale_ratio < 1.0
        ):
            raise SimulationRunError(
                "sand max_tangent_scale_ratio must be at least one"
            )
        if terrain_max_edge_ratio is None:
            terrain_max_edge_ratio = max_tangent_scale_ratio
        if (
            not math.isfinite(terrain_max_edge_ratio)
            or terrain_max_edge_ratio < 1.0
        ):
            raise SimulationRunError(
                "sand triangle terrain_max_edge_ratio must be at least one"
            )
        if contact_tolerance is None:
            contact_tolerance = self.sand_terrain_contact_tolerance
        if not math.isfinite(contact_tolerance) or contact_tolerance < 0.0:
            raise SimulationRunError(
                "sand triangle contact_tolerance must be finite and non-negative"
            )
        if boundary_mode not in {"clipped", "fully_active"}:
            raise SimulationRunError(
                "sand triangle boundary_mode must be clipped or fully_active"
            )
        if isinstance(subdivision_level, bool) or subdivision_level not in {0, 1}:
            raise SimulationRunError(
                "sand triangle subdivision_level must be zero or one"
            )

        nx, ny = self.grid.resolution
        dx, dy = self.grid.spacing
        if full_coverage_height is None:
            full_coverage_height = self.sand_epsilon + min(dx, dy)
        if (
            not math.isfinite(full_coverage_height)
            or full_coverage_height <= self.sand_epsilon
        ):
            raise SimulationRunError(
                "sand full_coverage_height must exceed sand_epsilon"
            )

        topology = _fixed_checkerboard_triangles(nx, ny)
        height_flat = np.asarray(frame.sand_height, dtype=np.float64).reshape(-1)
        active_flat = height_flat > self.sand_epsilon
        topology_has_sand = np.any(active_flat[topology], axis=1)
        candidate_ids = np.flatnonzero(topology_has_sand).astype(np.int32)
        candidate_topology = topology[candidate_ids]

        # Assemble only triangles adjacent to sand. On Garden this avoids
        # materializing per-vertex float attributes for roughly two million
        # wholly dry base triangles on every rendered frame.
        vertex_i = candidate_topology // ny
        vertex_j = candidate_topology % ny
        vertex_x = self.grid.domain_min[0] + (vertex_i + 0.5) * dx
        vertex_y = self.grid.domain_min[1] + (vertex_j + 0.5) * dy
        surface_flat = np.asarray(frame.sand_surface, dtype=np.float64).reshape(-1)
        vertex_z = surface_flat[candidate_topology]
        vertex_height = height_flat[candidate_topology]
        terrain_flat = np.asarray(frame.terrain, dtype=np.float64).reshape(-1)
        vertex_terrain = terrain_flat[candidate_topology]
        saturation_flat = np.asarray(frame.saturation, dtype=np.float64).reshape(-1)
        wetness_flat = np.asarray(
            frame.mixed_water_wetness, dtype=np.float64
        ).reshape(-1)
        values = np.stack(
            (
                vertex_x,
                vertex_y,
                vertex_z,
                vertex_height,
                saturation_flat[candidate_topology],
                wetness_flat[candidate_topology],
            ),
            axis=-1,
        )

        # Classify every original edge before shoreline clipping. These tests
        # answer three separate geometric questions instead of treating a
        # steep *sand surface* as an obstacle:
        #
        # 1. terrain_regular: the reconstructed terrain itself is continuous;
        # 2. contact_blocked: one sand column would enter a higher solid cell;
        # 3. buried_step: active sand on both sides completely covers a steep
        #    terrain step and restores a continuous visible sand surface.
        vertex_xy = values[..., :2]
        edge_xy = np.roll(vertex_xy, -1, axis=1) - vertex_xy
        horizontal_length = np.linalg.norm(edge_xy, axis=-1)
        horizontal_floor = min(dx, dy) * 1.0e-10
        next_terrain = np.roll(vertex_terrain, -1, axis=1)
        terrain_delta = next_terrain - vertex_terrain
        terrain_spatial_length = np.sqrt(
            horizontal_length * horizontal_length + terrain_delta * terrain_delta
        )
        terrain_edge_ratio = np.divide(
            terrain_spatial_length,
            horizontal_length,
            out=np.full_like(terrain_spatial_length, np.inf),
            where=horizontal_length > horizontal_floor,
        )
        terrain_regular = terrain_edge_ratio <= terrain_max_edge_ratio

        active_vertex = vertex_height > self.sand_epsilon
        next_active = np.roll(active_vertex, -1, axis=1)
        next_surface = np.roll(vertex_z, -1, axis=1)
        edge_relevant = active_vertex | next_active
        contact_blocked = edge_relevant & (
            (active_vertex & (next_terrain > vertex_z + contact_tolerance))
            | (
                next_active
                & (vertex_terrain > next_surface + contact_tolerance)
            )
        )
        buried_step = (
            active_vertex
            & next_active
            & (
                np.minimum(vertex_z, next_surface) + contact_tolerance
                >= np.maximum(vertex_terrain, next_terrain)
            )
        )
        terrain_cliff = (
            edge_relevant
            & ~contact_blocked
            & ~terrain_regular
            & ~buried_step
        )
        edge_connected = (
            ~edge_relevant
            | (~contact_blocked & (terrain_regular | buried_step))
        )
        candidate_contact_blocked = np.any(contact_blocked, axis=1)
        # Contact gets priority so the rejection categories are disjoint and
        # their counts can be audited against the aggregate rejected count.
        candidate_terrain_cliff = (
            ~candidate_contact_blocked & np.any(terrain_cliff, axis=1)
        )
        candidate_connected = np.all(edge_connected, axis=1)
        candidate_buried_step = candidate_connected & np.any(
            edge_relevant & ~terrain_regular & buried_step, axis=1
        )

        # Preserve total-surface steepness only as a diagnostic for the later
        # T2 subdivision policy. It must not delete the steep dam faces that T1
        # is meant to represent.
        surface_delta = np.roll(vertex_z, -1, axis=1) - vertex_z
        surface_spatial_length = np.sqrt(
            horizontal_length * horizontal_length + surface_delta * surface_delta
        )
        surface_edge_ratio = np.divide(
            surface_spatial_length,
            horizontal_length,
            out=np.full_like(surface_spatial_length, np.inf),
            where=horizontal_length > horizontal_floor,
        )
        candidate_steep_sand = candidate_connected & np.any(
            edge_relevant & (surface_edge_ratio > max_tangent_scale_ratio),
            axis=1,
        )

        # Boundary clipping returns global base-triangle ids. Dense boolean
        # maps let each generated piece inherit its source classification with
        # vectorized indexing and no Python loop over Garden's triangles.
        base_contact_blocked = np.zeros(topology.shape[0], dtype=np.bool_)
        base_terrain_cliff = np.zeros(topology.shape[0], dtype=np.bool_)
        base_connected = np.zeros(topology.shape[0], dtype=np.bool_)
        base_buried_step = np.zeros(topology.shape[0], dtype=np.bool_)
        base_steep_sand = np.zeros(topology.shape[0], dtype=np.bool_)
        base_contact_blocked[candidate_ids] = candidate_contact_blocked
        base_terrain_cliff[candidate_ids] = candidate_terrain_cliff
        base_connected[candidate_ids] = candidate_connected
        base_buried_step[candidate_ids] = candidate_buried_step
        base_steep_sand[candidate_ids] = candidate_steep_sand

        clipped, source_ids, piece_ids, boundary_class = _clip_triangle_values(
            values,
            vertex_height - self.sand_epsilon,
            candidate_ids,
            boundary_mode=boundary_mode,
        )

        vertices_sim = clipped[..., :3]
        if vertices_sim.shape[0] > 0:
            twice_area_sim = np.linalg.norm(
                np.cross(
                    vertices_sim[:, 1] - vertices_sim[:, 0],
                    vertices_sim[:, 2] - vertices_sim[:, 0],
                ),
                axis=-1,
            )
            piece_degenerate = twice_area_sim <= dx * dy * 1.0e-12
            piece_contact_blocked = base_contact_blocked[source_ids]
            piece_terrain_cliff = base_terrain_cliff[source_ids]
            piece_connected = base_connected[source_ids]
            valid = piece_connected & ~piece_degenerate
        else:
            valid = np.empty(0, dtype=np.bool_)
            piece_degenerate = np.empty(0, dtype=np.bool_)
            piece_contact_blocked = np.empty(0, dtype=np.bool_)
            piece_terrain_cliff = np.empty(0, dtype=np.bool_)
            piece_connected = np.empty(0, dtype=np.bool_)
        rejected_count = int(valid.size - np.count_nonzero(valid))
        # Count degeneracy only after connectivity acceptance so the three
        # rejection counters partition the aggregate rejected pieces.
        rejected_degenerate_count = int(
            np.count_nonzero(piece_degenerate & piece_connected)
        )
        rejected_contact_count = int(np.count_nonzero(piece_contact_blocked))
        rejected_terrain_cliff_count = int(np.count_nonzero(piece_terrain_cliff))
        accepted_buried_step_count = int(
            np.count_nonzero(valid & base_buried_step[source_ids])
        )
        steep_sand_triangle_count = int(
            np.count_nonzero(valid & base_steep_sand[source_ids])
        )
        clipped = clipped[valid]
        source_ids = source_ids[valid]
        piece_ids = piece_ids[valid]
        boundary_class = boundary_class[valid]
        pre_subdivision_triangle_count = int(clipped.shape[0])
        child_ids = None
        if subdivision_level == 1:
            # T2 performs the same split for every accepted parent. There is
            # no slope threshold here, so topology cannot flicker when a
            # dynamic sand surface crosses an adaptive decision boundary.
            (
                clipped,
                source_ids,
                piece_ids,
                boundary_class,
                child_ids,
            ) = _subdivide_triangle_values_once(
                clipped, source_ids, piece_ids, boundary_class
            )
        vertices_sim = clipped[..., :3]

        # For z=f(x,y), ||grad(f)|| = ||n_xy||/|n_z|. Compute it after
        # clipping/subdivision so every diagnostic value corresponds one-to-one
        # with the final Gaussian primitive rather than only its base triangle.
        final_normal_sim = np.cross(
            vertices_sim[:, 1] - vertices_sim[:, 0],
            vertices_sim[:, 2] - vertices_sim[:, 0],
        )
        triangle_slope = np.divide(
            np.linalg.norm(final_normal_sim[:, :2], axis=-1),
            np.abs(final_normal_sim[:, 2]),
            out=np.full(vertices_sim.shape[0], np.inf, dtype=np.float64),
            where=np.abs(final_normal_sim[:, 2])
            > np.finfo(np.float64).eps,
        )

        vertices_gs = apply_transform(vertices_sim, self.sim_to_gs)
        centers_gs = vertices_gs.mean(axis=1)
        first_edge_gs = vertices_gs[:, 1] - vertices_gs[:, 0]
        second_edge_gs = vertices_gs[:, 2] - vertices_gs[:, 0]
        normals_gs = np.cross(first_edge_gs, second_edge_gs)
        normal_length = np.linalg.norm(normals_gs, axis=-1, keepdims=True)
        if np.any(normal_length <= np.finfo(np.float64).eps):
            raise SimulationRunError(
                "sand triangle became degenerate after sim_to_gs transform"
            )
        normals_gs /= normal_length

        # An affine reflection reverses the cross-product orientation. Align
        # against the inverse-transformed simulation normal so the material
        # normal still points toward the physical height-field upper side.
        if vertices_sim.shape[0] > 0:
            normal_sim = np.cross(
                vertices_sim[:, 1] - vertices_sim[:, 0],
                vertices_sim[:, 2] - vertices_sim[:, 0],
            )
            reference_normal = normal_sim @ np.linalg.inv(
                self.sim_to_gs[:3, :3]
            )
            flip = np.sum(normals_gs * reference_normal, axis=-1) < 0.0
            normals_gs[flip] *= -1.0

        tangent_first = vertices_gs[:, 0] - centers_gs
        tangent_first /= np.linalg.norm(
            tangent_first, axis=-1, keepdims=True
        )
        tangent_second = np.cross(normals_gs, tangent_first)
        tangent_second /= np.linalg.norm(
            tangent_second, axis=-1, keepdims=True
        )
        rotations = np.stack(
            (tangent_first, tangent_second, normals_gs), axis=-1
        )
        quaternions = _rotation_matrices_to_quaternions(rotations)

        offsets = vertices_gs - centers_gs[:, None, :]
        extent_first = np.max(
            np.abs(np.sum(offsets * tangent_first[:, None, :], axis=-1)),
            axis=1,
        )
        extent_second = np.max(
            np.abs(np.sum(offsets * tangent_second[:, None, :], axis=-1)),
            axis=1,
        )
        scale_first = coverage_gamma * extent_first
        scale_second = coverage_gamma * extent_second
        scale_normal = normal_thickness_ratio * np.minimum(
            scale_first, scale_second
        )
        scales = np.stack(
            (scale_first, scale_second, scale_normal), axis=-1
        )
        if np.any(scales <= 0.0) or not np.isfinite(scales).all():
            raise SimulationRunError(
                "sand triangle produced a non-positive Gaussian scale"
            )

        center_height = clipped[..., 3].mean(axis=1)
        normalized_height = np.clip(
            (center_height - self.sand_epsilon)
            / (full_coverage_height - self.sand_epsilon),
            0.0,
            1.0,
        )
        coverage = normalized_height * normalized_height * (
            3.0 - 2.0 * normalized_height
        )
        primitive_sources = np.stack(
            (source_ids, piece_ids)
            if child_ids is None
            else (source_ids, piece_ids, child_ids),
            axis=-1,
        )
        return SandSurfaceGeometry(
            xyz_gs=np.ascontiguousarray(centers_gs, dtype=np.float32),
            normals_gs=np.ascontiguousarray(normals_gs, dtype=np.float32),
            rotations_wxyz=np.ascontiguousarray(quaternions, dtype=np.float32),
            scales_gs=np.ascontiguousarray(scales, dtype=np.float32),
            coverage=np.ascontiguousarray(coverage, dtype=np.float32),
            saturation=np.ascontiguousarray(
                clipped[..., 4].mean(axis=1), dtype=np.float32
            ),
            mixed_water_wetness=np.ascontiguousarray(
                clipped[..., 5].mean(axis=1), dtype=np.float32
            ),
            active_cell_indices=np.ascontiguousarray(
                primitive_sources, dtype=np.int32
            ),
            geometry_backend="triangle_games",
            boundary_class=np.ascontiguousarray(
                boundary_class, dtype=np.uint8
            ),
            triangle_vertices_gs=np.ascontiguousarray(
                vertices_gs, dtype=np.float32
            ),
            triangle_vertex_sand_height=np.ascontiguousarray(
                clipped[..., 3], dtype=np.float32
            ),
            triangle_slope=np.ascontiguousarray(
                triangle_slope, dtype=np.float32
            ),
            sand_epsilon=float(self.sand_epsilon),
            full_coverage_height=float(full_coverage_height),
            base_triangle_count=int(topology.shape[0]),
            candidate_triangle_count=int(candidate_ids.shape[0]),
            rejected_triangle_count=rejected_count,
            rejected_degenerate_count=rejected_degenerate_count,
            rejected_terrain_cliff_count=rejected_terrain_cliff_count,
            rejected_high_terrain_contact_count=rejected_contact_count,
            accepted_buried_step_count=accepted_buried_step_count,
            steep_sand_triangle_count=steep_sand_triangle_count,
            subdivision_level=int(subdivision_level),
            pre_subdivision_triangle_count=pre_subdivision_triangle_count,
        )

    def build_sand_geometry(
        self,
        frame: WaterSurfaceFrame,
        *,
        backend: str = "cell_tangent",
        coverage_factor: float = 0.55,
        normal_thickness_ratio: float = 0.10,
        max_tangent_scale_ratio: float = 4.0,
        full_coverage_height: float | None = None,
        triangle_coverage_gamma: float = 1.0,
        triangle_terrain_max_edge_ratio: float | None = None,
        triangle_contact_tolerance: float | None = None,
        triangle_diagonal_mode: str = "checkerboard",
        triangle_boundary_mode: str = "clipped",
        triangle_subdivision_level: int = 0,
    ) -> SandSurfaceGeometry:
        """Dispatch an explicit sand backend without changing legacy defaults."""

        if backend == "cell_tangent":
            return self.sand_geometry(
                frame,
                coverage_factor=coverage_factor,
                normal_thickness_ratio=normal_thickness_ratio,
                max_tangent_scale_ratio=max_tangent_scale_ratio,
                full_coverage_height=full_coverage_height,
            )
        if backend != "triangle_games":
            raise SimulationRunError(
                "sand geometry backend must be cell_tangent or triangle_games"
            )
        if isinstance(triangle_subdivision_level, bool) or int(
            triangle_subdivision_level
        ) != triangle_subdivision_level:
            raise SimulationRunError(
                "sand triangle subdivision_level must be an integer"
            )
        if int(triangle_subdivision_level) not in {0, 1}:
            raise SimulationRunError(
                "sand triangle subdivision_level must be zero or one"
            )
        return self.triangle_sand_geometry(
            frame,
            coverage_gamma=triangle_coverage_gamma,
            normal_thickness_ratio=normal_thickness_ratio,
            max_tangent_scale_ratio=max_tangent_scale_ratio,
            terrain_max_edge_ratio=triangle_terrain_max_edge_ratio,
            contact_tolerance=triangle_contact_tolerance,
            full_coverage_height=full_coverage_height,
            diagonal_mode=triangle_diagonal_mode,
            boundary_mode=triangle_boundary_mode,
            subdivision_level=int(triangle_subdivision_level),
        )
