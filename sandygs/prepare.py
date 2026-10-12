"""Writers for P1 terrain bundles and sand-water solver input."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .exchange import (
    FORMAT_VERSION,
    PROTOCOL_NAME,
    GridSpec,
    build_legacy_sim_to_gs,
    rainygs_endpoint_image_to_cell_field,
    sha256_file,
)


def _matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def base_metadata(
    *,
    scene_id: str,
    grid: GridSpec,
    sim_to_gs: np.ndarray,
    scale_status: str,
    meters_per_simulation_unit: float | None,
    scale_source: str,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "format_version": FORMAT_VERSION,
        "scene_id": scene_id,
        "units": {
            "horizontal": "simulation_unit",
            "vertical": "simulation_unit",
            "time": "second",
            "meters_per_simulation_unit": meters_per_simulation_unit,
            "scale_status": scale_status,
            "scale_source": scale_source,
        },
        "grid": {
            "axis_order": ["x", "y"],
            "resolution": list(grid.resolution),
            "domain_min": list(grid.domain_min),
            "domain_max": list(grid.domain_max),
            "spacing": list(grid.spacing),
            "centering": "cell",
            "cell_center_origin": list(grid.cell_center_origin),
            "storage_order": "C",
            "exchange_boundary_width": 0,
        },
        "coordinates": {
            "vector_convention": "column",
            "handedness": "right",
            "height_axis": "+z",
            "sim_to_gs": _matrix_list(sim_to_gs),
            "gs_to_sim": _matrix_list(np.linalg.inv(sim_to_gs)),
        },
        "frames": [],
    }


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)


def _interior_feather_weight(
    support: np.ndarray,
    boundary_feather_cells: int,
) -> np.ndarray:
    """Ramp from zero at a mask boundary to one in its eroded interior.

    The 3x3 erosion deliberately follows the same square-grid neighborhood as
    the local constitutive diagnostics. It lets an initialization ablation
    alter the sand core without replacing the physically distinct thin wet
    fringe by a new one-cell material discontinuity.
    """

    mask = np.asarray(support, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("interior feather support must be a 2D field")
    if boundary_feather_cells <= 0:
        return mask.astype(np.float64)

    weight = np.zeros(mask.shape, dtype=np.float64)
    interior = mask.copy()
    for ring in range(1, boundary_feather_cells + 1):
        padded = np.pad(interior, 1, mode="constant", constant_values=False)
        eroded = np.ones_like(interior)
        for offset_x in range(3):
            for offset_y in range(3):
                eroded &= padded[
                    offset_x:offset_x + mask.shape[0],
                    offset_y:offset_y + mask.shape[1],
                ]
        interior = eroded
        weight[interior] = ring / boundary_feather_cells

    # Smoothstep avoids a derivative jump at either end of the blend.
    return weight * weight * (3.0 - 2.0 * weight)


def write_field(
    bundle: Path,
    relative_path: str,
    array: np.ndarray,
) -> dict[str, Any]:
    output = bundle / relative_path
    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.ascontiguousarray(array, dtype=np.float32)
    np.save(output, values, allow_pickle=False)
    return {
        "path": relative_path,
        "centering": "cell",
        "dtype": "float32",
        "shape": list(values.shape),
        "units": "simulation_unit",
        "constraints": ["finite"],
        "sha256": sha256_file(output),
    }


def write_solver_terrain(
    bundle: Path,
    terrain_xy: np.ndarray,
    grid: GridSpec,
    *,
    boundary_width: int = 1,
) -> None:
    """Write the double NPY and YAML expected by sand-water-mixtures.

    Exchange arrays contain only physical cells. The solver grid adds boundary
    cells around them and samples the NPY with clamped edges for those cells.
    """

    dx, dy = grid.spacing
    if not np.isclose(dx, dy, rtol=0.0, atol=1e-12):
        raise ValueError("sand-water-mixtures currently requires square cells")
    if boundary_width < 1:
        raise ValueError("sand-water-mixtures requires boundary_width >= 1")
    solver_directory = bundle / "solver"
    solver_directory.mkdir(parents=True, exist_ok=True)
    np.save(
        solver_directory / "terrain.npy",
        np.ascontiguousarray(terrain_xy, dtype=np.float64),
        allow_pickle=False,
    )
    full_resolution = [
        grid.resolution[0] + 2 * boundary_width,
        grid.resolution[1] + 2 * boundary_width,
    ]
    center = [
        0.5 * (grid.domain_min[0] + grid.domain_max[0]),
        0.5 * (grid.domain_min[1] + grid.domain_max[1]),
    ]
    scene = {
        "name": "sandygs-p1-terrain",
        "grid": {
            "resolution": full_resolution,
            "spacing": float(dx),
            "boundary_width": boundary_width,
            "center": center,
        },
        "physics": {
            "gravity": 9.81,
            "enable_sand_internal_force": False,
        },
        "time": {
            "water_dt": 0.001,
            "sand_substeps": 1,
            "adaptive": True,
            "target_cfl": 0.45,
        },
        "initial": {
            "terrain": {
                "type": "npy",
                "file": "terrain.npy",
                "spacing": float(dx),
                "origin": list(grid.cell_center_origin),
            },
            "sand": 0.0,
            "total_water": 0.0,
        },
        "run": {
            "steps": 1,
            "frame_stride": 1,
            "diagnostic_snapshots": False,
            "write_obj": False,
        },
    }
    _write_yaml(solver_directory / "scene.yaml", scene)


def prepare_rainygs_terrain_bundle(
    *,
    source_height_map: Path,
    output: Path,
    scene_id: str,
    domain: tuple[float, float, float, float],
    rotation_degrees: tuple[float, float, float],
    translation: tuple[float, float, float],
    scale: tuple[float, float, float],
) -> dict[str, Any]:
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"output bundle is not empty: {output}")
    source = np.load(source_height_map, allow_pickle=False)
    if source.ndim != 2:
        raise ValueError("source height map must be two-dimensional")
    grid = GridSpec(
        resolution=(source.shape[1], source.shape[0]),
        domain_min=(domain[0], domain[2]),
        domain_max=(domain[1], domain[3]),
    )
    terrain = rainygs_endpoint_image_to_cell_field(source, grid)
    sim_to_gs = build_legacy_sim_to_gs(
        rotation_degrees, translation, scale
    )
    output.mkdir(parents=True, exist_ok=True)
    metadata = base_metadata(
        scene_id=scene_id,
        grid=grid,
        sim_to_gs=sim_to_gs,
        scale_status="unverified",
        meters_per_simulation_unit=None,
        scale_source="Legacy visual alignment; not metric calibrated",
    )
    descriptor = write_field(output, "frames/000000/terrain.npy", terrain)
    descriptor["semantic"] = "static_terrain_elevation"
    metadata["frames"] = [
        {
            "frame_index": 0,
            "step_index": 0,
            "time_seconds": 0.0,
            "fields": {"terrain": descriptor},
        }
    ]
    metadata["provenance"] = {
        "source_kind": "rainygs_endpoint_height_map",
        "source_file": source_height_map.name,
        "source_sha256": sha256_file(source_height_map),
        "source_axis_order": ["y", "x"],
        "source_sampling": "inclusive_endpoints",
        "conversion": "bilinear_to_cell_centers_then_yx_to_xy",
        "legacy_alignment": {
            "rotation_degrees_xyz": list(rotation_degrees),
            "translation": list(translation),
            "scale": list(scale),
        },
    }
    _write_yaml(output / "metadata.yaml", metadata)
    write_solver_terrain(output, terrain, grid)
    return metadata


def prepare_pure_water_case(
    *,
    exchange_bundle: Path,
    output: Path,
    initial_surface_level: float,
    water_dt: float,
    steps: int,
    frame_stride: int,
    target_cfl: float,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Create a no-sand solver scene from a validated P1 terrain bundle.

    ``initial_surface_level`` follows the legacy SimRain convention: it is an
    absolute hydraulic level, not a uniform thickness.  The actual input field
    is therefore ``max(level - terrain, 0)``.
    """

    from .exchange import validate_bundle

    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"output case directory is not empty: {output}")
    if not np.isfinite(initial_surface_level):
        raise ValueError("initial_surface_level must be finite")
    if not np.isfinite(water_dt) or water_dt <= 0.0:
        raise ValueError("water_dt must be positive")
    if steps < 1 or frame_stride < 1:
        raise ValueError("steps and frame_stride must be positive")
    if duration_seconds is not None and (
        not np.isfinite(duration_seconds) or duration_seconds <= 0.0
    ):
        raise ValueError("duration_seconds must be positive when provided")
    if not np.isfinite(target_cfl) or target_cfl <= 0.0:
        raise ValueError("target_cfl must be positive")

    exchange_bundle = exchange_bundle.resolve()
    metadata = validate_bundle(exchange_bundle)
    grid_data = metadata["grid"]
    grid = GridSpec(
        tuple(grid_data["resolution"]),
        tuple(grid_data["domain_min"]),
        tuple(grid_data["domain_max"]),
    )
    terrain_descriptor = metadata["frames"][0]["fields"].get("terrain")
    if not isinstance(terrain_descriptor, dict):
        raise ValueError("exchange frame 0 must contain a terrain field")
    terrain_path = exchange_bundle / terrain_descriptor["path"]
    terrain = np.load(terrain_path, allow_pickle=False)
    if terrain.shape != grid.resolution:
        raise ValueError("exchange terrain shape does not match grid resolution")
    total_water = np.maximum(
        float(initial_surface_level) - terrain.astype(np.float64), 0.0
    )

    dx, dy = grid.spacing
    if not np.isclose(dx, dy, rtol=0.0, atol=1e-12):
        raise ValueError("sand-water-mixtures currently requires square cells")
    boundary_width = 1
    output.mkdir(parents=True, exist_ok=True)
    np.save(
        output / "terrain.npy",
        np.ascontiguousarray(terrain, dtype=np.float64),
        allow_pickle=False,
    )
    np.save(
        output / "total_water.npy",
        np.ascontiguousarray(total_water, dtype=np.float64),
        allow_pickle=False,
    )
    center = [
        0.5 * (grid.domain_min[0] + grid.domain_max[0]),
        0.5 * (grid.domain_min[1] + grid.domain_max[1]),
    ]
    npy_geometry = {
        "spacing": float(dx),
        "origin": list(grid.cell_center_origin),
    }
    scene = {
        "name": "sandygs-p2-pure-water",
        "grid": {
            # The solver owns one ghost layer on each side.  Exchange arrays
            # remain physical-cell-only and are sampled with clamped edges.
            "resolution": [
                grid.resolution[0] + 2 * boundary_width,
                grid.resolution[1] + 2 * boundary_width,
            ],
            "spacing": float(dx),
            "boundary_width": boundary_width,
            "center": center,
        },
        "physics": {
            "gravity": 9.81,
            "diffusion_rate": 0.0,
            "momentum_exchange_rate": 0.0,
            "ground_friction": 0.0,
            "enable_sand_internal_force": False,
        },
        "time": {
            "water_dt": float(water_dt),
            "sand_substeps": 1,
            "adaptive": True,
            "target_cfl": float(target_cfl),
        },
        "initial": {
            "terrain": {
                "type": "npy",
                "file": "terrain.npy",
                **npy_geometry,
            },
            "sand": 0.0,
            "total_water": {
                "type": "npy",
                "file": "total_water.npy",
                **npy_geometry,
            },
        },
        "run": {
            "steps": int(steps),
            "frame_stride": int(frame_stride),
            "diagnostic_snapshots": False,
            "write_obj": False,
        },
    }
    if duration_seconds is not None:
        # Duration, unlike a step count, remains meaningful under adaptive CFL
        # stepping. The solver shortens its final step to hit this time exactly.
        scene["run"]["duration"] = float(duration_seconds)
    _write_yaml(output / "scene.yaml", scene)
    case_metadata = {
        "format_version": 1,
        "case": "sandygs.p2_pure_water",
        "scene_id": metadata["scene_id"],
        "exchange_bundle": str(exchange_bundle),
        "initialization": {
            "kind": "absolute_surface_level",
            "surface_level": float(initial_surface_level),
            "water_depth_min": float(total_water.min()),
            "water_depth_max": float(total_water.max()),
            "wet_cell_count": int(np.count_nonzero(total_water > 0.0)),
        },
        "run": {
            "water_dt": float(water_dt),
            "steps": int(steps),
            "duration_seconds": (
                None if duration_seconds is None else float(duration_seconds)
            ),
            "frame_stride": int(frame_stride),
            "target_cfl": float(target_cfl),
            "recommended_snapshot_level": "render",
        },
        "files": {
            "terrain_sha256": sha256_file(output / "terrain.npy"),
            "total_water_sha256": sha256_file(output / "total_water.npy"),
        },
    }
    _write_yaml(output / "case.yaml", case_metadata)
    return case_metadata


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def sample_height_spec(spec: Any, grid: GridSpec, *, name: str) -> np.ndarray:
    """Sample the solver's scalar/box/cylinder/Gaussian height syntax.

    P3 writes the sampled result to NPY instead of asking the C++ parser to
    evaluate the shape again. This produces a checksummed, inspectable initial
    material field and guarantees that simulation and rendering use the same
    cell centers.
    """

    nx, ny = grid.resolution
    dx, dy = grid.spacing
    x = grid.domain_min[0] + (np.arange(nx) + 0.5) * dx
    y = grid.domain_min[1] + (np.arange(ny) + 0.5) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        value = _finite_number(spec, name)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return np.full(grid.resolution, value, dtype=np.float64)
    if not isinstance(spec, dict):
        raise ValueError(f"{name} must be a non-negative scalar or shape mapping")

    shapes = spec.get("shapes")
    if shapes is not None:
        if not isinstance(shapes, list) or not shapes:
            raise ValueError(f"{name}.shapes must be a non-empty list")
        combine = spec.get("combine", "sum")
        if combine not in {"sum", "max", "subtract"}:
            raise ValueError(f"{name}.combine must be sum, max, or subtract")
        sampled = [
            sample_height_spec(shape, grid, name=f"{name}.shapes[{index}]")
            for index, shape in enumerate(shapes)
        ]
        if combine == "sum":
            return np.sum(sampled, axis=0)
        if combine == "max":
            return np.maximum.reduce(sampled)
        # A difference composition is useful for reproducible notches and
        # breaches. The first shape is the positive field; every later shape
        # removes material, and the physical height is clamped non-negative.
        return np.maximum(sampled[0] - np.sum(sampled[1:], axis=0), 0.0)

    shape_type = spec.get("type")
    height = _finite_number(spec.get("height"), f"{name}.height")
    if height < 0.0:
        raise ValueError(f"{name}.height must be non-negative")

    def vector2(key: str, default: tuple[float, float] | None = None) -> np.ndarray:
        value = spec.get(key, default)
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
        ):
            raise ValueError(f"{name}.{key} must contain two numbers")
        return np.asarray(
            [
                _finite_number(value[0], f"{name}.{key}[0]"),
                _finite_number(value[1], f"{name}.{key}[1]"),
            ],
            dtype=np.float64,
        )

    if shape_type == "gaussian":
        center = vector2("center", (0.0, 0.0))
        sigma = _finite_number(spec.get("sigma"), f"{name}.sigma")
        if sigma <= 0.0:
            raise ValueError(f"{name}.sigma must be positive")
        squared_distance = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        return height * np.exp(-squared_distance / (2.0 * sigma * sigma))

    if shape_type == "cylinder":
        center = vector2("center", (0.0, 0.0))
        radius = _finite_number(spec.get("radius"), f"{name}.radius")
        edge_width = _finite_number(
            spec.get("edge_width", 0.0), f"{name}.edge_width"
        )
        if radius <= 0.0 or edge_width < 0.0:
            raise ValueError(
                f"{name}.radius must be positive and edge_width non-negative"
            )
        distance = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
        if edge_width == 0.0:
            return np.where(distance <= radius, height, 0.0)
        coverage = np.clip(
            (radius + 0.5 * edge_width - distance) / edge_width, 0.0, 1.0
        )
        coverage = coverage * coverage * (3.0 - 2.0 * coverage)
        return height * coverage

    if shape_type == "box":
        minimum = vector2("min")
        maximum = vector2("max")
        edge_width = _finite_number(
            spec.get("edge_width", 0.0), f"{name}.edge_width"
        )
        if np.any(maximum <= minimum) or edge_width < 0.0:
            raise ValueError(
                f"{name}.max must exceed min and edge_width be non-negative"
            )
        if edge_width == 0.0:
            inside = (
                (xx >= minimum[0])
                & (xx <= maximum[0])
                & (yy >= minimum[1])
                & (yy <= maximum[1])
            )
            return np.where(inside, height, 0.0)
        coverage = np.ones(grid.resolution, dtype=np.float64)
        for coordinates, lower, upper in (
            (xx, minimum[0], maximum[0]),
            (yy, minimum[1], maximum[1]),
        ):
            lower_weight = np.clip(
                (coordinates - lower) / edge_width + 0.5, 0.0, 1.0
            )
            upper_weight = np.clip(
                (upper - coordinates) / edge_width + 0.5, 0.0, 1.0
            )
            lower_weight = lower_weight**2 * (3.0 - 2.0 * lower_weight)
            upper_weight = upper_weight**2 * (3.0 - 2.0 * upper_weight)
            coverage *= lower_weight * upper_weight
        return height * coverage

    raise ValueError(
        f"{name}.type must be gaussian, cylinder or box; got {shape_type!r}"
    )


def prepare_sand_water_case(
    *,
    exchange_bundle: Path,
    output: Path,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Create an auditable P3 sand/water solver scene from a P1 terrain bundle."""

    from .exchange import validate_bundle

    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"output case directory is not empty: {output}")
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError("P3 profile must be a schema_version 1 mapping")
    initial = profile.get("initial")
    solver = profile.get("solver")
    if not isinstance(initial, dict) or not isinstance(solver, dict):
        raise ValueError("P3 profile must contain initial and solver mappings")
    sand_initializers = [
        key for key in ("sand", "sand_surface") if key in initial
    ]
    if len(sand_initializers) != 1:
        raise ValueError(
            "P3 initial must define exactly one of sand (thickness) or "
            "sand_surface (absolute terrain+sand elevation)"
        )
    water_initializers = [
        key
        for key in (
            "total_water", "surface_level", "water_surface", "saturation"
        )
        if key in initial
    ]
    if len(water_initializers) > 1:
        raise ValueError(
            "P3 initial may define only one of total_water, surface_level, "
            "water_surface, or saturation"
        )
    saturation_override_spec = initial.get("sand_saturation_override")
    if saturation_override_spec is not None:
        if not isinstance(saturation_override_spec, dict):
            raise ValueError(
                "initial.sand_saturation_override must be a mapping"
            )
        unknown = set(saturation_override_spec) - {
            "value",
            "sand_height_epsilon",
            "boundary_feather_cells",
        }
        if unknown:
            raise ValueError(
                "initial.sand_saturation_override has unknown keys: "
                + ", ".join(sorted(unknown))
            )
        if "value" not in saturation_override_spec:
            raise ValueError(
                "initial.sand_saturation_override must define value"
            )
        if "saturation" in initial:
            raise ValueError(
                "sand_saturation_override is redundant with initial.saturation"
            )

    exchange_bundle = exchange_bundle.resolve()
    metadata = validate_bundle(exchange_bundle)
    grid_data = metadata["grid"]
    grid = GridSpec(
        tuple(grid_data["resolution"]),
        tuple(grid_data["domain_min"]),
        tuple(grid_data["domain_max"]),
    )
    terrain_descriptor = metadata["frames"][0]["fields"].get("terrain")
    if not isinstance(terrain_descriptor, dict):
        raise ValueError("exchange frame 0 must contain a terrain field")
    terrain = np.load(
        exchange_bundle / terrain_descriptor["path"], allow_pickle=False
    ).astype(np.float64)
    target_sand_surface = None
    if "sand_surface" in initial:
        # A reconstructed terrain contains real relief and may also contain
        # foreground objects.  Sampling the desired absolute top surface first
        # prevents those elevations from being added a second time to a
        # prescribed sand thickness.  Cells where terrain already lies above
        # the target simply contain no sand.
        target_sand_surface = sample_height_spec(
            initial["sand_surface"], grid, name="initial.sand_surface"
        )
        sand = np.maximum(target_sand_surface - terrain, 0.0)
        sand_kind = "absolute_surface"
    elif isinstance(initial["sand"], dict) and "exchange_field" in initial["sand"]:
        spec = initial["sand"]
        if spec != {"exchange_field": "sand_height"}:
            raise ValueError("initial.sand exchange reference must be exactly {exchange_field: sand_height}")
        descriptor = metadata["frames"][0]["fields"].get("sand_height")
        if not isinstance(descriptor, dict) or descriptor.get("semantic") != "sand_thickness":
            raise ValueError("exchange frame 0 needs a sand_height field with sand_thickness semantic")
        if "nonnegative" not in descriptor.get("constraints", []):
            raise ValueError("exchange sand_height must declare the nonnegative constraint")
        sand = np.load(exchange_bundle / descriptor["path"], allow_pickle=False).astype(np.float64)
        sand_kind = "exchange_sand_height"
    else:
        # Retain the original thickness interpretation for old profiles and
        # synthetic tests where terrain-relative material depth is intended.
        sand = sample_height_spec(initial["sand"], grid, name="initial.sand")
        sand_kind = "terrain_relative_thickness"
    target_water_surface = None
    water_depth_limit = None
    water_initial_mask = None
    if "water_surface" in initial:
        # Unlike a uniform water thickness, a target free-surface elevation
        # fills low connected terrain without coating reconstructed objects.
        # A max-composed field can also add a smooth upstream head while
        # remaining independent from the prescribed sand height.
        water_surface_spec = initial["water_surface"]
        constrained_surface = (
            isinstance(water_surface_spec, dict)
            and "surface" in water_surface_spec
        )
        if constrained_surface:
            unknown = set(water_surface_spec) - {"surface", "max_depth", "mask"}
            if unknown:
                raise ValueError(
                    "initial.water_surface constraint has unknown keys: "
                    + ", ".join(sorted(unknown))
                )
            target_water_surface = sample_height_spec(
                water_surface_spec["surface"],
                grid,
                name="initial.water_surface.surface",
            )
        else:
            target_water_surface = sample_height_spec(
                water_surface_spec, grid, name="initial.water_surface"
            )
        total_water = np.maximum(target_water_surface - terrain, 0.0)
        if constrained_surface and "max_depth" in water_surface_spec:
            # Photogrammetric height fields can contain unobserved pits.  A
            # target hydraulic level alone would fill those pits with metres
            # of water, so experiments may cap the intended shallow layer with
            # a scalar or spatially varying depth field.
            water_depth_limit = sample_height_spec(
                water_surface_spec["max_depth"],
                grid,
                name="initial.water_surface.max_depth",
            )
            total_water = np.minimum(total_water, water_depth_limit)
        if constrained_surface and "mask" in water_surface_spec:
            # The mask is a [0,1] coverage field rather than a hard crop.  A
            # smooth box therefore avoids a one-cell water cliff at the edge
            # of the reconstructed, physically trustworthy Garden region.
            water_initial_mask = sample_height_spec(
                water_surface_spec["mask"],
                grid,
                name="initial.water_surface.mask",
            )
            if np.any(water_initial_mask > 1.0):
                raise ValueError("initial.water_surface.mask must not exceed 1")
            total_water *= water_initial_mask
        water_kind = (
            "constrained_absolute_surface_field"
            if constrained_surface
            else "absolute_surface_field"
        )
    elif "surface_level" in initial:
        surface_level = _finite_number(
            initial["surface_level"], "initial.surface_level"
        )
        total_water = np.maximum(surface_level - terrain, 0.0)
        water_kind = "absolute_surface_level"
    elif "saturation" in initial:
        saturation = _finite_number(
            initial["saturation"], "initial.saturation"
        )
        if not 0.0 <= saturation < 1.0:
            raise ValueError("initial.saturation must lie in [0,1)")
        # This is the solver's documented column indicator:
        # s = water / (water + sand). Solving for water creates a fully
        # reproducible mixed-water initial condition without duplicating a
        # second sand-shaped primitive in YAML.
        total_water = saturation / (1.0 - saturation) * sand
        water_kind = "sand_saturation"
    else:
        total_water = sample_height_spec(
            initial.get("total_water", 0.0),
            grid,
            name="initial.total_water",
        )
        water_kind = "height_field"

    base_total_water = None
    saturation_override_mask = None
    saturation_override_weight = None
    saturation_override_value = None
    saturation_override_height_epsilon = None
    saturation_override_boundary_feather_cells = None
    if saturation_override_spec is not None:
        saturation_override_value = _finite_number(
            saturation_override_spec["value"],
            "initial.sand_saturation_override.value",
        )
        if not 0.0 <= saturation_override_value < 1.0:
            raise ValueError(
                "initial.sand_saturation_override.value must lie in [0,1)"
            )
        saturation_override_height_epsilon = _finite_number(
            saturation_override_spec.get("sand_height_epsilon", 0.0),
            "initial.sand_saturation_override.sand_height_epsilon",
        )
        if saturation_override_height_epsilon < 0.0:
            raise ValueError(
                "initial.sand_saturation_override.sand_height_epsilon "
                "must be non-negative"
            )
        raw_boundary_feather_cells = saturation_override_spec.get(
            "boundary_feather_cells", 0
        )
        if (
            isinstance(raw_boundary_feather_cells, bool)
            or not isinstance(raw_boundary_feather_cells, int)
            or raw_boundary_feather_cells < 0
        ):
            raise ValueError(
                "initial.sand_saturation_override.boundary_feather_cells "
                "must be a non-negative integer"
            )
        saturation_override_boundary_feather_cells = (
            raw_boundary_feather_cells
        )

        # This is an initialization-only diagnostic.  First construct the
        # ordinary absolute hydraulic surface, then replace water only in
        # sand-bearing cells by the amount that gives the requested column
        # saturation phi = h_w / (h_w + h_s).  Sand-free cells retain the
        # original reservoir exactly.  A uniform phi and a globally planar
        # H+h_w cannot both hold over non-planar terrain, so keep the base
        # field beside the override for an explicit hydraulic audit.
        base_total_water = total_water.copy()
        saturation_override_mask = sand > saturation_override_height_epsilon
        if not np.any(saturation_override_mask):
            raise ValueError(
                "initial.sand_saturation_override selected no sand cells"
            )
        water_to_sand_ratio = (
            saturation_override_value / (1.0 - saturation_override_value)
        )
        saturation_override_weight = _interior_feather_weight(
            saturation_override_mask,
            saturation_override_boundary_feather_cells,
        )
        if not np.any(saturation_override_weight > 0.0):
            raise ValueError(
                "initial.sand_saturation_override boundary feather removed "
                "the entire sand interior"
            )
        uniform_total_water = water_to_sand_ratio * sand
        total_water = total_water.copy()
        # Keep the original hydraulic state at the thin-sand boundary and
        # smoothly approach uniform saturation toward the dam core.  A hard
        # replacement through the last nonzero-sand cell creates a one-cell
        # jump in both water depth and constitutive cohesion, contaminating
        # the diagnostic with a large artificial edge impulse.
        total_water += saturation_override_weight * (
            uniform_total_water - total_water
        )
        water_kind += "_with_uniform_sand_saturation_override"

    dx, dy = grid.spacing
    if not np.isclose(dx, dy, rtol=0.0, atol=1e-12):
        raise ValueError("sand-water-mixtures currently requires square cells")
    boundary_width = 1
    output.mkdir(parents=True, exist_ok=True)
    output_fields = [
        ("terrain.npy", terrain),
        ("sand.npy", sand),
        ("total_water.npy", total_water),
    ]
    if target_sand_surface is not None:
        # This field is not consumed by the solver, but keeping it beside the
        # derived thickness makes initialization errors directly auditable.
        output_fields.append(
            ("target_sand_surface.npy", target_sand_surface)
        )
    if target_water_surface is not None:
        output_fields.append(
            ("target_water_surface.npy", target_water_surface)
        )
    if water_depth_limit is not None:
        output_fields.append(("water_depth_limit.npy", water_depth_limit))
    if water_initial_mask is not None:
        output_fields.append(("water_initial_mask.npy", water_initial_mask))
    if base_total_water is not None:
        output_fields.extend(
            [
                (
                    "base_total_water_before_saturation_override.npy",
                    base_total_water,
                ),
                (
                    "sand_saturation_override_mask.npy",
                    saturation_override_mask.astype(np.float64),
                ),
                (
                    "sand_saturation_override_weight.npy",
                    saturation_override_weight,
                ),
            ]
        )
    for filename, values in output_fields:
        np.save(
            output / filename,
            np.ascontiguousarray(values, dtype=np.float64),
            allow_pickle=False,
        )

    center = [
        0.5 * (grid.domain_min[0] + grid.domain_max[0]),
        0.5 * (grid.domain_min[1] + grid.domain_max[1]),
    ]
    npy_geometry = {
        "spacing": float(dx),
        "origin": list(grid.cell_center_origin),
    }
    physics = {
        "gravity": 9.81,
        "diffusion_rate": 0.0,
        "momentum_exchange_rate": 1.0,
        "ground_friction": 0.42,
        # Only relax the sand/dry-terrain contact classifier. Profiles may use
        # a small positive value for reconstructed ground without altering H.
        "sand_terrain_contact_tolerance": 0.0,
        "young_modulus": 75000.0,
        "poisson": 0.25,
        "drucker_prager_friction": 0.5,
        "enable_sand_internal_force": True,
    }
    profile_physics = profile.get("physics", {})
    if not isinstance(profile_physics, dict):
        raise ValueError("P3 profile physics must be a mapping")
    physics.update(profile_physics)
    run_steps = int(solver.get("steps", 1))
    frame_stride = int(solver["frame_stride"])
    duration = solver.get("duration_seconds")
    if run_steps < 1 or frame_stride < 1:
        raise ValueError("solver steps and frame_stride must be positive")
    scene = {
        "name": str(profile.get("profile_id", "sandygs-p3-sand")),
        "grid": {
            "resolution": [
                grid.resolution[0] + 2 * boundary_width,
                grid.resolution[1] + 2 * boundary_width,
            ],
            "spacing": float(dx),
            "boundary_width": boundary_width,
            "center": center,
        },
        "physics": physics,
        "time": {
            "water_dt": float(solver["water_dt"]),
            "sand_substeps": int(solver.get("sand_substeps", 3)),
            "adaptive": True,
            "target_cfl": float(solver["target_cfl"]),
            "internal_force_height_fraction": float(
                solver.get("internal_force_height_fraction", 0.0004)
            ),
        },
        "exchange_scheme": str(profile.get("exchange_scheme", "paper")),
        "initial": {
            "terrain": {
                "type": "npy",
                "file": "terrain.npy",
                **npy_geometry,
            },
            "sand": {
                "type": "npy",
                "file": "sand.npy",
                **npy_geometry,
            },
            "total_water": {
                "type": "npy",
                "file": "total_water.npy",
                **npy_geometry,
            },
        },
        "run": {
            "steps": run_steps,
            "frame_stride": frame_stride,
            "diagnostic_snapshots": False,
            "write_obj": False,
        },
    }
    if duration is not None:
        duration = _finite_number(duration, "solver.duration_seconds")
        if duration <= 0.0:
            raise ValueError("solver.duration_seconds must be positive")
        scene["run"]["duration"] = duration
    _write_yaml(output / "scene.yaml", scene)

    cell_area = dx * dy
    case_metadata = {
        "format_version": 1,
        "case": "sandygs.p3_sand_water",
        "scene_id": metadata["scene_id"],
        "profile_id": profile.get("profile_id"),
        "exchange_bundle": str(exchange_bundle),
        "initialization": {
            "sand_kind": sand_kind,
            "sand_cell_count": int(np.count_nonzero(sand > 0.0)),
            "sand_height_min": float(sand.min()),
            "sand_height_max": float(sand.max()),
            "sand_volume": float(sand.sum() * cell_area),
            "water_kind": water_kind,
            "water_cell_count": int(np.count_nonzero(total_water > 0.0)),
            "water_height_max": float(total_water.max()),
            "water_volume": float(total_water.sum() * cell_area),
        },
        "run": {
            "water_dt": float(solver["water_dt"]),
            "sand_substeps": int(solver.get("sand_substeps", 3)),
            "steps": run_steps,
            "duration_seconds": duration,
            "frame_stride": frame_stride,
            "target_cfl": float(solver["target_cfl"]),
            "recommended_snapshot_level": "render",
        },
        "files": {
            filename: sha256_file(output / filename)
            for filename, _ in output_fields
        },
    }
    if target_sand_surface is not None:
        case_metadata["initialization"].update(
            {
                "target_surface_min": float(target_sand_surface.min()),
                "target_surface_max": float(target_sand_surface.max()),
            }
        )
    if target_water_surface is not None:
        wet = total_water > 0.0
        reconstructed_surface = terrain + total_water
        # A nonzero difference is intentional wherever the optional mask or
        # depth cap prevents an unreliable terrain depression from filling to
        # the requested hydraulic level.  Report it as a residual, not an
        # initialization error, for constrained profiles.
        surface_residual = (
            float(np.max(np.abs(reconstructed_surface[wet] - target_water_surface[wet])))
            if np.any(wet)
            else 0.0
        )
        case_metadata["initialization"].update(
            {
                "target_water_surface_min": float(target_water_surface.min()),
                "target_water_surface_max": float(target_water_surface.max()),
                "target_water_surface_wet_residual_max": surface_residual,
            }
        )
        if water_depth_limit is None and water_initial_mask is None:
            # Preserve the original strict equality metric for unconstrained
            # absolute-surface profiles and their existing regression tests.
            case_metadata["initialization"][
                "target_water_surface_wet_error_max"
            ] = surface_residual
    if water_depth_limit is not None:
        case_metadata["initialization"].update(
            {
                "water_depth_limit_min": float(water_depth_limit.min()),
                "water_depth_limit_max": float(water_depth_limit.max()),
                "water_depth_capped_cell_count": int(
                    np.count_nonzero(
                        np.maximum(target_water_surface - terrain, 0.0)
                        > water_depth_limit
                    )
                ),
            }
        )
    if water_initial_mask is not None:
        case_metadata["initialization"].update(
            {
                "water_mask_cell_count": int(
                    np.count_nonzero(water_initial_mask > 0.0)
                ),
                "water_mask_fraction": float(
                    np.count_nonzero(water_initial_mask > 0.0)
                    / water_initial_mask.size
                ),
            }
        )
    if saturation_override_mask is not None:
        material_height = sand + total_water
        realized_saturation = np.divide(
            total_water,
            material_height,
            out=np.zeros_like(total_water),
            where=material_height > 0.0,
        )
        before_volume = float(base_total_water.sum() * cell_area)
        after_volume = float(total_water.sum() * cell_area)
        modified_mask = saturation_override_weight > 0.0
        full_override_mask = saturation_override_weight >= 1.0 - 1.0e-12
        diagnostic_mask = (
            full_override_mask
            if np.any(full_override_mask)
            else modified_mask
        )
        override_values = realized_saturation[diagnostic_mask]
        case_metadata["initialization"].update(
            {
                "sand_saturation_override": saturation_override_value,
                "sand_saturation_override_height_epsilon": (
                    saturation_override_height_epsilon
                ),
                "sand_saturation_override_boundary_feather_cells": (
                    saturation_override_boundary_feather_cells
                ),
                "sand_saturation_override_support_cell_count": int(
                    np.count_nonzero(saturation_override_mask)
                ),
                # Retain the original key for manifests produced before the
                # feathered diagnostic was introduced.
                "sand_saturation_override_cell_count": int(
                    np.count_nonzero(saturation_override_mask)
                ),
                "sand_saturation_override_modified_cell_count": int(
                    np.count_nonzero(modified_mask)
                ),
                "sand_saturation_override_full_cell_count": int(
                    np.count_nonzero(full_override_mask)
                ),
                "sand_saturation_override_realized_min": float(
                    override_values.min()
                ),
                "sand_saturation_override_realized_max": float(
                    override_values.max()
                ),
                "base_water_volume_before_saturation_override": before_volume,
                "water_volume_change_from_saturation_override": (
                    after_volume - before_volume
                ),
            }
        )
    _write_yaml(output / "case.yaml", case_metadata)
    return case_metadata
