"""Version 1 SandyGS simulation-exchange protocol and coordinate helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


PROTOCOL_NAME = "sandygs.simulation_exchange"
FORMAT_VERSION = 1


class ExchangeValidationError(ValueError):
    """Raised when an exchange bundle violates the protocol."""

    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class GridSpec:
    """Geometry of a two-dimensional, cell-centered scalar grid."""

    resolution: tuple[int, int]
    domain_min: tuple[float, float]
    domain_max: tuple[float, float]

    def __post_init__(self) -> None:
        if len(self.resolution) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 2
            for value in self.resolution
        ):
            raise ValueError("resolution must contain two integers >= 2")
        bounds = (*self.domain_min, *self.domain_max)
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError("grid bounds must be finite")
        if any(
            upper <= lower
            for lower, upper in zip(self.domain_min, self.domain_max)
        ):
            raise ValueError("domain_max must be greater than domain_min")

    @property
    def spacing(self) -> tuple[float, float]:
        return tuple(
            (upper - lower) / count
            for lower, upper, count in zip(
                self.domain_min, self.domain_max, self.resolution
            )
        )

    @property
    def cell_center_origin(self) -> tuple[float, float]:
        return tuple(
            lower + 0.5 * spacing
            for lower, spacing in zip(self.domain_min, self.spacing)
        )


def _rotation_matrix_xyz(euler_degrees: Iterable[float]) -> np.ndarray:
    angles = tuple(float(value) for value in euler_degrees)
    if len(angles) != 3 or not all(math.isfinite(value) for value in angles):
        raise ValueError("Euler rotation must contain three finite angles")
    rx, ry, rz = np.deg2rad(angles)
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(rx), -np.sin(rx)],
         [0.0, np.sin(rx), np.cos(rx)]],
        dtype=np.float64,
    )
    rotation_y = np.array(
        [[np.cos(ry), 0.0, np.sin(ry)], [0.0, 1.0, 0.0],
         [-np.sin(ry), 0.0, np.cos(ry)]],
        dtype=np.float64,
    )
    rotation_z = np.array(
        [[np.cos(rz), -np.sin(rz), 0.0],
         [np.sin(rz), np.cos(rz), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rotation_z @ rotation_y @ rotation_x


def build_legacy_sim_to_gs(
    rotation_degrees: Iterable[float],
    translation: Iterable[float],
    scale: Iterable[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Return the explicit inverse of the height-field alignment.

    The mapping is ``p_sim = T @ R @ S @ p_gs`` for column vectors, where
    ``R = Rz @ Ry @ Rx``. Water points are mapped back with the inverse. Garden
    uses unit scale, but scale remains explicit so metadata never hides it.
    """

    translation_array = np.asarray(tuple(translation), dtype=np.float64)
    scale_array = np.asarray(tuple(scale), dtype=np.float64)
    if translation_array.shape != (3,) or not np.isfinite(translation_array).all():
        raise ValueError("translation must contain three finite values")
    if (
        scale_array.shape != (3,)
        or not np.isfinite(scale_array).all()
        or np.any(scale_array == 0.0)
    ):
        raise ValueError("scale must contain three finite non-zero values")

    gs_to_sim = np.eye(4, dtype=np.float64)
    gs_to_sim[:3, :3] = _rotation_matrix_xyz(rotation_degrees) @ np.diag(
        scale_array
    )
    gs_to_sim[:3, 3] = translation_array
    return np.linalg.inv(gs_to_sim)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a column-vector 4x4 transform to one point or an array of points."""

    points_array = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if points_array.shape[-1:] != (3,) or not np.isfinite(points_array).all():
        raise ValueError("points must be finite and have a final dimension of 3")
    homogeneous = np.concatenate(
        [points_array, np.ones((*points_array.shape[:-1], 1), dtype=np.float64)],
        axis=-1,
    )
    transformed = homogeneous @ matrix.T
    return transformed[..., :3] / transformed[..., 3:4]


def cell_center_coordinates(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return x and y center coordinates without image-axis transposition."""

    dx, dy = grid.spacing
    x = grid.domain_min[0] + (np.arange(grid.resolution[0]) + 0.5) * dx
    y = grid.domain_min[1] + (np.arange(grid.resolution[1]) + 0.5) * dy
    return x, y


def rainygs_endpoint_image_to_cell_field(
    image_yx: np.ndarray, grid: GridSpec
) -> np.ndarray:
    """Convert an endpoint-sampled ``[y,x]`` map to ``[x,y]`` cells.

    Source map coordinates use inclusive ``linspace`` endpoints.
    The exchange protocol uses cell centers. The two separable interpolation
    passes make that half-cell change explicit instead of silently relabelling
    the samples.
    """

    source = np.asarray(image_yx)
    if source.ndim != 2:
        raise ValueError("Endpoint height map must be a two-dimensional array")
    if source.shape != (grid.resolution[1], grid.resolution[0]):
        raise ValueError(
            "Height map [height,width] does not match exchange [nx,ny]: "
            f"{source.shape} versus {grid.resolution}"
        )
    if not np.issubdtype(source.dtype, np.number) or not np.isfinite(source).all():
        raise ValueError("Endpoint height map must contain only finite numbers")

    source_x = np.linspace(
        grid.domain_min[0], grid.domain_max[0], source.shape[1], dtype=np.float64
    )
    source_y = np.linspace(
        grid.domain_min[1], grid.domain_max[1], source.shape[0], dtype=np.float64
    )
    target_x, target_y = cell_center_coordinates(grid)

    interpolated_x = np.empty((source.shape[0], target_x.size), dtype=np.float64)
    for source_y_index, row in enumerate(source):
        interpolated_x[source_y_index] = np.interp(target_x, source_x, row)
    target_yx = np.empty((target_y.size, target_x.size), dtype=np.float64)
    for target_x_index in range(target_x.size):
        target_yx[:, target_x_index] = np.interp(
            target_y, source_y, interpolated_x[:, target_x_index]
        )
    return np.ascontiguousarray(target_yx.T, dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(bundle: Path) -> dict[str, Any]:
    metadata_path = Path(bundle) / "metadata.yaml"
    try:
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ExchangeValidationError([f"cannot load {metadata_path}: {exc}"]) from exc
    if not isinstance(metadata, dict):
        raise ExchangeValidationError(["metadata.yaml must contain a mapping"])
    return metadata


def _sequence(
    value: Any, length: int, name: str, errors: list[str]
) -> list[Any] | None:
    if not isinstance(value, list) or len(value) != length:
        errors.append(f"{name} must be a list of length {length}")
        return None
    return value


def _safe_bundle_path(
    bundle: Path, relative: Any, name: str, errors: list[str]
) -> Path | None:
    if not isinstance(relative, str) or not relative:
        errors.append(f"{name} must be a non-empty relative path")
        return None
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        errors.append(f"{name} must stay inside the exchange bundle")
        return None
    path = bundle / candidate
    try:
        path.resolve(strict=False).relative_to(bundle.resolve())
    except ValueError:
        errors.append(f"{name} resolves outside the exchange bundle")
        return None
    return path


def validate_bundle(bundle: Path, *, verify_checksums: bool = True) -> dict[str, Any]:
    """Validate metadata, arrays, coordinates, constraints and frame time."""

    bundle = Path(bundle)
    metadata = load_metadata(bundle)
    errors: list[str] = []

    if metadata.get("protocol") != PROTOCOL_NAME:
        errors.append(f"protocol must be {PROTOCOL_NAME!r}")
    if metadata.get("format_version") != FORMAT_VERSION:
        errors.append(f"format_version must be {FORMAT_VERSION}")

    grid_data = metadata.get("grid")
    if not isinstance(grid_data, dict):
        errors.append("grid must be a mapping")
        grid_data = {}
    resolution = _sequence(grid_data.get("resolution"), 2, "grid.resolution", errors)
    domain_min = _sequence(grid_data.get("domain_min"), 2, "grid.domain_min", errors)
    domain_max = _sequence(grid_data.get("domain_max"), 2, "grid.domain_max", errors)
    spacing = _sequence(grid_data.get("spacing"), 2, "grid.spacing", errors)
    center_origin = _sequence(
        grid_data.get("cell_center_origin"), 2, "grid.cell_center_origin", errors
    )
    grid: GridSpec | None = None
    if resolution is not None and domain_min is not None and domain_max is not None:
        try:
            grid = GridSpec(
                tuple(resolution), tuple(float(v) for v in domain_min),
                tuple(float(v) for v in domain_max)
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid grid: {exc}")
    if grid is not None:
        if spacing is not None and not np.allclose(
            np.asarray(spacing, dtype=float), grid.spacing, rtol=0.0, atol=1e-9
        ):
            errors.append("grid.spacing is inconsistent with domain/resolution")
        if center_origin is not None and not np.allclose(
            np.asarray(center_origin, dtype=float),
            grid.cell_center_origin,
            rtol=0.0,
            atol=1e-9,
        ):
            errors.append("grid.cell_center_origin is not the first cell center")
    expected_grid_values = {
        "axis_order": ["x", "y"],
        "centering": "cell",
        "storage_order": "C",
        "exchange_boundary_width": 0,
    }
    for key, expected in expected_grid_values.items():
        if grid_data.get(key) != expected:
            errors.append(f"grid.{key} must be {expected!r}")

    units = metadata.get("units")
    if not isinstance(units, dict):
        errors.append("units must be a mapping")
        units = {}
    if units.get("scale_status") not in {"unverified", "assumed", "measured"}:
        errors.append("units.scale_status must be unverified, assumed or measured")
    metric_scale = units.get("meters_per_simulation_unit")
    if metric_scale is not None:
        if (
            isinstance(metric_scale, bool)
            or not isinstance(metric_scale, (int, float))
            or not math.isfinite(metric_scale)
            or metric_scale <= 0.0
        ):
            errors.append("meters_per_simulation_unit must be null or positive")
    elif units.get("scale_status") == "measured":
        errors.append("measured scale requires meters_per_simulation_unit")

    coordinates = metadata.get("coordinates")
    if not isinstance(coordinates, dict):
        errors.append("coordinates must be a mapping")
        coordinates = {}
    matrices: dict[str, np.ndarray] = {}
    for name in ("sim_to_gs", "gs_to_sim"):
        try:
            matrix = np.asarray(coordinates.get(name), dtype=np.float64)
        except (TypeError, ValueError):
            matrix = np.empty((0, 0))
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            errors.append(f"coordinates.{name} must be a finite 4x4 matrix")
        else:
            matrices[name] = matrix
    if len(matrices) == 2 and not np.allclose(
        matrices["sim_to_gs"] @ matrices["gs_to_sim"],
        np.eye(4),
        rtol=0.0,
        atol=1e-9,
    ):
        errors.append("sim_to_gs and gs_to_sim are not inverses")
    if coordinates.get("vector_convention") != "column":
        errors.append("coordinates.vector_convention must be 'column'")
    if coordinates.get("handedness") != "right":
        errors.append("coordinates.handedness must be 'right'")
    if coordinates.get("height_axis") != "+z":
        errors.append("coordinates.height_axis must be '+z'")

    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        errors.append("frames must be a non-empty list")
        frames = []
    seen_frame_indices: set[int] = set()
    previous_time = -math.inf
    expected_shape = tuple(grid.resolution) if grid is not None else None
    for frame_position, frame in enumerate(frames):
        prefix = f"frames[{frame_position}]"
        if not isinstance(frame, dict):
            errors.append(f"{prefix} must be a mapping")
            continue
        frame_index = frame.get("frame_index")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            errors.append(f"{prefix}.frame_index must be a non-negative integer")
        elif frame_index in seen_frame_indices:
            errors.append(f"{prefix}.frame_index duplicates {frame_index}")
        else:
            seen_frame_indices.add(frame_index)
        step_index = frame.get("step_index")
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            errors.append(f"{prefix}.step_index must be a non-negative integer")
        time_seconds = frame.get("time_seconds")
        if (
            isinstance(time_seconds, bool)
            or not isinstance(time_seconds, (int, float))
            or not math.isfinite(time_seconds)
            or time_seconds < 0.0
        ):
            errors.append(f"{prefix}.time_seconds must be finite and non-negative")
        elif time_seconds < previous_time:
            errors.append(f"{prefix}.time_seconds is not monotonic")
        else:
            previous_time = float(time_seconds)

        fields = frame.get("fields")
        if not isinstance(fields, dict) or not fields:
            errors.append(f"{prefix}.fields must be a non-empty mapping")
            continue
        for field_name, descriptor in fields.items():
            field_prefix = f"{prefix}.fields.{field_name}"
            if not isinstance(descriptor, dict):
                errors.append(f"{field_prefix} must be a mapping")
                continue
            path = _safe_bundle_path(
                bundle, descriptor.get("path"), f"{field_prefix}.path", errors
            )
            if descriptor.get("centering") != "cell":
                errors.append(f"{field_prefix}.centering must be 'cell'")
            if descriptor.get("dtype") != "float32":
                errors.append(f"{field_prefix}.dtype must be 'float32'")
            descriptor_shape = descriptor.get("shape")
            if not isinstance(descriptor_shape, list):
                descriptor_shape_tuple: tuple[Any, ...] = ()
            else:
                descriptor_shape_tuple = tuple(descriptor_shape)
            if (
                expected_shape is not None
                and descriptor_shape_tuple != expected_shape
            ):
                errors.append(
                    f"{field_prefix}.shape must be {list(expected_shape)}"
                )
            if not isinstance(descriptor.get("semantic"), str) or not descriptor[
                "semantic"
            ]:
                errors.append(f"{field_prefix}.semantic must be a non-empty string")
            if not isinstance(descriptor.get("units"), str) or not descriptor["units"]:
                errors.append(f"{field_prefix}.units must be a non-empty string")
            if path is None:
                continue
            try:
                array = np.load(path, allow_pickle=False)
            except (OSError, ValueError) as exc:
                errors.append(f"cannot load {path}: {exc}")
                continue
            if array.dtype != np.float32:
                errors.append(f"{path} has dtype {array.dtype}, expected float32")
            if expected_shape is not None and array.shape != expected_shape:
                errors.append(f"{path} has shape {array.shape}, expected {expected_shape}")
            if not np.isfinite(array).all():
                errors.append(f"{path} contains non-finite values")
            constraints = descriptor.get("constraints", [])
            if not isinstance(constraints, list):
                errors.append(f"{field_prefix}.constraints must be a list")
                constraints = []
            if "finite" not in constraints:
                errors.append(f"{field_prefix}.constraints must include 'finite'")
            if "nonnegative" in constraints and np.any(array < 0.0):
                errors.append(f"{path} violates the nonnegative constraint")
            checksum = descriptor.get("sha256")
            if verify_checksums and checksum is not None:
                if not isinstance(checksum, str) or sha256_file(path) != checksum:
                    errors.append(f"{path} does not match its sha256")

    if errors:
        raise ExchangeValidationError(errors)
    return metadata
