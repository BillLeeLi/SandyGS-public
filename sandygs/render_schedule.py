"""CPU-only expansion of declarative time/camera render schedules.

The renderer used to advance physical time implicitly whenever it advanced to
the next camera.  That made a moving-camera clip possible, but it could not
express a fixed-camera time series or synchronized views of one solver state.
This module turns a compact profile into an auditable per-frame table before
CUDA or a PGSR checkpoint is loaded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class ScheduledRenderFrame:
    sequence: str
    sequence_type: str
    output_group: str
    output_name: str
    time_seconds: float
    camera_index: int
    save_debug: bool
    save_intermediates: bool

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _sample_values(spec: Any, name: str) -> list[float]:
    if isinstance(spec, list):
        if not spec:
            raise ValueError(f"{name} must not be empty")
        values = [_finite_float(value, f"{name}[{index}]") for index, value in enumerate(spec)]
    elif isinstance(spec, dict):
        start = _finite_float(spec.get("start"), f"{name}.start")
        end = _finite_float(spec.get("end"), f"{name}.end")
        if end < start:
            raise ValueError(f"{name}.end must not precede start")
        if "count" in spec:
            count = _positive_int(spec["count"], f"{name}.count")
        else:
            fps = _finite_float(spec.get("fps"), f"{name}.fps")
            if fps <= 0.0:
                raise ValueError(f"{name}.fps must be positive")
            count = int(round((end - start) * fps)) + 1
        if count == 1:
            values = [start]
        else:
            step = (end - start) / (count - 1)
            values = [start + index * step for index in range(count)]
    else:
        raise ValueError(f"{name} must be a list or start/end mapping")
    if any(value < 0.0 for value in values):
        raise ValueError(f"{name} cannot contain negative physical time")
    if any(later < earlier for earlier, later in zip(values, values[1:])):
        raise ValueError(f"{name} must be non-decreasing")
    return values


def _camera_indices(spec: Any, name: str) -> list[int]:
    if isinstance(spec, list):
        values = spec
    elif isinstance(spec, dict):
        start = int(spec.get("start", 0))
        end = int(spec.get("end"))
        count = _positive_int(spec.get("count"), f"{name}.count")
        if start < 0 or end < start:
            raise ValueError(f"{name} must satisfy 0 <= start <= end")
        if count == 1:
            values = [start]
        else:
            values = [
                int(round(start + index * (end - start) / (count - 1)))
                for index in range(count)
            ]
    else:
        raise ValueError(f"{name} must be a list or start/end/count mapping")
    indices = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"{name}[{index}] must be a non-negative integer")
        indices.append(int(value))
    if not indices:
        raise ValueError(f"{name} must not be empty")
    return indices


def _debug_indices(spec: Any, frame_count: int, name: str) -> set[int]:
    if spec in (None, False, "none"):
        return set()
    if spec in (True, "all"):
        return set(range(frame_count))
    if not isinstance(spec, list):
        raise ValueError(f"{name} must be all, none, or a list of local indices")
    result = set()
    for position, value in enumerate(spec):
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{name}[{position}] must be an integer")
        index = int(value)
        if index < 0:
            index += frame_count
        if not 0 <= index < frame_count:
            raise ValueError(f"{name}[{position}] is outside this sequence")
        result.add(index)
    return result


def expand_render_schedule(spec: dict[str, Any]) -> tuple[list[ScheduledRenderFrame], int]:
    """Expand fixed, synchronized and moving-camera sequence declarations."""

    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("render_schedule must be a schema_version 1 mapping")
    camera_path = spec.get("camera_path", "generated_orbit")
    if camera_path not in {"generated_orbit", "observed_cameras"}:
        raise ValueError("camera_path must be generated_orbit or observed_cameras")
    camera_path_frames = _positive_int(
        spec.get("camera_path_frames"), "render_schedule.camera_path_frames"
    )
    azimuth_range = spec.get("camera_path_azimuth_degrees")
    if camera_path == "observed_cameras" and azimuth_range is not None:
        raise ValueError("observed_cameras does not use camera_path_azimuth_degrees")
    if azimuth_range is not None:
        if not isinstance(azimuth_range, (list, tuple)) or len(azimuth_range) != 2:
            raise ValueError(
                "render_schedule.camera_path_azimuth_degrees must contain "
                "[start, end]"
            )
        azimuth_start = _finite_float(
            azimuth_range[0],
            "render_schedule.camera_path_azimuth_degrees[0]",
        )
        azimuth_end = _finite_float(
            azimuth_range[1],
            "render_schedule.camera_path_azimuth_degrees[1]",
        )
        if azimuth_end <= azimuth_start:
            raise ValueError(
                "render_schedule.camera_path_azimuth_degrees must be increasing"
            )
    sequences = spec.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("render_schedule.sequences must be a non-empty list")

    frames: list[ScheduledRenderFrame] = []
    names: set[str] = set()
    for sequence_index, sequence in enumerate(sequences):
        prefix = f"render_schedule.sequences[{sequence_index}]"
        if not isinstance(sequence, dict):
            raise ValueError(f"{prefix} must be a mapping")
        name = sequence.get("name")
        if not isinstance(name, str) or not name or any(
            part in name for part in ("/", "\\", "..")
        ):
            raise ValueError(f"{prefix}.name must be a safe non-empty name")
        if name in names:
            raise ValueError(f"duplicate render sequence name: {name}")
        names.add(name)
        sequence_type = sequence.get("type")
        times = _sample_values(sequence.get("times"), f"{prefix}.times")
        debug = _debug_indices(sequence.get("debug", "none"), len(times), f"{prefix}.debug")
        intermediate_mode = sequence.get("intermediates", "debug")
        if intermediate_mode not in {"all", "none", "debug"}:
            raise ValueError(f"{prefix}.intermediates must be all, none, or debug")

        def flags(local_index: int) -> tuple[bool, bool]:
            save_debug = local_index in debug
            save_intermediates = intermediate_mode == "all" or (
                intermediate_mode == "debug" and save_debug
            )
            return save_debug, save_intermediates

        if sequence_type == "fixed_camera":
            cameras = _camera_indices([sequence.get("camera_index")], f"{prefix}.camera_index")
            camera_index = cameras[0]
            for local_index, time_seconds in enumerate(times):
                save_debug, save_intermediates = flags(local_index)
                frames.append(ScheduledRenderFrame(
                    sequence=name,
                    sequence_type=sequence_type,
                    output_group=name,
                    output_name=f"{local_index:05d}",
                    time_seconds=time_seconds,
                    camera_index=camera_index,
                    save_debug=save_debug,
                    save_intermediates=save_intermediates,
                ))
        elif sequence_type == "synchronized_multiview":
            cameras = _camera_indices(sequence.get("camera_indices"), f"{prefix}.camera_indices")
            for time_index, time_seconds in enumerate(times):
                save_debug, save_intermediates = flags(time_index)
                for camera_index in cameras:
                    frames.append(ScheduledRenderFrame(
                        sequence=name,
                        sequence_type=sequence_type,
                        output_group=f"{name}/time_{time_index:04d}",
                        output_name=f"camera_{camera_index:03d}",
                        time_seconds=time_seconds,
                        camera_index=camera_index,
                        save_debug=save_debug,
                        save_intermediates=save_intermediates,
                    ))
        elif sequence_type == "moving_camera":
            cameras = _camera_indices(sequence.get("camera_indices"), f"{prefix}.camera_indices")
            if len(cameras) != len(times):
                raise ValueError(
                    f"{prefix} moving_camera needs one camera index per time"
                )
            for local_index, (time_seconds, camera_index) in enumerate(zip(times, cameras)):
                save_debug, save_intermediates = flags(local_index)
                frames.append(ScheduledRenderFrame(
                    sequence=name,
                    sequence_type=sequence_type,
                    output_group=name,
                    output_name=f"{local_index:05d}",
                    time_seconds=time_seconds,
                    camera_index=camera_index,
                    save_debug=save_debug,
                    save_intermediates=save_intermediates,
                ))
        else:
            raise ValueError(
                f"{prefix}.type must be fixed_camera, synchronized_multiview, "
                "or moving_camera"
            )

    invalid = [frame.camera_index for frame in frames if frame.camera_index >= camera_path_frames]
    if invalid:
        path_description = "generated path" if camera_path == "generated_orbit" else "observed camera list"
        raise ValueError(
            f"camera index {invalid[0]} exceeds {path_description} size {camera_path_frames}"
        )
    return frames, camera_path_frames


def resolve_profile_render_window(
    render: dict[str, Any],
    solver: dict[str, Any],
    *,
    max_views: int,
) -> dict[str, Any]:
    """Resolve legacy sequential timing or an explicit schedule before a run."""

    if not isinstance(render, dict) or not isinstance(solver, dict):
        raise ValueError("render and solver must be mappings")
    if isinstance(max_views, bool) or not isinstance(max_views, int) or max_views <= 0:
        raise ValueError("max_views must be a positive integer")
    schedule = render.get("render_schedule")
    if schedule is None:
        first = _finite_float(
            render.get("frame_time_seconds"), "render.frame_time_seconds"
        )
        fps = _finite_float(render.get("fps"), "render.fps")
        if fps <= 0.0:
            raise ValueError("render.fps must be positive")
        return {
            "frames": None,
            "camera_path_frames": None,
            "first_time_seconds": first,
            "last_time_seconds": first + (max_views - 1) / fps,
        }

    frames, camera_path_frames = expand_render_schedule(schedule)
    duration = solver.get("duration_seconds")
    if duration is None:
        raise ValueError(
            "an explicit render schedule requires solver.duration_seconds"
        )
    duration_seconds = _finite_float(duration, "solver.duration_seconds")
    times = [frame.time_seconds for frame in frames]
    first = min(times)
    last = max(times)
    if last > duration_seconds + 1.0e-12:
        raise ValueError("render schedule ends after solver.duration_seconds")
    return {
        "frames": frames,
        "camera_path_frames": camera_path_frames,
        "first_time_seconds": first,
        "last_time_seconds": last,
    }
