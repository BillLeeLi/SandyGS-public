#!/usr/bin/env python3
"""Render and encode a sparse-diagnostic existing-sand E4 demo."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sandygs.exchange import sha256_file
from sandygs.existing_sand.workspace import ExistingSandError, _load_yaml
from sandygs.pipeline import build_renderer_environment
from sandygs.render_schedule import ScheduledRenderFrame, expand_render_schedule
from sandygs.storage_guard import StorageGuardError, validate_artifact_storage
from scripts.experiments.existing_sand_e4_render import build_plan


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExistingSandError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ExistingSandError(f"{label} must be a finite number")
    return result


def _demo_settings(config: dict) -> dict:
    raw = config.get("demo")
    if not isinstance(raw, dict):
        raise ExistingSandError("E4 demo config requires a demo mapping")
    fps = _finite_number(raw.get("fps"), "demo.fps")
    start = _finite_number(
        raw.get("start_time_seconds"), "demo.start_time_seconds"
    )
    end = _finite_number(
        raw.get("end_time_seconds"), "demo.end_time_seconds"
    )
    if fps <= 0.0 or start < 0.0 or end <= start:
        raise ExistingSandError(
            "E4 demo needs fps>0 and 0<=start_time_seconds<end_time_seconds"
        )
    interval_count_float = (end - start) * fps
    interval_count = int(round(interval_count_float))
    if (
        interval_count <= 0
        or not math.isclose(
            interval_count_float, interval_count, rel_tol=0.0, abs_tol=1e-8
        )
    ):
        raise ExistingSandError(
            "E4 demo duration times fps must be an integer number of intervals"
        )
    times = [start + index / fps for index in range(interval_count + 1)]
    times[-1] = end
    diagnostic_times = raw.get("diagnostic_times_seconds")
    if not isinstance(diagnostic_times, list) or not diagnostic_times:
        raise ExistingSandError(
            "demo.diagnostic_times_seconds must be a non-empty list"
        )
    diagnostic_indices: list[int] = []
    checked_diagnostic_times: list[float] = []
    for position, value in enumerate(diagnostic_times):
        time_seconds = _finite_number(
            value, f"demo.diagnostic_times_seconds[{position}]"
        )
        index_float = (time_seconds - start) * fps
        index = int(round(index_float))
        if (
            index < 0
            or index > interval_count
            or not math.isclose(index_float, index, rel_tol=0.0, abs_tol=1e-8)
        ):
            raise ExistingSandError(
                "every E4 demo diagnostic time must lie on a rendered frame"
            )
        diagnostic_indices.append(index)
        checked_diagnostic_times.append(time_seconds)
    if diagnostic_indices != sorted(set(diagnostic_indices)):
        raise ExistingSandError(
            "demo.diagnostic_times_seconds must be strictly increasing"
        )

    video = raw.get("video", {})
    if not isinstance(video, dict):
        raise ExistingSandError("demo.video must be a mapping")
    codec = video.get("codec", "libx264")
    preset = video.get("preset", "slow")
    pixel_format = video.get("pixel_format", "yuv420p")
    crf = video.get("crf", 18)
    if codec != "libx264":
        raise ExistingSandError("the maintained E4 demo codec is libx264")
    if preset not in {
        "ultrafast", "superfast", "veryfast", "faster", "fast", "medium",
        "slow", "slower", "veryslow",
    }:
        raise ExistingSandError("demo.video.preset is not a libx264 preset")
    if pixel_format != "yuv420p":
        raise ExistingSandError("the maintained E4 demo pixel format is yuv420p")
    if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= 51:
        raise ExistingSandError("demo.video.crf must be an integer in [0,51]")
    multiview = video.get("multiview", True)
    if not isinstance(multiview, bool):
        raise ExistingSandError("demo.video.multiview must be a boolean")
    return {
        "fps": fps,
        "start_time_seconds": start,
        "end_time_seconds": end,
        "times_seconds": times,
        "diagnostic_times_seconds": checked_diagnostic_times,
        "diagnostic_indices": diagnostic_indices,
        "codec": codec,
        "preset": preset,
        "pixel_format": pixel_format,
        "crf": crf,
        "multiview": multiview,
    }


def build_demo_schedule(config: dict, settings: dict) -> dict:
    camera_ids = config.get("camera_ids")
    if not isinstance(camera_ids, list) or not camera_ids:
        raise ExistingSandError("E4 demo camera_ids must be a non-empty list")
    return {
        "schema_version": 1,
        "camera_path": "observed_cameras",
        "camera_path_frames": len(camera_ids),
        "sequences": [{
            "name": "demo_multiview",
            "type": "synchronized_multiview",
            "camera_indices": list(range(len(camera_ids))),
            "times": settings["times_seconds"],
            "debug": settings["diagnostic_indices"],
            "intermediates": "debug",
        }],
    }


def build_demo_plan(run_root: Path, config_path: Path) -> dict:
    config = _load_yaml(config_path, "E4 demo config")
    settings = _demo_settings(config)
    schedule = build_demo_schedule(config, settings)
    plan = build_plan(
        run_root,
        config_path,
        "water",
        times_override=settings["times_seconds"],
        render_schedule_override=schedule,
    )
    plan["demo"] = settings
    plan["diagnostic_frame_count"] = (
        len(settings["diagnostic_indices"]) * len(config["camera_ids"])
    )
    return plan


def _expected_rgb(output: Path, frames: list[ScheduledRenderFrame]) -> list[Path]:
    return [
        output / "release_check" / frame.output_group / "renders"
        / f"{frame.output_name}.png"
        for frame in frames
    ]


def render_demo(plan: dict) -> Path:
    output = Path(plan["output"])
    if output.exists():
        raise ExistingSandError(f"refusing to overwrite E4 demo render: {output}")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "render.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(plan["renderer"], stream, sort_keys=False)
    audit = {key: value for key, value in plan.items() if key != "renderer"}
    audit["output"] = str(output)
    with (output / "plan.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(audit, stream, sort_keys=False)
    command = [str(value) for value in plan["command"]]
    print("+", " ".join(command), flush=True)
    subprocess.run(
        command, cwd=ROOT, env=build_renderer_environment(), check=True
    )
    frames, _ = expand_render_schedule(plan["renderer"]["render_schedule"])
    missing = [path for path in _expected_rgb(output, frames) if not path.is_file()]
    if missing:
        raise ExistingSandError(
            f"E4 demo render is missing {len(missing)} RGB frames, first: {missing[0]}"
        )
    diagnostic_frames = [frame for frame in frames if frame.save_debug]
    missing_stats = [
        output / "release_check" / frame.output_group / "water_debug"
        / "water_failure_stats" / f"{frame.output_name}.yaml"
        for frame in diagnostic_frames
    ]
    missing_stats = [path for path in missing_stats if not path.is_file()]
    if missing_stats:
        raise ExistingSandError(
            "E4 demo sparse diagnostics are incomplete; first missing file: "
            f"{missing_stats[0]}"
        )
    result = {
        "schema_version": 1,
        "status": "completed_e4_demo_render",
        "case": "water",
        "frame_count": len(frames),
        "rgb_frame_count": len(frames),
        "diagnostic_frame_count": len(diagnostic_frames),
        "fps": plan["demo"]["fps"],
        "start_time_seconds": plan["demo"]["start_time_seconds"],
        "end_time_seconds": plan["demo"]["end_time_seconds"],
        "diagnostic_times_seconds": plan["demo"][
            "diagnostic_times_seconds"
        ],
        "camera_map": plan["camera_map"],
    }
    with (output / "result.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(result, stream, sort_keys=False)
    print(f"E4 demo rendered {len(frames)} RGB frames: {output}")
    return output


def _find_program(name: str) -> Path | None:
    candidate = shutil.which(name)
    if candidate:
        return Path(candidate).resolve()
    adjacent = Path(sys.executable).resolve().parent / name
    if adjacent.is_file() and os.access(adjacent, os.X_OK):
        return adjacent
    return None


def _camera_frame_table(
    output: Path, schedule: dict
) -> dict[int, list[tuple[int, Path]]]:
    frames, _ = expand_render_schedule(schedule)
    table: dict[int, list[tuple[int, Path]]] = {}
    time_index: dict[float, int] = {}
    for frame in frames:
        if frame.sequence != "demo_multiview":
            continue
        if frame.time_seconds not in time_index:
            time_index[frame.time_seconds] = len(time_index)
        path = (
            output / "release_check" / frame.output_group / "renders"
            / f"{frame.output_name}.png"
        )
        if not path.is_file():
            raise ExistingSandError(f"E4 demo RGB frame is missing: {path}")
        table.setdefault(frame.camera_index, []).append(
            (time_index[frame.time_seconds], path)
        )
    if not table:
        raise ExistingSandError("E4 demo schedule has no multiview RGB frames")
    expected_count = len(time_index)
    for camera_index, values in table.items():
        if [index for index, _ in values] != list(range(expected_count)):
            raise ExistingSandError(
                f"E4 demo camera {camera_index} does not contain every time"
            )
    return table


def _encoder_common(settings: dict) -> list[str]:
    return [
        "-c:v", settings["codec"],
        "-preset", settings["preset"],
        "-crf", str(settings["crf"]),
        "-pix_fmt", settings["pixel_format"],
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-movflags", "+faststart",
    ]


def _probe_video(ffprobe: Path | None, path: Path) -> dict:
    if ffprobe is None:
        return {}
    completed = subprocess.run(
        [
            str(ffprobe), "-v", "error", "-count_frames",
            "-select_streams", "v:0", "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames,duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    return streams[0] if len(streams) == 1 else {}


def encode_demo(run_root: Path, config_path: Path) -> Path:
    config = _load_yaml(config_path, "E4 demo config")
    settings = _demo_settings(config)
    render_id = config.get("render_id")
    if not isinstance(render_id, str) or not render_id:
        raise ExistingSandError("E4 demo config has no render_id")
    output = Path(run_root).resolve() / "visualizations" / render_id / "water"
    result = _load_yaml(output / "result.yaml", "E4 demo render result")
    if result.get("status") != "completed_e4_demo_render":
        raise ExistingSandError("E4 demo render is not complete")
    render_config = _load_yaml(output / "render.yaml", "E4 demo render config")
    schedule = render_config.get("render_schedule")
    if not isinstance(schedule, dict):
        raise ExistingSandError("E4 demo render has no saved schedule")
    table = _camera_frame_table(output, schedule)
    camera_map = result.get("camera_map")
    if not isinstance(camera_map, list) or len(camera_map) != len(table):
        raise ExistingSandError("E4 demo camera map differs from rendered streams")
    ffmpeg = _find_program("ffmpeg")
    if ffmpeg is None:
        raise ExistingSandError(
            "FFmpeg is unavailable; activate the sandygs Conda environment "
            "and rerun the encode action"
        )
    ffprobe = _find_program("ffprobe")
    videos = output / "videos"
    if videos.exists():
        raise ExistingSandError(f"refusing to overwrite E4 demo videos: {videos}")
    videos.mkdir(parents=True, exist_ok=False)
    manifest_videos = []
    try:
        with tempfile.TemporaryDirectory(
            prefix="e4_demo_encode_", dir=output
        ) as temporary:
            staging = Path(temporary)
            patterns: dict[int, Path] = {}
            for camera_index, values in sorted(table.items()):
                frame_dir = staging / f"camera_{camera_index:03d}"
                frame_dir.mkdir()
                for frame_index, source in values:
                    os.link(source, frame_dir / f"frame_{frame_index:05d}.png")
                patterns[camera_index] = frame_dir / "frame_%05d.png"

            for camera_index, pattern in sorted(patterns.items()):
                camera_id = str(camera_map[camera_index]["camera_id"])
                output_name = (
                    f"camera_{camera_index:03d}_{Path(camera_id).stem}.mp4"
                )
                final_path = videos / output_name
                temporary_path = staging / output_name
                command = [
                    str(ffmpeg), "-hide_banner", "-loglevel", "warning",
                    "-framerate", f"{settings['fps']:g}",
                    "-start_number", "0", "-i", str(pattern),
                    "-frames:v", str(len(table[camera_index])),
                    *_encoder_common(settings), str(temporary_path),
                ]
                print("+", " ".join(command), flush=True)
                subprocess.run(command, check=True)
                temporary_path.replace(final_path)
                manifest_videos.append({
                    "type": "single_camera",
                    "camera_index": camera_index,
                    "camera_id": camera_id,
                    "path": str(final_path.relative_to(output)),
                    "bytes": final_path.stat().st_size,
                    "sha256": sha256_file(final_path),
                    "probe": _probe_video(ffprobe, final_path),
                })

            if settings["multiview"]:
                camera_labels = [
                    Path(str(camera_map[index]["camera_id"])).stem.removeprefix(
                        "frame_"
                    )
                    for index in sorted(patterns)
                ]
                temporary_path = staging / (
                    "multiview_" + "_".join(camera_labels) + ".mp4"
                )
                final_path = videos / temporary_path.name
                inputs: list[str] = []
                for camera_index in sorted(patterns):
                    inputs.extend([
                        "-framerate", f"{settings['fps']:g}",
                        "-start_number", "0", "-i",
                        str(patterns[camera_index]),
                    ])
                command = [
                    str(ffmpeg), "-hide_banner", "-loglevel", "warning",
                    *inputs,
                    "-filter_complex", f"hstack=inputs={len(patterns)}",
                    "-frames:v", str(len(next(iter(table.values())))),
                    *_encoder_common(settings), str(temporary_path),
                ]
                print("+", " ".join(command), flush=True)
                subprocess.run(command, check=True)
                temporary_path.replace(final_path)
                manifest_videos.append({
                    "type": "multiview_horizontal",
                    "camera_indices": sorted(patterns),
                    "path": str(final_path.relative_to(output)),
                    "bytes": final_path.stat().st_size,
                    "sha256": sha256_file(final_path),
                    "probe": _probe_video(ffprobe, final_path),
                })
    except Exception:
        shutil.rmtree(videos, ignore_errors=True)
        raise

    frame_count_per_camera = len(next(iter(table.values())))
    manifest = {
        "schema_version": 1,
        "status": "completed_e4_demo_encoding",
        "ffmpeg": str(ffmpeg),
        "ffmpeg_version": subprocess.run(
            [str(ffmpeg), "-version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0],
        "fps": settings["fps"],
        "frame_count_per_camera": frame_count_per_camera,
        "duration_seconds_on_timeline": settings["end_time_seconds"]
        - settings["start_time_seconds"],
        "encoded_duration_seconds": frame_count_per_camera / settings["fps"],
        "codec": settings["codec"],
        "preset": settings["preset"],
        "crf": settings["crf"],
        "pixel_format": settings["pixel_format"],
        "config": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(Path(config_path).resolve()),
        "videos": manifest_videos,
    }
    with (videos / "manifest.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)
    print(f"E4 demo encoded {len(manifest_videos)} videos: {videos}")
    return videos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "render", "encode", "all"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/existing_sand/indoor02_e4_demo_v1.yaml",
    )
    args = parser.parse_args()
    try:
        validate_artifact_storage(ROOT / "configs/paths.local.yaml")
        if args.action == "encode":
            encode_demo(args.run, args.config)
            return 0
        plan = build_demo_plan(args.run, args.config)
        if args.action == "check":
            ffmpeg = _find_program("ffmpeg")
            summary = {
                "output": str(plan["output"]),
                "rgb_frame_count": plan["frame_count"],
                "diagnostic_frame_count": plan["diagnostic_frame_count"],
                "fps": plan["demo"]["fps"],
                "time_range_seconds": [
                    plan["demo"]["start_time_seconds"],
                    plan["demo"]["end_time_seconds"],
                ],
                "diagnostic_times_seconds": plan["demo"][
                    "diagnostic_times_seconds"
                ],
                "camera_map": plan["camera_map"],
                "ffmpeg": None if ffmpeg is None else str(ffmpeg),
                "render_command": [str(value) for value in plan["command"]],
            }
            print(json.dumps(summary, indent=2))
            return 0
        render_demo(plan)
        if args.action == "all":
            encode_demo(args.run, args.config)
    except (
        ExistingSandError,
        StorageGuardError,
        OSError,
        ValueError,
        KeyError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(2, f"existing-sand E4 demo: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
