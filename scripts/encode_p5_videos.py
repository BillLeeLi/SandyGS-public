#!/usr/bin/env python3
"""Encode fixed/moving P5 PNG sequences without changing render artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        type=Path,
        help="P5 artifact directory containing render/release_check",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="sequence name to encode; repeat to select several",
    )
    args = parser.parse_args()
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg is not available on PATH")

    release = args.artifact.resolve() / "render" / "release_check"
    schedule_path = release / "render_schedule.yaml"
    if not schedule_path.is_file():
        parser.error(f"missing render schedule: {schedule_path}")
    with schedule_path.open("r", encoding="utf-8") as stream:
        schedule = yaml.safe_load(stream)
    frames = schedule.get("frames") if isinstance(schedule, dict) else None
    if not isinstance(frames, list):
        parser.error("render schedule has no frame table")

    sequence_types: dict[str, str] = {}
    for frame in frames:
        if isinstance(frame, dict):
            sequence_types[str(frame.get("sequence"))] = str(
                frame.get("sequence_type")
            )
    selected = set(args.sequence or ())
    if selected:
        unknown = selected - set(sequence_types)
        if unknown:
            parser.error("unknown sequence(s): " + ", ".join(sorted(unknown)))
    else:
        # A synchronized multi-view group is a comparison grid, not a temporal
        # camera stream. Encode only sequences with one image per timestamp.
        selected = {
            name
            for name, kind in sequence_types.items()
            if kind in {"fixed_camera", "moving_camera"}
        }

    video_root = release / "videos"
    video_root.mkdir(parents=True, exist_ok=True)
    for name in sorted(selected):
        input_pattern = release / name / "renders" / "%05d.png"
        if not (release / name / "renders" / "00000.png").is_file():
            parser.error(f"sequence has no frame 00000: {name}")
        output = video_root / f"{name}.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                f"{args.fps:g}",
                "-i",
                str(input_pattern),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(output),
            ],
            check=True,
        )
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
