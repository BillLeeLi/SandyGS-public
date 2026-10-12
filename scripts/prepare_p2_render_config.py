#!/usr/bin/env python3
"""Generate SandyGS arguments for direct pure-water run rendering."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.simulation import SandWaterRunAdapter
from sandygs.configuration import (
    renderer_arguments_from_manifest,
    resolve_water_rendering_backend,
    validate_current_runtime_config,
)


def load_mapping(path: Path, name: str) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a YAML mapping")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolved-scene",
        type=Path,
        required=True,
        help="portable scene after scripts/resolve_scene_config.py",
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--exchange", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    resolved = load_mapping(args.resolved_scene, "resolved scene")
    profile = load_mapping(args.profile, "simulation profile")
    render = profile.get("render")
    if not isinstance(render, dict):
        raise ValueError("profile must contain render settings")
    water_backend = resolve_water_rendering_backend(
        render, source=f"P2 profile {args.profile.resolve()}"
    )
    renderer_arguments = renderer_arguments_from_manifest(resolved)
    validate_current_runtime_config(
        {**renderer_arguments, **render},
        source=f"P2 profile {args.profile.resolve()}",
    )

    exchange = args.exchange.resolve()
    run = args.run.resolve()
    frame_time = float(render["frame_time_seconds"])
    epsilon = float(render["free_water_epsilon"])
    adapter = SandWaterRunAdapter(
        exchange,
        run,
        require_pure_water=True,
        free_water_epsilon=epsilon,
        water_normal_gradient_mode=str(
            render.get("water_normal_gradient_mode", "legacy_full")
        ),
    )
    # Sample now so a missing field or out-of-range time fails before loading a
    # multi-million-Gaussian checkpoint in render_sand_water.py.
    sampled = adapter.sample(frame_time)

    output_config = dict(renderer_arguments)
    output_config.update(
        {
            "simulation_backend": "sand_water",
            "simulation_exchange_path": str(exchange),
            "sand_water_run_path": str(run),
            "simulation_start_time": frame_time,
            "simulation_render_fps": float(render["fps"]),
            "free_water_epsilon": epsilon,
            "water_normal_gradient_mode": str(
                render.get("water_normal_gradient_mode", "legacy_full")
            ),
            "water_rendering_backend": water_backend,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output_config, stream, sort_keys=False)
    print(
        f"wrote {args.output.resolve()} for t={frame_time}, "
        f"solver step {sampled.lower_step}->{sampled.upper_step}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
