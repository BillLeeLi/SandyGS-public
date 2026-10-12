#!/usr/bin/env python3
"""Prepare a no-sand P2 solver scene from a P1 exchange bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.prepare import prepare_pure_water_case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.profile.open("r", encoding="utf-8") as stream:
        profile = yaml.safe_load(stream)
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError("simulation profile must be a schema_version 1 mapping")
    solver = profile.get("solver")
    if not isinstance(solver, dict):
        raise ValueError("simulation profile is missing solver settings")
    metadata = prepare_pure_water_case(
        exchange_bundle=args.exchange,
        output=args.output.resolve(),
        initial_surface_level=float(profile["initial_surface_level"]),
        water_dt=float(solver["water_dt"]),
        steps=int(solver["steps"]),
        frame_stride=int(solver["frame_stride"]),
        target_cfl=float(solver["target_cfl"]),
        duration_seconds=(
            None
            if solver.get("duration_seconds") is None
            else float(solver["duration_seconds"])
        ),
    )
    initialization = metadata["initialization"]
    print(
        f"prepared {args.output.resolve()} with "
        f"{initialization['wet_cell_count']} wet cells and depth range "
        f"[{initialization['water_depth_min']}, "
        f"{initialization['water_depth_max']}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
