#!/usr/bin/env python3
"""Prepare a P3 sand/water solver scene from a P1 exchange bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.prepare import prepare_sand_water_case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.profile.open("r", encoding="utf-8") as stream:
        profile = yaml.safe_load(stream)
    metadata = prepare_sand_water_case(
        exchange_bundle=args.exchange,
        output=args.output.resolve(),
        profile=profile,
    )
    initial = metadata["initialization"]
    print(
        f"prepared {args.output.resolve()} with "
        f"{initial['sand_cell_count']} sand cells, "
        f"max height {initial['sand_height_max']:.9g}, "
        f"volume {initial['sand_volume']:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
