#!/usr/bin/env python3
"""Compare raw, planar, and locally fitted E4 tabletop heights without GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sandygs.existing_sand.e4_local_fit import run_local_tabletop_fit
from sandygs.existing_sand.workspace import ExistingSandError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--outer-min", type=int, default=8)
    parser.add_argument("--outer-max", type=int, default=32)
    parser.add_argument("--inner-min", type=int, default=32)
    parser.add_argument("--alpha-min", type=float, default=.98)
    parser.add_argument("--residual-max", type=float, default=.02)
    parser.add_argument("--local-deviation-max", type=float, default=.005)
    args = parser.parse_args()
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    try:
        result = run_local_tabletop_fit(
            args.run, args.source_id, args.stage_id, repo_root=ROOT,
            outer_min=args.outer_min, outer_max=args.outer_max, inner_min=args.inner_min,
            alpha_min=args.alpha_min, residual_max=args.residual_max,
            local_deviation_max=args.local_deviation_max, command=command,
        )
    except (ExistingSandError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"existing-sand E4 local fit: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
