#!/usr/bin/env python3
"""Compare selected-only, full-scene, and guarded-hybrid E4 sand tops on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sandygs.existing_sand.e4_sand_surface import run_sand_surface_comparison
from sandygs.existing_sand.workspace import ExistingSandError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ortho-id", required=True)
    parser.add_argument("--base-id", required=True)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--alpha-min", type=float, default=.98)
    parser.add_argument("--agreement-max", type=float, default=.02)
    parser.add_argument("--core-distance", type=float, default=6.0)
    parser.add_argument("--spike-max", type=float, default=.05)
    args = parser.parse_args()
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    try:
        result = run_sand_surface_comparison(
            args.run, args.ortho_id, args.base_id, args.stage_id,
            repo_root=ROOT, alpha_min=args.alpha_min,
            agreement_max=args.agreement_max, core_distance=args.core_distance,
            spike_max=args.spike_max, command=command,
        )
    except (ExistingSandError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"existing-sand E4 sand surface: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
