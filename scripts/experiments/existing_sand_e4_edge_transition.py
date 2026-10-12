#!/usr/bin/env python3
"""Compare two harmonic sand-thickness edge bands using existing E4 artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sandygs.existing_sand.e4_edge_transition import run_edge_transition_comparison
from sandygs.existing_sand.workspace import ExistingSandError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--surface-id", required=True)
    parser.add_argument("--ortho-id", required=True)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--widths", nargs=2, type=int, default=(5, 6), metavar=("INNER", "OUTER"))
    args = parser.parse_args()
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    try:
        result = run_edge_transition_comparison(
            args.run, args.surface_id, args.ortho_id, args.stage_id,
            tuple(args.widths), repo_root=ROOT, command=command,
        )
    except (ExistingSandError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"existing-sand E4 edge transition: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
