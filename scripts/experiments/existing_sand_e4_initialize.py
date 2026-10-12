#!/usr/bin/env python3
"""Prepare a reviewed E4 exchange and paired dry/water solver initial cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sandygs.existing_sand.e4_initialization import run_e4_initialization
from sandygs.existing_sand.workspace import ExistingSandError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repair-id", required=True)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    try:
        result = run_e4_initialization(args.run, args.repair_id, args.stage_id,
                                       args.config, repo_root=ROOT, command=command)
    except (ExistingSandError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"existing-sand E4 initialization: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
