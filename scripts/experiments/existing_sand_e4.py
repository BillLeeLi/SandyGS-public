#!/usr/bin/env python3
"""Export same-camera PGSR orthographic sand/base surfaces for E4 review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sandygs.existing_sand.e4_height import export_orthographic_surfaces
from sandygs.existing_sand.workspace import ExistingSandError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--derived-id", required=True)
    parser.add_argument("--fit-id", required=True)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--bounds", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"), required=True)
    parser.add_argument("--resolution", nargs=2, type=int, metavar=("NX", "NY"), required=True)
    parser.add_argument("--camera-z", type=float, required=True, help="orthographic camera height in simulation coordinates")
    parser.add_argument("--alpha-min", type=float, default=0.5)
    args = parser.parse_args()
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    try:
        result = export_orthographic_surfaces(
            args.run, args.derived_id, args.fit_id, args.stage_id,
            tuple(args.bounds), args.resolution[0], args.resolution[1],
            args.camera_z, alpha_min=args.alpha_min, repo_root=ROOT,
            command=command,
        )
    except (ExistingSandError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"existing-sand E4: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
