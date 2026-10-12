#!/usr/bin/env python3
"""Capture a few source PGSR views into an existing-sand observations stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.existing_sand import ExistingSandError  # noqa: E402
from sandygs.existing_sand.capture import capture_source_views  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--camera-id", action="append", required=True,
                        help="run.yaml camera ID; repeat for a few selected views")
    parser.add_argument("--dataset-name", required=True,
                        help="exact PGSR training dataset_name, e.g. garden")
    parser.add_argument("--images-dir", default="images",
                        help="dataset-relative image directory used for training")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--alpha-min", type=float, default=0.1)
    parser.add_argument("--white-background", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = capture_source_views(
            args.run, args.capture_id, args.camera_id,
            dataset_name=args.dataset_name, width=args.width, height=args.height,
            sh_degree=args.sh_degree, alpha_min=args.alpha_min,
            white_background=args.white_background, repo_root=REPO_ROOT,
            images_dir=args.images_dir,
            command=[sys.executable, str(Path(__file__).resolve()),
                     *(argv if argv is not None else sys.argv[1:])],
        )
    except (ExistingSandError, ImportError, OSError, RuntimeError) as exc:
        parser.exit(2, f"existing-sand capture: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
