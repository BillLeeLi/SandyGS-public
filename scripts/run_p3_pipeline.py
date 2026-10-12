#!/usr/bin/env python3
"""Compatibility entry for the shared SandyGS P3/P4/P5 pipeline."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.pipeline import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
