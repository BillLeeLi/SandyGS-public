#!/usr/bin/env python3
"""Fail fast unless the machine-local artifact volume is safely writable."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.storage_guard import StorageGuardError, validate_artifact_storage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        type=Path,
        default=REPO_ROOT / "configs/paths.local.yaml",
    )
    args = parser.parse_args()
    try:
        result = validate_artifact_storage(args.paths)
    except StorageGuardError as exc:
        parser.exit(2, f"SandyGS artifact storage guard failed: {exc}\n")
    if result["enabled"]:
        print(
            "SandyGS artifact storage guard passed: "
            f"{result['artifact_root']}"
        )
    else:
        print("SandyGS artifact storage guard is not enabled on this machine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
