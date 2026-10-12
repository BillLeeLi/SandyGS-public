#!/usr/bin/env python3
"""Read-only validator for a SandyGS simulation-exchange bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.exchange import ExchangeValidationError, validate_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--skip-checksums", action="store_true", help="do not hash field arrays"
    )
    args = parser.parse_args()
    try:
        metadata = validate_bundle(
            args.bundle.resolve(), verify_checksums=not args.skip_checksums
        )
    except ExchangeValidationError as exc:
        for error in exc.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"OK: {metadata['scene_id']} format v{metadata['format_version']}, "
        f"{len(metadata['frames'])} frame(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
