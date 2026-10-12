#!/usr/bin/env python3
"""Initialize and verify offline existing-sand experiment manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.existing_sand import (  # noqa: E402
    ExistingSandError,
    initialize_run,
    record_stage,
    verify_run,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="create an exclusive run manifest")
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--run-id", required=True)
    record = sub.add_parser("record", help="register completed stage files")
    record.add_argument("--run", type=Path, required=True)
    record.add_argument("--record", type=Path, required=True)
    verify = sub.add_parser("verify", help="check source and stage file hashes")
    verify.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = initialize_run(args.config, args.output, run_id=args.run_id,
                                    repo_root=REPO_ROOT)
            summary = {"run_id": result["run_id"],
                       "gaussian_count": result["source"]["gaussian_count"],
                       "checkpoint_sha256": result["source"]["checkpoint_sha256"],
                       "manifest": str(args.output.resolve() / "run.yaml")}
        elif args.command == "record":
            result = record_stage(args.run, args.record, repo_root=REPO_ROOT)
            summary = {"stage": result["stage"], "stage_id": result["stage_id"],
                       "artifacts": len(result["artifacts"])}
        else:
            summary = verify_run(args.run)
    except (ExistingSandError, OSError) as exc:
        parser.exit(2, f"existing-sand: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
