#!/usr/bin/env python3
"""Convert a legacy SandyGS scene manifest to schema v2 offline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.configuration import (  # noqa: E402
    ConfigError,
    load_yaml_mapping,
    normalize_scene_config,
)


def migrate_scene_file(source: Path, output: Path) -> dict:
    if output.exists():
        raise ConfigError(f"refusing to overwrite migration output: {output}")
    raw = load_yaml_mapping(source, description="scene configuration")
    normalized, migration = normalize_scene_config(raw, source=str(source))
    if not migration["converted"]:
        raise ConfigError(f"{source} already uses scene schema v2")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(normalized, stream, sort_keys=False, allow_unicode=True)
    return migration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        migration = migrate_scene_file(args.input.resolve(), args.output.resolve())
    except ConfigError as exc:
        print(f"configuration migration error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote scene schema v2: {args.output.resolve()}")
    print("removed fields:")
    for value in migration["removed_fields"]:
        print(f"  - {value}")
    print("removed paths:")
    for value in migration["removed_paths"]:
        print(f"  - {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
