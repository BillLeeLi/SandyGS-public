#!/usr/bin/env python3
"""Validate only the resources consumed by one SandyGS operation."""

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
    build_scene_manifest,
    load_yaml_mapping,
    validate_reconstruction_resources,
    validate_sand_water_resources,
    validate_terrain_exchange_resources,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        type=Path,
        default=REPO_ROOT / "configs/paths.local.yaml",
    )
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument(
        "--operation",
        choices=(
            "terrain_exchange",
            "simulation",
            "reconstruction",
            "sand_water_render",
        ),
        required=True,
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="simulation profile; used to validate opt-in render resources",
    )
    args = parser.parse_args()
    try:
        manifest, renderer, _ = build_scene_manifest(
            args.paths.resolve(),
            args.scene.resolve(),
            check_inputs=True,
            run_id="resource-validation-only",
            operation=args.operation,
            repo_root=REPO_ROOT,
        )
        render_options = None
        if args.profile is not None:
            profile = load_yaml_mapping(
                args.profile.resolve(), description="simulation profile"
            )
            render_options = profile.get("render")
            if not isinstance(render_options, dict):
                raise ConfigError("simulation profile must contain render")
        elif args.operation == "sand_water_render":
            raise ConfigError(
                "sand_water_render validation requires --profile so optional "
                "texture resources are checked"
            )
        paths = manifest["paths"]
        if args.operation == "terrain_exchange":
            result = {
                "terrain_exchange": validate_terrain_exchange_resources(
                    paths, renderer
                )
            }
        elif args.operation == "simulation":
            result = validate_sand_water_resources(
                paths,
                renderer,
                render_options,
                include_render=False,
                expected_scene_id=str(manifest["scene_id"]),
            )
        elif args.operation == "reconstruction":
            result = {
                "render": validate_reconstruction_resources(
                    paths, renderer, render_options
                )
            }
        else:
            result = validate_sand_water_resources(
                paths,
                renderer,
                render_options,
                include_render=True,
                expected_scene_id=str(manifest["scene_id"]),
            )
    except (OSError, ValueError) as exc:
        print(f"resource validation error: {exc}", file=sys.stderr)
        return 2
    print(yaml.safe_dump(result, sort_keys=False, allow_unicode=True).rstrip())
    print(f"{args.operation} resource validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
