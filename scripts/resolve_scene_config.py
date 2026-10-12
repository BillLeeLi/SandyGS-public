#!/usr/bin/env python3
"""Resolve machine-local roots and a portable SandyGS scene manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.configuration import (  # noqa: E402
    ConfigError,
    build_scene_manifest,
)
from sandygs.storage_guard import validate_artifact_output  # noqa: E402


def build_manifest(
    paths_config: Path,
    scene_config: Path,
    check_inputs: bool,
    run_id: str,
    operation: str = "all",
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Compatibility export for callers predating ``sandygs.configuration``."""

    return build_scene_manifest(
        paths_config,
        scene_config,
        check_inputs=check_inputs,
        run_id=run_id,
        operation=operation,
        repo_root=REPO_ROOT,
    )


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve a portable scene manifest into machine-local paths."
    )
    parser.add_argument(
        "--paths",
        type=Path,
        default=REPO_ROOT / "configs/paths.local.yaml",
        help="machine-local roots file (default: configs/paths.local.yaml)",
    )
    parser.add_argument("--scene", type=Path, required=True, help="portable scene YAML")
    parser.add_argument(
        "--operation",
        choices=(
            "all",
            "resolve",
            "terrain_exchange",
            "simulation",
            "reconstruction",
            "sand_water_render",
        ),
        default="all",
        help="validate only resources consumed by this operation",
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="check inputs required by --operation",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        help="new artifact directory name (default: current UTC timestamp)",
    )
    parser.add_argument("--output", type=Path, help="write the resolved audit manifest")
    parser.add_argument(
        "--renderer-output",
        type=Path,
        help="write the flat runtime YAML accepted by render_sand_water.py",
    )
    parser.add_argument("--max-views", type=int, help="append --max_views")
    parser.add_argument(
        "--disable-preload-img",
        action="store_true",
        help="append --disable_preload_img",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        renderer_output = args.renderer_output
        for output in (args.output, renderer_output):
            if output is not None:
                validate_artifact_output(args.paths, output)
        manifest, renderer_arguments, command = build_manifest(
            args.paths,
            args.scene,
            args.check_inputs,
            args.run_id,
            args.operation,
        )
        render_overrides: dict[str, Any] = {}
        if args.max_views is not None:
            if args.max_views <= 0:
                raise ConfigError("--max-views must be greater than zero")
            command.extend(["--max_views", str(args.max_views)])
            render_overrides["max_views"] = args.max_views
        if args.disable_preload_img:
            command.append("--disable_preload_img")
            render_overrides["disable_preload_img"] = True
        if render_overrides:
            manifest["render_overrides"] = render_overrides

        if args.output:
            write_yaml(args.output, manifest)
        if renderer_output:
            write_yaml(renderer_output, renderer_arguments)

        printable_command = list(command)
        if renderer_output:
            placeholder_index = printable_command.index(
                "<generated-render-config.yaml>"
            )
            printable_command[placeholder_index] = str(renderer_output.resolve())
        print(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True).rstrip())
        print("\nRenderer command:")
        print(shlex.join(printable_command))
        if not renderer_output:
            print(
                "\nThe command is a template. Pass --renderer-output to replace "
                "<generated-render-config.yaml> with a real file."
            )
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
