#!/usr/bin/env python3
"""Run the P2 height-map -> pure-water solver -> SandyGS render pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.configuration import (
    build_scene_manifest,
    resolve_water_rendering_backend,
    validate_sand_water_resources,
)
from sandygs.prepare import prepare_pure_water_case
from sandygs.render_schedule import resolve_profile_render_window
from sandygs.simulation import SandWaterRunAdapter
from sandygs.solver import resolve_sand_water_solver
from sandygs.storage_guard import validate_artifact_storage


def load_profile(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        profile = yaml.safe_load(stream)
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError("P2 profile must be a schema_version 1 mapping")
    if not isinstance(profile.get("solver"), dict) or not isinstance(
        profile.get("render"), dict
    ):
        raise ValueError("P2 profile must contain solver and render mappings")
    render = dict(profile["render"])
    render["water_rendering_backend"] = resolve_water_rendering_backend(
        render, source=f"P2 profile {path}"
    )
    profile = dict(profile)
    profile["render"] = render
    return profile


def write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)


def ensure_new_directory(path: Path, name: str) -> None:
    if path.exists():
        raise FileExistsError(
            f"{name} already exists; choose a new --run-id: {path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        type=Path,
        default=REPO_ROOT / "configs/paths.local.yaml",
    )
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--solver",
        type=Path,
        default=None,
        help=(
            "explicit solver executable; otherwise use paths.local.yaml, then "
            "the verified in-tree release build"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        help="new immutable run name",
    )
    parser.add_argument("--max-views", type=int, default=1)
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="stop after solver/adapter validation (CPU-only preflight)",
    )
    args = parser.parse_args()
    if args.max_views <= 0:
        raise ValueError("--max-views must be positive")

    # Validate removable/mounted artifact storage before build_manifest or any
    # stage directory has a chance to create a fallback on the WSL root disk.
    validate_artifact_storage(args.paths)

    profile = load_profile(args.profile.resolve())
    operation = "simulation" if args.skip_render else "sand_water_render"
    # Resolve the versioned scene once.  Operation-scoped validation keeps a
    # CPU-only simulation independent of PGSR and all SandyGS paths independent
    # of historical SimRain frames.
    resolved, renderer_arguments, _ = build_scene_manifest(
        args.paths.resolve(),
        args.scene.resolve(),
        check_inputs=True,
        run_id=args.run_id,
        operation=operation,
        repo_root=REPO_ROOT,
    )
    render_plan = resolve_profile_render_window(
        profile["render"], profile["solver"], max_views=args.max_views
    )
    scene_id = str(resolved["scene_id"])
    roots = resolved["roots"]
    paths = resolved["paths"]
    resource_validation = validate_sand_water_resources(
        paths,
        renderer_arguments,
        profile["render"],
        include_render=not args.skip_render,
        expected_scene_id=scene_id,
    )
    solver_executable = resolve_sand_water_solver(
        cli_path=args.solver,
        paths_config=args.paths,
        repo_root=REPO_ROOT,
    )
    exchange = Path(paths["p1_exchange"]).resolve()
    derived_sandygs_root = (Path(roots["derived"]) / "sandygs").resolve()
    try:
        # A P1 bundle conventionally ends in its version directory. Reusing
        # its parent keeps dataset/scene organization portable (for example
        # mipnerf360/garden) without parsing scene_id text.
        scene_relative = exchange.relative_to(derived_sandygs_root).parent
    except ValueError as exc:
        raise ValueError(
            f"P1 exchange must be below {derived_sandygs_root}: {exchange}"
        ) from exc
    input_directory = (
        derived_sandygs_root
        / scene_relative
        / "p2"
        / args.run_id
        / "input"
    )
    artifact_directory = (
        Path(roots["artifacts"]) / "p2" / scene_relative / args.run_id
    )
    run_directory = artifact_directory / "run"
    render_directory = artifact_directory / "render"
    ensure_new_directory(input_directory, "P2 derived input")
    ensure_new_directory(artifact_directory, "P2 artifact")
    artifact_directory.mkdir(parents=True)

    solver = profile["solver"]
    render = profile["render"]
    prepare_pure_water_case(
        exchange_bundle=exchange,
        output=input_directory,
        initial_surface_level=float(profile["initial_surface_level"]),
        water_dt=float(solver["water_dt"]),
        steps=int(solver["steps"]),
        frame_stride=int(solver["frame_stride"]),
        target_cfl=float(solver["target_cfl"]),
        duration_seconds=(
            None
            if solver.get("duration_seconds") is None
            else float(solver["duration_seconds"])
        ),
    )

    solver_command = [
        str(solver_executable.path),
        "--scene",
        str(input_directory / "scene.yaml"),
        "--output",
        str(run_directory),
        "--snapshot",
        str(solver.get("snapshot_level", "render")),
        "--no-obj",
    ]
    render_config_path = artifact_directory / "sand_water_render.yaml"
    render_command = [
        sys.executable,
        str(REPO_ROOT / "render_sand_water.py"),
        "-s",
        paths["dataset"],
        "-m",
        paths["checkpoint"],
        "--render_output_path",
        str(render_directory),
        "--config_file",
        str(render_config_path),
        "--max_views",
        str(args.max_views),
        "--disable_preload_img",
    ]
    audit = {
        "schema_version": 1,
        "stage": "P2",
        "status": "prepared",
        "scene_id": scene_id,
        "run_id": args.run_id,
        "source_files": {
            "paths": str(args.paths.resolve()),
            "scene": str(args.scene.resolve()),
            "profile": str(args.profile.resolve()),
        },
        "paths": {
            "exchange": str(exchange),
            "input": str(input_directory),
            "solver_run": str(run_directory),
            "render": str(render_directory),
        },
        "commands": {
            "solver": solver_command,
            "render": render_command,
        },
        "resource_validation": resource_validation,
        "solver_build": solver_executable.audit(),
    }
    if render_plan["frames"] is not None:
        frames = render_plan["frames"]
        audit["render_schedule"] = {
            "frame_count": len(frames),
            "camera_path_frames": render_plan["camera_path_frames"],
            "unique_camera_count": len({frame.camera_index for frame in frames}),
            "unique_time_count": len({frame.time_seconds for frame in frames}),
            "first_time_seconds": render_plan["first_time_seconds"],
            "last_time_seconds": render_plan["last_time_seconds"],
        }
    write_yaml(artifact_directory / "pipeline.yaml", audit)

    subprocess.run(solver_command, check=True)
    frame_time = float(render_plan["first_time_seconds"])
    epsilon = float(render["free_water_epsilon"])
    adapter = SandWaterRunAdapter(
        exchange,
        run_directory,
        require_pure_water=True,
        free_water_epsilon=epsilon,
        water_normal_gradient_mode=str(
            render.get("water_normal_gradient_mode", "legacy_full")
        ),
    )
    sampled = adapter.sample(frame_time)
    final_render_time = float(render_plan["last_time_seconds"])
    final_sampled = adapter.sample(final_render_time)

    p2_arguments = dict(renderer_arguments)
    # Forward the complete render mapping so a regression-only P2 profile can
    # select an explicit time/camera schedule and raw-array capture without
    # changing the pure-water initial state or solver fields.
    p2_arguments.update(render)
    p2_arguments.update(
        {
            "simulation_backend": "sand_water",
            "simulation_exchange_path": str(exchange),
            "sand_water_run_path": str(run_directory),
            "simulation_start_time": frame_time,
            "simulation_render_fps": float(render["fps"]),
            "free_water_epsilon": epsilon,
            "water_normal_gradient_mode": str(
                render.get("water_normal_gradient_mode", "legacy_full")
            ),
            "water_rendering_backend": str(render["water_rendering_backend"]),
        }
    )
    write_yaml(render_config_path, p2_arguments)
    audit["solver_sample"] = {
        "time_seconds": frame_time,
        "lower_step": sampled.lower_step,
        "upper_step": sampled.upper_step,
        "interpolation_weight": sampled.interpolation_weight,
        "active_water_cells": int(sampled.free_water_mask.sum()),
        "final_time_seconds": final_render_time,
        "final_lower_step": final_sampled.lower_step,
        "final_upper_step": final_sampled.upper_step,
        "final_interpolation_weight": final_sampled.interpolation_weight,
    }
    audit["status"] = "solver_validated"
    write_yaml(artifact_directory / "pipeline.yaml", audit)

    if not args.skip_render:
        # The shell wrapper configures GCC/CUDA before this subprocess. Keeping
        # the renderer as a process also releases its large GPU allocation when
        # a stage finishes or fails.
        subprocess.run(render_command, check=True)
        audit["status"] = "completed"
        write_yaml(artifact_directory / "pipeline.yaml", audit)
    print(f"P2 {audit['status']}: {artifact_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
