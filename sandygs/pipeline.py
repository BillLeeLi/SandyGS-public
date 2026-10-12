"""Shared P3/P4/P5 terrain-to-solver-to-render pipeline implementation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.configuration import (
    build_scene_manifest,
    validate_sand_water_resources,
)
from sandygs.prepare import prepare_sand_water_case
from sandygs.preview import (
    write_initialization_preview,
    write_solver_evolution_preview,
)
from sandygs.render_schedule import expand_render_schedule
from sandygs.simulation import SandWaterRunAdapter
from sandygs.solver import resolve_sand_water_solver
from sandygs.storage_guard import validate_artifact_storage


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_profile(path: Path, experiment_phase: str | None = None) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        profile = yaml.safe_load(stream)
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError("simulation profile must be a schema_version 1 mapping")
    for key in ("initial", "solver", "render"):
        if not isinstance(profile.get(key), dict):
            raise ValueError(f"simulation profile must contain a {key} mapping")
    if experiment_phase is not None:
        phases = profile.get("phases")
        if not isinstance(phases, dict) or not isinstance(
            phases.get(experiment_phase), dict
        ):
            raise ValueError(
                f"profile has no mapping for phase {experiment_phase!r}"
            )
        profile = _deep_merge(profile, phases[experiment_phase])
        profile["selected_phase"] = experiment_phase
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


def build_renderer_environment() -> dict[str, str]:
    """Mirror the audited CUDA JIT environment for the renderer child.

    The stage entry scripts launch the active Conda Python by absolute path,
    which does not guarantee that the shell activation helper was sourced.
    In particular, CUDA 11.8 rejects Ubuntu 24.04's default GCC/G++ 13 when
    NVDIFFREC first builds ``renderutils_plugin``. Keep the CPU solver on the
    system compiler, but give only the renderer subprocess GCC/G++ 11 and a
    conservative Ninja job count suitable for the 8 GB WSL workstation.
    """

    environment = os.environ.copy()
    environment_bin_path = Path(sys.executable).resolve().parent
    environment_prefix = environment_bin_path.parent
    environment.setdefault("CUDA_HOME", "/usr/local/cuda")
    environment.setdefault("CC", "/usr/bin/gcc-11")
    environment.setdefault("CXX", "/usr/bin/g++-11")
    environment.setdefault("MAX_JOBS", "2")
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    cuda_home = Path(environment["CUDA_HOME"])

    def prepend_path(name: str, entries: list[Path]) -> None:
        current = environment.get(name, "").split(os.pathsep)
        ordered = [str(entry) for entry in entries if entry.is_dir()]
        ordered.extend(value for value in current if value and value not in ordered)
        environment[name] = os.pathsep.join(ordered)

    prepend_path("PATH", [environment_bin_path, cuda_home / "bin"])
    prepend_path(
        "LIBRARY_PATH", [environment_prefix / "lib", cuda_home / "lib64"]
    )
    prepend_path(
        "LD_LIBRARY_PATH",
        [environment_prefix / "lib", Path("/usr/lib/wsl/lib"), cuda_home / "lib64"],
    )
    return environment


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
        "--stage", choices=("p3", "p4", "p5"), default="p3",
        help="artifact namespace and stage-specific contract checks",
    )
    parser.add_argument(
        "--experiment-phase",
        default=None,
        help=(
            "select a named phase override from the P5 experiment profile "
            "(for example e1, e2, short, or server)"
        ),
    )
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
    parser.add_argument("--max-views", type=int, default=2)
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="stop after solver and analytic surfel validation",
    )
    args = parser.parse_args()
    if args.max_views <= 0:
        raise ValueError("--max-views must be positive")

    # P3/P4/P5 all create immutable artifacts. Fail before preparing derived
    # inputs or launching the solver if the machine-local volume is unsafe.
    validate_artifact_storage(args.paths)

    profile = load_profile(args.profile.resolve(), args.experiment_phase)  # 加载 simulation profile
    stage = args.stage.lower()
    stage_label = stage.upper()
    if stage == "p5" and args.experiment_phase is None:
        raise ValueError(
            "P5 requires --experiment-phase naming an entry under profile.phases"
        )
    if stage != "p5" and args.experiment_phase is not None:
        raise ValueError("--experiment-phase is only supported by P5")
    prepare_only = stage == "p5" and args.experiment_phase == "e1"
    include_render = not (args.skip_render or prepare_only)
    operation = "sand_water_render" if include_render else "simulation"
    resolved, renderer_arguments, _ = build_scene_manifest(
        args.paths.resolve(),
        args.scene.resolve(),
        check_inputs=True,
        run_id=args.run_id,
        operation=operation,
        repo_root=REPO_ROOT,
    )
    stage_namespace = "p5-experiments" if stage == "p5" else stage
    schedule_spec = profile["render"].get("render_schedule")
    schedule_frames = None
    camera_path_frames = None
    if schedule_spec is not None:
        # Expand and range-check before launching the potentially long CPU
        # solver. A server job should not spend hours producing states for an
        # invalid camera table or timestamps outside its saved duration.
        schedule_frames, camera_path_frames = expand_render_schedule(
            schedule_spec
        )
        duration = profile["solver"].get("duration_seconds")
        if duration is None:
            raise ValueError("a render schedule requires solver.duration_seconds")
        last_scheduled_time = max(
            frame.time_seconds for frame in schedule_frames
        )
        if last_scheduled_time > float(duration) + 1.0e-12:
            raise ValueError(
                "render schedule ends after solver.duration_seconds"
            )
    roots = resolved["roots"]
    paths = resolved["paths"]
    resource_validation = validate_sand_water_resources(
        paths,
        renderer_arguments,
        profile["render"],
        include_render=include_render,
        expected_scene_id=str(resolved["scene_id"]),
    )
    solver_executable = resolve_sand_water_solver(
        cli_path=args.solver,
        paths_config=args.paths,
        repo_root=REPO_ROOT,
    )
    exchange = Path(paths["p1_exchange"]).resolve()
    derived_sandygs_root = (Path(roots["derived"]) / "sandygs").resolve()
    try:
        scene_relative = exchange.relative_to(derived_sandygs_root).parent
    except ValueError as exc:
        raise ValueError(
            f"P1 exchange must be below {derived_sandygs_root}: {exchange}"
        ) from exc

    input_directory = (
        derived_sandygs_root
        / scene_relative
        / stage_namespace
        / args.run_id
        / "input"
    )
    artifact_directory = (
        Path(roots["artifacts"])
        / stage_namespace
        / scene_relative
        / args.run_id
    )
    run_directory = artifact_directory / "run"
    render_directory = artifact_directory / "render"
    ensure_new_directory(input_directory, f"{stage_label} derived input")
    ensure_new_directory(artifact_directory, f"{stage_label} artifact")
    artifact_directory.mkdir(parents=True)

    case = prepare_sand_water_case(
        exchange_bundle=exchange,
        output=input_directory,
        profile=profile,
    )
    initialization_preview = (
        write_initialization_preview(input_directory)
        if stage == "p5"
        else None
    )
    solver = profile["solver"]
    render = profile["render"]
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
    render_config_path = artifact_directory / (
        "sand_water_render.yaml"
        if stage in {"p4", "p5"}
        else "sand_render.yaml"
    )
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
        "stage": stage_label,
        "status": "prepared",
        "scene_id": resolved["scene_id"],
        "run_id": args.run_id,
        "experiment_phase": args.experiment_phase,
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
        "commands": {"solver": solver_command, "render": render_command},
        "initialization": case["initialization"],
        "resource_validation": resource_validation,
        "solver_build": solver_executable.audit(),
    }
    if initialization_preview is not None:
        audit["initialization_preview"] = initialization_preview
    write_yaml(artifact_directory / "pipeline.yaml", audit)

    if prepare_only:
        audit["status"] = "initialized"
        write_yaml(artifact_directory / "resolved_profile.yaml", profile)
        write_yaml(artifact_directory / "pipeline.yaml", audit)
        print(f"{stage_label}/E1 initialized: {artifact_directory}")
        return 0

    # The solver process exits before PGSR loads, keeping the 8 GiB workstation
    # from holding the C++ scratch grids and 3.8M source Gaussians together.
    subprocess.run(solver_command, check=True)
    if stage == "p5":
        # Render-level solver snapshots contain enough material fields for a
        # CPU evolution audit even when CUDA/PGSR is unavailable on this host.
        audit["physics_preview"] = write_solver_evolution_preview(
            run_directory,
            artifact_directory / "physics_preview",
        )
        write_yaml(artifact_directory / "pipeline.yaml", audit)
    if schedule_frames is not None:
        scheduled_times = [frame.time_seconds for frame in schedule_frames]
        frame_time = min(scheduled_times)
        final_render_time = max(scheduled_times)
        audit["render_schedule"] = {
            "frame_count": len(schedule_frames),
            "camera_path_frames": camera_path_frames,
            "unique_camera_count": len(
                {frame.camera_index for frame in schedule_frames}
            ),
            "unique_time_count": len(set(scheduled_times)),
            "first_time_seconds": frame_time,
            "last_time_seconds": final_render_time,
        }
    else:
        frame_time = float(render["frame_time_seconds"])
        final_render_time = frame_time + (
            (args.max_views - 1) / float(render["fps"])
        )
    adapter = SandWaterRunAdapter(
        exchange,
        run_directory,
        require_pure_water=False,
        free_water_epsilon=float(render["free_water_epsilon"]),
        sand_epsilon=float(render["sand_epsilon"]),
        water_normal_gradient_mode=str(
            render.get("water_normal_gradient_mode", "legacy_full")
        ),
    )
    first = adapter.sample(frame_time)
    final = adapter.sample(final_render_time)
    def build_sand_geometry(frame):
        # CPU preflight must exercise the same backend and parameters as the
        # CUDA renderer; otherwise a server job could validate cell surfels and
        # fail only after loading PGSR for the triangle path.
        return adapter.build_sand_geometry(
            frame,
            backend=str(render.get("sand_geometry_backend", "cell_tangent")),
            coverage_factor=float(render["sand_coverage_factor"]),
            normal_thickness_ratio=float(render["sand_normal_thickness_ratio"]),
            max_tangent_scale_ratio=float(
                render["sand_max_tangent_scale_ratio"]
            ),
            full_coverage_height=float(render["sand_full_coverage_height"]),
            triangle_coverage_gamma=float(
                render.get("sand_triangle_coverage_gamma", 1.0)
            ),
            triangle_terrain_max_edge_ratio=(
                None
                if render.get("sand_triangle_terrain_max_edge_ratio") is None
                else float(render["sand_triangle_terrain_max_edge_ratio"])
            ),
            triangle_contact_tolerance=(
                None
                if render.get("sand_triangle_contact_tolerance") is None
                else float(render["sand_triangle_contact_tolerance"])
            ),
            triangle_diagonal_mode=str(
                render.get("sand_triangle_diagonal_mode", "checkerboard")
            ),
            triangle_boundary_mode=str(
                render.get("sand_triangle_boundary_mode", "clipped")
            ),
            triangle_subdivision_level=int(
                render.get("sand_triangle_subdivision_level", 0)
            ),
        )

    first_geometry = build_sand_geometry(first)
    final_geometry = build_sand_geometry(final)
    if stage in {"p4", "p5"}:
        # P4/P5 require the overlapping opaque-sand/free-water case. A
        # profile with only mixed pore water is a valid P3 wet-sand test but
        # cannot exercise SSR, refraction or layered depth ownership.
        if not bool(render.get("render_layered_water", False)):
            raise ValueError(
                f"{stage_label} profile must enable render_layered_water"
            )
        if render.get("water_rendering_backend") != "sandygs_p4":
            raise ValueError(
                f"{stage_label} profile must set "
                "water_rendering_backend=sandygs_p4"
            )
        if first_geometry.xyz_gs.shape[0] == 0:
            raise ValueError(
                f"{stage_label} profile produced no active sand cells"
            )
        if not first.free_water_mask.any():
            raise ValueError(
                f"{stage_label} profile produced no free-water cells"
            )

    p3_arguments = dict(renderer_arguments)
    # Renderer-facing material values intentionally remain in the profile so
    # visual experiments can change without touching solver output arrays.
    p3_arguments.update(render)
    p3_arguments.update(
        {
            "simulation_backend": "sand_water",
            "simulation_exchange_path": str(exchange),
            "sand_water_run_path": str(run_directory),
            "simulation_start_time": frame_time,
            "simulation_render_fps": float(render["fps"]),
            "render_sand": True,
        }
    )
    write_yaml(render_config_path, p3_arguments)
    if stage == "p5":
        write_yaml(artifact_directory / "resolved_profile.yaml", profile)
    audit["solver_sample"] = {
        "time_seconds": frame_time,
        "lower_step": first.lower_step,
        "upper_step": first.upper_step,
        "interpolation_weight": first.interpolation_weight,
        "sand_geometry_backend": first_geometry.geometry_backend,
        "active_sand_cells": int(first.sand_mask.sum()),
        "sand_gaussian_count": int(first_geometry.xyz_gs.shape[0]),
        "sand_candidate_triangle_count": (
            first_geometry.candidate_triangle_count
        ),
        "sand_rejected_triangle_count": first_geometry.rejected_triangle_count,
        "sand_rejected_degenerate_count": (
            first_geometry.rejected_degenerate_count
        ),
        "sand_rejected_terrain_cliff_count": (
            first_geometry.rejected_terrain_cliff_count
        ),
        "sand_rejected_high_terrain_contact_count": (
            first_geometry.rejected_high_terrain_contact_count
        ),
        "sand_accepted_buried_step_count": (
            first_geometry.accepted_buried_step_count
        ),
        "sand_steep_triangle_count": first_geometry.steep_sand_triangle_count,
        "sand_subdivision_level": first_geometry.subdivision_level,
        "sand_pre_subdivision_triangle_count": (
            first_geometry.pre_subdivision_triangle_count
        ),
        "active_water_cells": int(first.free_water_mask.sum()),
        "final_time_seconds": final_render_time,
        "final_lower_step": final.lower_step,
        "final_upper_step": final.upper_step,
        "final_interpolation_weight": final.interpolation_weight,
        "final_active_sand_cells": int(final.sand_mask.sum()),
        "final_sand_gaussian_count": int(final_geometry.xyz_gs.shape[0]),
        "final_sand_candidate_triangle_count": (
            final_geometry.candidate_triangle_count
        ),
        "final_sand_rejected_triangle_count": (
            final_geometry.rejected_triangle_count
        ),
        "final_sand_rejected_degenerate_count": (
            final_geometry.rejected_degenerate_count
        ),
        "final_sand_rejected_terrain_cliff_count": (
            final_geometry.rejected_terrain_cliff_count
        ),
        "final_sand_rejected_high_terrain_contact_count": (
            final_geometry.rejected_high_terrain_contact_count
        ),
        "final_sand_accepted_buried_step_count": (
            final_geometry.accepted_buried_step_count
        ),
        "final_sand_steep_triangle_count": (
            final_geometry.steep_sand_triangle_count
        ),
        "final_sand_subdivision_level": final_geometry.subdivision_level,
        "final_sand_pre_subdivision_triangle_count": (
            final_geometry.pre_subdivision_triangle_count
        ),
        "final_active_water_cells": int(final.free_water_mask.sum()),
        "scale_min": [
            float(value) for value in first_geometry.scales_gs.min(axis=0)
        ],
        "scale_max": [
            float(value) for value in first_geometry.scales_gs.max(axis=0)
        ],
    }
    audit["status"] = "solver_validated"
    write_yaml(artifact_directory / "pipeline.yaml", audit)

    if not args.skip_render:
        subprocess.run(
            render_command,
            check=True,
            cwd=REPO_ROOT,
            env=build_renderer_environment(),
        )
        audit["status"] = "completed"  # 渲染成功后更新状态
        write_yaml(artifact_directory / "pipeline.yaml", audit)
    print(f"{stage_label} {audit['status']}: {artifact_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
