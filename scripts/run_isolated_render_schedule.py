#!/usr/bin/env python3
"""Render an immutable solver run one time/view per renderer process.

This driver is intentionally slower than the ordinary scheduled renderer: the
PGSR scene is loaded once per frame.  In return, all CUDA tensors, NumPy
snapshot caches, and WSL GPU-host allocations are released at every process
boundary.  It is intended for memory-constrained rerenders and supports
``--resume`` after a WSL restart without rerunning completed frames.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.render_schedule import ScheduledRenderFrame, expand_render_schedule
from sandygs.storage_guard import validate_artifact_storage


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)


def command_option(command: list[Any], option: str) -> str:
    values = [str(value) for value in command]
    try:
        return values[values.index(option) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"source render command has no {option}") from exc


def safe_artifact_relative(value: Any, *, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} must stay below the artifact root")
    return path


def isolated_schedule(
    frame: ScheduledRenderFrame,
    source_schedule: dict[str, Any],
) -> dict[str, Any]:
    """Build a one-frame table with a predictable temporary output name."""

    schedule: dict[str, Any] = {
        "schema_version": 1,
        "camera_path": source_schedule.get("camera_path", "generated_orbit"),
        "camera_path_frames": source_schedule["camera_path_frames"],
        "sequences": [
            {
                "name": "isolated_frame",
                "type": "moving_camera",
                "camera_indices": [frame.camera_index],
                "times": [frame.time_seconds],
                "debug": "all" if frame.save_debug else "none",
                "intermediates": "all" if frame.save_intermediates else "none",
            }
        ],
    }
    if "camera_path_azimuth_degrees" in source_schedule:
        schedule["camera_path_azimuth_degrees"] = source_schedule[
            "camera_path_azimuth_degrees"
        ]
    return schedule


def merge_isolated_frame(
    job_render: Path,
    final_render: Path,
    frame: ScheduledRenderFrame,
) -> int:
    """Copy one isolated render tree into its original scheduled location."""

    source = job_render / "release_check/isolated_frame"
    if not source.is_dir():
        raise FileNotFoundError(f"isolated renderer produced no frame tree: {source}")
    expected_rgb = source / "renders/00000.png"
    if not expected_rgb.is_file():
        raise FileNotFoundError(f"isolated renderer produced no RGB frame: {expected_rgb}")

    destination = final_render / "release_check" / frame.output_group
    copied = 0
    for source_path in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_path.relative_to(source)
        if source_path.stem == "00000":
            relative = relative.with_name(frame.output_name + source_path.suffix)
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied += 1
    raw_manifest = (
        destination
        / "regression_raw"
        / "manifests"
        / f"{frame.output_name}.yaml"
    )
    if raw_manifest.is_file():
        manifest = load_yaml(raw_manifest)
        manifest["frame"] = frame.output_name
        arrays = manifest.get("arrays")
        if not isinstance(arrays, list):
            raise ValueError(f"invalid regression raw manifest: {raw_manifest}")
        for entry in arrays:
            if not isinstance(entry, dict) or not isinstance(
                entry.get("relative_path"), str
            ):
                raise ValueError(
                    f"invalid regression raw manifest entry: {raw_manifest}"
                )
            relative = Path(entry["relative_path"])
            if relative.stem == "00000":
                entry["relative_path"] = relative.with_name(
                    frame.output_name + relative.suffix
                ).as_posix()
        write_yaml(raw_manifest, manifest)
    return copied


def available_memory_gib() -> float:
    """Return Linux MemAvailable without importing a monitoring dependency."""

    with Path("/proc/meminfo").open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / (1024.0**2)
    raise RuntimeError("/proc/meminfo does not report MemAvailable")


def build_renderer_environment(paths: dict[str, Any]) -> dict[str, str]:
    """Mirror the repository's audited Conda/CUDA renderer environment."""

    environment = os.environ.copy()
    environment_bin = Path(sys.executable).resolve().parent
    environment_prefix = environment_bin.parent
    environment.setdefault("CUDA_HOME", "/usr/local/cuda")
    environment.setdefault("CC", "/usr/bin/gcc-11")
    environment.setdefault("CXX", "/usr/bin/g++-11")
    # A missing JIT extension must compile serially on this 8 GiB WSL machine.
    environment.setdefault("MAX_JOBS", "1")
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    environment.setdefault(
        "SANDYGS_DATA_ROOT",
        str(Path(paths["roots"]["datasets"]).expanduser().resolve().parent),
    )
    cuda_home = Path(environment["CUDA_HOME"])

    def prepend(name: str, entries: list[Path]) -> None:
        current = environment.get(name, "").split(os.pathsep)
        ordered = [str(entry) for entry in entries if entry.is_dir()]
        ordered.extend(value for value in current if value and value not in ordered)
        environment[name] = os.pathsep.join(ordered)

    prepend("PATH", [environment_bin, cuda_home / "bin"])
    prepend("LIBRARY_PATH", [environment_prefix / "lib", cuda_home / "lib64"])
    prepend(
        "LD_LIBRARY_PATH",
        [environment_prefix / "lib", Path("/usr/lib/wsl/lib"), cuda_home / "lib64"],
    )
    return environment


def run_and_log(command: list[str], log_path: Path, environment: dict[str, str]) -> None:
    """Stream renderer output to the terminal and retain an exact frame log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument(
        "--paths",
        type=Path,
        default=REPO_ROOT / "configs/paths.local.yaml",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("isolated-render-%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing run-id and skip frames with completion markers.",
    )
    args = parser.parse_args()

    spec_path = args.spec.expanduser().resolve()
    paths_path = args.paths.expanduser().resolve()
    spec = load_yaml(spec_path)
    if spec.get("schema_version") != 1:
        raise ValueError("isolated render spec must use schema_version: 1")
    validate_artifact_storage(paths_path)
    paths = load_yaml(paths_path)
    artifacts = Path(paths["roots"]["artifacts"]).expanduser().resolve()
    source_relative = safe_artifact_relative(
        spec["source"]["artifact_relative"], label="source.artifact_relative"
    )
    output_relative = safe_artifact_relative(
        spec["output_parent_relative"], label="output_parent_relative"
    )
    source = artifacts / source_relative
    output = artifacts / output_relative / args.run_id
    if not (source / "run/metadata.yaml").is_file():
        raise FileNotFoundError(f"source solver run is incomplete: {source / 'run'}")
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"refusing to overwrite isolated render output {output}; use --resume"
        )
    output.mkdir(parents=True, exist_ok=True)

    base_config = load_yaml(source / "sand_water_render.yaml")
    source_pipeline = load_yaml(source / "pipeline.yaml")
    source_command = source_pipeline["commands"]["render"]
    dataset = command_option(source_command, "-s")
    checkpoint = command_option(source_command, "-m")
    render_schedule = spec["render_schedule"]
    frames, _ = expand_render_schedule(render_schedule)
    render_overrides = spec.get("render_overrides", {})
    if not isinstance(render_overrides, dict):
        raise ValueError("render_overrides must be a mapping")

    jobs: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        job_id = f"frame_{index:04d}_t{frame.time_seconds:.6f}_c{frame.camera_index:03d}"
        job_root = output / "_isolated_jobs" / job_id
        config_path = job_root / "render_config.yaml"
        job_render = job_root / "render"
        config = dict(base_config)
        config.update(render_overrides)
        config["render_schedule"] = isolated_schedule(frame, render_schedule)
        write_yaml(config_path, config)
        command = [
            sys.executable,
            str(REPO_ROOT / "render_sand_water.py"),
            "-s",
            dataset,
            "-m",
            checkpoint,
            "--render_output_path",
            str(job_render),
            "--config_file",
            str(config_path),
            "--max_views",
            "1",
            "--disable_preload_img",
        ]
        jobs.append(
            {
                "id": job_id,
                "index": index,
                "status": "pending",
                "frame": frame.to_mapping(),
                "command": command,
                "config": str(config_path),
                "log": str(job_root / "renderer.log"),
            }
        )

    audit_path = output / "experiment.yaml"
    prior_jobs: dict[str, dict[str, Any]] = {}
    if args.resume and audit_path.is_file():
        prior = load_yaml(audit_path)
        if Path(str(prior.get("source_artifact", ""))) != source:
            raise ValueError("--resume source artifact differs from the existing run")
        prior_jobs = {str(job["id"]): job for job in prior.get("jobs", [])}
        for job in jobs:
            previous = prior_jobs.get(job["id"])
            if previous is not None and previous.get("status") == "completed":
                job.update(previous)

    audit: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "run_id": args.run_id,
        "status": "prepared",
        "source_artifact": str(source),
        "spec": str(spec_path),
        "isolation": "one_physical_time_and_camera_per_process",
        "minimum_available_memory_gib": float(
            spec.get("minimum_available_memory_gib", 2.0)
        ),
        "final_render": str(output / "render"),
        "jobs": jobs,
    }
    write_yaml(audit_path, audit)
    if not args.execute:
        print(f"Isolated render prepared: {output}")
        return 0

    environment = build_renderer_environment(paths)
    final_render = output / "render"
    minimum_memory = audit["minimum_available_memory_gib"]
    audit["status"] = "running"
    write_yaml(audit_path, audit)
    try:
        for job, frame in zip(jobs, frames):
            expected_rgb = (
                final_render
                / "release_check"
                / frame.output_group
                / "renders"
                / f"{frame.output_name}.png"
            )
            if job["status"] == "completed" and expected_rgb.is_file():
                print(f"[skip completed] {job['id']}")
                continue
            memory_gib = available_memory_gib()
            if memory_gib < minimum_memory:
                raise RuntimeError(
                    f"only {memory_gib:.2f} GiB MemAvailable before {job['id']}; "
                    f"require at least {minimum_memory:.2f} GiB"
                )
            job["status"] = "running"
            job["mem_available_before_gib"] = memory_gib
            write_yaml(audit_path, audit)
            print(
                f"[{job['index'] + 1}/{len(jobs)}] {job['id']} "
                f"(MemAvailable={memory_gib:.2f} GiB)"
            )
            run_and_log(
                job["command"], Path(job["log"]), environment
            )
            job_root = Path(job["config"]).parent
            job["copied_product_count"] = merge_isolated_frame(
                job_root / "render", final_render, frame
            )
            metrics_path = job_root / "render/release_check/render_metrics.yaml"
            if metrics_path.is_file():
                job["metrics"] = load_yaml(metrics_path)
            job["status"] = "completed"
            write_yaml(audit_path, audit)

        # Publish the original full table beside the merged frame tree so the
        # final output remains self-describing despite isolated child runs.
        schedule_path = final_render / "release_check/render_schedule.yaml"
        write_yaml(
            schedule_path,
            {
                "schema_version": 1,
                "frames": [frame.to_mapping() for frame in frames],
                "execution": "isolated_process_per_frame",
            },
        )
        completed_metrics = [
            job.get("metrics", {}) for job in jobs if job["status"] == "completed"
        ]
        write_yaml(
            final_render / "release_check/isolated_render_metrics.yaml",
            {
                "schema_version": 1,
                "frame_count": len(jobs),
                "completed_frame_count": len(completed_metrics),
                "render_seconds_sum": sum(
                    float(value.get("render_seconds", 0.0))
                    for value in completed_metrics
                ),
                "peak_cuda_memory_allocated_mib_max": max(
                    (
                        float(value.get("peak_cuda_memory_allocated_mib", 0.0))
                        for value in completed_metrics
                    ),
                    default=0.0,
                ),
                "peak_cuda_memory_reserved_mib_max": max(
                    (
                        float(value.get("peak_cuda_memory_reserved_mib", 0.0))
                        for value in completed_metrics
                    ),
                    default=0.0,
                ),
            },
        )
        audit["status"] = "completed"
        write_yaml(audit_path, audit)
    except BaseException as exc:
        audit["status"] = "failed"
        audit["error"] = f"{type(exc).__name__}: {exc}"
        running = next((job for job in jobs if job["status"] == "running"), None)
        if running is not None:
            running["status"] = "failed"
            audit["failed_job"] = running["id"]
        write_yaml(audit_path, audit)
        raise

    print(f"Isolated render completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
