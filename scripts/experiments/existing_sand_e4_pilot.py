#!/usr/bin/env python3
"""Run one checked E4 dry/water solver case with the existing binary."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sandygs.exchange import sha256_file, validate_bundle
from sandygs.existing_sand.workspace import ExistingSandError, _load_yaml, _stage_location
from sandygs.storage_guard import StorageGuardError, validate_artifact_storage


def _artifact_path(stage: Path, artifacts: dict, relative: str) -> Path:
    entry = artifacts.get(relative)
    if not isinstance(entry, dict):
        raise ExistingSandError(f"E4 initialization artifact missing: {relative}")
    path = (stage / relative).resolve()
    if not path.is_relative_to(stage) or not path.is_file():
        raise ExistingSandError(f"E4 initialization artifact unavailable: {relative}")
    if sha256_file(path) != entry.get("sha256"):
        raise ExistingSandError(f"E4 initialization artifact changed: {relative}")
    return path


def prepare_command(run_root: Path, config_path: Path, case: str,
                    binary: Path, *, experiment_id: str | None = None) -> dict:
    run_root = Path(run_root).resolve()
    config_path = Path(config_path).resolve()
    binary = Path(binary).resolve()
    config = _load_yaml(config_path, "E4 pilot config")
    run = _load_yaml(run_root / "run.yaml", "existing-sand run")
    if config.get("schema_version") != 1 or config.get("scene_id") != run.get("scene_id"):
        raise ExistingSandError("E4 pilot config scene does not match the run")
    if case not in ("dry", "water") or case not in config.get("cases", []):
        raise ExistingSandError("E4 pilot case must be dry or water and listed in config")
    stage_id = config.get("initialization_stage_id")
    stage = _stage_location(run_root, "modeling", stage_id)
    manifest = _load_yaml(stage / "manifest.yaml", "E4 initialization manifest")
    if (manifest.get("stage_id") != stage_id
            or manifest.get("prompt", {}).get("type") != "e4_reviewed_heightfield_solver_initialization"):
        raise ExistingSandError("E4 pilot needs a sealed E4 initialization stage")
    artifacts = {item["path"]: item for item in manifest.get("artifacts", [])}
    scene = _artifact_path(stage, artifacts, f"cases/{case}/input/scene.yaml")
    case_yaml = _load_yaml(_artifact_path(stage, artifacts, f"cases/{case}/input/case.yaml"),
                           "E4 solver case")
    for name in ("terrain.npy", "sand.npy", "total_water.npy"):
        _artifact_path(stage, artifacts, f"cases/{case}/input/{name}")
    exchange = stage / "exchange"
    _artifact_path(stage, artifacts, "exchange/metadata.yaml")
    validate_bundle(exchange)
    if Path(case_yaml.get("exchange_bundle", "")).resolve() != exchange:
        raise ExistingSandError("solver case points to a different exchange bundle")
    scene_data = _load_yaml(scene, "E4 solver scene")
    settings = config.get("solver")
    if not isinstance(settings, dict) or set(settings) != {
            "duration_seconds", "frame_stride", "snapshot_level", "write_obj"}:
        raise ExistingSandError("E4 pilot solver settings are incomplete or ambiguous")
    duration = settings["duration_seconds"]
    stride = settings["frame_stride"]
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or duration <= 0
            or isinstance(stride, bool) or not isinstance(stride, int) or stride < 1
            or settings["snapshot_level"] != "render" or settings["write_obj"] is not False):
        raise ExistingSandError("E4 pilot needs positive duration/stride, render snapshots and no OBJ")
    if (not np.isclose(float(scene_data["run"]["duration"]), duration, rtol=0, atol=1e-12)
            or scene_data["run"]["frame_stride"] != stride
            or scene_data["run"]["write_obj"] is not False):
        raise ExistingSandError("E4 pilot settings differ from sealed solver scene")
    if not binary.is_file() or not (binary.stat().st_mode & 0o111):
        raise ExistingSandError(f"solver binary is missing or not executable: {binary}")
    identifier = experiment_id or config.get("experiment_id")
    # Reuse the established stage ID syntax even though solver output has its
    # own simulations/ namespace rather than an offline modeling stage.
    if not isinstance(identifier, str) or not identifier or any(
        not (character.islower() or character.isdigit() or character in "_-")
        for character in identifier
    ) or not identifier[0].islower():
        raise ExistingSandError("experiment_id must be a lowercase stage-style identifier")
    output = run_root / "simulations" / identifier / case
    if output.exists():
        raise ExistingSandError(f"refusing to overwrite solver output: {output}")
    command = [str(binary), "--scene", str(scene), "--output", str(output),
               "--duration", str(duration), "--frame-stride", str(stride),
               "--snapshot", "render", "--no-obj"]
    return {"command": command, "output": output, "stage": stage,
            "stage_id": stage_id, "binary_sha256": sha256_file(binary),
            "config_sha256": sha256_file(config_path),
            "manifest_sha256": sha256_file(stage / "manifest.yaml"),
            "duration_seconds": float(duration)}


def summarize_solver_run(plan: dict, *, case: str) -> dict:
    output = plan["output"]
    with (output / "metrics.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ExistingSandError("solver produced no time step after initialization")
    first, last = rows[0], rows[-1]
    fields = ("time", "sand_volume", "total_water_volume", "max_velocity", "min_det_f")
    if any(not math.isfinite(float(row[field])) for row in rows for field in fields):
        raise ExistingSandError("solver metrics contain nonfinite values")
    if (int(first["step"]) != 0 or not math.isclose(float(first["time"]), 0, abs_tol=1e-12)
            or not math.isclose(float(last["time"]), plan["duration_seconds"], abs_tol=1e-9)):
        raise ExistingSandError("solver time range differs from E4 pilot plan")
    frames = sorted((output / "frames").glob("[0-9]*"))
    if (len(frames) < 2 or frames[0].name != "000000"
            or int(frames[-1].name) != int(last["step"])):
        raise ExistingSandError("solver did not write initial and final render snapshots")
    metadata = _load_yaml(output / "metadata.yaml", "solver metadata")
    boundary = int(metadata["grid"]["boundary_width"])
    exchange_metadata = validate_bundle(plan["stage"] / "exchange")
    shape = tuple(exchange_metadata["grid"]["resolution"])
    final = {}
    for name in ("terrain", "sand_height", "pure_water_height", "mixed_water_height", "total_water"):
        values = np.load(frames[-1] / f"{name}.npy", allow_pickle=False)
        interior = values[boundary:-boundary, boundary:-boundary]
        if interior.shape != shape or not np.isfinite(interior).all():
            raise ExistingSandError(f"final solver field {name} has wrong shape or nonfinite values")
        if name != "terrain" and np.any(interior < -1e-10):
            raise ExistingSandError(f"final solver field {name} has negative values")
        final[name] = interior
    first_water = np.load(frames[0] / "total_water.npy", allow_pickle=False)
    first_water = first_water[boundary:-boundary, boundary:-boundary]
    expected = np.load(plan["stage"] / "cases" / case / "input" / "total_water.npy",
                       allow_pickle=False)
    if not np.allclose(first_water, expected, rtol=0, atol=1e-12):
        raise ExistingSandError("initial solver water differs from prepared E4 case")
    if case == "dry" and np.max(np.abs(final["total_water"])) > 1e-12:
        raise ExistingSandError("dry solver case developed water")
    fields = ("sand_volume", "total_water_volume", "max_velocity", "min_det_f")
    report = {
        "status": "completed_solver_pilot",
        "case": case,
        "initialization_stage_id": plan["stage_id"],
        "initialization_manifest_sha256": plan["manifest_sha256"],
        "solver_binary_sha256": plan["binary_sha256"],
        "pilot_config_sha256": plan["config_sha256"],
        "command": plan["command"],
        "initial_step": int(first["step"]), "final_step": int(last["step"]),
        "final_time_seconds": float(last["time"]),
        "saved_frames": len(frames),
        "initial": {field: float(first[field]) for field in fields},
        "final": {field: float(last[field]) for field in fields},
        "sand_volume_change": float(last["sand_volume"]) - float(first["sand_volume"]),
        "water_volume_change": float(last["total_water_volume"]) - float(first["total_water_volume"]),
        "final_water_on_initial_sand_cells": int(np.count_nonzero(
            (final["total_water"] > 1e-6) &
            (np.load(plan["stage"] / "cases" / case / "input" / "sand.npy",
                     allow_pickle=False) > 1e-6)
        )),
    }
    with (output / "pilot_report.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/existing_sand/indoor02_e4_pilot.yaml")
    parser.add_argument("--case", choices=("dry", "water"), required=True)
    parser.add_argument("--experiment-id", help="new ID when repeating the dry/water pilot")
    parser.add_argument("--binary", type=Path, default=ROOT / "solvers/sand_water/build/linux/x86_64/release/demo")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        validate_artifact_storage(ROOT / "configs/paths.local.yaml")
        plan = prepare_command(args.run, args.config, args.case, args.binary,
                               experiment_id=args.experiment_id)
        if args.check_only:
            print(json.dumps({"output": str(plan["output"]), "command": plan["command"],
                              "initialization_stage_id": plan["stage_id"],
                              "solver_binary_sha256": plan["binary_sha256"]}, indent=2))
            return 0
        plan["output"].parent.mkdir(parents=True, exist_ok=True)
        print("+", " ".join(plan["command"]), flush=True)
        subprocess.run(plan["command"], check=True)
        print(json.dumps(summarize_solver_run(plan, case=args.case), indent=2))
    except (ExistingSandError, StorageGuardError, OSError, ValueError, KeyError,
            subprocess.CalledProcessError) as exc:
        parser.exit(2, f"existing-sand E4 pilot: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
