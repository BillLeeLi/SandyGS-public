#!/usr/bin/env python3
"""Review intermediate E4 solver fields, including an interrupted run."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sandygs.existing_sand.workspace import ExistingSandError, _load_yaml


FIELDS = ("sand_height", "total_water", "mixed_water_height")
HISTORY_FIELDS = ("sand_volume", "total_water_volume", "min_det_f",
                  "max_velocity", "cfl", "sand_velocity_clamp_count")


def _case(directory: Path) -> dict:
    metadata = _load_yaml(directory / "metadata.yaml", "E4 solver metadata")
    grid = metadata["grid"]
    boundary = int(grid["boundary_width"])
    shape = tuple(int(value) - 2 * boundary for value in grid["resolution"])
    if boundary < 1 or len(shape) != 2 or min(shape) < 1:
        raise ExistingSandError(f"invalid E4 solver grid: {directory}")
    with (directory / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(field not in rows[0] for field in
                       ("step", "time", *HISTORY_FIELDS)):
        raise ExistingSandError(f"E4 solver metrics are unavailable: {directory}")
    valid_rows = []
    for row in rows:
        try:
            step = int(row["step"])
            time = float(row["time"])
            values = {
                name: float(row[name]) if row[name] not in (None, "") else float("nan")
                for name in HISTORY_FIELDS
            }
        except (TypeError, ValueError):
            break  # An interrupted writer can leave one incomplete trailing row.
        if not math.isfinite(time):
            break
        if valid_rows and (step <= valid_rows[-1]["step"] or time <= valid_rows[-1]["time"]):
            raise ExistingSandError(f"invalid metric step/time order: {directory}")
        valid_rows.append({"step": step, "time": time, **values})
    if not valid_rows:
        raise ExistingSandError(f"E4 solver has no complete metric rows: {directory}")
    steps = {row["step"]: row["time"] for row in valid_rows}
    frames = []
    for path in sorted((directory / "frames").glob("[0-9]*")):
        if not path.is_dir() or not path.name.isdigit():
            continue
        step = int(path.name)
        if step in steps and all((path / f"{name}.npy").is_file() for name in FIELDS):
            frames.append((steps[step], step, path))
    if not frames:
        raise ExistingSandError(f"E4 solver has no complete snapshots: {directory}")
    return {"directory": directory, "metadata": metadata, "shape": shape,
            "boundary": boundary, "rows": valid_rows, "frames": frames, "cache": {}}


def _arrays(case: dict, path: Path) -> dict[str, np.ndarray]:
    if path not in case["cache"]:
        expected = tuple(size + 2 * case["boundary"] for size in case["shape"])
        loaded = {}
        for name in FIELDS:
            raw = np.load(path / f"{name}.npy", allow_pickle=False)
            if raw.shape != expected:
                raise ExistingSandError(f"wrong E4 snapshot shape: {path}/{name}.npy")
            b = case["boundary"]
            loaded[name] = raw[b:-b, b:-b].astype(np.float64, copy=False)
        case["cache"][path] = loaded
    return case["cache"][path]


def _sample(case: dict, time: float) -> tuple[dict[str, np.ndarray], dict] | None:
    frames = case["frames"]
    times = [item[0] for item in frames]
    if time < times[0] - 1e-9 or time > times[-1] + 1e-9:
        return None
    index = bisect_left(times, time)
    if index == len(frames):
        index -= 1
    if abs(times[index] - time) <= 1e-9 or index == 0:
        selected = frames[index]
        return _arrays(case, selected[2]), {
            "step_bracket": [selected[1], selected[1]],
            "time_bracket": [selected[0], selected[0]],
            "interpolation_weight": 0.0,
        }
    before, after = frames[index - 1], frames[index]
    weight = (time - before[0]) / (after[0] - before[0])
    left, right = _arrays(case, before[2]), _arrays(case, after[2])
    fields = {name: left[name] * (1.0 - weight) + right[name] * weight
              for name in FIELDS}
    return fields, {"step_bracket": [before[1], after[1]],
                    "time_bracket": [before[0], after[0]],
                    "interpolation_weight": float(weight)}


def _finite_max(arrays: list[np.ndarray], *, absolute: bool = False) -> float:
    finite = [np.abs(array[np.isfinite(array)]) if absolute else
              array[np.isfinite(array)] for array in arrays]
    finite = [values for values in finite if values.size]
    return max((float(np.max(values)) for values in finite), default=0.0) or 1e-12


def _history_figure(cases: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    items = (
        ("sand_volume", "sand volume change", True),
        ("total_water_volume", "water volume change", True),
        ("min_det_f", "minimum det(F)", False),
        ("max_velocity", "maximum velocity", False),
        ("cfl", "maximum reported CFL", False),
        ("sand_velocity_clamp_count", "cumulative clamped steps", False),
    )
    for ax, (field, title, relative) in zip(axes.flat, items):
        for name, case in cases.items():
            times = np.asarray([row["time"] for row in case["rows"]])
            values = np.asarray([row[field] for row in case["rows"]])
            if relative:
                values = (values - values[0]) / max(abs(values[0]), 1e-12)
            elif field == "sand_velocity_clamp_count":
                values = np.cumsum(values > 0)
            ax.plot(times, values, label=name, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("solver time (s)")
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(output / "metrics_history.png", dpi=130)
    plt.close(fig)


def _field_figure(output: Path, selected: list[dict], initial_sand: np.ndarray,
                  extent: list[float]) -> None:
    limits = {
        "sand": _finite_max([entry[case]["sand_height"]
                             for entry in selected for case in ("dry", "water")]),
        "water": _finite_max([entry["water"]["total_water"] for entry in selected]),
        "mixed": _finite_max([entry["water"]["mixed_water_height"] for entry in selected]),
        "difference": _finite_max([entry["water"]["sand_height"] -
                                   entry["dry"]["sand_height"] for entry in selected],
                                  absolute=True),
        "dry_change": _finite_max([entry["dry"]["sand_height"] - initial_sand
                                  for entry in selected], absolute=True),
    }
    contour = initial_sand.T > 1e-6
    for index, entry in enumerate(selected):
        fields = (
            (entry["dry"]["sand_height"], "dry sand", "YlOrBr", 0, limits["sand"]),
            (entry["water"]["sand_height"], "wet sand", "YlOrBr", 0, limits["sand"]),
            (entry["water"]["sand_height"] - entry["dry"]["sand_height"],
             "wet minus dry sand", "coolwarm", -limits["difference"], limits["difference"]),
            (entry["water"]["total_water"], "total water", "Blues", 0, limits["water"]),
            (entry["water"]["mixed_water_height"], "mixed water", "PuBuGn", 0, limits["mixed"]),
            (entry["dry"]["sand_height"] - initial_sand,
             "dry sand change", "coolwarm", -limits["dry_change"], limits["dry_change"]),
        )
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        for ax, (values, title, cmap, low, high) in zip(axes.flat, fields):
            image = ax.imshow(np.ma.masked_invalid(values.T), origin="lower",
                              extent=extent, cmap=cmap, vmin=low, vmax=high)
            ax.contour(contour, levels=[.5], origin="lower", extent=extent,
                       colors="black", linewidths=.4)
            ax.set_title(title)
            ax.set_xlabel("simulation x")
            ax.set_ylabel("simulation y")
            fig.colorbar(image, ax=ax, shrink=.75)
        fig.suptitle(f"requested t = {entry['time_seconds']:.3f} s")
        fig.savefig(output / f"time_{index:04d}.png", dpi=120)
        plt.close(fig)


def review(run_root: Path, experiment_id: str, output_id: str,
           targets: list[float]) -> dict:
    if any(re.fullmatch(r"[a-z][a-z0-9_-]*", item) is None
           for item in (experiment_id, output_id)):
        raise ExistingSandError("E4 diagnostic IDs must be lowercase stage-style names")
    if (not targets or any(not math.isfinite(value) or value < 0 for value in targets)
            or any(b <= a for a, b in zip(targets, targets[1:]))):
        raise ExistingSandError("E4 diagnostic times must be increasing and nonnegative")
    run_root = Path(run_root).resolve()
    experiment = run_root / "simulations" / experiment_id
    output = run_root / "diagnostics" / output_id
    if output.exists():
        raise ExistingSandError(f"refusing to overwrite E4 diagnostic: {output}")
    cases = {name: _case(experiment / name) for name in ("dry", "water")}
    dry_grid = cases["dry"]["metadata"]["grid"]
    if dry_grid != cases["water"]["metadata"]["grid"]:
        raise ExistingSandError("E4 dry/water grids differ")
    if cases["dry"]["shape"] != cases["water"]["shape"]:
        raise ExistingSandError("E4 dry/water interior shapes differ")
    first = _sample(cases["dry"], 0.0)
    if first is None:
        raise ExistingSandError("E4 dry case has no initial snapshot")
    initial_sand = first[0]["sand_height"]
    selected = []
    skipped = []
    for target in targets:
        dry, water = _sample(cases["dry"], target), _sample(cases["water"], target)
        if dry is None or water is None:
            skipped.append(float(target))
            continue
        dry_fields, dry_source = dry
        water_fields, water_source = water
        difference = water_fields["sand_height"] - dry_fields["sand_height"]
        valid = np.isfinite(difference)
        summary = {
            "time_seconds": float(target),
            "dry_source": dry_source,
            "water_source": water_source,
            "dry_nonfinite_sand_cells": int(np.count_nonzero(~np.isfinite(dry_fields["sand_height"]))),
            "water_nonfinite_sand_cells": int(np.count_nonzero(~np.isfinite(water_fields["sand_height"]))),
            "water_on_initial_sand_cells": int(np.count_nonzero(
                (initial_sand > 1e-6) & (water_fields["total_water"] > 1e-6))),
            "max_abs_wet_minus_dry_sand": float(np.max(np.abs(difference[valid]))) if valid.any() else None,
        }
        selected.append({"time_seconds": float(target), "dry": dry_fields,
                         "water": water_fields, "summary": summary})
    if not selected:
        raise ExistingSandError("no requested E4 time is present in both solver outputs")
    spacing = float(dry_grid["spacing"])
    nx, ny = cases["dry"]["shape"]
    center = [float(value) for value in dry_grid["center"]]
    extent = [center[0] - nx * spacing / 2, center[0] + nx * spacing / 2,
              center[1] - ny * spacing / 2, center[1] + ny * spacing / 2]
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = Path(tempfile.mkdtemp(prefix=f".{output_id}_", dir=output.parent))
    try:
        _history_figure(cases, payload)
        _field_figure(payload, selected, initial_sand, extent)
        report = {
            "status": "partial" if skipped else "requested_times_available",
            "experiment_id": experiment_id,
            "output_id": output_id,
            "completed_pilot_reports": {
                name: (case["directory"] / "pilot_report.yaml").is_file()
                for name, case in cases.items()
            },
            "available_time_seconds": {
                name: float(case["frames"][-1][0]) for name, case in cases.items()
            },
            "requested_times_seconds": [float(value) for value in targets],
            "skipped_times_seconds": skipped,
            "grid_resolution_xy": [nx, ny],
            "extent_xy": extent,
            "panels": ["dry sand", "wet sand", "wet minus dry sand",
                       "total water", "mixed water", "dry sand change"],
            "times": [entry["summary"] for entry in selected],
        }
        with (payload / "summary.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(report, stream, sort_keys=False)
        payload.rename(output)
    finally:
        if payload.exists():
            shutil.rmtree(payload)
    return {"output": str(output), "reviewed_times_seconds":
            [entry["time_seconds"] for entry in selected], "skipped_times_seconds": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--times", nargs="+", type=float, required=True)
    args = parser.parse_args()
    try:
        result = review(args.run, args.experiment_id, args.output_id, args.times)
    except (ExistingSandError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"existing-sand E4 time series: {exc}\n")
    print(yaml.safe_dump(result, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
