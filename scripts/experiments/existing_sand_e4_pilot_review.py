#!/usr/bin/env python3
"""Compare completed E4 dry/water render snapshots and volume histories."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sandygs.exchange import GridSpec, validate_bundle
from sandygs.existing_sand.workspace import ExistingSandError, _load_yaml, _stage_location


def _metrics(path: Path) -> list[dict[str, float]]:
    with (path / "metrics.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
    if len(rows) < 2:
        raise ExistingSandError(f"solver metrics need initial and evolved rows: {path}")
    return rows


def _frame(path: Path, *, final: bool, boundary: int, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    frames = sorted((path / "frames").glob("[0-9]*"))
    if len(frames) < 2:
        raise ExistingSandError(f"solver run needs at least two render frames: {path}")
    frame = frames[-1] if final else frames[0]
    fields = {}
    for name in ("terrain", "sand_height", "pure_water_height", "mixed_water_height", "total_water"):
        raw = np.load(frame / f"{name}.npy", allow_pickle=False)
        values = raw[boundary:-boundary, boundary:-boundary]
        if values.shape != shape or not np.isfinite(values).all():
            raise ExistingSandError(f"invalid {name} in {frame}")
        fields[name] = values
    return fields


def build_review(run_root: Path, experiment_id: str, initialization_stage_id: str) -> dict:
    run_root = Path(run_root).resolve()
    stage = _stage_location(run_root, "modeling", initialization_stage_id)
    exchange_meta = validate_bundle(stage / "exchange")
    grid_data = exchange_meta["grid"]
    grid = GridSpec(tuple(grid_data["resolution"]), tuple(grid_data["domain_min"]),
                    tuple(grid_data["domain_max"]))
    experiment = run_root / "simulations" / experiment_id
    output = experiment / "review"
    if output.exists():
        raise ExistingSandError(f"refusing to overwrite review: {output}")
    runs = {case: experiment / case for case in ("dry", "water")}
    histories = {case: _metrics(path) for case, path in runs.items()}
    reports = {}
    for case, path in runs.items():
        report = _load_yaml(path / "pilot_report.yaml", f"{case} pilot report")
        reports[case] = report
        metadata = _load_yaml(path / "metadata.yaml", f"{case} solver metadata")
        if (report.get("status") != "completed_solver_pilot"
                or report.get("initialization_stage_id") != initialization_stage_id
                or report.get("case") != case
                or metadata["grid"]["boundary_width"] != 1
                or metadata["grid"]["resolution"] != [grid.resolution[0] + 2, grid.resolution[1] + 2]):
            raise ExistingSandError(f"{case} pilot does not match the E4 exchange grid")
    for key in ("initialization_manifest_sha256", "solver_binary_sha256", "pilot_config_sha256"):
        if not reports["dry"].get(key) or reports["dry"][key] != reports["water"].get(key):
            raise ExistingSandError(f"dry and water pilots differ in {key}")
    times = [history[-1]["time"] for history in histories.values()]
    if not np.isclose(times[0], times[1], rtol=0, atol=1e-9):
        raise ExistingSandError("dry and water pilots end at different times")
    dry_first = _frame(runs["dry"], final=False, boundary=1, shape=grid.resolution)
    dry_last = _frame(runs["dry"], final=True, boundary=1, shape=grid.resolution)
    water_first = _frame(runs["water"], final=False, boundary=1, shape=grid.resolution)
    water_last = _frame(runs["water"], final=True, boundary=1, shape=grid.resolution)
    if (not np.allclose(dry_first["sand_height"], water_first["sand_height"], rtol=0, atol=1e-12)
            or np.any(dry_first["total_water"] != 0)):
        raise ExistingSandError("dry and water cases do not share the same sand or dry control")
    initial_sand = dry_first["sand_height"]
    sand_difference = water_last["sand_height"] - dry_last["sand_height"]
    water_final = water_last["total_water"]
    overlap = (initial_sand > 1e-6) & (water_final > 1e-6)
    cell_area = grid.spacing[0] * grid.spacing[1]
    summary = {
        "status": "dry_water_solver_field_comparison",
        "experiment_id": experiment_id,
        "initialization_stage_id": initialization_stage_id,
        "final_time_seconds": float(times[0]),
        "dry_final_step": int(histories["dry"][-1]["step"]),
        "water_final_step": int(histories["water"][-1]["step"]),
        "dry_sand_volume_change": float(histories["dry"][-1]["sand_volume"] - histories["dry"][0]["sand_volume"]),
        "water_sand_volume_change": float(histories["water"][-1]["sand_volume"] - histories["water"][0]["sand_volume"]),
        "water_volume_change": float(histories["water"][-1]["total_water_volume"] - histories["water"][0]["total_water_volume"]),
        "water_on_initial_sand_cells": int(overlap.sum()),
        "water_on_initial_sand_volume": float(water_final[overlap].sum() * cell_area),
        "final_mixed_water_volume": float(water_last["mixed_water_height"].sum() * cell_area),
        "max_abs_wet_minus_dry_sand_thickness": float(np.max(np.abs(sand_difference))),
    }
    extent = [grid.domain_min[0], grid.domain_max[0], grid.domain_min[1], grid.domain_max[1]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = (
        (initial_sand, "initial sand thickness", "YlOrBr"),
        (water_first["total_water"], "initial water thickness", "Blues"),
        (water_final, "final water thickness", "Blues"),
        (dry_last["sand_height"] - initial_sand, "dry sand change", "coolwarm"),
        (sand_difference, "wet minus dry sand", "coolwarm"),
        (water_last["mixed_water_height"], "final mixed water", "PuBuGn"),
    )
    for ax, (field, title, palette) in zip(axes.flat, panels):
        limit = float(np.max(np.abs(field))) if palette == "coolwarm" else None
        image = ax.imshow(field.T, origin="lower", extent=extent, cmap=palette,
                          vmin=-limit if limit else None, vmax=limit if limit else None)
        ax.contour(initial_sand.T > 1e-6, levels=[.5], origin="lower", extent=extent,
                   colors="black", linewidths=.4)
        ax.set_title(title)
        ax.set_xlabel("simulation x")
        ax.set_ylabel("simulation y")
        fig.colorbar(image, ax=ax, shrink=.75)
    output.mkdir(parents=True)
    fig.savefig(output / "overview.png", dpi=140)
    plt.close(fig)
    with (output / "metrics.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(summary, stream, sort_keys=False)
    return {"review_dir": str(output), **summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--experiment-id", default="e4_water_pilot_001")
    parser.add_argument("--initialization-stage-id", default="e4_exchange_initial_001")
    args = parser.parse_args()
    try:
        result = build_review(args.run, args.experiment_id, args.initialization_stage_id)
    except (ExistingSandError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"existing-sand E4 pilot review: {exc}\n")
    print(yaml.safe_dump(result, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
