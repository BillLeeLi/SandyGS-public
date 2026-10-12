#!/usr/bin/env python3
"""Export comparable final sand thickness maps from completed E4 pilot runs."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sandygs.exchange import sha256_file
from sandygs.existing_sand.workspace import ExistingSandError, _load_yaml


def _sand_field(run_dir: Path, step: int, boundary: int, resolution: tuple[int, int]) -> tuple[np.ndarray, Path]:
    path = run_dir / "frames" / f"{step:06d}" / "sand_height.npy"
    raw = np.load(path, allow_pickle=False)
    expected = tuple(n + 2 * boundary for n in resolution)
    if raw.shape != expected or boundary < 1:
        raise ExistingSandError(f"unexpected E4 solver sand grid: {path}")
    interior = raw[boundary:-boundary, boundary:-boundary]
    if not np.isfinite(interior).all() or np.min(interior) < -1e-10:
        raise ExistingSandError(f"invalid final sand thickness: {path}")
    return interior, path


def export_final_sand(run_root: Path, experiment_ids: list[str], output_id: str) -> dict:
    run_root = Path(run_root).resolve()
    identifiers = [output_id, *experiment_ids]
    if any(re.fullmatch(r"[a-z][a-z0-9_-]*", identifier) is None for identifier in identifiers):
        raise ExistingSandError("E4 diagnostic IDs must be lowercase stage-style identifiers")
    output = run_root / "diagnostics" / output_id
    if output.exists():
        raise ExistingSandError(f"refusing to overwrite E4 diagnostic: {output}")
    if not experiment_ids or len(set(experiment_ids)) != len(experiment_ids):
        raise ExistingSandError("provide unique E4 pilot experiment IDs")

    cases = {}
    records = []
    reference = None
    for experiment_id in experiment_ids:
        experiment = run_root / "simulations" / experiment_id
        metrics = _load_yaml(experiment / "review" / "metrics.yaml", "E4 pilot review")
        if metrics.get("experiment_id") != experiment_id or metrics.get("status") != "dry_water_solver_field_comparison":
            raise ExistingSandError(f"E4 pilot review is missing or incomplete: {experiment_id}")
        loaded = {}
        for case in ("dry", "water"):
            run_dir = experiment / case
            report = _load_yaml(run_dir / "pilot_report.yaml", "E4 pilot report")
            metadata = _load_yaml(run_dir / "metadata.yaml", "E4 solver metadata")
            grid = metadata["grid"]
            boundary = int(grid["boundary_width"])
            resolution = tuple(int(n) - 2 * boundary for n in grid["resolution"])
            spacing = float(grid["spacing"])
            center = tuple(float(v) for v in grid["center"])
            key = (resolution, spacing, center)
            if (report.get("status") != "completed_solver_pilot" or report.get("case") != case
                    or report.get("initialization_stage_id") != metrics.get("initialization_stage_id")
                    or not np.isclose(report["final_time_seconds"], metrics["final_time_seconds"], atol=1e-9, rtol=0)
                    or int(report["final_step"]) != int(metrics[f"{case}_final_step"])):
                raise ExistingSandError(f"E4 pilot output disagrees with its review: {experiment_id}/{case}")
            if reference is None:
                reference = key
            elif key != reference:
                raise ExistingSandError("E4 pilot grids differ; a shared image extent would be misleading")
            field, path = _sand_field(run_dir, int(report["final_step"]), boundary, resolution)
            loaded[case] = field
            records.append({"experiment_id": experiment_id, "case": case,
                            "step": int(report["final_step"]),
                            "time_seconds": float(report["final_time_seconds"]),
                            "source": str(path), "source_sha256": sha256_file(path),
                            "gravity": metadata["physics"]["gravity"],
                            "dry_cohesion_c0": metadata["physics"]["cohesion"]["c0"]})
        cases[experiment_id] = loaded

    assert reference is not None
    resolution, spacing, center = reference
    extent = [center[0] - resolution[0] * spacing / 2,
              center[0] + resolution[0] * spacing / 2,
              center[1] - resolution[1] * spacing / 2,
              center[1] + resolution[1] * spacing / 2]
    vmax = max(float(np.max(field)) for pair in cases.values() for field in pair.values())
    fig, axes = plt.subplots(2, len(experiment_ids), figsize=(5 * len(experiment_ids), 9),
                             squeeze=False, constrained_layout=True)
    for column, experiment_id in enumerate(experiment_ids):
        for row, case in enumerate(("dry", "water")):
            ax = axes[row, column]
            image = ax.imshow(cases[experiment_id][case].T, origin="lower", extent=extent,
                              cmap="YlOrBr", vmin=0, vmax=vmax)
            ax.set_title(f"{experiment_id}\n{case} final sand thickness")
            ax.set_xlabel("simulation x")
            ax.set_ylabel("simulation y")
    fig.colorbar(image, ax=axes, shrink=0.75, label="sand thickness (simulation units)")

    output.mkdir(parents=True)
    fig.savefig(output / "final_sand_comparison.png", dpi=140)
    plt.close(fig)
    for experiment_id in experiment_ids:
        for case in ("dry", "water"):
            np.save(output / f"{experiment_id}_{case}_final_sand_xy.npy",
                    cases[experiment_id][case], allow_pickle=False)
    manifest = {"status": "E4 final sand thickness diagnostic", "axis_order": "[x,y]",
                "ghost_cells_removed": True, "grid_resolution_xy": list(resolution),
                "extent_xy": extent, "shared_color_scale": [0.0, vmax],
                "source_frames": records}
    with (output / "manifest.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)
    return {"output": str(output), "experiments": experiment_ids,
            "grid_resolution_xy": list(resolution), "shared_vmax": vmax}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--experiment-id", action="append", required=True)
    parser.add_argument("--output-id", required=True)
    args = parser.parse_args()
    try:
        result = export_final_sand(args.run, args.experiment_id, args.output_id)
    except (ExistingSandError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"existing-sand E4 final sand: {exc}\n")
    print(yaml.safe_dump(result, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
