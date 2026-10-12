#!/usr/bin/env python3
"""Attribute E4 sand occlusion to disjoint E3 source-Gaussian groups.

This diagnostic never edits the accepted checkpoint. It writes temporary
counterfactual checkpoint copies and renders the same dry t=0 frames for each.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.ndimage import binary_erosion
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sandygs.exchange import sha256_file
from sandygs.pipeline import build_renderer_environment
from sandygs.storage_guard import validate_artifact_storage


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _save_yaml(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)


def _counterfactual_cloud(original: np.ndarray, keep: np.ndarray, target: Path) -> None:
    target.parent.mkdir(parents=True)
    PlyData([PlyElement.describe(original[keep], "vertex")], text=False,
            byte_order="<").write(str(target))


def _png_mask(path: Path, value: np.ndarray) -> None:
    Image.fromarray(value.astype(np.uint8) * 255).save(path)


def _maps(folder: Path, camera_index: int) -> dict[str, np.ndarray]:
    base = folder / "release_check/selected_views/time_0000/sand_debug"
    stem = f"camera_{camera_index:03d}"
    with Image.open(base / "sand_coverage" / f"{stem}.png") as image:
        coverage = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    with Image.open(base / "sand_owner" / f"{stem}.png") as image:
        owner = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
    delta = np.load(base / "sand_resolved_depth_delta_raw" / f"{stem}.npy",
                    allow_pickle=False)
    sand_depth = np.load(base / "sand_plane_depth_raw" / f"{stem}.npy",
                         allow_pickle=False)
    if delta.shape != coverage.shape or sand_depth.shape != coverage.shape:
        raise ValueError(f"diagnostic map shape mismatch: {folder}, {stem}")
    return {"coverage": coverage, "owner": owner, "delta": delta,
            "sand_depth": sand_depth}


def _summarize(stage: Path, camera_ids: list[str], depth_bias: float) -> dict:
    cases = ["full", "without_retained", "without_55", "without_139"]
    report = {"definition": "delta = source camera-Z - sand camera-Z + depth_bias; "
                            "negative delta means source lies in front where source alpha is valid",
              "depth_bias": depth_bias,
              "cameras": []}
    for camera_index, camera_id in enumerate(camera_ids):
        maps = {case: _maps(stage / "cases" / case, camera_index)
                for case in cases}
        full = maps["full"]
        # The erosion separates interior holes from the intended 5-pixel taper.
        interior = binary_erosion(full["coverage"] >= .95, iterations=5)
        hidden = interior & ~full["owner"] & np.isfinite(full["delta"]) & (full["delta"] < 0)
        if not interior.any():
            raise ValueError(f"empty interior sand coverage for {camera_id}")
        camera_report = {"camera_id": camera_id,
                         "interior_pixels": int(interior.sum()),
                         "baseline_hidden_pixels": int(hidden.sum()),
                         "cases": {}}
        camera_dir = stage / "review" / camera_id
        camera_dir.mkdir(parents=True)
        _png_mask(camera_dir / "baseline_hidden.png", hidden)
        original = Image.open(
            stage / "cases/full/release_check/selected_views/time_0000/renders"
            / f"camera_{camera_index:03d}.png").convert("RGB")
        no55 = Image.open(
            stage / "cases/without_55/release_check/selected_views/time_0000/renders"
            / f"camera_{camera_index:03d}.png").convert("RGB")
        overview = np.asarray(original).copy()
        overview[hidden] = (
            .25 * overview[hidden] + .75 * np.array([255, 30, 30])
        ).astype(np.uint8)
        Image.fromarray(np.concatenate((np.asarray(original), np.asarray(no55),
                                        overview), axis=1)).save(camera_dir / "overview.png")
        for case, data in maps.items():
            exposed = hidden & data["owner"]
            still_behind = hidden & ~data["owner"]
            finite = hidden & np.isfinite(data["delta"])
            delta = data["delta"][finite]
            case_report = {
                "hidden_pixels_now_sand_owned": int(exposed.sum()),
                "hidden_pixels_still_source_owned": int(still_behind.sum()),
                "hidden_pixels_delta_negative": int((delta < 0).sum()),
                "hidden_delta_p50": float(np.median(delta)) if delta.size else None,
                "hidden_delta_p05_p95": [float(v) for v in np.quantile(delta, [.05, .95])]
                if delta.size else None,
                "interior_sand_owned_pixels": int((interior & data["owner"]).sum()),
                "interior_source_front_pixels": int((interior & np.isfinite(data["delta"])
                                                      & (data["delta"] < 0)).sum()),
            }
            camera_report["cases"][case] = case_report
            _png_mask(camera_dir / f"{case}_newly_exposed.png", exposed)
            _png_mask(camera_dir / f"{case}_sand_owner.png", data["owner"] & interior)
            # A delta map is more precise than an RGB overlay; its value is
            # meaningful only where the source render has valid opacity.
            np.save(camera_dir / f"{case}_delta.npy", data["delta"],
                    allow_pickle=False)
        report["cameras"].append(camera_report)
    _save_yaml(stage / "metrics.yaml", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--diagnostic-id", default="e4_source_ablation_001")
    args = parser.parse_args()
    run = args.run.resolve()
    stage = run / "diagnostics" / args.diagnostic_id
    if stage.exists():
        parser.error(f"refusing to overwrite: {stage}")
    validate_artifact_storage(ROOT / "configs/paths.local.yaml")
    fit = run / "stages/background/e3_hole_joint450_depth010_fit450_001"
    init55 = run / "stages/background/e3_init_frame00055_p050_s23_cleanup020_001"
    init139 = run / "stages/background/e3_hole_joint450_depth010_init_001"
    baseline = run / "visualizations/e4_render_v2_surface_recovery_001/dry/render.yaml"
    fit_record = _read_yaml(fit / "record.yaml")
    q55 = _read_yaml(init55 / "quality.yaml")
    q139 = _read_yaml(init139 / "quality.yaml")
    retained = int(fit_record["parameters"]["new_start_index"])
    first_new = int(q55["new_gaussians"])
    later_new = int(q139["appended_gaussians"])
    if (retained, first_new, later_new) != (279190, 2409, 400):
        raise ValueError("E3 source group counts changed; review this diagnostic")
    cloud = fit / "checkpoint/point_cloud/iteration_30000/point_cloud.ply"
    ply = PlyData.read(str(cloud), mmap="r")
    if ply.text or ply.byte_order != "<" or len(ply.elements) != 1:
        raise ValueError("expected binary little-endian E3 vertex PLY")
    vertices = ply["vertex"].data
    if len(vertices) != retained + first_new + later_new:
        raise ValueError("E3 group counts do not sum to checkpoint PLY size")
    renderer = _read_yaml(baseline)
    if renderer.get("dataset_name") != "indoor02":
        raise ValueError("unexpected E4 baseline renderer scene")
    schedule = renderer["render_schedule"]
    if schedule["camera_path_frames"] != 3:
        raise ValueError("baseline camera count changed")
    schedule["sequences"][0]["times"] = [0.0]
    renderer["sand_debug_output_keys"] = [
        "sand_coverage", "sand_owner", "sand_occlusion", "sand_only",
        "sand_resolved_depth_delta_raw", "sand_plane_depth_raw",
    ]
    camera_ids = ["frame_00055.jpg", "frame_00139.jpg", "frame_00126.jpg"]
    for expected, camera_file in zip(camera_ids, renderer["observed_camera_files"]):
        if _read_yaml(Path(camera_file))["camera_id"] != expected:
            raise ValueError("baseline camera order changed")
    stage.mkdir(parents=True, exist_ok=False)
    manifest = {
        "diagnostic_id": args.diagnostic_id,
        "source_checkpoint": str(cloud),
        "source_checkpoint_sha256": sha256_file(cloud),
        "baseline_renderer": str(baseline),
        "baseline_renderer_sha256": sha256_file(baseline),
        "group_counts": {"retained": retained, "added_55": first_new,
                         "added_139": later_new},
        "camera_ids": camera_ids, "physical_time_seconds": 0.0,
        "source": "dry solver snapshot and triangle recovery renderer from completed E4 run",
    }
    _save_yaml(stage / "manifest.yaml", manifest)
    n = len(vertices)
    groups = {
        "full": np.ones(n, dtype=bool),
        "without_retained": np.arange(n) >= retained,
        "without_55": (np.arange(n) < retained) | (np.arange(n) >= retained + first_new),
        "without_139": np.arange(n) < retained + first_new,
    }
    dataset = Path(_read_yaml(run / "run.yaml")["source"]["dataset"])
    for case, keep in groups.items():
        output = stage / "cases" / case
        checkpoint = output / "checkpoint"
        checkpoint.mkdir(parents=True)
        original_checkpoint = fit / "checkpoint"
        for item in original_checkpoint.iterdir():
            if item.name != "point_cloud":
                (checkpoint / item.name).symlink_to(item)
        _counterfactual_cloud(vertices, keep,
                              checkpoint / "point_cloud/iteration_30000/point_cloud.ply")
        _save_yaml(output / "render.yaml", renderer)
        command = [sys.executable, str(ROOT / "render_sand_water.py"),
                   "-s", str(dataset), "-m", str(checkpoint),
                   "--iteration", "30000", "--config_file", str(output / "render.yaml"),
                   "--render_output_path", str(output), "--disable_preload_img"]
        with (output / "render.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT,
                                    env=build_renderer_environment(),
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"{case} render failed ({result.returncode}); see {output / 'render.log'}")
        print(f"{case}: {int(keep.sum())} source Gaussians, three frames rendered", flush=True)
    report = _summarize(stage, camera_ids, float(renderer.get("sand_depth_bias", .002)))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Review: {stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
