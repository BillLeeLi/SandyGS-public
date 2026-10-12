#!/usr/bin/env python3
"""Locate the subset of camera-55 E3 points responsible for E4 occlusion."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sandygs.exchange import sha256_file
from sandygs.pipeline import build_renderer_environment
from sandygs.storage_guard import validate_artifact_storage
from existing_sand_e4_source_ablation import (
    _counterfactual_cloud, _maps, _read_yaml, _save_yaml,
)


def _projection_candidates(run: Path, parent: Path, vertices: np.ndarray,
                           renderer: dict) -> tuple[dict[str, np.ndarray], dict]:
    first, count = 279190, 2409
    group = vertices[first:first + count]
    xyz = np.column_stack([group[key] for key in ("x", "y", "z")]).astype(np.float64)
    scale = np.exp(np.column_stack([group[f"scale_{axis}"] for axis in range(3)]))
    center_front = np.zeros(count, dtype=bool)
    footprint = np.zeros(count, dtype=bool)
    audit = {"projection_radius": "3 * maximum Gaussian scale * maximum focal length / center camera-Z; approximate conservative screen-space radius",
             "views": []}
    for camera_index in (1, 2):
        camera = _read_yaml(Path(renderer["observed_camera_files"][camera_index]))
        K = np.asarray(camera["pgsr_K"], dtype=np.float64)
        w2c = np.asarray(camera["gs_w2c"], dtype=np.float64)
        cam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
            v = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
        hidden_path = parent / "review" / camera["camera_id"] / "baseline_hidden.png"
        with Image.open(hidden_path) as image:
            hidden = np.asarray(image.convert("L")) >= 128
        projected = np.isfinite(u) & np.isfinite(v) & (cam[:, 2] > 0)
        ui = np.zeros(count, dtype=np.int32)
        vi = np.zeros(count, dtype=np.int32)
        ui[projected] = np.rint(u[projected]).astype(np.int32)
        vi[projected] = np.rint(v[projected]).astype(np.int32)
        projected &= ((ui >= 0) & (ui < hidden.shape[1])
                      & (vi >= 0) & (vi < hidden.shape[0]))
        distance = np.full(count, np.inf, dtype=np.float64)
        distance[projected] = distance_transform_edt(~hidden)[vi[projected], ui[projected]]
        radius = 3.0 * scale.max(axis=1) * max(K[0, 0], K[1, 1]) / np.maximum(cam[:, 2], 1e-6)
        nearby = projected & (distance <= radius)
        footprint |= nearby
        sand_depth = _maps(parent / "cases/full", camera_index)["sand_depth"]
        direct = projected & (distance == 0)
        center_depth_difference = np.full(count, np.nan)
        center_depth_difference[direct] = cam[direct, 2] - sand_depth[vi[direct], ui[direct]]
        center_front |= direct & np.isfinite(center_depth_difference) & (center_depth_difference < 0)
        audit["views"].append({"camera_id": camera["camera_id"],
                               "projected_center_in_hole": int(direct.sum()),
                               "estimated_footprint_overlaps_hole": int(nearby.sum()),
                               "center_in_hole_in_front_of_sand": int((direct & (center_depth_difference < 0)).sum())})
    exchange = run / "stages/modeling/e4_exchange_initial_v2_001/exchange"
    metadata = _read_yaml(exchange / "metadata.yaml")
    if not np.allclose(metadata["coordinates"]["sim_to_gs"], np.eye(4), rtol=0, atol=1e-9):
        raise ValueError("E4 source-subset height classification requires identity sim_to_gs")
    grid = metadata["grid"]
    terrain = np.load(exchange / "frames/000000/terrain.npy", allow_pickle=False)
    sand = np.load(exchange / "frames/000000/sand_height.npy", allow_pickle=False)
    x_index = (xyz[:, 0] - grid["cell_center_origin"][0]) / grid["spacing"][0]
    y_index = (xyz[:, 1] - grid["cell_center_origin"][1]) / grid["spacing"][1]
    if not ((x_index >= 0).all() and (x_index <= terrain.shape[0] - 1).all()
            and (y_index >= 0).all() and (y_index <= terrain.shape[1] - 1).all()):
        raise ValueError("camera-55 points leave E4 exchange domain; inspect before classifying")
    top_height = map_coordinates(terrain + sand, [x_index, y_index],
                                 order=1, mode="nearest")
    above_top = xyz[:, 2] > top_height
    groups = {"center_front": center_front,
              "footprint_above_top": footprint & above_top,
              "footprint_below_top": footprint & ~above_top,
              "footprint_all": footprint}
    audit["groups"] = {name: int(mask.sum()) for name, mask in groups.items()}
    audit["height_definition"] = "55 Gaussian center world +z vs bilinear (terrain + initial sand thickness) on E4 cell centers; sim_to_gs is identity"
    return groups, audit


def _geometry_audit(run: Path, final_indices: np.ndarray) -> dict:
    exchange = run / "stages/modeling/e4_exchange_initial_v2_001/exchange"
    grid = _read_yaml(exchange / "metadata.yaml")["grid"]
    terrain = np.load(exchange / "frames/000000/terrain.npy", allow_pickle=False)
    sand = np.load(exchange / "frames/000000/sand_height.npy", allow_pickle=False)
    background = run / "stages/background"
    paths = {
        "e3_first_initialization": background / "e3_init_frame00055_p050_s23_cleanup020_001/point_cloud.ply",
        "after_55_fit": background / "e3_fit_frame00055_p050_s23_100p50_001/checkpoint/point_cloud/iteration_30000/point_cloud.ply",
        "after_three_view_joint": background / "e3_mv_001_noadd_55_97_139_joint450_001/checkpoint/point_cloud/iteration_30000/point_cloud.ply",
        "final_background": background / "e3_hole_joint450_depth010_fit450_001/checkpoint/point_cloud/iteration_30000/point_cloud.ply",
    }
    report = {"candidate_count": int(len(final_indices)),
              "candidate_definition": "55-group centers with approximate 3 sigma projected footprint overlapping 139 or 126 interior hole and final center +z above E4 initial terrain+sand top",
              "height_units": "simulation unit; PGSR-to-simulation identity, physical metric scale unverified",
              "history": {}}
    def quantile(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.quantile(values, [.05, .5, .95])]
    for name, path in paths.items():
        vertices = PlyData.read(str(path), mmap="r")["vertex"].data[final_indices]
        xyz = np.column_stack([vertices[key] for key in ("x", "y", "z")]).astype(np.float64)
        xi = (xyz[:, 0] - grid["cell_center_origin"][0]) / grid["spacing"][0]
        yi = (xyz[:, 1] - grid["cell_center_origin"][1]) / grid["spacing"][1]
        ground = map_coordinates(terrain, [xi, yi], order=1, mode="nearest")
        thickness = map_coordinates(sand, [xi, yi], order=1, mode="nearest")
        gap = xyz[:, 2] - ground - thickness
        row = {"center_above_e4_sand_top_count": int((gap > 0).sum()),
               "center_minus_e4_sand_top_p05_p50_p95": quantile(gap),
               "center_minus_terrain_p05_p50_p95": quantile(xyz[:, 2] - ground)}
        if name == "final_background":
            scales = np.exp(np.column_stack([vertices[f"scale_{i}"] for i in range(3)]).astype(np.float64))
            opacity = 1 / (1 + np.exp(-vertices["opacity"].astype(np.float64)))
            quaternion = np.column_stack([vertices[f"rot_{i}"] for i in range(4)]).astype(np.float64)
            quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
            rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()
            normal = rotation[np.arange(len(vertices)), :, scales.argmin(axis=1)]
            tilt = np.degrees(np.arccos(np.clip(np.abs(normal[:, 2]), 0, 1)))
            row.update({"e4_initial_sand_thickness_p05_p50_p95": quantile(thickness),
                        "center_above_top_by_gt_0p005_count": int((gap > .005).sum()),
                        "center_above_top_by_gt_0p01_count": int((gap > .01).sum()),
                        "center_above_top_by_gt_0p02_count": int((gap > .02).sum()),
                        "opacity_p05_p50_p95": quantile(opacity),
                        "world_normal_tilt_from_vertical_degrees_p05_p50_p95": quantile(tilt),
                        "maximum_gaussian_scale_p05_p50_p95": quantile(scales.max(axis=1))})
        report["history"][name] = row
    return report


def _score(stage: Path, parent: Path, groups: dict[str, np.ndarray]) -> dict:
    from scipy.ndimage import binary_erosion

    report = {"definition": "baseline hidden pixels are fixed from full E4 render; owner is compared after subset removal",
              "cameras": []}
    for camera_index, camera_id in enumerate(("frame_00055.jpg", "frame_00139.jpg", "frame_00126.jpg")):
        full = _maps(parent / "cases/full", camera_index)
        interior = binary_erosion(full["coverage"] >= .95, iterations=5)
        hidden = interior & ~full["owner"] & np.isfinite(full["delta"]) & (full["delta"] < 0)
        view = {"camera_id": camera_id, "baseline_hidden_pixels": int(hidden.sum()),
                "cases": {}}
        for case in groups:
            data = _maps(stage / "cases" / case, camera_index)
            if (not np.array_equal(full["sand_depth"], data["sand_depth"], equal_nan=True)
                    or not np.array_equal(full["coverage"], data["coverage"])):
                raise ValueError(f"sand surface changed for {camera_id}, {case}")
            exposed = hidden & data["owner"]
            view["cases"][case] = {"removed_55_points": int(groups[case].sum()),
                                   "hidden_pixels_newly_sand_owned": int(exposed.sum()),
                                   "hidden_pixels_still_source_owned": int((hidden & ~data["owner"]).sum())}
            output = stage / "review" / camera_id
            output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(exposed.astype(np.uint8) * 255).save(output / f"{case}_newly_exposed.png")
        report["cameras"].append(view)
    _save_yaml(stage / "metrics.yaml", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--diagnostic-id", default="e4_source_subset_001")
    parser.add_argument("--parent-id", default="e4_source_ablation_001")
    args = parser.parse_args()
    run = args.run.resolve()
    stage = run / "diagnostics" / args.diagnostic_id
    parent = run / "diagnostics" / args.parent_id
    if stage.exists():
        parser.error(f"refusing to overwrite: {stage}")
    validate_artifact_storage(ROOT / "configs/paths.local.yaml")
    checkpoint = run / "stages/background/e3_hole_joint450_depth010_fit450_001/checkpoint"
    cloud = checkpoint / "point_cloud/iteration_30000/point_cloud.ply"
    parent_manifest = _read_yaml(parent / "manifest.yaml")
    if (parent_manifest["source_checkpoint"] != str(cloud)
            or parent_manifest["source_checkpoint_sha256"] != sha256_file(cloud)):
        raise ValueError("source PLY differs from parent diagnostic")
    renderer = _read_yaml(parent / "cases/full/render.yaml")
    vertices = PlyData.read(str(cloud), mmap="r")["vertex"].data
    if len(vertices) != 281999:
        raise ValueError("final E3 PLY point count changed")
    groups, audit = _projection_candidates(run, parent, vertices, renderer)
    stage.mkdir(parents=True, exist_ok=False)
    _save_yaml(stage / "manifest.yaml", {"parent": str(parent),
                                        "source_cloud_sha256": sha256_file(cloud),
                                        "candidate_definition": audit})
    _save_yaml(stage / "geometry_metrics.yaml", _geometry_audit(
        run, np.flatnonzero(groups["footprint_above_top"]) + 279190))
    for name, mask in groups.items():
        np.save(stage / f"{name}_final_indices.npy", np.flatnonzero(mask) + 279190,
                allow_pickle=False)
        output = stage / "cases" / name
        model = output / "checkpoint"
        model.mkdir(parents=True)
        for item in checkpoint.iterdir():
            if item.name != "point_cloud":
                (model / item.name).symlink_to(item)
        keep = np.ones(len(vertices), dtype=bool)
        keep[279190:281599] &= ~mask
        _counterfactual_cloud(vertices, keep,
                              model / "point_cloud/iteration_30000/point_cloud.ply")
        _save_yaml(output / "render.yaml", renderer)
        command = [sys.executable, str(ROOT / "render_sand_water.py"),
                   "-s", _read_yaml(run / "run.yaml")["source"]["dataset"],
                   "-m", str(model), "--iteration", "30000",
                   "--config_file", str(output / "render.yaml"),
                   "--render_output_path", str(output), "--disable_preload_img"]
        with (output / "render.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=build_renderer_environment(),
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"{name} render failed ({result.returncode}): {output / 'render.log'}")
        print(f"{name}: removed {int(mask.sum())} camera-55 points", flush=True)
    report = _score(stage, parent, groups)
    print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
    print(f"Review: {stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
