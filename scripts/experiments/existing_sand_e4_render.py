#!/usr/bin/env python3
"""Render selected PGSR observation cameras over an existing E4 solver run."""
from __future__ import annotations

import argparse
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
from sandygs.pipeline import build_renderer_environment
from sandygs.render_schedule import expand_render_schedule
from sandygs.storage_guard import StorageGuardError, validate_artifact_storage


_TEXTURE_RENDER_KEYS = {
    "sand_texture_enabled", "sand_texture_diagnostics",
    "sand_texture_material", "sand_texture_material_mode",
    "sand_texture_root", "sand_texture_projection",
    "sand_texture_triplanar_sharpness", "sand_texture_phase_mode",
    "sand_texture_period_sim", "sand_texture_rotation_degrees",
    "sand_texture_offset", "sand_texture_primary_albedo_gain",
    "sand_texture_primary_roughness_scale", "sand_texture_normal_strength",
    "sand_texture_normal_highpass_enabled",
    "sand_texture_normal_highpass_sigma_texels",
    "sand_texture_normal_highpass_strength", "sand_texture_roughness_min",
    "sand_texture_roughness_max", "sand_texture_lod_bias",
    "sand_texture_wet_adjustment_enabled",
    "sand_texture_wet_albedo_scale", "sand_texture_wet_response_rate",
    "sand_texture_ao_enabled",
}
_TEXTURE_BOOLEAN_KEYS = {
    "sand_texture_enabled", "sand_texture_diagnostics",
    "sand_texture_normal_highpass_enabled",
    "sand_texture_wet_adjustment_enabled", "sand_texture_ao_enabled",
}
_SURFACE_RENDER_KEYS = {
    "sand_geometry_backend", "sand_triangle_diagonal_mode",
    "sand_triangle_boundary_mode", "sand_triangle_subdivision_level",
    "sand_triangle_coverage_gamma", "sand_triangle_terrain_max_edge_ratio",
    "sand_plane_depth_stability_fallback", "sand_plane_orientation_recovery",
    "sand_soft_boundary_guidance", "sand_boundary_fade_height",
    "sand_physical_reference_depth_recovery",
    "sand_physical_reference_normal_recovery",
    "sand_center_trusted_invalid_depth_recovery",
    "sand_center_trusted_invalid_normal_recovery",
    "water_depth_bias", "water_terrain_depth_proxy",
    "water_terrain_proxy_source_tolerance", "water_terrain_proxy_depth_bias",
    "water_physical_reference_depth_recovery",
    "water_physical_reference_erosion_pixels",
    "water_physical_reference_gaussian_coverage_min",
} | _TEXTURE_RENDER_KEYS
_SURFACE_BOOLEAN_KEYS = {
    "sand_plane_depth_stability_fallback", "sand_plane_orientation_recovery",
    "sand_soft_boundary_guidance", "sand_physical_reference_depth_recovery",
    "sand_physical_reference_normal_recovery",
    "sand_center_trusted_invalid_depth_recovery",
    "sand_center_trusted_invalid_normal_recovery",
    "water_terrain_depth_proxy",
    "water_physical_reference_depth_recovery",
} | _TEXTURE_BOOLEAN_KEYS
_SURFACE_DIAGNOSTIC_KEYS = [
    "sand_triangle_reference_coverage", "sand_guided_coverage",
    "sand_physical_reference_recovery_used",
    "sand_physical_reference_normal_recovery_used",
    "sand_center_trusted_invalid_recovery_used",
    "sand_center_trusted_invalid_normal_recovery_used",
    "sand_failure_classification", "sand_depth_competition_classification",
]
_WATER_PROXY_DIAGNOSTIC_KEYS = [
    "water_depth_in_front_without_proxy", "water_terrain_proxy_coverage",
    "water_terrain_proxy_used", "water_terrain_proxy_rejected_foreground",
    "water_occlusion_depth_raw", "water_terrain_proxy_depth_raw",
    "water_terrain_proxy_source_delta_raw", "water_occlusion_bias_raw",
]
_WATER_REFERENCE_DIAGNOSTIC_KEYS = [
    "water_depth_in_front_without_physical_reference",
    "water_physical_reference_coverage",
    "water_physical_reference_trusted",
    "water_physical_reference_recovered",
    "water_physical_reference_removed",
    "water_physical_reference_rejected_opaque",
    "water_physical_reference_depth_raw",
]
_TEXTURE_DIAGNOSTIC_KEYS = [
    "sand_texture_valid", "sand_texture_lod",
    "sand_texture_albedo_final", "sand_texture_roughness_final",
    "sand_texture_normal_world", "sand_texture_triplanar_weights",
    "sand_texture_triplanar_lod", "sand_texture_wet_response",
    "sand_texture_wet_brightness", "sand_texture_final",
]


def _validate_texture_overrides(overrides: dict) -> None:
    choices = {
        "sand_texture_material_mode": {"perturbation", "texture_primary"},
        "sand_texture_projection": {"simulation_xy", "simulation_triplanar"},
        "sand_texture_phase_mode": {"world_locked"},
    }
    for key, valid in choices.items():
        if key in overrides and overrides[key] not in valid:
            raise ExistingSandError(f"invalid {key}: {overrides[key]}")

    for key in (
        "sand_texture_triplanar_sharpness", "sand_texture_period_sim",
        "sand_texture_primary_roughness_scale",
        "sand_texture_normal_highpass_sigma_texels",
        "sand_texture_wet_response_rate",
    ):
        if key in overrides and (
            isinstance(overrides[key], bool)
            or not isinstance(overrides[key], (int, float))
            or not math.isfinite(overrides[key])
            or overrides[key] <= 0.0
        ):
            raise ExistingSandError(f"{key} must be finite and positive")

    for key in (
        "sand_texture_normal_strength",
        "sand_texture_normal_highpass_strength",
    ):
        if key in overrides and (
            isinstance(overrides[key], bool)
            or not isinstance(overrides[key], (int, float))
            or not math.isfinite(overrides[key])
            or overrides[key] < 0.0
        ):
            raise ExistingSandError(f"{key} must be finite and non-negative")

    for key in (
        "sand_texture_roughness_min", "sand_texture_roughness_max",
        "sand_texture_wet_albedo_scale",
    ):
        if key in overrides and (
            isinstance(overrides[key], bool)
            or not isinstance(overrides[key], (int, float))
            or not math.isfinite(overrides[key])
            or not 0.0 <= overrides[key] <= 1.0
        ):
            raise ExistingSandError(f"{key} must be finite and lie in [0,1]")

    for key in ("sand_texture_rotation_degrees", "sand_texture_lod_bias"):
        if key in overrides and (
            isinstance(overrides[key], bool)
            or not isinstance(overrides[key], (int, float))
            or not math.isfinite(overrides[key])
        ):
            raise ExistingSandError(f"{key} must be finite")

    for key, length, non_negative in (
        ("sand_texture_offset", 2, False),
        ("sand_texture_primary_albedo_gain", 3, True),
    ):
        value = overrides.get(key)
        if value is not None and (
            not isinstance(value, list)
            or len(value) != length
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                or (non_negative and item < 0.0)
                for item in value
            )
        ):
            qualifier = " finite non-negative" if non_negative else " finite"
            raise ExistingSandError(
                f"{key} must contain {length}{qualifier} values"
            )

    if (
        "sand_texture_roughness_min" in overrides
        and "sand_texture_roughness_max" in overrides
        and overrides["sand_texture_roughness_min"]
        > overrides["sand_texture_roughness_max"]
    ):
        raise ExistingSandError("sand texture roughness bounds must be ordered")

    if overrides.get("sand_texture_enabled", False):
        for key in ("sand_texture_material", "sand_texture_root"):
            if not isinstance(overrides.get(key), str) or not overrides[key]:
                raise ExistingSandError(
                    f"sand_texture_enabled requires a non-empty {key}"
                )
    if overrides.get("sand_texture_diagnostics", False) and not overrides.get(
        "sand_texture_enabled", False
    ):
        raise ExistingSandError(
            "sand_texture_diagnostics requires sand_texture_enabled"
        )
    if overrides.get(
        "sand_texture_wet_adjustment_enabled", False
    ) and overrides.get("sand_texture_material_mode") != "texture_primary":
        raise ExistingSandError(
            "sand_texture_wet_adjustment_enabled requires "
            "sand_texture_material_mode=texture_primary"
        )


def _surface_render_overrides(config: dict) -> dict:
    overrides = config.get("surface_render_overrides", {})
    if not isinstance(overrides, dict):
        raise ExistingSandError("surface_render_overrides must be a mapping")
    unknown = set(overrides) - _SURFACE_RENDER_KEYS
    if unknown:
        raise ExistingSandError(f"unsupported surface render override: {sorted(unknown)}")
    for key in _SURFACE_BOOLEAN_KEYS & set(overrides):
        if not isinstance(overrides[key], bool):
            raise ExistingSandError(f"{key} must be a boolean")
    choices = {
        "sand_geometry_backend": {"cell_tangent", "triangle_games"},
        "sand_triangle_diagonal_mode": {"checkerboard"},
        "sand_triangle_boundary_mode": {"clipped", "fully_active"},
    }
    for key, valid in choices.items():
        if key in overrides and overrides[key] not in valid:
            raise ExistingSandError(f"invalid {key}: {overrides[key]}")
    _validate_texture_overrides(overrides)
    if "sand_triangle_subdivision_level" in overrides and (
            isinstance(overrides["sand_triangle_subdivision_level"], bool)
            or overrides["sand_triangle_subdivision_level"] not in {0, 1}):
        raise ExistingSandError("sand_triangle_subdivision_level must be 0 or 1")
    for key in ("sand_triangle_coverage_gamma", "sand_triangle_terrain_max_edge_ratio",
                "sand_boundary_fade_height", "water_terrain_proxy_source_tolerance"):
        if key in overrides and (
                isinstance(overrides[key], bool)
                or not isinstance(overrides[key], (int, float))
                or not math.isfinite(overrides[key]) or overrides[key] <= 0):
            raise ExistingSandError(f"{key} must be finite and positive")
    if "water_depth_bias" in overrides and (
            isinstance(overrides["water_depth_bias"], bool)
            or not isinstance(overrides["water_depth_bias"], (int, float))
            or not math.isfinite(overrides["water_depth_bias"])
            or overrides["water_depth_bias"] < 0):
        raise ExistingSandError("water_depth_bias must be finite and non-negative")
    if "water_terrain_proxy_depth_bias" in overrides and (
            isinstance(overrides["water_terrain_proxy_depth_bias"], bool)
            or not isinstance(overrides["water_terrain_proxy_depth_bias"], (int, float))
            or not math.isfinite(overrides["water_terrain_proxy_depth_bias"])
            or overrides["water_terrain_proxy_depth_bias"] < 0):
        raise ExistingSandError(
            "water_terrain_proxy_depth_bias must be finite and non-negative"
        )
    if "water_physical_reference_erosion_pixels" in overrides and (
            isinstance(overrides["water_physical_reference_erosion_pixels"], bool)
            or not isinstance(
                overrides["water_physical_reference_erosion_pixels"], int
            )
            or overrides["water_physical_reference_erosion_pixels"] < 0):
        raise ExistingSandError(
            "water_physical_reference_erosion_pixels must be a "
            "non-negative integer"
        )
    if "water_physical_reference_gaussian_coverage_min" in overrides and (
            isinstance(
                overrides["water_physical_reference_gaussian_coverage_min"],
                bool,
            )
            or not isinstance(
                overrides["water_physical_reference_gaussian_coverage_min"],
                (int, float),
            )
            or not math.isfinite(
                overrides["water_physical_reference_gaussian_coverage_min"]
            )
            or not 0.0 <= overrides[
                "water_physical_reference_gaussian_coverage_min"
            ] <= 1.0):
        raise ExistingSandError(
            "water_physical_reference_gaussian_coverage_min must be in [0,1]"
        )
    if overrides.get(
        "water_physical_reference_depth_recovery", False
    ) and "water_physical_reference_erosion_pixels" not in overrides:
        raise ExistingSandError(
            "water physical reference recovery requires an explicit "
            "water_physical_reference_erosion_pixels"
        )
    if overrides.get(
        "water_physical_reference_depth_recovery", False
    ) and "water_physical_reference_gaussian_coverage_min" not in overrides:
        raise ExistingSandError(
            "water physical reference recovery requires an explicit "
            "water_physical_reference_gaussian_coverage_min"
        )
    if overrides.get("water_terrain_depth_proxy", False) and (
            "water_terrain_proxy_source_tolerance" not in overrides):
        raise ExistingSandError(
            "water_terrain_depth_proxy requires an explicit "
            "water_terrain_proxy_source_tolerance"
        )
    if overrides.get("water_terrain_depth_proxy", False) and (
            "water_terrain_proxy_depth_bias" not in overrides):
        raise ExistingSandError(
            "water_terrain_depth_proxy requires an explicit "
            "water_terrain_proxy_depth_bias"
        )
    if ("sand_triangle_terrain_max_edge_ratio" in overrides
            and overrides["sand_triangle_terrain_max_edge_ratio"] < 1.0):
        raise ExistingSandError("sand_triangle_terrain_max_edge_ratio must be at least one")
    if overrides.get("sand_geometry_backend", "cell_tangent") != "triangle_games" and (
            overrides.get("sand_triangle_subdivision_level", 0) != 0
            or any(overrides.get(key, False) for key in (
                "sand_soft_boundary_guidance",
                "sand_physical_reference_depth_recovery",
                "sand_center_trusted_invalid_depth_recovery"))):
        raise ExistingSandError("triangle subdivision and surface recovery require triangle_games")
    if any(overrides.get(key, False) for key in (
            "sand_physical_reference_depth_recovery",
            "sand_center_trusted_invalid_depth_recovery")) and not overrides.get(
                "sand_soft_boundary_guidance", False):
        raise ExistingSandError("physical triangle recovery requires sand_soft_boundary_guidance")
    for normal_key, depth_key in (
            ("sand_physical_reference_normal_recovery",
             "sand_physical_reference_depth_recovery"),
            ("sand_center_trusted_invalid_normal_recovery",
             "sand_center_trusted_invalid_depth_recovery")):
        if overrides.get(normal_key, False) and not overrides.get(depth_key, False):
            raise ExistingSandError(f"{normal_key} requires {depth_key}")
    return overrides


def _id(value: object, label: str) -> str:
    if (not isinstance(value, str) or not value or not value[0].islower()
            or any(not (character.islower() or character.isdigit() or character in "_-")
                   for character in value)):
        raise ExistingSandError(f"{label} must be a lowercase stage-style ID")
    return value


def _checked_stage_file(stage: Path, manifest: dict, relative: str) -> Path:
    matches = [item for item in manifest.get("artifacts", []) if item.get("path") == relative]
    if len(matches) != 1:
        raise ExistingSandError(f"sealed stage lacks artifact {relative}: {stage}")
    path = (stage / relative).resolve()
    if not path.is_relative_to(stage) or not path.is_file() or sha256_file(path) != matches[0].get("sha256"):
        raise ExistingSandError(f"sealed stage artifact changed: {path}")
    return path


def _repo_config(relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ExistingSandError("E4 render base config must be a repository-relative path")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ExistingSandError(f"E4 render base config unavailable: {relative}")
    return path


def build_plan(
    run_root: Path,
    config_path: Path,
    case: str,
    *,
    times_override: list[float] | None = None,
    render_schedule_override: dict | None = None,
) -> dict:
    run_root = Path(run_root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path, "E4 render config")
    run = _load_yaml(run_root / "run.yaml", "existing-sand run")
    if config.get("schema_version") != 1 or config.get("scene_id") != run.get("scene_id"):
        raise ExistingSandError("E4 render config scene does not match the run")
    if case not in {"dry", "water"}:
        raise ExistingSandError("E4 render case must be dry or water")
    render_id = _id(config.get("render_id"), "render_id")
    obs_id = _id(config.get("observation_stage_id"), "observation_stage_id")
    background_id = _id(config.get("background_stage_id"), "background_stage_id")
    initialization_id = _id(config.get("initialization_stage_id"), "initialization_stage_id")
    simulation_id = _id(config.get("simulation_id"), "simulation_id")
    camera_ids = config.get("camera_ids")
    if (not isinstance(camera_ids, list) or not camera_ids
            or len(camera_ids) != len(set(camera_ids))
            or any(not isinstance(value, str) or not value for value in camera_ids)):
        raise ExistingSandError("E4 render camera_ids must be distinct non-empty names")
    times = (
        config.get("times_seconds")
        if times_override is None
        else times_override
    )
    if (not isinstance(times, list) or not times
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in times)
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ExistingSandError("E4 render times_seconds must be strictly increasing non-negative times")

    obs = _stage_location(run_root, "observations", obs_id)
    obs_record = _load_yaml(obs / "record.yaml", "E4 observation record")
    obs_manifest = _load_yaml(obs / "manifest.yaml", "E4 observation manifest")
    if (obs_record.get("stage_id") != obs_id or obs_record.get("parameters", {}).get("dataset_name") != run["scene_id"]
            or any(cid not in obs_record.get("camera_ids", []) for cid in camera_ids)):
        raise ExistingSandError("E4 render cameras are not in the captured PGSR observation stage")
    camera_files = []
    camera_map = []
    for camera_id in camera_ids:
        candidates = []
        for relative in (item.get("path", "") for item in obs_manifest.get("artifacts", [])):
            if relative.endswith("/camera.yaml"):
                path = _checked_stage_file(obs, obs_manifest, relative)
                camera = _load_yaml(path, "E4 observation camera")
                if camera.get("camera_id") == camera_id:
                    candidates.append((path, camera))
        if len(candidates) != 1:
            raise ExistingSandError(f"camera {camera_id} has {len(candidates)} captured views")
        camera_path, camera = candidates[0]
        if not np.allclose(camera["sim_to_gs"], run["coordinate"]["sim_to_gs"], rtol=0, atol=1e-6):
            raise ExistingSandError(f"camera simulation transform differs from run: {camera_id}")
        camera_files.append(str(camera_path))
        camera_map.append({"camera_index": len(camera_map), "camera_id": camera_id,
                           "camera_file": str(camera_path), "camera_sha256": sha256_file(camera_path)})

    background = _stage_location(run_root, "background", background_id)
    background_manifest = _load_yaml(background / "manifest.yaml", "E4 background manifest")
    background_record = _load_yaml(background / "record.yaml", "E4 background record")
    if (background_record.get("stage_id") != background_id
            or background_record.get("prompt", {}).get("type") != "pgsr_local_background_optimization"):
        raise ExistingSandError("E4 render needs the reviewed E3 background fit")
    cloud = _checked_stage_file(background, background_manifest,
                                "checkpoint/point_cloud/iteration_30000/point_cloud.ply")
    checkpoint = background / "checkpoint"
    initial = _stage_location(run_root, "modeling", initialization_id)
    initial_manifest = _load_yaml(initial / "manifest.yaml", "E4 initialization manifest")
    if initial_manifest.get("stage_id") != initialization_id:
        raise ExistingSandError("E4 initialization manifest ID differs")
    exchange = initial / "exchange"
    exchange_metadata = validate_bundle(exchange)
    if not np.allclose(exchange_metadata["coordinates"]["sim_to_gs"],
                       run["coordinate"]["sim_to_gs"], rtol=0, atol=1e-6):
        raise ExistingSandError("E4 exchange transform differs from the run")
    simulation = run_root / "simulations" / simulation_id
    simulation_case = simulation / case
    report = _load_yaml(simulation_case / "pilot_report.yaml", "E4 solver report")
    if (report.get("status") != "completed_solver_pilot" or report.get("case") != case
            or report.get("initialization_stage_id") != initialization_id
            or report.get("initialization_manifest_sha256") != sha256_file(initial / "manifest.yaml")
            or max(times) > float(report["final_time_seconds"]) + 1e-9):
        raise ExistingSandError("E4 solver run does not match the selected initialization or time window")

    scene_config = _repo_config(config.get("base_scene_config"))
    profile_config = _repo_config(config.get("base_render_profile"))
    scene = _load_yaml(scene_config, "P4 base scene")
    profile = _load_yaml(profile_config, "P4 base profile")
    renderer = dict(scene["renderer"])
    renderer.update(profile["render"])
    grid = exchange_metadata["grid"]
    default_schedule = {
        "schema_version": 1, "camera_path": "observed_cameras",
        "camera_path_frames": len(camera_files),
        "sequences": [{
            "name": "selected_views", "type": "synchronized_multiview",
            "camera_indices": list(range(len(camera_files))),
            "times": [float(value) for value in times],
            "debug": "all", "intermediates": "all",
        }],
    }
    render_schedule = (
        default_schedule
        if render_schedule_override is None
        else render_schedule_override
    )
    renderer.update({
        "dataset_name": run["scene_id"],
        "height_map_range": [grid["domain_min"][0], grid["domain_max"][0],
                             grid["domain_min"][1], grid["domain_max"][1]],
        "align_rotation_angles": [0.0, 0.0, 0.0],
        "align_translation_vector": [0.0, 0.0, 0.0],
        "align_scale": [1.0, 1.0, 1.0],
        "is_render_360": True, "is_sky_model": False,
        "is_render_height_map": False, "is_render_height_map_post": False,
        "simulation_backend": "sand_water",
        "simulation_exchange_path": str(exchange),
        "sand_water_run_path": str(simulation_case),
        "render_sand": True,
        "observed_camera_files": camera_files,
        "sand_debug_output_keys": [
            "sand_only", "sand_alpha", "sand_coverage", "sand_depth", "sand_owner",
            "sand_occlusion", "water_coverage", "water_depth_in_front", "water_normal",
        ],
        "render_schedule": render_schedule,
    })
    surface_overrides = _surface_render_overrides(config)
    renderer.update(surface_overrides)
    if surface_overrides:
        renderer["sand_debug_output_keys"] += _SURFACE_DIAGNOSTIC_KEYS
    if surface_overrides.get("water_terrain_depth_proxy", False):
        renderer["sand_debug_output_keys"] += _WATER_PROXY_DIAGNOSTIC_KEYS
    if surface_overrides.get(
        "water_physical_reference_depth_recovery", False
    ):
        renderer["sand_debug_output_keys"] += _WATER_REFERENCE_DIAGNOSTIC_KEYS
    if surface_overrides.get("sand_texture_diagnostics", False):
        renderer["sand_debug_output_keys"] += _TEXTURE_DIAGNOSTIC_KEYS
    renderer["sand_debug_output_keys"] = list(dict.fromkeys(
        renderer["sand_debug_output_keys"]
    ))
    schedule, schedule_camera_count = expand_render_schedule(
        renderer["render_schedule"]
    )
    if (
        renderer["render_schedule"].get("camera_path") != "observed_cameras"
        or schedule_camera_count != len(camera_files)
    ):
        raise ExistingSandError(
            "E4 observed-camera schedule must use every configured camera file"
        )
    output = run_root / "visualizations" / render_id / case
    if output.exists():
        raise ExistingSandError(f"refusing to overwrite E4 render: {output}")
    dataset = Path(run["source"]["dataset"]).resolve()
    if not dataset.is_dir():
        raise ExistingSandError(f"source dataset unavailable: {dataset}")
    for item in camera_map:
        camera = _load_yaml(Path(item["camera_file"]), "E4 observation camera")
        if not (dataset / obs_record["parameters"]["images_dir"] / camera["source_image"]).is_file():
            raise ExistingSandError(f"source image unavailable for {item['camera_id']}")
    command = [sys.executable, str(ROOT / "render_sand_water.py"),
               "-s", str(dataset), "-m", str(checkpoint),
               "--iteration", "30000", "--config_file", str(output / "render.yaml"),
               "--render_output_path", str(output), "--disable_preload_img"]
    return {"output": output, "renderer": renderer, "command": command,
            "source_cloud": str(cloud), "source_cloud_sha256": sha256_file(cloud),
            "exchange": str(exchange), "simulation_case": str(simulation_case),
            "camera_map": camera_map, "frame_count": len(schedule),
            "times_seconds": [float(value) for value in times],
            "config_sha256": sha256_file(config_path),
            "base_scene_sha256": sha256_file(scene_config),
            "base_profile_sha256": sha256_file(profile_config),
            "case": case, "render_id": render_id}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/existing_sand/indoor02_e4_render_v2.yaml")
    parser.add_argument("--case", choices=("dry", "water"), required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        validate_artifact_storage(ROOT / "configs/paths.local.yaml")
        plan = build_plan(args.run, args.config, args.case)
        if args.check_only:
            print(json.dumps({key: plan[key] for key in
                              ("output", "command", "source_cloud_sha256", "frame_count",
                               "times_seconds", "camera_map")}, default=str, indent=2))
            return 0
        output = plan["output"]
        output.mkdir(parents=True, exist_ok=False)
        with (output / "render.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(plan["renderer"], stream, sort_keys=False)
        audit = {key: value for key, value in plan.items() if key != "renderer"}
        audit["output"] = str(output)
        with (output / "plan.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(audit, stream, sort_keys=False)
        print("+", " ".join(plan["command"]), flush=True)
        subprocess.run(plan["command"], cwd=ROOT,
                       env=build_renderer_environment(), check=True)
        rendered = list((output / "release_check" / "selected_views").glob("time_*/renders/*.png"))
        if len(rendered) != plan["frame_count"]:
            raise ExistingSandError(f"E4 render saved {len(rendered)} RGB frames, expected {plan['frame_count']}")
        with (output / "result.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump({"status": "completed", "case": args.case,
                            "frame_count": len(rendered), "times_seconds": plan["times_seconds"],
                            "camera_map": plan["camera_map"]}, stream, sort_keys=False)
        print(f"E4 {args.case} rendered {len(rendered)} frames: {output}")
    except (ExistingSandError, StorageGuardError, OSError, ValueError, KeyError,
            subprocess.CalledProcessError) as exc:
        parser.exit(2, f"existing-sand E4 render: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
