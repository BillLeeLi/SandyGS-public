"""Versioned scene configuration and operation-scoped resource validation.

Scene schema v2 replaces the misleading ``rainygs_arguments`` boundary with a
renderer mapping. Production loading accepts only v2; the offline migration
tool alone calls the v1 normalization path. Business modules consume only the
normalized v2 representation.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a local paths file or scene configuration is invalid."""


VARIABLE_PATTERN = re.compile(
    r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))"
)

REMOVED_SCENE_PATHS = {"fluid_frames"}
REMOVED_RENDERER_FIELDS = {
    "fluid_root",
    "src_height_map_path",
    "src_height_map_high_path",
    "time_duration_raindrop",
    "interval_dis_raindrop",
    "diffuse_raindrop",
    "opacity_raindrop",
    "scale_raindrop",
    "roughness_raindrop",
    "specular_raindrop",
    "diffuse_moisture",
    "opacity_moisture",
    "scale_moisture",
    "roughness_moisture",
    "specular_moisture",
    "diffuse_rainsplash",
    "opacity_rainsplash",
    "scale_rainsplash",
    "roughness_rainsplash",
    "specular_rainsplash",
}

REQUIRED_RENDERER_FIELDS = {
    "dataset_name",
    "height_map_range",
    "align_rotation_angles",
    "align_translation_vector",
    "align_scale",
    "is_render_360",
    "is_sky_model",
    "is_render_height_map",
    "is_render_height_map_post",
    "ortho_cam_z",
    "low_height_map_z",
    "high_height_map_z",
    "height_map_width",
    "height_map_height",
    "brdf_envmap_res",
    "diffuse_fluid",
    "opacity_fluid",
    "scale_fluid",
    "roughness_fluid",
    "specular_fluid",
}

OPERATION_REQUIRED_PATHS = {
    "resolve": set(),
    "terrain_exchange": {"low_height_map"},
    "simulation": {"p1_exchange"},
    "reconstruction": {"dataset", "checkpoint"},
    "sand_water_render": {"dataset", "checkpoint", "p1_exchange"},
}


def load_yaml_mapping(path: Path, *, description: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{description} must contain a YAML mapping: {path}")
    return value


def resolve_water_rendering_backend(
    renderer: dict[str, Any], *, source: str
) -> str:
    """Resolve the only water backend supported by the current mainline.

    Missing selectors now mean ``sandygs_p4``.  An explicit legacy selector is
    rejected instead of being silently reinterpreted, so historical results
    remain tied to the pre-refactor tag.
    """

    backend = renderer.get("water_rendering_backend", "sandygs_p4")
    if backend == "rainygs_legacy":
        raise ConfigError(
            f"{source} selects removed water backend rainygs_legacy; use the "
            "pre-refactor-reconstruction-v1 tag for historical rendering or "
            "migrate explicitly to sandygs_p4"
        )
    if backend != "sandygs_p4":
        raise ConfigError(
            f"{source} has unsupported water_rendering_backend={backend!r}; "
            "expected sandygs_p4"
        )
    return backend


def validate_current_runtime_config(
    value: dict[str, Any], *, source: str
) -> tuple[str, str]:
    """Reject runtime settings that only the pre-refactor code can execute."""

    if not isinstance(value, dict):
        raise ConfigError(f"{source} must contain a configuration mapping")
    removed = sorted(REMOVED_RENDERER_FIELDS & value.keys())
    if removed:
        raise ConfigError(
            f"{source} contains removed legacy water field(s): "
            + ", ".join(removed)
            + "; use scripts/migrate_scene_config.py for scene manifests or "
            "the pre-refactor-reconstruction-v1 tag for historical rendering"
        )
    simulation_backend = value.get("simulation_backend", "sand_water")
    if simulation_backend == "simrain":
        raise ConfigError(
            f"{source} selects removed simulation backend simrain; use the "
            "pre-refactor-reconstruction-v1 tag for historical rendering"
        )
    if simulation_backend != "sand_water":
        raise ConfigError(
            f"{source} has unsupported simulation_backend="
            f"{simulation_backend!r}; expected sand_water"
        )
    water_backend = resolve_water_rendering_backend(value, source=source)
    return simulation_backend, water_backend


def normalize_scene_config(
    value: dict[str, Any], *, source: str = "scene configuration"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a normalized schema-v2 scene and an auditable migration record."""

    version = value.get("schema_version")
    if version not in {1, 2}:
        raise ConfigError(f"{source} must declare schema_version: 1 or 2")
    scene_id = value.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ConfigError(f"{source} must define a non-empty scene_id")
    paths = value.get("paths")
    if not isinstance(paths, dict):
        raise ConfigError(f"{source} must define a paths mapping")

    has_legacy = "rainygs_arguments" in value
    has_current = "renderer" in value
    if has_legacy and has_current:
        raise ConfigError(
            f"{source} cannot contain both rainygs_arguments and renderer"
        )

    if version == 2:
        if has_legacy:
            raise ConfigError(
                f"{source} uses removed field rainygs_arguments in schema v2"
            )
        if not has_current or not isinstance(value.get("renderer"), dict):
            raise ConfigError(f"{source} schema v2 must define renderer")
        renderer = dict(value["renderer"])
        removed_fields = sorted(REMOVED_RENDERER_FIELDS & renderer.keys())
        removed_paths = sorted(REMOVED_SCENE_PATHS & paths.keys())
        if removed_fields or removed_paths:
            details = [f"renderer.{name}" for name in removed_fields]
            details.extend(f"paths.{name}" for name in removed_paths)
            raise ConfigError(
                f"{source} schema v2 contains removed field(s): "
                + ", ".join(details)
            )
        migration = {
            "source_schema_version": 2,
            "normalized_schema_version": 2,
            "converted": False,
            "removed_fields": [],
            "removed_paths": [],
        }
        normalized_paths = dict(paths)
    else:
        legacy = value.get("rainygs_arguments")
        if not isinstance(legacy, dict):
            raise ConfigError(
                f"{source} schema v1 must define rainygs_arguments"
            )
        renderer = {
            key: item
            for key, item in legacy.items()
            if key not in REMOVED_RENDERER_FIELDS
        }
        removed_fields = sorted(REMOVED_RENDERER_FIELDS & legacy.keys())
        removed_paths = sorted(REMOVED_SCENE_PATHS & paths.keys())
        normalized_paths = {
            key: item for key, item in paths.items() if key not in REMOVED_SCENE_PATHS
        }
        migration = {
            "source_schema_version": 1,
            "normalized_schema_version": 2,
            "converted": True,
            "removed_fields": [
                f"rainygs_arguments.{name}" for name in removed_fields
            ],
            "removed_paths": [f"paths.{name}" for name in removed_paths],
        }

    validate_current_runtime_config(renderer, source=source)
    missing_renderer = sorted(REQUIRED_RENDERER_FIELDS - renderer.keys())
    if missing_renderer:
        raise ConfigError(
            f"{source} renderer is missing field(s): {', '.join(missing_renderer)}"
        )

    normalized = {
        key: item
        for key, item in value.items()
        if key not in {"schema_version", "rainygs_arguments", "renderer", "paths"}
    }
    normalized = {
        "schema_version": 2,
        **normalized,
        "paths": normalized_paths,
        "renderer": renderer,
    }
    return normalized, migration


def load_scene_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = load_yaml_mapping(path, description="scene configuration")
    if value.get("schema_version") != 2:
        raise ConfigError(
            f"{path} uses retired scene schema v1; migrate it offline with "
            "scripts/migrate_scene_config.py"
        )
    return normalize_scene_config(value, source=str(path))


def load_paths_config(path: Path) -> dict[str, Any]:
    value = load_yaml_mapping(path, description="local paths configuration")
    if value.get("schema_version") != 1:
        raise ConfigError(f"{path} must declare schema_version: 1")
    if not isinstance(value.get("roots"), dict) or not value["roots"]:
        raise ConfigError(f"{path} must define a non-empty roots mapping")
    return value


def expand_local_root(value: Any, config_dir: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"root {name!r} must be a non-empty string")
    missing = sorted(
        {
            match.group(1) or match.group(2)
            for match in VARIABLE_PATTERN.finditer(value)
            if (match.group(1) or match.group(2)) not in os.environ
        }
    )
    if missing:
        raise ConfigError(
            f"root {name!r} uses unset environment variable(s): "
            + ", ".join(missing)
        )
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute():
        expanded = config_dir / expanded
    return expanded.resolve(strict=False)


def resolve_path_spec(
    name: str, spec: Any, roots: dict[str, Path]
) -> tuple[Path, str, bool]:
    if not isinstance(spec, dict):
        raise ConfigError(f"paths.{name} must be a mapping")
    root_name = spec.get("root")
    relative_value = spec.get("path")
    kind = spec.get("kind", "any")
    required = spec.get("required", True)
    if root_name not in roots:
        raise ConfigError(f"paths.{name}.root references unknown root {root_name!r}")
    if not isinstance(relative_value, str) or not relative_value.strip():
        raise ConfigError(f"paths.{name}.path must be a non-empty string")
    relative_path = Path(relative_value)
    if relative_path.is_absolute():
        raise ConfigError(f"paths.{name}.path must be relative to root {root_name!r}")
    if kind not in {"file", "directory", "any"}:
        raise ConfigError(f"paths.{name}.kind must be file, directory, or any")
    if not isinstance(required, bool):
        raise ConfigError(f"paths.{name}.required must be true or false")
    root = roots[root_name]
    resolved = (root / relative_path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigError(f"paths.{name}.path escapes root {root_name!r}") from exc
    return resolved, kind, required


def validate_input(name: str, path: Path, kind: str, required: bool) -> str:
    if not path.exists():
        if required:
            raise ConfigError(f"required {name} does not exist: {path}")
        return "missing (optional)"
    if kind == "file" and not path.is_file():
        raise ConfigError(f"{name} must be a file: {path}")
    if kind == "directory" and not path.is_dir():
        raise ConfigError(f"{name} must be a directory: {path}")
    return "present"


def _required_paths_for_operation(
    operation: str, raw_paths: dict[str, Any]
) -> set[str]:
    if operation == "all":
        return {
            name
            for name, spec in raw_paths.items()
            if not isinstance(spec, dict) or spec.get("required", True)
        }
    if operation not in OPERATION_REQUIRED_PATHS:
        choices = ", ".join(["all", *sorted(OPERATION_REQUIRED_PATHS)])
        raise ConfigError(f"unknown operation {operation!r}; expected one of {choices}")
    return set(OPERATION_REQUIRED_PATHS[operation])


def build_scene_manifest(
    paths_config: Path,
    scene_config: Path,
    *,
    check_inputs: bool,
    run_id: str,
    operation: str = "all",
    repo_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    local = load_paths_config(paths_config)
    scene, migration = load_scene_config(scene_config)
    roots = {
        name: expand_local_root(value, paths_config.parent, name)
        for name, value in local["roots"].items()
    }
    raw_paths = scene["paths"]
    required_paths = _required_paths_for_operation(operation, raw_paths)
    required_keys = required_paths | {"artifacts"}
    missing = sorted(required_keys - raw_paths.keys())
    if missing:
        raise ConfigError(
            f"{scene_config} is missing path(s) required for {operation}: "
            + ", ".join(missing)
        )

    resolved: dict[str, str] = {}
    statuses: dict[str, str] = {}
    for name, spec in raw_paths.items():
        path, kind, declared_required = resolve_path_spec(name, spec, roots)
        resolved[name] = str(path)
        if not check_inputs:
            statuses[name] = "not checked"
        elif name in required_paths:
            statuses[name] = validate_input(name, path, kind, True)
        elif operation == "all":
            statuses[name] = validate_input(name, path, kind, declared_required)
        else:
            statuses[name] = f"not required for {operation}"

    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ConfigError("run ID must be one non-empty path component")
    run_output = (Path(resolved["artifacts"]) / run_id).resolve(strict=False)
    resolved["run_output"] = str(run_output)
    statuses["run_output"] = "present" if run_output.is_dir() else "not created"

    renderer = dict(scene["renderer"])
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    command = [
        os.fspath(Path(sys.executable)),
        str(root / "render_sand_water.py"),
        "-s",
        resolved.get("dataset", "<dataset-required-for-render>"),
        "-m",
        resolved.get("checkpoint", "<checkpoint-required-for-render>"),
        "--render_output_path",
        resolved["run_output"],
        "--config_file",
        "<generated-render-config.yaml>",
    ]
    manifest = {
        "schema_version": 2,
        "scene_id": scene.get("scene_id"),
        "run_id": run_id,
        "operation": operation,
        "source_files": {
            "paths": str(paths_config.resolve()),
            "scene": str(scene_config.resolve()),
        },
        "configuration_migration": migration,
        "roots": {name: str(path) for name, path in roots.items()},
        "paths": resolved,
        "input_status": statuses,
        "renderer_arguments": renderer,
        "command_template": command,
    }
    return manifest, renderer, command


def renderer_arguments_from_manifest(value: dict[str, Any]) -> dict[str, Any]:
    """Read renderer arguments from a current schema-v2 resolved manifest."""

    if value.get("schema_version") != 2:
        raise ConfigError(
            "resolved manifest must use schema_version 2; regenerate it from "
            "a migrated scene configuration"
        )
    current = value.get("renderer_arguments")
    legacy = value.get("rainygs_arguments")
    if current is not None and legacy is not None:
        raise ConfigError(
            "resolved manifest cannot contain both renderer_arguments and "
            "rainygs_arguments"
        )
    if value.get("schema_version") == 2 and legacy is not None:
        raise ConfigError(
            "schema-v2 resolved manifest uses removed field rainygs_arguments"
        )
    if not isinstance(current, dict):
        raise ConfigError("resolved manifest has no renderer arguments mapping")
    removed = sorted(REMOVED_RENDERER_FIELDS & current.keys())
    if removed:
        raise ConfigError(
            "resolved manifest contains removed renderer field(s): "
            + ", ".join(removed)
        )
    validate_current_runtime_config(current, source="resolved manifest")
    return dict(current)


def validate_sand_water_resources(
    paths: dict[str, str],
    renderer: dict[str, Any],
    render_options: dict[str, Any] | None,
    *,
    include_render: bool,
    expected_scene_id: str | None = None,
) -> dict[str, Any]:
    """Validate only resources consumed by a SandyGS simulation/render action."""

    from sandygs.exchange import validate_bundle

    runtime = dict(renderer)
    if render_options is not None:
        runtime.update(render_options)
    validate_current_runtime_config(runtime, source="sand-water runtime")

    exchange = Path(paths["p1_exchange"])
    metadata = validate_bundle(exchange)
    if (
        expected_scene_id is not None
        and metadata.get("scene_id") != expected_scene_id
    ):
        raise ConfigError(
            f"P1 exchange scene_id {metadata.get('scene_id')!r} does not match "
            f"scene configuration {expected_scene_id!r}"
        )
    result: dict[str, Any] = {
        "exchange": {
            "path": str(exchange),
            "scene_id": metadata["scene_id"],
            "resolution": metadata["grid"]["resolution"],
        }
    }
    if not include_render:
        return result

    result["render"] = validate_reconstruction_resources(
        paths, renderer, render_options
    )
    return result


def validate_terrain_exchange_resources(
    paths: dict[str, str], renderer: dict[str, Any]
) -> dict[str, Any]:
    """Validate the height-map input actually consumed while creating P1."""

    import numpy as np

    path = Path(paths["low_height_map"])
    try:
        height = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot load terrain height map {path}: {exc}") from exc
    if height.ndim != 2 or not np.issubdtype(height.dtype, np.number):
        raise ConfigError(f"terrain height map must be a numeric 2-D array: {path}")
    if not np.isfinite(height).all():
        raise ConfigError(f"terrain height map contains non-finite values: {path}")
    for field, length in (
        ("height_map_range", 4),
        ("align_rotation_angles", 3),
        ("align_translation_vector", 3),
        ("align_scale", 3),
    ):
        value = renderer.get(field)
        if not isinstance(value, list) or len(value) != length:
            raise ConfigError(f"renderer.{field} must contain {length} values")
    return {
        "height_map": str(path),
        "dtype": str(height.dtype),
        "shape": list(height.shape),
    }


def validate_reconstruction_resources(
    paths: dict[str, str],
    renderer: dict[str, Any],
    render_options: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate dataset/checkpoint/environment and optional material textures."""

    dataset = Path(paths["dataset"])
    checkpoint = Path(paths["checkpoint"])
    required = [
        dataset / "sparse/0/cameras.bin",
        dataset / "sparse/0/images.bin",
        dataset / "sparse/0/points3D.bin",
        checkpoint / "cfg_args",
    ]
    point_cloud_root = checkpoint / "point_cloud"
    iterations = []
    if point_cloud_root.is_dir():
        for directory in point_cloud_root.glob("iteration_*"):
            try:
                iteration = int(directory.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if (directory / "point_cloud.ply").is_file():
                iterations.append(iteration)
    if not iterations:
        raise ConfigError(f"checkpoint has no point-cloud iteration: {checkpoint}")
    iteration = max(iterations)
    point_cloud = checkpoint / f"point_cloud/iteration_{iteration}/point_cloud.ply"
    required.append(point_cloud)
    environment_name = (
        "overcast_soil_puresky_1k.hdr"
        if renderer["dataset_name"] == "siat_city"
        else "overcast_soil_puresky_1k_3.png"
    )
    environment = (
        checkpoint / f"brdf_mlp/iteration_{iteration}/{environment_name}"
    )
    required.append(environment)
    if bool(renderer.get("is_sky_model")):
        required.append(checkpoint / "env_light_chkpnt30000.pth")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ConfigError("missing render resource(s): " + ", ".join(missing))

    texture_files: list[str] = []
    options = render_options or {}
    if bool(options.get("sand_texture_enabled")):
        raw_root = options.get("sand_texture_root")
        if not isinstance(raw_root, str) or not raw_root:
            raise ConfigError("sand_texture_enabled requires sand_texture_root")
        expanded = Path(os.path.expandvars(os.path.expanduser(raw_root))).resolve()
        unresolved = VARIABLE_PATTERN.findall(str(expanded))
        if unresolved:
            raise ConfigError(f"unresolved sand texture root: {raw_root}")
        texture_paths = [
            expanded / "albedo.jpg",
            expanded / "roughness.jpg",
            expanded / "normal_gl.png",
        ]
        missing_texture = [str(path) for path in texture_paths if not path.is_file()]
        if missing_texture:
            raise ConfigError(
                "missing sand texture resource(s): " + ", ".join(missing_texture)
            )
        texture_files = [str(path) for path in texture_paths]
    return {
        "dataset_sparse": [str(path) for path in required[:3]],
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": iteration,
        "point_cloud": str(point_cloud),
        "environment": str(environment),
        "sand_textures": texture_files,
    }
