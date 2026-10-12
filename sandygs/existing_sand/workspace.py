"""Immutable run and stage manifests for the offline existing-sand workflow.

This module records inputs and output files. It does not load PGSR, run models,
or edit a source checkpoint. Later offline stages can write their own payloads
under ``stages/<stage>/<stage_id>/`` and register them with ``record_stage``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np
import yaml


class ExistingSandError(ValueError):
    """An existing-sand input or artifact violates the offline contract."""


_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
_STAGES = {"observations", "segmentation", "background", "modeling"}
_CODE_FILES = (
    "sandygs/existing_sand/__init__.py",
    "sandygs/existing_sand/workspace.py",
    "scripts/experiments/existing_sand.py",
    "sandygs/existing_sand/geometry.py",
    "sandygs/existing_sand/capture.py",
    "scripts/experiments/existing_sand_capture.py",
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExistingSandError(f"{label} must be a mapping")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExistingSandError(f"{label} must be a non-empty string")
    return value


def _identifier(value: Any, label: str) -> str:
    value = _string(value, label)
    if _ID.fullmatch(value) is None:
        raise ExistingSandError(f"{label} must match {_ID.pattern}")
    return value


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ExistingSandError(f"cannot load {label} {path}: {exc}") from exc
    return _mapping(value, label)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ExistingSandError(f"cannot hash file {path}: {exc}") from exc
    return digest.hexdigest()


def _content_digest(files: list[dict[str, Any]]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_relative(root: Path, value: Any, label: str) -> tuple[str, Path]:
    relative = Path(_string(value, label))
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ExistingSandError(f"{label} must be a path inside {root}")
    root = root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ExistingSandError(f"{label} escapes {root}")
    return relative.as_posix(), candidate


def _file_record(path: Path, relative: str, *, role: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ExistingSandError(f"missing file: {path}")
    record: dict[str, Any] = {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if role is not None:
        record["role"] = role
    return record


def _directory_records(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise ExistingSandError(f"missing checkpoint directory: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ExistingSandError(f"checkpoint contains a symlink: {path}")
        if path.is_file():
            files.append(_file_record(path, path.relative_to(root).as_posix()))
    if not files:
        raise ExistingSandError(f"checkpoint directory is empty: {root}")
    return files


def _resolve_input(base: Path, value: Any, label: str) -> Path:
    path = Path(_string(value, label)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _ply_vertex_count(path: Path) -> int:
    vertex_count: int | None = None
    saw_end = False
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ExistingSandError(f"point cloud is not a PLY file: {path}")
        for _ in range(4096):
            line = stream.readline()
            if not line:
                break
            if stream.tell() > 1024 * 1024:
                break
            fields = line.split()
            if fields[:2] == [b"element", b"vertex"]:
                if len(fields) != 3 or vertex_count is not None:
                    raise ExistingSandError(f"invalid PLY vertex declaration: {path}")
                try:
                    vertex_count = int(fields[2])
                except ValueError as exc:
                    raise ExistingSandError(f"invalid PLY vertex count: {path}") from exc
            if fields == [b"end_header"]:
                saw_end = True
                break
    if not saw_end or vertex_count is None or vertex_count <= 0:
        raise ExistingSandError(f"PLY needs a header and positive vertex count: {path}")
    return vertex_count


def _matrix(value: Any) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ExistingSandError("coordinate.sim_to_gs must be a numeric 4x4 matrix") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ExistingSandError("coordinate.sim_to_gs must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ExistingSandError("coordinate.sim_to_gs must be affine")
    try:
        np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ExistingSandError("coordinate.sim_to_gs is singular") from exc
    return matrix


def _camera_ids(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ExistingSandError("camera_ids must be a non-empty list")
    cameras = [_string(item, "camera_ids item") for item in value]
    if len(set(cameras)) != len(cameras):
        raise ExistingSandError("camera_ids contains duplicates")
    return cameras


def _code_record(repo_root: Path, extra: Any = None) -> dict[str, Any]:
    files = list(_CODE_FILES)
    if extra is not None:
        if not isinstance(extra, list):
            raise ExistingSandError("code_files must be a list")
        files.extend(_string(item, "code_files item") for item in extra)
    records = []
    for relative in sorted(set(files)):
        normalized, path = _safe_relative(repo_root, relative, "code file")
        records.append(_file_record(path, normalized))
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ExistingSandError(f"cannot read Git revision in {repo_root}")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )
    if status.returncode != 0:
        raise ExistingSandError(f"cannot read Git status in {repo_root}")
    return {
        "git_commit": result.stdout.strip(),
        "dirty_paths": [line[3:] for line in status.stdout.splitlines()],
        "files": records,
        "content_sha256": _content_digest(records),
    }


def _disjoint(output: Path, source: Path, label: str) -> None:
    if output == source or output in source.parents or source in output.parents:
        raise ExistingSandError(f"output overlaps {label}: {source}")


def initialize_run(config_path: Path, output: Path, *, run_id: str,
                   repo_root: Path) -> dict[str, Any]:
    """Create an exclusive run manifest from a small authored YAML config."""
    config_path = Path(config_path).resolve()
    output = Path(output).resolve()
    repo_root = Path(repo_root).resolve()
    run_id = _identifier(run_id, "run_id")
    config = _load_yaml(config_path, "existing-sand config")
    if config.get("schema_version") != 1:
        raise ExistingSandError("existing-sand config must use schema_version 1")
    scene_id = _identifier(config.get("scene_id"), "scene_id")
    source = _mapping(config.get("source"), "source")
    dataset = _resolve_input(config_path.parent, source.get("dataset"), "source.dataset")
    checkpoint = _resolve_input(
        config_path.parent, source.get("checkpoint"), "source.checkpoint"
    )
    if not dataset.is_dir():
        raise ExistingSandError(f"missing dataset directory: {dataset}")
    _disjoint(output, repo_root, "source repository")
    for name, path in (("dataset", dataset), ("checkpoint", checkpoint)):
        _disjoint(output, path, name)
    point_cloud_relative, point_cloud = _safe_relative(
        checkpoint, source.get("point_cloud"), "source.point_cloud"
    )
    if not point_cloud.is_file():
        raise ExistingSandError(f"missing source point cloud: {point_cloud}")
    gaussian_count = _ply_vertex_count(point_cloud)
    checkpoint_files = _directory_records(checkpoint)
    dataset_files: list[dict[str, Any]] = []
    specified_files = source.get("dataset_files", [])
    if not isinstance(specified_files, list) or not specified_files:
        raise ExistingSandError("source.dataset_files must be a non-empty list of camera/input files")
    for item in specified_files:
        relative, path = _safe_relative(dataset, item, "source.dataset_files item")
        dataset_files.append(_file_record(path, relative))
    if len({item["path"] for item in dataset_files}) != len(dataset_files):
        raise ExistingSandError("source.dataset_files contains duplicates")
    coordinate = _mapping(config.get("coordinate"), "coordinate")
    sim_to_gs = _matrix(coordinate.get("sim_to_gs"))
    cameras = _camera_ids(config.get("camera_ids"))
    if output.exists():
        raise ExistingSandError(f"refusing to overwrite existing run: {output}")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "scene_id": scene_id,
        "source": {
            "dataset": str(dataset),
            "dataset_files": dataset_files,
            "checkpoint": str(checkpoint),
            "checkpoint_files": checkpoint_files,
            "checkpoint_sha256": _content_digest(checkpoint_files),
            "point_cloud": point_cloud_relative,
            "point_cloud_sha256": _sha256(point_cloud),
            "gaussian_count": gaussian_count,
        },
        "camera_ids": cameras,
        "coordinate": {
            "sim_to_gs": sim_to_gs.tolist(),
            "gs_to_sim": np.linalg.inv(sim_to_gs).tolist(),
        },
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "code": _code_record(repo_root),
        "stage_root": "stages",
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)
    return manifest


def _stage_location(run_root: Path, stage: Any, stage_id: Any) -> Path:
    stage = _string(stage, "stage")
    if stage not in _STAGES:
        raise ExistingSandError(f"stage must be one of {sorted(_STAGES)}")
    return run_root / "stages" / stage / _identifier(stage_id, "stage_id")


def _validate_stage_descriptor(record: dict[str, Any], run: dict[str, Any],
                               stage_dir: Path, record_path: Path,
                               repo_root: Path) -> dict[str, Any]:
    if record.get("schema_version") != 1:
        raise ExistingSandError("stage record must use schema_version 1")
    stage = _string(record.get("stage"), "stage")
    stage_id = _identifier(record.get("stage_id"), "stage_id")
    if stage_dir != _stage_location(stage_dir.parents[2], stage, stage_id):
        raise ExistingSandError("stage record does not match its directory")
    cameras = record.get("camera_ids", [])
    if not isinstance(cameras, list):
        raise ExistingSandError("stage camera_ids must be a list")
    cameras = [_string(item, "stage camera_ids item") for item in cameras]
    if len(set(cameras)) != len(cameras):
        raise ExistingSandError("stage camera_ids must be a list without duplicates")
    if any(camera not in run["camera_ids"] for camera in cameras):
        raise ExistingSandError("stage references an unknown camera_id")
    seed = record.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ExistingSandError("stage seed must be an integer or null")
    for key in ("prompt", "parameters"):
        if not isinstance(record.get(key, {}), dict):
            raise ExistingSandError(f"stage {key} must be a mapping")
    commands = record.get("commands", [])
    if not isinstance(commands, list) or any(
        not isinstance(command, list) or not command or
        any(not isinstance(part, str) for part in command)
        for command in commands
    ):
        raise ExistingSandError("stage commands must be a list of argv lists")
    command_cwd = _resolve_input(stage_dir, record.get("command_cwd", str(repo_root)),
                                 "command_cwd")
    if not command_cwd.is_dir():
        raise ExistingSandError(f"missing command working directory: {command_cwd}")
    environment = _mapping(record.get("environment", {}), "environment")
    if any(not isinstance(key, str) or not isinstance(value, str)
           for key, value in environment.items()):
        raise ExistingSandError("environment must map string keys to string values")
    models = record.get("models", [])
    if not isinstance(models, list):
        raise ExistingSandError("stage models must be a list")
    model_records = []
    for index, model in enumerate(models):
        model = _mapping(model, f"models[{index}]")
        entry = {
            "name": _string(model.get("name"), "model.name"),
            "version": _string(model.get("version"), "model.version"),
        }
        if "checkpoint" in model:
            path = _resolve_input(stage_dir, model["checkpoint"], "model.checkpoint")
            entry["checkpoint"] = str(path)
            entry["sha256"] = _sha256(path)
        model_records.append(entry)
    artifacts = record.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        raise ExistingSandError("stage artifacts must be a non-empty list")
    artifact_records = []
    for index, item in enumerate(artifacts):
        item = _mapping(item, f"artifacts[{index}]")
        relative, path = _safe_relative(stage_dir, item.get("path"), "artifact.path")
        if path in {stage_dir / "manifest.yaml", record_path}:
            raise ExistingSandError("stage control files cannot be artifacts")
        artifact_records.append(
            _file_record(path, relative, role=_string(item.get("role"), "artifact.role"))
        )
    if len({item["path"] for item in artifact_records}) != len(artifact_records):
        raise ExistingSandError("stage artifacts contains duplicate paths")
    parents = record.get("parents", [])
    if not isinstance(parents, list):
        raise ExistingSandError("stage parents must be a list")
    parent_records = []
    for parent in parents:
        parent = _mapping(parent, "parent")
        parent_stage = _string(parent.get("stage"), "parent.stage")
        parent_id = _identifier(parent.get("stage_id"), "parent.stage_id")
        path = _stage_location(stage_dir.parents[2], parent_stage, parent_id) / "manifest.yaml"
        parent_records.append({
            "stage": parent_stage, "stage_id": parent_id, "sha256": _sha256(path)
        })
    if len({(item["stage"], item["stage_id"]) for item in parent_records}) != len(parent_records):
        raise ExistingSandError("stage parents contains duplicates")
    return {
        "schema_version": 1,
        "stage": stage,
        "stage_id": stage_id,
        "run_id": run["run_id"],
        "source_checkpoint_sha256": run["source"]["checkpoint_sha256"],
        "source_gaussian_count": run["source"]["gaussian_count"],
        "camera_ids": cameras,
        "prompt": record.get("prompt", {}),
        "seed": seed,
        "models": model_records,
        "parameters": record.get("parameters", {}),
        "commands": commands,
        "command_cwd": str(command_cwd),
        "environment": environment,
        "parents": parent_records,
        "record": _file_record(record_path, record_path.relative_to(stage_dir).as_posix()),
        "artifacts": artifact_records,
        "code": _code_record(repo_root, record.get("code_files")),
    }


def record_stage(run_root: Path, record_path: Path, *, repo_root: Path) -> dict[str, Any]:
    """Seal one completed stage directory without modifying its payload files."""
    run_root = Path(run_root).resolve()
    repo_root = Path(repo_root).resolve()
    run = _load_yaml(run_root / "run.yaml", "run manifest")
    record_path = Path(record_path).resolve()
    record = _load_yaml(record_path, "stage record")
    stage_dir = _stage_location(run_root, record.get("stage"), record.get("stage_id"))
    if not record_path.is_relative_to(stage_dir) or not stage_dir.is_dir():
        raise ExistingSandError(f"stage record must be inside {stage_dir}")
    manifest_path = stage_dir / "manifest.yaml"
    if manifest_path.exists():
        raise ExistingSandError(f"refusing to overwrite stage manifest: {manifest_path}")
    manifest = _validate_stage_descriptor(record, run, stage_dir, record_path, repo_root)
    with manifest_path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)
    return manifest


def verify_run(run_root: Path, *, verify_source: bool = True) -> dict[str, Any]:
    """Read manifests and verify immutable source/stage file hashes."""
    run_root = Path(run_root).resolve()
    run = _load_yaml(run_root / "run.yaml", "run manifest")
    if run.get("schema_version") != 1:
        raise ExistingSandError("run manifest must use schema_version 1")
    source = _mapping(run.get("source"), "run source")
    if verify_source:
        checkpoint = Path(source["checkpoint"])
        files = _directory_records(checkpoint)
        if files != source["checkpoint_files"] or _content_digest(files) != source["checkpoint_sha256"]:
            raise ExistingSandError("source checkpoint has changed")
        point_cloud = checkpoint / source["point_cloud"]
        if _ply_vertex_count(point_cloud) != source["gaussian_count"]:
            raise ExistingSandError("source Gaussian count has changed")
        for item in source["dataset_files"]:
            path = Path(source["dataset"]) / item["path"]
            if _file_record(path, item["path"]) != item:
                raise ExistingSandError(f"source dataset file has changed: {path}")
    stage_root = run_root / "stages"
    stages = []
    if stage_root.exists():
        for stage_dir in sorted(path for path in stage_root.glob("*/*") if path.is_dir()):
            manifest_path = stage_dir / "manifest.yaml"
            if not manifest_path.is_file():
                raise ExistingSandError(f"unregistered stage directory: {stage_dir}")
            manifest = _load_yaml(manifest_path, "stage manifest")
            if stage_dir != _stage_location(run_root, manifest.get("stage"), manifest.get("stage_id")):
                raise ExistingSandError(f"stage manifest location mismatch: {manifest_path}")
            if manifest.get("run_id") != run["run_id"] or (
                manifest.get("source_checkpoint_sha256") != source["checkpoint_sha256"]
            ) or manifest.get("source_gaussian_count") != source["gaussian_count"]:
                raise ExistingSandError(f"stage source binding mismatch: {manifest_path}")
            descriptor = manifest["record"]
            descriptor_relative, descriptor_path = _safe_relative(
                stage_dir, descriptor["path"], "stage record path"
            )
            if _file_record(descriptor_path, descriptor_relative) != descriptor:
                raise ExistingSandError(f"stage record has changed: {descriptor_path}")
            for item in manifest["artifacts"]:
                relative, path = _safe_relative(stage_dir, item["path"], "artifact.path")
                if _file_record(path, relative, role=item["role"]) != item:
                    raise ExistingSandError(f"stage artifact has changed: {path}")
            for parent in manifest["parents"]:
                path = _stage_location(run_root, parent["stage"], parent["stage_id"]) / "manifest.yaml"
                if _sha256(path) != parent["sha256"]:
                    raise ExistingSandError(f"parent stage has changed: {path}")
            for model in manifest["models"]:
                if "checkpoint" in model and _sha256(Path(model["checkpoint"])) != model["sha256"]:
                    raise ExistingSandError(f"model checkpoint has changed: {model['checkpoint']}")
            stages.append({"stage": manifest["stage"], "stage_id": manifest["stage_id"]})
    return {"run_id": run["run_id"], "scene_id": run["scene_id"], "stages": stages}
