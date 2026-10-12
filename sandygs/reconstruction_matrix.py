"""Resolve and validate the small pre-refactor reconstruction matrix."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

import yaml

from sandygs.pile_benchmark import select_experiment_phase
from sandygs.regression_capture import (
    RegressionCaptureError,
    validate_regression_output_keys,
)
from sandygs.render_schedule import expand_render_schedule


class ReconstructionMatrixError(RuntimeError):
    """Raised when the authored reconstruction matrix is inconsistent."""


_SAFE_ID = re.compile(r"^[a-z][a-z0-9_-]*$")


def _load_yaml(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ReconstructionMatrixError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReconstructionMatrixError(f"{label} must contain a YAML mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_repo_file(repo_root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReconstructionMatrixError(f"{label} must be a non-empty repository path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReconstructionMatrixError(f"{label} must be a safe repository-relative path")
    root = repo_root.expanduser().resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReconstructionMatrixError(f"{label} resolves outside {root}") from exc
    if not path.is_file():
        raise ReconstructionMatrixError(f"missing {label}: {path}")
    return path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _select_profile_phase(
    profile: dict[str, Any], phase: Any, *, label: str
) -> dict[str, Any]:
    if profile.get("schema_version") != 1:
        raise ReconstructionMatrixError(f"{label} must use schema_version 1")
    resolved = deepcopy(profile)
    if phase is None:
        return resolved
    if not isinstance(phase, str) or not phase:
        raise ReconstructionMatrixError(f"{label}.base_phase must be a non-empty string")
    phases = profile.get("phases")
    if not isinstance(phases, dict) or not isinstance(phases.get(phase), dict):
        raise ReconstructionMatrixError(f"{label} has no phase {phase!r}")
    resolved = _deep_merge(resolved, phases[phase])
    resolved["selected_phase"] = phase
    return resolved


def _nested_value(value: dict[str, Any], dotted: str, *, label: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ReconstructionMatrixError(f"{label} does not contain {dotted}")
        current = current[part]
    return current


def _changed_paths(before: Any, after: Any, prefix: str = "") -> list[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        paths: list[str] = []
        for key in sorted(set(before) | set(after)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                paths.append(child)
            else:
                paths.extend(_changed_paths(before[key], after[key], child))
        return paths
    return [] if before == after else [prefix]


def _expected_samples(value: Any, *, label: str) -> list[tuple[int, float]]:
    if not isinstance(value, list) or not value:
        raise ReconstructionMatrixError(f"{label} must be a non-empty list")
    samples: list[tuple[int, float]] = []
    for index, sample in enumerate(value):
        if not isinstance(sample, dict):
            raise ReconstructionMatrixError(f"{label}[{index}] must be a mapping")
        camera = sample.get("camera_index")
        time = sample.get("time_seconds")
        if isinstance(camera, bool) or not isinstance(camera, int) or camera < 0:
            raise ReconstructionMatrixError(
                f"{label}[{index}].camera_index must be non-negative"
            )
        try:
            seconds = float(time)
        except (TypeError, ValueError) as exc:
            raise ReconstructionMatrixError(
                f"{label}[{index}].time_seconds must be numeric"
            ) from exc
        samples.append((camera, seconds))
    if len(set(samples)) != len(samples):
        raise ReconstructionMatrixError(f"{label} contains duplicate camera/time samples")
    return samples


def _resolve_render_case(
    case: dict[str, Any],
    *,
    repo_root: Path,
    capture_sets: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    identifier = case.get("id")
    if not isinstance(identifier, str) or _SAFE_ID.fullmatch(identifier) is None:
        raise ReconstructionMatrixError(
            "each render case id must use lowercase letters, digits, '_' or '-'"
        )
    runner = case.get("runner")
    if runner not in {"p2", "p4", "p5"}:
        raise ReconstructionMatrixError(f"{identifier}.runner must be p2, p4, or p5")
    profile_path = _safe_repo_file(
        repo_root, case.get("base_profile"), label=f"{identifier}.base_profile"
    )
    scene_path = _safe_repo_file(
        repo_root, case.get("scene"), label=f"{identifier}.scene"
    )
    source = _load_yaml(profile_path, label=f"{identifier} base profile")
    base = _select_profile_phase(
        source, case.get("base_phase"), label=f"{identifier} base profile"
    )
    resolved = deepcopy(base)
    input_files = [
        {
            "role": "scene",
            "path": scene_path.relative_to(repo_root).as_posix(),
            "sha256": _sha256(scene_path),
        },
        {
            "role": "base_profile",
            "path": profile_path.relative_to(repo_root).as_posix(),
            "sha256": _sha256(profile_path),
        }
    ]

    water_profile_value = case.get("water_render_profile")
    if water_profile_value is not None:
        water_path = _safe_repo_file(
            repo_root,
            water_profile_value,
            label=f"{identifier}.water_render_profile",
        )
        water_profile = _load_yaml(water_path, label="water render profile")
        water_render = water_profile.get("render")
        if not isinstance(water_render, dict):
            raise ReconstructionMatrixError("water render profile has no render mapping")
        resolved.setdefault("render", {}).update(
            {
                key: deepcopy(value)
                for key, value in water_render.items()
                if key.startswith("water_")
            }
        )
        input_files.append(
            {
                "role": "water_render_profile",
                "path": water_path.relative_to(repo_root).as_posix(),
                "sha256": _sha256(water_path),
            }
        )

    overrides = case.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ReconstructionMatrixError(f"{identifier}.overrides must be a mapping")
    resolved = _deep_merge(resolved, overrides)
    render = resolved.get("render")
    solver = resolved.get("solver")
    if not isinstance(render, dict) or not isinstance(solver, dict):
        raise ReconstructionMatrixError(
            f"{identifier} resolved profile needs solver and render mappings"
        )
    schedule = case.get("render_schedule")
    if not isinstance(schedule, dict):
        raise ReconstructionMatrixError(f"{identifier}.render_schedule must be a mapping")
    render["render_schedule"] = deepcopy(schedule)

    capture_name = case.get("capture_set")
    if not isinstance(capture_name, str) or capture_name not in capture_sets:
        raise ReconstructionMatrixError(f"{identifier}.capture_set is not declared")
    try:
        capture_keys = validate_regression_output_keys(capture_sets[capture_name])
    except RegressionCaptureError as exc:
        raise ReconstructionMatrixError(f"invalid capture set {capture_name}: {exc}") from exc
    render["regression_raw_output_keys"] = capture_keys

    expected_backend = case.get("expected_water_backend")
    if expected_backend != "sandygs_p4":
        raise ReconstructionMatrixError(
            f"{identifier} reconstruction baseline must use sandygs_p4"
        )
    if render.get("water_rendering_backend") != expected_backend:
        raise ReconstructionMatrixError(
            f"{identifier} must resolve water_rendering_backend={expected_backend}"
        )
    if runner == "p2" and render.get("render_sand") is not False:
        raise ReconstructionMatrixError(
            f"{identifier} pure-water profile must set render_sand=false"
        )
    frames, camera_path_frames = expand_render_schedule(schedule)
    actual_samples = [(frame.camera_index, frame.time_seconds) for frame in frames]
    expected = _expected_samples(
        case.get("expected_samples"), label=f"{identifier}.expected_samples"
    )
    if len(actual_samples) != len(expected) or any(
        camera != expected_camera or abs(time - expected_time) > 1.0e-12
        for (camera, time), (expected_camera, expected_time) in zip(
            actual_samples, expected
        )
    ):
        raise ReconstructionMatrixError(
            f"{identifier} expanded schedule does not match expected_samples"
        )
    if not all(frame.save_debug and frame.save_intermediates for frame in frames):
        raise ReconstructionMatrixError(
            f"{identifier} must save debug maps and intermediates for every sample"
        )
    duration = solver.get("duration_seconds")
    if duration is None or max(frame.time_seconds for frame in frames) > float(duration) + 1.0e-12:
        raise ReconstructionMatrixError(
            f"{identifier} schedule exceeds solver.duration_seconds"
        )

    for dotted in case.get("preserve", []):
        if _nested_value(base, dotted, label=f"{identifier} base") != _nested_value(
            resolved, dotted, label=f"{identifier} resolved"
        ):
            raise ReconstructionMatrixError(
                f"{identifier} changed preserved field {dotted}"
            )

    materialized = deepcopy(resolved)
    materialized["profile_id"] = f"reconstruction-regression-{identifier}"
    materialized.pop("selected_phase", None)
    experiment_phase = None
    if runner == "p5":
        experiment_phase = "reconstruction_regression"
        materialized["phases"] = {experiment_phase: {}}
    record = {
        "id": identifier,
        "runner": runner,
        "scene": scene_path.relative_to(repo_root).as_posix(),
        "base_phase": case.get("base_phase"),
        "input_files": input_files,
        "changed_paths": _changed_paths(base, resolved),
        "camera_path_frames": camera_path_frames,
        "samples": [
            {"camera_index": camera, "time_seconds": time}
            for camera, time in actual_samples
        ],
        "raw_output_keys": capture_keys,
        "experiment_phase": experiment_phase,
        "materialization_only_paths": (
            ["profile_id", "phases"] if runner == "p5" else ["profile_id"]
        ),
    }
    return materialized, record


def resolve_reconstruction_matrix(
    spec_path: Path, *, repo_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate the authored matrix and return materialized profiles and audit."""

    repo_root = repo_root.expanduser().resolve()
    spec_path = spec_path.expanduser().resolve()
    spec = _load_yaml(spec_path, label="reconstruction matrix")
    if spec.get("schema_version") != 1:
        raise ReconstructionMatrixError("reconstruction matrix schema_version must be 1")
    matrix_id = spec.get("matrix_id")
    if not isinstance(matrix_id, str) or not matrix_id:
        raise ReconstructionMatrixError("matrix_id must be a non-empty string")
    capture_sets = spec.get("capture_sets")
    render_cases = spec.get("render_cases")
    if not isinstance(capture_sets, dict) or not isinstance(render_cases, list):
        raise ReconstructionMatrixError("matrix needs capture_sets and render_cases")

    profiles: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for case in render_cases:
        if not isinstance(case, dict):
            raise ReconstructionMatrixError("each render case must be a mapping")
        profile, record = _resolve_render_case(
            case, repo_root=repo_root, capture_sets=capture_sets
        )
        if record["id"] in profiles:
            raise ReconstructionMatrixError(f"duplicate render case id: {record['id']}")
        profiles[record["id"]] = profile
        records.append(record)

    fig4 = spec.get("fig4")
    if not isinstance(fig4, dict):
        raise ReconstructionMatrixError("matrix needs a fig4 mapping")
    fig4_path = _safe_repo_file(
        repo_root, fig4.get("base_config"), label="fig4.base_config"
    )
    fig4_source = _load_yaml(fig4_path, label="Fig.4 config")
    try:
        fig4_resolved = select_experiment_phase(fig4_source, str(fig4.get("phase")))
    except (KeyError, TypeError, ValueError) as exc:
        raise ReconstructionMatrixError(f"cannot resolve Fig.4 phase: {exc}") from exc
    saturations = [float(value) for value in fig4.get("saturations", [])]
    times = [float(value) for value in fig4.get("times_seconds", [])]
    if saturations != [0.0, 0.25, 0.45] or times != [0.0, 0.5, 2.0]:
        raise ReconstructionMatrixError(
            "Fig.4 reconstruction samples must remain phi=0/0.25/0.45 at 0/0.5/2 s"
        )
    duration = float(fig4_resolved["solver_scene"]["run"]["duration"])
    if max(times) > duration + 1.0e-12:
        raise ReconstructionMatrixError("Fig.4 samples exceed the selected phase duration")
    available_views = {
        str(view["name"]): deepcopy(view)
        for view in fig4_resolved["render"]["views"]
    }
    requested_views = fig4.get("views")
    if not isinstance(requested_views, list) or any(
        name not in available_views for name in requested_views
    ):
        raise ReconstructionMatrixError("Fig.4 requested view is absent from base config")

    baseline_records = []
    baselines = spec.get("preserved_render_baselines")
    if not isinstance(baselines, list) or not baselines:
        raise ReconstructionMatrixError("preserved_render_baselines must be non-empty")
    for index, value in enumerate(baselines):
        path = _safe_repo_file(
            repo_root, value, label=f"preserved_render_baselines[{index}]"
        )
        baseline = _load_yaml(path, label="render baseline")
        if not baseline.get("baseline_id") or not isinstance(baseline.get("cases"), list):
            raise ReconstructionMatrixError(f"not a frozen render baseline spec: {path}")
        baseline_records.append(
            {
                "baseline_id": baseline["baseline_id"],
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "schema_version": 1,
        "matrix_id": matrix_id,
        "source_spec": {"path": str(spec_path), "sha256": _sha256(spec_path)},
        "render_cases": records,
        "fig4": {
            "base_config": fig4_path.relative_to(repo_root).as_posix(),
            "base_config_sha256": _sha256(fig4_path),
            "phase": fig4.get("phase"),
            "saturations": saturations,
            "times_seconds": times,
            "views": [available_views[name] for name in requested_views],
            "command_templates": [
                [
                    "python",
                    "scripts/experiments/run_fig4_piles.py",
                    "--config",
                    fig4_path.relative_to(repo_root).as_posix(),
                    "--phase",
                    str(fig4.get("phase")),
                    "--saturation",
                    str(saturation),
                    "--run-id",
                    f"<new-fig4-phi-{round(saturation * 1000):03d}-run-id>",
                ]
                for saturation in saturations
            ],
        },
        "preserved_render_baselines": baseline_records,
    }
    return profiles, manifest


def materialize_reconstruction_matrix(
    spec_path: Path, *, repo_root: Path, output_root: Path
) -> Path:
    """Write resolved profiles and a hash-bearing manifest to a new directory."""

    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise ReconstructionMatrixError(
            f"refusing to overwrite reconstruction matrix output: {output_root}"
        )
    profiles, manifest = resolve_reconstruction_matrix(
        spec_path, repo_root=repo_root
    )
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid4().hex}"
    try:
        temporary.mkdir(parents=True)
        for identifier, profile in profiles.items():
            destination = temporary / "profiles" / f"{identifier}.yaml"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(profile, stream, sort_keys=False, allow_unicode=True)
            record = next(
                item for item in manifest["render_cases"] if item["id"] == identifier
            )
            record["materialized_profile"] = (
                Path("profiles") / destination.name
            ).as_posix()
            record["materialized_profile_sha256"] = _sha256(destination)
            script = {
                "p2": "scripts/run_p2_pipeline.sh",
                "p4": "scripts/run_p4_pipeline.sh",
                "p5": "scripts/run_p5_experiment.sh",
            }[record["runner"]]
            command = [
                script,
                "--scene",
                record["scene"],
                "--profile",
                str(output_root / record["materialized_profile"]),
            ]
            if record["experiment_phase"] is not None:
                command.extend(
                    ["--experiment-phase", record["experiment_phase"]]
                )
            command.extend(["--run-id", f"<new-{identifier}-run-id>"])
            record["command_template"] = command
        with (temporary / "matrix_manifest.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)
        temporary.rename(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_root
