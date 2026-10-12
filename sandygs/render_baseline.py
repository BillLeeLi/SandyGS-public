"""Freeze and verify immutable sand-render regression baselines.

The freezer copies selected rendered passes plus only the solver snapshots
needed to sample their physical times. Every copied payload is listed with its
SHA-256 digest, and an existing baseline is never overwritten implicitly.
"""

from __future__ import annotations

from bisect import bisect_left
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import subprocess
from typing import Any
from uuid import uuid4

import yaml


class RenderBaselineError(RuntimeError):
    """Raised when a baseline specification or frozen payload is invalid."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except OSError as exc:
        raise RenderBaselineError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RenderBaselineError(f"{path} must contain a YAML mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RenderBaselineError(f"{label} must be a safe relative path: {value}")
    return path


def _normalise_output_name(value: Any) -> str:
    text = str(value)
    # PyYAML can parse unquoted zero-padded frame names as integers. The files
    # and schedule semantics still use five-digit names.
    if text.isdigit():
        return text.zfill(5)
    return text


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=repo_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    revision = run("rev-parse", "HEAD")
    status = run("status", "--short")
    return {
        "revision_at_freeze": revision,
        "worktree_dirty_at_freeze": bool(status),
        "worktree_status_at_freeze": status.splitlines() if status else [],
        "scope_note": (
            "This records the checkout used to freeze the artifact. The source "
            "P5 run predates Git-revision capture and therefore has an unknown "
            "render-time revision."
        ),
    }


def _schedule_index(schedule_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    schedule = _load_yaml(schedule_path)
    frames = schedule.get("frames")
    if not isinstance(frames, list):
        raise RenderBaselineError(f"{schedule_path} has no frames list")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            raise RenderBaselineError(f"{schedule_path} contains a non-mapping frame")
        key = (
            str(frame.get("output_group")),
            _normalise_output_name(frame.get("output_name")),
        )
        if key in result:
            raise RenderBaselineError(f"duplicate render schedule output {key}")
        result[key] = frame
    return result


def _solver_samples(
    run_root: Path, requested_times: list[float]
) -> tuple[dict[float, dict[str, Any]], list[int]]:
    metrics_path = run_root / "metrics.csv"
    try:
        with metrics_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise RenderBaselineError(f"cannot read {metrics_path}: {exc}") from exc
    step_times = {int(row["step"]): float(row["time"]) for row in rows}
    frames_root = run_root / "frames"
    try:
        steps = sorted(int(path.name) for path in frames_root.iterdir() if path.is_dir())
    except (OSError, ValueError) as exc:
        raise RenderBaselineError(f"cannot enumerate solver frames in {frames_root}") from exc
    if not steps:
        raise RenderBaselineError(f"no solver frames in {frames_root}")
    try:
        times = [step_times[step] for step in steps]
    except KeyError as exc:
        raise RenderBaselineError(f"solver frame {exc.args[0]} is absent from metrics.csv") from exc

    samples: dict[float, dict[str, Any]] = {}
    needed: set[int] = set()
    for requested in sorted(set(requested_times)):
        position = bisect_left(times, requested)
        if position == 0:
            lower = upper = 0
        elif position == len(times):
            if requested > times[-1] + 1e-12:
                raise RenderBaselineError(
                    f"requested time {requested} exceeds solver range ending at {times[-1]}"
                )
            lower = upper = len(times) - 1
        elif times[position] == requested:
            lower = upper = position
        else:
            lower, upper = position - 1, position
        lower_step, upper_step = steps[lower], steps[upper]
        if lower == upper:
            weight = 0.0
        else:
            weight = (requested - times[lower]) / (times[upper] - times[lower])
        samples[requested] = {
            "requested_time_seconds": requested,
            "lower_step": lower_step,
            "lower_time_seconds": times[lower],
            "upper_step": upper_step,
            "upper_time_seconds": times[upper],
            "interpolation_weight": weight,
        }
        needed.update((lower_step, upper_step))
    return samples, sorted(needed)


def _copy_payload(
    source: Path,
    destination: Path,
    *,
    category: str,
    source_label: str,
    entries: list[dict[str, Any]],
    case_id: str | None = None,
    product: str | None = None,
) -> None:
    if not source.is_file():
        raise RenderBaselineError(f"required baseline input is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_digest = _sha256(source)
    frozen_digest = _sha256(destination)
    if source_digest != frozen_digest:
        raise RenderBaselineError(f"copy verification failed for {source}")
    entry: dict[str, Any] = {
        "category": category,
        "source": source_label,
        "frozen_relative": destination.as_posix(),
        "size_bytes": destination.stat().st_size,
        "sha256": frozen_digest,
    }
    if case_id is not None:
        entry["case_id"] = case_id
    if product is not None:
        entry["product"] = product
    entries.append(entry)


def freeze_render_baseline(
    *, spec_path: Path, artifacts_root: Path, repo_root: Path
) -> Path:
    """Create a new immutable baseline and return its final directory."""

    spec_path = spec_path.resolve()
    artifacts_root = artifacts_root.resolve()
    repo_root = repo_root.resolve()
    spec = _load_yaml(spec_path)
    if spec.get("schema_version") != 1:
        raise RenderBaselineError("baseline specification schema_version must be 1")

    source_config = spec.get("source")
    output_config = spec.get("output")
    if not isinstance(source_config, dict) or not isinstance(output_config, dict):
        raise RenderBaselineError("baseline specification needs source and output mappings")
    source_relative = _safe_relative(
        str(source_config.get("artifact_relative")), label="source.artifact_relative"
    )
    # A derived renderer comparison may live under ``comparisons/`` while its
    # immutable physical snapshots still belong to the original P5 run.  Keep
    # both roots explicit instead of copying or addressing the solver through
    # unsafe ``..`` paths in a baseline specification.
    solver_source_relative = _safe_relative(
        str(source_config.get("solver_artifact_relative", source_relative)),
        label="source.solver_artifact_relative",
    )
    output_relative = _safe_relative(
        str(output_config.get("artifact_relative")), label="output.artifact_relative"
    )
    source_root = artifacts_root / source_relative
    solver_source_root = artifacts_root / solver_source_relative
    output_root = artifacts_root / output_relative
    if output_root.exists():
        raise RenderBaselineError(
            f"refusing to overwrite frozen baseline {output_root}; use a new baseline_id"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = output_root.parent / f".{output_root.name}.tmp-{uuid4().hex}"
    temporary_root.mkdir()

    entries: list[dict[str, Any]] = []
    try:
        release_relative = _safe_relative(
            str(source_config.get("release_relative")), label="source.release_relative"
        )
        run_relative = _safe_relative(
            str(source_config.get("solver_run_relative")),
            label="source.solver_run_relative",
        )
        schedule_relative = _safe_relative(
            str(source_config.get("schedule_relative")),
            label="source.schedule_relative",
        )
        release_root = source_root / release_relative
        run_root = solver_source_root / run_relative
        schedule = _schedule_index(source_root / schedule_relative)

        products = spec.get("products")
        cases = spec.get("cases")
        if not isinstance(products, dict) or not products:
            raise RenderBaselineError("baseline specification needs products")
        if not isinstance(cases, list) or not cases:
            raise RenderBaselineError("baseline specification needs cases")

        selected_cases: list[dict[str, Any]] = []
        requested_times: list[float] = []
        seen_ids: set[str] = set()
        for case in cases:
            if not isinstance(case, dict):
                raise RenderBaselineError("each baseline case must be a mapping")
            case_id = str(case.get("id"))
            if not case_id or case_id in seen_ids or "/" in case_id or ".." in case_id:
                raise RenderBaselineError(f"invalid or duplicate case id: {case_id}")
            seen_ids.add(case_id)
            output_group = _safe_relative(
                str(case.get("output_group")), label=f"case {case_id} output_group"
            )
            output_name = _normalise_output_name(case.get("output_name"))
            frame = schedule.get((output_group.as_posix(), output_name))
            if frame is None:
                raise RenderBaselineError(
                    f"case {case_id} is absent from the source render schedule"
                )
            expected_camera = int(case["camera_index"])
            expected_time = float(case["time_seconds"])
            if int(frame.get("camera_index")) != expected_camera:
                raise RenderBaselineError(f"case {case_id} camera does not match schedule")
            if abs(float(frame.get("time_seconds")) - expected_time) > 1e-12:
                raise RenderBaselineError(f"case {case_id} time does not match schedule")
            if not frame.get("save_debug"):
                raise RenderBaselineError(f"case {case_id} was not rendered with debug passes")
            requested_times.append(expected_time)
            selected_cases.append(
                {
                    "id": case_id,
                    "camera_index": expected_camera,
                    "time_seconds": expected_time,
                    "output_group": output_group.as_posix(),
                    "output_name": output_name,
                    "purpose": str(case.get("purpose", "")),
                }
            )
            for product, template in products.items():
                relative = _safe_relative(
                    str(template).format(output_name=output_name),
                    label=f"product {product}",
                )
                source = release_root / output_group / relative
                destination_relative = Path("cases") / case_id / f"{product}{source.suffix}"
                _copy_payload(
                    source,
                    temporary_root / destination_relative,
                    category="render_product",
                    source_label=source.relative_to(source_root).as_posix(),
                    entries=entries,
                    case_id=case_id,
                    product=str(product),
                )
                entries[-1]["frozen_relative"] = destination_relative.as_posix()

        samples, solver_steps = _solver_samples(run_root, requested_times)
        fields = spec.get("solver_fields")
        if not isinstance(fields, list) or not fields:
            raise RenderBaselineError("baseline specification needs solver_fields")
        for step in solver_steps:
            for field in fields:
                source = run_root / "frames" / f"{step:06d}" / f"{field}.npy"
                destination_relative = (
                    Path("solver_snapshots")
                    / "frames"
                    / f"{step:06d}"
                    / f"{field}.npy"
                )
                _copy_payload(
                    source,
                    temporary_root / destination_relative,
                    category="solver_snapshot",
                    source_label=(
                        Path("solver_artifact")
                        / source.relative_to(solver_source_root)
                    ).as_posix(),
                    entries=entries,
                )
                entries[-1]["frozen_relative"] = destination_relative.as_posix()

        for value in spec.get("provenance_files", []):
            relative = _safe_relative(str(value), label="provenance file")
            destination_relative = Path("provenance/source_artifact") / relative
            _copy_payload(
                source_root / relative,
                temporary_root / destination_relative,
                category="source_provenance",
                source_label=relative.as_posix(),
                entries=entries,
            )
            entries[-1]["frozen_relative"] = destination_relative.as_posix()

        for value in spec.get("solver_provenance_files", []):
            relative = _safe_relative(
                str(value), label="solver provenance file"
            )
            destination_relative = Path("provenance/solver_artifact") / relative
            _copy_payload(
                solver_source_root / relative,
                temporary_root / destination_relative,
                category="solver_source_provenance",
                source_label=(Path("solver_artifact") / relative).as_posix(),
                entries=entries,
            )
            entries[-1]["frozen_relative"] = destination_relative.as_posix()

        implementation_files: list[dict[str, Any]] = []
        for value in spec.get("reference_implementation_files", []):
            relative = _safe_relative(str(value), label="implementation file")
            destination_relative = Path("provenance/reference_implementation") / relative
            _copy_payload(
                repo_root / relative,
                temporary_root / destination_relative,
                category="reference_implementation_at_freeze",
                source_label=relative.as_posix(),
                entries=entries,
            )
            entries[-1]["frozen_relative"] = destination_relative.as_posix()
            implementation_files.append(
                {"path": relative.as_posix(), "sha256": entries[-1]["sha256"]}
            )

        spec_destination = temporary_root / "baseline_spec.yaml"
        shutil.copy2(spec_path, spec_destination)
        entries.append(
            {
                "category": "baseline_spec",
                "source": spec_path.as_posix(),
                "frozen_relative": "baseline_spec.yaml",
                "size_bytes": spec_destination.stat().st_size,
                "sha256": _sha256(spec_destination),
            }
        )

        manifest = {
            "schema_version": 1,
            "baseline_id": str(spec.get("baseline_id")),
            "description": str(spec.get("description", "")),
            # Keep replacement history inside the new immutable artifact. The
            # superseded directory itself is left untouched so its hashes and
            # audit trail remain meaningful.
            "supersedes": spec.get("supersedes"),
            "status": "frozen",
            "overwrite_policy": "forbidden",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "geometry_backend": spec.get("geometry_backend"),
            "source": {
                "artifact_relative": source_relative.as_posix(),
                "solver_artifact_relative": (
                    solver_source_relative.as_posix()
                ),
                "render_time_git_revision": None,
                "render_time_revision_status": "not_recorded_by_source_artifact",
            },
            "repository_at_freeze": _git_snapshot(repo_root),
            "reference_implementation_files": implementation_files,
            "cases": selected_cases,
            "solver_samples": [samples[value] for value in sorted(samples)],
            "solver_snapshot_steps": solver_steps,
            "files": sorted(entries, key=lambda entry: entry["frozen_relative"]),
        }
        with (temporary_root / "manifest.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(manifest, stream, sort_keys=False)
        with (temporary_root / "SHA256SUMS").open("w", encoding="utf-8") as stream:
            for entry in manifest["files"]:
                stream.write(f"{entry['sha256']}  {entry['frozen_relative']}\n")
        temporary_root.replace(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return output_root


def verify_render_baseline(output_root: Path) -> dict[str, Any]:
    """Verify every frozen payload against ``manifest.yaml``."""

    output_root = output_root.resolve()
    manifest = _load_yaml(output_root / "manifest.yaml")
    if manifest.get("status") != "frozen":
        raise RenderBaselineError(f"{output_root} is not marked frozen")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RenderBaselineError(f"{output_root}/manifest.yaml has no file list")
    for entry in files:
        relative = _safe_relative(str(entry["frozen_relative"]), label="manifest path")
        path = output_root / relative
        if not path.is_file():
            raise RenderBaselineError(f"frozen payload is missing: {path}")
        actual = _sha256(path)
        if actual != entry.get("sha256"):
            raise RenderBaselineError(
                f"frozen payload hash mismatch: {relative} expected "
                f"{entry.get('sha256')}, got {actual}"
            )
    return manifest
