"""Machine-local guards for artifact roots backed by removable storage."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

import yaml


class StorageGuardError(RuntimeError):
    """Raised before an experiment could write to an unsafe artifact root."""


def _expanded_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise StorageGuardError(f"{label} must be a non-empty path string")
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        raise StorageGuardError(f"{label} must be absolute: {value}")
    return path


def _load_paths(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except OSError as exc:
        raise StorageGuardError(f"cannot read paths config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StorageGuardError(f"paths config must contain a mapping: {path}")
    return value


def _guard_config(paths: dict[str, Any]) -> dict[str, Any] | None:
    storage_guard = paths.get("storage_guard")
    if storage_guard is None:
        return None
    if not isinstance(storage_guard, dict):
        raise StorageGuardError("storage_guard must be a mapping")
    artifact_guard = storage_guard.get("artifacts")
    if artifact_guard is None:
        return None
    if not isinstance(artifact_guard, dict):
        raise StorageGuardError("storage_guard.artifacts must be a mapping")
    if artifact_guard.get("enabled", True) is False:
        return None
    return artifact_guard


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_artifact_storage(paths_file: Path) -> dict[str, Any]:
    """Validate the configured artifact volume before creating any output.

    The guard is optional so the same tracked code works on ordinary Linux
    servers. A workstation enables it only in the ignored ``paths.local.yaml``.
    """

    paths_file = Path(paths_file).expanduser().resolve()
    paths = _load_paths(paths_file)
    guard = _guard_config(paths)
    if guard is None:
        return {"enabled": False, "paths_file": str(paths_file)}

    roots = paths.get("roots")
    if not isinstance(roots, dict):
        raise StorageGuardError("paths config must define roots")
    artifact_link = _expanded_path(
        roots.get("artifacts"), label="roots.artifacts"
    )
    mountpoint = _expanded_path(
        guard.get("mountpoint"), label="storage_guard.artifacts.mountpoint"
    )
    expected_root = _expanded_path(
        guard.get("expected_real_path"),
        label="storage_guard.artifacts.expected_real_path",
    )

    if not mountpoint.is_dir() or not os.path.ismount(mountpoint):
        raise StorageGuardError(
            f"required artifact volume is not mounted at {mountpoint}; "
            "refusing to write artifacts"
        )
    try:
        resolved_mountpoint = mountpoint.resolve(strict=True)
        resolved_expected = expected_root.resolve(strict=True)
    except OSError as exc:
        raise StorageGuardError(
            f"artifact mount target is incomplete: {exc}"
        ) from exc
    if not _is_relative_to(resolved_expected, resolved_mountpoint):
        raise StorageGuardError(
            f"expected artifact root {resolved_expected} is outside mounted "
            f"volume {resolved_mountpoint}"
        )

    if bool(guard.get("require_symlink", True)) and not artifact_link.is_symlink():
        raise StorageGuardError(
            f"artifact root must remain a symbolic link: {artifact_link}"
        )
    try:
        resolved_link = artifact_link.resolve(strict=True)
    except OSError as exc:
        raise StorageGuardError(
            f"artifact link is missing or dangling: {artifact_link}: {exc}"
        ) from exc
    if resolved_link != resolved_expected:
        raise StorageGuardError(
            f"artifact link resolves to {resolved_link}, expected {resolved_expected}"
        )

    # Check the kernel filesystem flag before the sentinel so a read-only VHD
    # reports its primary mount problem even when the identity file has not yet
    # been initialized.
    try:
        filesystem = os.statvfs(resolved_expected)
    except OSError as exc:
        raise StorageGuardError(
            f"cannot inspect artifact filesystem {resolved_expected}: {exc}"
        ) from exc
    readonly_flag = getattr(os, "ST_RDONLY", 1)
    if filesystem.f_flag & readonly_flag:
        raise StorageGuardError(
            f"artifact filesystem is mounted read-only at {mountpoint}"
        )
    if not os.access(resolved_expected, os.W_OK | os.X_OK):
        raise StorageGuardError(
            f"artifact root is not writable by the current user: {resolved_expected}"
        )

    volume_id_file = guard.get("volume_id_file")
    expected_volume_id = guard.get("volume_id")
    if (volume_id_file is None) != (expected_volume_id is None):
        raise StorageGuardError(
            "volume_id_file and volume_id must either both be set or both omitted"
        )
    if volume_id_file is not None:
        sentinel = Path(str(volume_id_file))
        if sentinel.is_absolute() or ".." in sentinel.parts:
            raise StorageGuardError(
                "storage_guard.artifacts.volume_id_file must be a safe "
                "mount-relative path"
            )
        sentinel = resolved_mountpoint / sentinel
        try:
            actual_volume_id = sentinel.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise StorageGuardError(
                f"cannot read artifact volume identity {sentinel}: {exc}"
            ) from exc
        if actual_volume_id != str(expected_volume_id):
            raise StorageGuardError(
                f"artifact volume identity mismatch in {sentinel}: "
                f"expected {expected_volume_id!r}, got {actual_volume_id!r}"
            )

    if bool(guard.get("write_probe", True)):
        # Permission bits and os.access alone do not prove that a filesystem is
        # writable. Create and immediately unlink a private zero-length probe
        # before the expensive solver or renderer starts.
        try:
            with tempfile.NamedTemporaryFile(
                prefix=".sandygs-write-probe-",
                dir=resolved_expected,
                delete=True,
            ):
                pass
        except OSError as exc:
            raise StorageGuardError(
                f"artifact write probe failed in {resolved_expected}: {exc}"
            ) from exc

    return {
        "enabled": True,
        "paths_file": str(paths_file),
        "mountpoint": str(resolved_mountpoint),
        "artifact_link": str(artifact_link),
        "artifact_root": str(resolved_expected),
        "writable": True,
    }


def validate_artifact_output(
    paths_file: Path, output_path: Path
) -> dict[str, Any]:
    """Apply the guard only when an explicit output belongs to artifacts."""

    paths_file = Path(paths_file).expanduser().resolve()
    if not paths_file.is_file():
        return {"enabled": False, "paths_file": str(paths_file)}
    paths = _load_paths(paths_file)
    guard = _guard_config(paths)
    if guard is None:
        return {"enabled": False, "paths_file": str(paths_file)}
    roots = paths.get("roots")
    if not isinstance(roots, dict):
        raise StorageGuardError("paths config must define roots")
    artifact_link = _expanded_path(
        roots.get("artifacts"), label="roots.artifacts"
    )
    expected_root = _expanded_path(
        guard.get("expected_real_path"),
        label="storage_guard.artifacts.expected_real_path",
    )
    output = Path(output_path).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = Path(os.path.abspath(output))
    applies = _is_relative_to(output, artifact_link) or _is_relative_to(
        output, expected_root
    )
    if not applies:
        return {
            "enabled": False,
            "paths_file": str(paths_file),
            "reason": "output is outside configured artifacts root",
        }
    return validate_artifact_storage(paths_file)
