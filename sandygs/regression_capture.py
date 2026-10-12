"""Persist explicitly selected renderer tensors for reconstruction regression."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


class RegressionCaptureError(RuntimeError):
    """Raised when a requested raw regression output cannot be captured."""


_SAFE_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SAFE_STEM = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_regression_output_keys(value: Any) -> list[str]:
    """Validate and normalise an opt-in renderer-output allow-list."""

    if not isinstance(value, (list, tuple)) or not value:
        raise RegressionCaptureError(
            "regression_raw_output_keys must be a non-empty list"
        )
    keys: list[str] = []
    for index, key in enumerate(value):
        if not isinstance(key, str) or _SAFE_KEY.fullmatch(key) is None:
            raise RegressionCaptureError(
                f"regression_raw_output_keys[{index}] is not a safe tensor key"
            )
        if key in keys:
            raise RegressionCaptureError(
                f"duplicate regression raw output key: {key}"
            )
        keys.append(key)
    return keys


def _as_numpy(value: Any, *, key: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    elif all(hasattr(value, method) for method in ("detach", "cpu", "numpy")):
        array = value.detach().cpu().numpy()
    else:
        raise RegressionCaptureError(
            f"renderer output {key!r} is not a NumPy array or tensor"
        )
    if not isinstance(array, np.ndarray) or array.dtype.hasobject:
        raise RegressionCaptureError(
            f"renderer output {key!r} cannot be stored as a safe NPY array"
        )
    return np.ascontiguousarray(array)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_regression_raw_arrays(
    render_outputs: Mapping[str, Any],
    output_root: Path,
    image_stem: str,
    output_keys: Sequence[str],
) -> dict[str, Any]:
    """Save one scheduled frame's selected raw tensors without overwriting."""

    keys = validate_regression_output_keys(output_keys)
    stem = str(image_stem)
    if _SAFE_STEM.fullmatch(stem) is None or stem in {".", ".."}:
        raise RegressionCaptureError(f"unsafe regression frame stem: {stem!r}")
    missing = [key for key in keys if key not in render_outputs]
    if missing:
        raise RegressionCaptureError(
            f"renderer did not produce requested regression outputs: {missing}"
        )

    root = output_root.expanduser().resolve() / "regression_raw"
    manifest_path = root / "manifests" / f"{stem}.yaml"
    targets = [root / key / f"{stem}.npy" for key in keys]
    existing = [path for path in [manifest_path, *targets] if path.exists()]
    if existing:
        raise RegressionCaptureError(
            f"refusing to overwrite regression raw output: {existing[0]}"
        )

    entries: list[dict[str, Any]] = []
    for key, target in zip(keys, targets):
        array = _as_numpy(render_outputs[key], key=key)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
        except OSError as exc:
            raise RegressionCaptureError(f"cannot write {target}: {exc}") from exc
        entries.append(
            {
                "key": key,
                "relative_path": target.relative_to(root).as_posix(),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": _sha256(target),
            }
        )

    manifest = {
        "schema_version": 1,
        "frame": stem,
        "arrays": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with manifest_path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(manifest, stream, sort_keys=False)
    except OSError as exc:
        raise RegressionCaptureError(
            f"cannot write {manifest_path}: {exc}"
        ) from exc
    return manifest
