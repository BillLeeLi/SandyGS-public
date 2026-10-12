"""Compare reconstruction outputs against an immutable reference payload.

This module deliberately does not freeze artifacts.  ``render_baseline`` owns
copying and integrity checking; this module only evaluates numerical agreement
between an already frozen reference tree and a candidate run tree.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import yaml


class ReconstructionRegressionError(RuntimeError):
    """Raised when a regression specification or input payload is invalid."""


_ARRAY_SUFFIXES = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_KINDS = {"exact", "numeric", "depth", "normal", "file_sha256"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except OSError as exc:
        raise ReconstructionRegressionError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ReconstructionRegressionError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReconstructionRegressionError(f"{path} must contain a YAML mapping")
    return value


def _safe_relative(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReconstructionRegressionError(f"{label} must be a non-empty path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ReconstructionRegressionError(
            f"{label} must be a safe relative path: {value}"
        )
    return path


def _resolve_input(root: Path, value: Any, *, label: str) -> Path:
    relative = _safe_relative(value, label=label)
    resolved_root = root.expanduser().resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ReconstructionRegressionError(
            f"{label} resolves outside {resolved_root}: {value}"
        ) from exc
    if not resolved.is_file():
        raise ReconstructionRegressionError(f"missing {label}: {resolved}")
    return resolved


def _validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise ReconstructionRegressionError("schema_version must be 1")
    if not isinstance(spec.get("regression_id"), str) or not spec["regression_id"]:
        raise ReconstructionRegressionError("regression_id must be a non-empty string")
    baseline = spec.get("baseline")
    if not isinstance(baseline, dict):
        raise ReconstructionRegressionError("baseline must be a mapping")
    for key in ("sandygs_revision", "solver_revision", "repeat_count"):
        if key not in baseline:
            raise ReconstructionRegressionError(f"baseline.{key} is required")
    for key in ("sandygs_revision", "solver_revision"):
        revision = baseline[key]
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision.lower())
        ):
            raise ReconstructionRegressionError(
                f"baseline.{key} must be a full 40-character Git commit"
            )
    repeat_count = baseline["repeat_count"]
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count < 1:
        raise ReconstructionRegressionError("baseline.repeat_count must be a positive integer")
    environment = baseline.get("environment")
    if not isinstance(environment, dict):
        raise ReconstructionRegressionError("baseline.environment must be a mapping")
    for key in (
        "python",
        "pytorch",
        "cuda",
        "gpu",
        "driver",
        "compiler",
        "build_options",
        "solver_threads",
    ):
        if key not in environment:
            raise ReconstructionRegressionError(f"baseline.environment.{key} is required")
    solver_threads = environment["solver_threads"]
    if (
        isinstance(solver_threads, bool)
        or not isinstance(solver_threads, int)
        or solver_threads < 1
    ):
        raise ReconstructionRegressionError(
            "baseline.environment.solver_threads must be a positive integer"
        )
    resources = baseline.get("resources")
    if not isinstance(resources, list):
        raise ReconstructionRegressionError("baseline.resources must be a list")
    resource_ids: set[str] = set()
    for index, resource in enumerate(resources):
        label = f"baseline.resources[{index}]"
        if not isinstance(resource, dict):
            raise ReconstructionRegressionError(f"{label} must be a mapping")
        identifier = resource.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ReconstructionRegressionError(f"{label}.id must be non-empty")
        if identifier in resource_ids:
            raise ReconstructionRegressionError(f"duplicate resource id: {identifier}")
        resource_ids.add(identifier)
        if not isinstance(resource.get("path"), str) or not resource["path"]:
            raise ReconstructionRegressionError(f"{label}.path must be non-empty")
        digest = resource.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise ReconstructionRegressionError(f"{label}.sha256 must be a SHA-256 digest")

    comparisons = spec.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ReconstructionRegressionError("comparisons must be a non-empty list")
    identifiers: set[str] = set()
    for index, comparison in enumerate(comparisons):
        label = f"comparisons[{index}]"
        if not isinstance(comparison, dict):
            raise ReconstructionRegressionError(f"{label} must be a mapping")
        identifier = comparison.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ReconstructionRegressionError(f"{label}.id must be non-empty")
        if identifier in identifiers:
            raise ReconstructionRegressionError(f"duplicate comparison id: {identifier}")
        identifiers.add(identifier)
        kind = comparison.get("kind")
        if kind not in _KINDS:
            raise ReconstructionRegressionError(
                f"{label}.kind must be one of {sorted(_KINDS)}"
            )
        _safe_relative(comparison.get("reference"), label=f"{label}.reference")
        _safe_relative(comparison.get("candidate"), label=f"{label}.candidate")
        if not isinstance(comparison.get("case"), dict):
            raise ReconstructionRegressionError(f"{label}.case must be a mapping")
        if kind != "file_sha256":
            for key in ("field", "units", "dtype"):
                if not isinstance(comparison.get(key), str) or not comparison[key]:
                    raise ReconstructionRegressionError(f"{label}.{key} must be non-empty")
            shape = comparison.get("shape")
            if (
                not isinstance(shape, list)
                or not shape
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item < 0
                    for item in shape
                )
            ):
                raise ReconstructionRegressionError(
                    f"{label}.shape must be a non-empty list of non-negative integers"
                )
        if kind in {"numeric", "depth"}:
            _validate_tolerances(comparison.get("tolerances"), label=label)
        if kind == "normal":
            tolerances = comparison.get("tolerances", {})
            if not isinstance(tolerances, dict):
                raise ReconstructionRegressionError(f"{label}.tolerances must be a mapping")
            allowed = {"max_angle_degrees", "mean_angle_degrees"}
            unknown = set(tolerances) - allowed
            if unknown:
                raise ReconstructionRegressionError(
                    f"{label}.tolerances has unsupported keys: {sorted(unknown)}"
                )
            for key in ("max_angle_degrees", "mean_angle_degrees"):
                _nonnegative_float(tolerances.get(key, 0.0), label=f"{label}.{key}")
        if kind in {"exact", "file_sha256"} and "tolerances" in comparison:
            raise ReconstructionRegressionError(f"{label}.tolerances is unused for {kind}")
        validity = comparison.get("validity")
        if validity is not None:
            if kind not in {"depth", "normal"}:
                raise ReconstructionRegressionError(
                    f"{label}.validity is only valid for depth or normal"
                )
            if not isinstance(validity, dict):
                raise ReconstructionRegressionError(f"{label}.validity must be a mapping")
            _safe_relative(validity.get("reference"), label=f"{label}.validity.reference")
            _safe_relative(validity.get("candidate"), label=f"{label}.validity.candidate")
        regions = comparison.get("regions", {})
        if not isinstance(regions, dict):
            raise ReconstructionRegressionError(f"{label}.regions must be a mapping")
        if regions and kind != "numeric":
            raise ReconstructionRegressionError(
                f"{label}.regions is only valid for numeric comparisons"
            )
        for name, region in regions.items():
            if not isinstance(region, dict):
                raise ReconstructionRegressionError(f"{label}.regions.{name} must be a mapping")
            _safe_relative(
                region.get("reference"), label=f"{label}.regions.{name}.reference"
            )
            _safe_relative(
                region.get("candidate"), label=f"{label}.regions.{name}.candidate"
            )
            if "tolerances" in region:
                _validate_tolerances(
                    region["tolerances"], label=f"{label}.regions.{name}"
                )
        if "valid_min" in comparison and kind != "depth":
            raise ReconstructionRegressionError(
                f"{label}.valid_min is only valid for depth comparisons"
            )


def _nonnegative_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ReconstructionRegressionError(f"{label} must be a non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReconstructionRegressionError(
            f"{label} must be a non-negative number"
        ) from exc
    if not np.isfinite(result) or result < 0.0:
        raise ReconstructionRegressionError(f"{label} must be a non-negative number")
    return result


def _validate_tolerances(value: Any, *, label: str) -> dict[str, float]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ReconstructionRegressionError(f"{label}.tolerances must be a mapping")
    allowed = {"max_abs", "mean_abs", "max_rel", "relative_floor"}
    unknown = set(value) - allowed
    if unknown:
        raise ReconstructionRegressionError(
            f"{label}.tolerances has unsupported keys: {sorted(unknown)}"
        )
    result = {
        "max_abs": _nonnegative_float(value.get("max_abs", 0.0), label=f"{label}.max_abs"),
        "mean_abs": _nonnegative_float(value.get("mean_abs", 0.0), label=f"{label}.mean_abs"),
        "max_rel": _nonnegative_float(value.get("max_rel", 0.0), label=f"{label}.max_rel"),
        "relative_floor": _nonnegative_float(
            value.get("relative_floor", 1.0e-12), label=f"{label}.relative_floor"
        ),
    }
    if result["relative_floor"] == 0.0:
        raise ReconstructionRegressionError(f"{label}.relative_floor must be positive")
    return result


def load_reconstruction_regression_spec(path: Path) -> dict[str, Any]:
    """Load and validate a versioned reconstruction-regression specification."""

    spec = _load_yaml(path)
    _validate_spec(spec)
    return spec


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix not in _ARRAY_SUFFIXES:
        raise ReconstructionRegressionError(
            f"unsupported array format {path.suffix!r} for {path}; use NPY or an image"
        )
    try:
        if suffix == ".npy":
            value = np.load(path, allow_pickle=False)
        else:
            with Image.open(path) as image:
                value = np.array(image)
    except (OSError, ValueError) as exc:
        raise ReconstructionRegressionError(f"cannot load array {path}: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise ReconstructionRegressionError(f"{path} did not contain an ndarray")
    return value


def _metadata(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    return {
        "reference": {
            "dtype": str(reference.dtype),
            "shape": list(reference.shape),
        },
        "candidate": {
            "dtype": str(candidate.dtype),
            "shape": list(candidate.shape),
        },
        "same_dtype": reference.dtype == candidate.dtype,
        "same_shape": reference.shape == candidate.shape,
    }


def _apply_array_contract(
    comparison: dict[str, Any], detail: dict[str, Any]
) -> dict[str, Any]:
    expected_dtype = comparison["dtype"]
    expected_shape = comparison["shape"]
    contract = {
        "expected_dtype": expected_dtype,
        "expected_shape": expected_shape,
        "reference_matches": bool(
            detail["reference"]["dtype"] == expected_dtype
            and detail["reference"]["shape"] == expected_shape
        ),
        "candidate_matches": bool(
            detail["candidate"]["dtype"] == expected_dtype
            and detail["candidate"]["shape"] == expected_shape
        ),
    }
    contract["passed"] = bool(contract["reference_matches"] and contract["candidate_matches"])
    detail["contract"] = contract
    detail["passed"] = bool(detail["passed"] and contract["passed"])
    return detail


def _first_locations(mask: np.ndarray, *, limit: int = 16) -> list[list[int]]:
    if mask.ndim == 0:
        return [[]] if bool(mask) else []
    return np.argwhere(mask)[:limit].astype(int).tolist()


def _exact_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    result = _metadata(reference, candidate)
    if reference.shape != candidate.shape:
        result.update({"passed": False, "mismatch_count": None, "first_mismatches": []})
        return result
    mismatches = np.not_equal(reference, candidate)
    mismatch_count = int(np.count_nonzero(mismatches))
    result.update(
        {
            "passed": bool(result["same_dtype"] and mismatch_count == 0),
            "mismatch_count": mismatch_count,
            "first_mismatches": _first_locations(mismatches),
        }
    )
    return result


def _numeric_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    selection: np.ndarray,
    tolerances: dict[str, float],
) -> dict[str, Any]:
    selected_count = int(np.count_nonzero(selection))
    if selected_count == 0:
        max_abs = mean_abs = max_rel = 0.0
        nonfinite_count = 0
    else:
        ref = np.asarray(reference[selection], dtype=np.float64)
        cand = np.asarray(candidate[selection], dtype=np.float64)
        finite = np.isfinite(ref) & np.isfinite(cand)
        nonfinite_count = int(selected_count - np.count_nonzero(finite))
        if np.any(finite):
            absolute = np.abs(cand[finite] - ref[finite])
            relative = absolute / np.maximum(
                np.abs(ref[finite]), tolerances["relative_floor"]
            )
            max_abs = float(np.max(absolute))
            mean_abs = float(np.mean(absolute))
            max_rel = float(np.max(relative))
        else:
            max_abs = mean_abs = max_rel = 0.0
    passed = bool(
        nonfinite_count == 0
        and max_abs <= tolerances["max_abs"]
        and mean_abs <= tolerances["mean_abs"]
        and max_rel <= tolerances["max_rel"]
    )
    return {
        "passed": passed,
        "sample_count": selected_count,
        "nonfinite_count": nonfinite_count,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "tolerances": tolerances,
    }


def _normalise_axis(axis: Any, ndim: int, *, label: str) -> int:
    try:
        result = int(axis)
    except (TypeError, ValueError) as exc:
        raise ReconstructionRegressionError(f"{label} must be an integer") from exc
    if result < 0:
        result += ndim
    if result < 0 or result >= ndim:
        raise ReconstructionRegressionError(f"{label} is outside an array with {ndim} axes")
    return result


def _mask_for_array(
    mask: np.ndarray, array: np.ndarray, *, channel_axis: int | None
) -> np.ndarray:
    if mask.dtype != np.bool_:
        if not np.issubdtype(mask.dtype, np.number) or not np.all(
            (mask == 0) | (mask == 1)
        ):
            raise ReconstructionRegressionError(
                "validity and region masks must be boolean or binary"
            )
        mask = mask.astype(bool)
    if mask.shape == array.shape:
        return mask
    if channel_axis is not None:
        expected = array.shape[:channel_axis] + array.shape[channel_axis + 1 :]
        if (
            mask.ndim == array.ndim
            and mask.shape[channel_axis] == 1
            and mask.shape[:channel_axis] + mask.shape[channel_axis + 1 :] == expected
        ):
            mask = np.squeeze(mask, axis=channel_axis)
        if mask.shape == expected:
            return np.broadcast_to(np.expand_dims(mask, channel_axis), array.shape)
    elif mask.ndim == array.ndim + 1:
        matching = [
            axis
            for axis, size in enumerate(mask.shape)
            if size == 1 and np.squeeze(mask, axis=axis).shape == array.shape
        ]
        if len(matching) == 1:
            return np.squeeze(mask, axis=matching[0])
    raise ReconstructionRegressionError(
        f"mask shape {list(mask.shape)} is incompatible with array shape {list(array.shape)}"
    )


def _load_pair(
    comparison: dict[str, Any], reference_root: Path, candidate_root: Path
) -> tuple[np.ndarray, np.ndarray]:
    identifier = comparison["id"]
    reference_path = _resolve_input(
        reference_root, comparison["reference"], label=f"{identifier}.reference"
    )
    candidate_path = _resolve_input(
        candidate_root, comparison["candidate"], label=f"{identifier}.candidate"
    )
    return _load_array(reference_path), _load_array(candidate_path)


def _compare_numeric(
    comparison: dict[str, Any], reference_root: Path, candidate_root: Path
) -> dict[str, Any]:
    reference, candidate = _load_pair(comparison, reference_root, candidate_root)
    result = _metadata(reference, candidate)
    if reference.shape != candidate.shape:
        result.update({"passed": False, "metrics": None, "regions": {}})
        return result
    if not np.issubdtype(reference.dtype, np.number) or not np.issubdtype(
        candidate.dtype, np.number
    ):
        raise ReconstructionRegressionError(f"{comparison['id']} requires numeric arrays")
    tolerances = _validate_tolerances(comparison.get("tolerances"), label=comparison["id"])
    selection = np.ones(reference.shape, dtype=bool)
    metrics = _numeric_metrics(reference, candidate, selection, tolerances)
    passed = bool(metrics["passed"] and result["same_dtype"])
    regions_report: dict[str, Any] = {}
    regions = comparison.get("regions", {})
    if not isinstance(regions, dict):
        raise ReconstructionRegressionError(f"{comparison['id']}.regions must be a mapping")
    channel_axis: int | None = None
    if regions:
        channel_axis = _normalise_axis(
            comparison.get("channel_axis", -1), reference.ndim, label="channel_axis"
        )
    for name, region in regions.items():
        if not isinstance(region, dict):
            raise ReconstructionRegressionError(f"region {name} must be a mapping")
        ref_mask = _load_array(
            _resolve_input(
                reference_root,
                region.get("reference"),
                label=f"region {name}.reference",
            )
        )
        cand_mask = _load_array(
            _resolve_input(
                candidate_root,
                region.get("candidate"),
                label=f"region {name}.candidate",
            )
        )
        mask_exact = _exact_arrays(ref_mask, cand_mask)
        ref_selection = _mask_for_array(ref_mask, reference, channel_axis=channel_axis)
        _mask_for_array(cand_mask, candidate, channel_axis=channel_axis)
        region_tolerances = _validate_tolerances(
            region.get("tolerances", comparison.get("tolerances")),
            label=f"{comparison['id']}.regions.{name}",
        )
        region_metrics = _numeric_metrics(
            reference, candidate, ref_selection, region_tolerances
        )
        region_passed = bool(mask_exact["passed"] and region_metrics["passed"])
        regions_report[str(name)] = {
            "passed": region_passed,
            "mask": mask_exact,
            "metrics": region_metrics,
        }
        passed = passed and region_passed
    result.update({"passed": passed, "metrics": metrics, "regions": regions_report})
    return result


def _validity_pair(
    comparison: dict[str, Any],
    reference: np.ndarray,
    candidate: np.ndarray,
    reference_root: Path,
    candidate_root: Path,
    *,
    channel_axis: int | None,
    inferred_reference: np.ndarray,
    inferred_candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    validity = comparison.get("validity")
    if validity is None:
        reference_mask = inferred_reference
        candidate_mask = inferred_candidate
        source = "inferred"
    else:
        if not isinstance(validity, dict):
            raise ReconstructionRegressionError(
                f"{comparison['id']}.validity must be a mapping"
            )
        reference_mask = _load_array(
            _resolve_input(
                reference_root,
                validity.get("reference"),
                label=f"{comparison['id']}.validity.reference",
            )
        )
        candidate_mask = _load_array(
            _resolve_input(
                candidate_root,
                validity.get("candidate"),
                label=f"{comparison['id']}.validity.candidate",
            )
        )
        source = "explicit"
    exact = _exact_arrays(reference_mask, candidate_mask)
    reference_mask = _mask_for_array(reference_mask, reference, channel_axis=channel_axis)
    candidate_mask = _mask_for_array(candidate_mask, candidate, channel_axis=channel_axis)
    mismatch = np.not_equal(reference_mask, candidate_mask)
    report = {
        "source": source,
        "passed": exact["passed"],
        "same_dtype": exact["same_dtype"],
        "same_shape": exact["same_shape"],
        "reference_valid_count": int(np.count_nonzero(reference_mask)),
        "candidate_valid_count": int(np.count_nonzero(candidate_mask)),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "first_mismatches": _first_locations(mismatch),
    }
    return reference_mask, candidate_mask, report


def _compare_depth(
    comparison: dict[str, Any], reference_root: Path, candidate_root: Path
) -> dict[str, Any]:
    reference, candidate = _load_pair(comparison, reference_root, candidate_root)
    result = _metadata(reference, candidate)
    if reference.shape != candidate.shape:
        result.update({"passed": False, "validity": None, "metrics": None})
        return result
    if not np.issubdtype(reference.dtype, np.number) or not np.issubdtype(
        candidate.dtype, np.number
    ):
        raise ReconstructionRegressionError(f"{comparison['id']} requires numeric arrays")
    valid_min = float(comparison.get("valid_min", 0.0))
    inferred_reference = np.isfinite(reference) & (reference > valid_min)
    inferred_candidate = np.isfinite(candidate) & (candidate > valid_min)
    reference_mask, candidate_mask, validity = _validity_pair(
        comparison,
        reference,
        candidate,
        reference_root,
        candidate_root,
        channel_axis=None,
        inferred_reference=inferred_reference,
        inferred_candidate=inferred_candidate,
    )
    tolerances = _validate_tolerances(comparison.get("tolerances"), label=comparison["id"])
    metrics = _numeric_metrics(reference, candidate, reference_mask & candidate_mask, tolerances)
    result.update(
        {
            "passed": bool(
                result["same_dtype"] and validity["passed"] and metrics["passed"]
            ),
            "validity": validity,
            "metrics": metrics,
        }
    )
    return result


def _compare_normal(
    comparison: dict[str, Any], reference_root: Path, candidate_root: Path
) -> dict[str, Any]:
    reference, candidate = _load_pair(comparison, reference_root, candidate_root)
    result = _metadata(reference, candidate)
    if reference.shape != candidate.shape:
        result.update({"passed": False, "validity": None, "metrics": None})
        return result
    if not np.issubdtype(reference.dtype, np.number) or not np.issubdtype(
        candidate.dtype, np.number
    ):
        raise ReconstructionRegressionError(f"{comparison['id']} requires numeric arrays")
    channel_axis = _normalise_axis(
        comparison.get("channel_axis", -1), reference.ndim, label="channel_axis"
    )
    ref_vectors = np.moveaxis(reference.astype(np.float64), channel_axis, -1)
    cand_vectors = np.moveaxis(candidate.astype(np.float64), channel_axis, -1)
    if ref_vectors.shape[-1] != 3:
        raise ReconstructionRegressionError(
            f"{comparison['id']} normal channel must contain three components"
        )
    ref_norm = np.linalg.norm(ref_vectors, axis=-1)
    cand_norm = np.linalg.norm(cand_vectors, axis=-1)
    inferred_reference = np.all(np.isfinite(ref_vectors), axis=-1) & (ref_norm > 1.0e-12)
    inferred_candidate = np.all(np.isfinite(cand_vectors), axis=-1) & (cand_norm > 1.0e-12)
    # Validity masks describe pixels, so compare them against the arrays with
    # the vector channel removed rather than broadcasting them over channels.
    reference_pixels = np.empty(ref_vectors.shape[:-1], dtype=np.uint8)
    candidate_pixels = np.empty(cand_vectors.shape[:-1], dtype=np.uint8)
    ref_valid, cand_valid, validity = _validity_pair(
        comparison,
        reference_pixels,
        candidate_pixels,
        reference_root,
        candidate_root,
        channel_axis=None,
        inferred_reference=inferred_reference,
        inferred_candidate=inferred_candidate,
    )
    common = ref_valid & cand_valid
    count = int(np.count_nonzero(common))
    if count:
        ref_unit = ref_vectors[common] / ref_norm[common, None]
        cand_unit = cand_vectors[common] / cand_norm[common, None]
        cosine = np.clip(np.sum(ref_unit * cand_unit, axis=-1), -1.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
        exactly_equal = np.all(ref_vectors[common] == cand_vectors[common], axis=-1)
        angles[exactly_equal] = 0.0
        max_angle = float(np.max(angles))
        mean_angle = float(np.mean(angles))
        nonfinite_count = int(np.count_nonzero(~np.isfinite(angles)))
    else:
        max_angle = mean_angle = 0.0
        nonfinite_count = 0
    tolerance_values = comparison.get("tolerances", {})
    max_tolerance = _nonnegative_float(
        tolerance_values.get("max_angle_degrees", 0.0), label="max_angle_degrees"
    )
    mean_tolerance = _nonnegative_float(
        tolerance_values.get("mean_angle_degrees", 0.0), label="mean_angle_degrees"
    )
    metrics = {
        "sample_count": count,
        "nonfinite_count": nonfinite_count,
        "max_angle_degrees": max_angle,
        "mean_angle_degrees": mean_angle,
        "tolerances": {
            "max_angle_degrees": max_tolerance,
            "mean_angle_degrees": mean_tolerance,
        },
    }
    metrics["passed"] = bool(
        nonfinite_count == 0
        and max_angle <= max_tolerance
        and mean_angle <= mean_tolerance
    )
    result.update(
        {
            "passed": bool(
                result["same_dtype"]
                and validity["passed"]
                and metrics["passed"]
            ),
            "validity": validity,
            "metrics": metrics,
        }
    )
    return result


def _compare_one(
    comparison: dict[str, Any], reference_root: Path, candidate_root: Path
) -> dict[str, Any]:
    kind = comparison["kind"]
    if kind == "file_sha256":
        reference = _resolve_input(
            reference_root, comparison["reference"], label=f"{comparison['id']}.reference"
        )
        candidate = _resolve_input(
            candidate_root, comparison["candidate"], label=f"{comparison['id']}.candidate"
        )
        reference_digest = _sha256(reference)
        candidate_digest = _sha256(candidate)
        detail = {
            "passed": reference_digest == candidate_digest,
            "reference_sha256": reference_digest,
            "candidate_sha256": candidate_digest,
        }
    elif kind == "exact":
        detail = _exact_arrays(*_load_pair(comparison, reference_root, candidate_root))
    elif kind == "numeric":
        detail = _compare_numeric(comparison, reference_root, candidate_root)
    elif kind == "depth":
        detail = _compare_depth(comparison, reference_root, candidate_root)
    elif kind == "normal":
        detail = _compare_normal(comparison, reference_root, candidate_root)
    else:  # Protected by specification validation.
        raise AssertionError(kind)
    if kind != "file_sha256":
        detail = _apply_array_contract(comparison, detail)
    return {
        "id": comparison["id"],
        "kind": kind,
        "reference": comparison["reference"],
        "candidate": comparison["candidate"],
        "field": comparison.get("field"),
        "units": comparison.get("units"),
        "dtype": comparison.get("dtype"),
        "shape": comparison.get("shape"),
        "case": comparison.get("case"),
        **detail,
    }


def compare_reconstruction_regression(
    *, spec_path: Path, reference_root: Path, candidate_root: Path
) -> dict[str, Any]:
    """Compare all fields in ``spec_path`` and return a serialisable report."""

    spec_path = spec_path.expanduser().resolve()
    spec = load_reconstruction_regression_spec(spec_path)
    reference_root = reference_root.expanduser().resolve()
    candidate_root = candidate_root.expanduser().resolve()
    if not reference_root.is_dir():
        raise ReconstructionRegressionError(
            f"reference root is not a directory: {reference_root}"
        )
    if not candidate_root.is_dir():
        raise ReconstructionRegressionError(
            f"candidate root is not a directory: {candidate_root}"
        )
    comparisons = [
        _compare_one(item, reference_root, candidate_root)
        for item in spec["comparisons"]
    ]
    failed = [item["id"] for item in comparisons if not item["passed"]]
    return {
        "schema_version": 1,
        "regression_id": spec["regression_id"],
        "status": "passed" if not failed else "failed",
        "spec": {"path": str(spec_path), "sha256": _sha256(spec_path)},
        "reference_root": str(reference_root),
        "candidate_root": str(candidate_root),
        "baseline": spec["baseline"],
        "comparison_count": len(comparisons),
        "failed_count": len(failed),
        "failed_comparisons": failed,
        "comparisons": comparisons,
    }


def write_reconstruction_regression_report(report: dict[str, Any], output: Path) -> Path:
    """Write a report once; existing reports are never overwritten implicitly."""

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(report, stream, sort_keys=False, allow_unicode=True)
    except FileExistsError as exc:
        raise ReconstructionRegressionError(
            f"refusing to overwrite existing report: {output}"
        ) from exc
    except OSError as exc:
        raise ReconstructionRegressionError(f"cannot write {output}: {exc}") from exc
    return output
