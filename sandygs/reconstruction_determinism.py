"""Measure intrinsic variation across repeated reconstruction runs.

This module is intentionally tolerance-free.  It reports the variation that
already exists between pre-refactor runs; choosing and freezing tolerances is
a separate, reviewed step.
"""

from __future__ import annotations

import csv
import hashlib
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class ReconstructionDeterminismError(RuntimeError):
    """Raised when repeated-run inputs cannot be audited safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_locations(mask: np.ndarray, limit: int = 16) -> list[list[int]]:
    if mask.ndim == 0:
        return [[]] if bool(mask) else []
    return np.argwhere(mask)[:limit].astype(int).tolist()


def _numeric_difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    finite = np.isfinite(ref) & np.isfinite(cand)
    nonfinite_mismatch = np.not_equal(np.isfinite(ref), np.isfinite(cand))
    if np.any(finite):
        absolute = np.abs(cand[finite] - ref[finite])
        relative = absolute / np.maximum(np.abs(ref[finite]), 1.0e-12)
        maximum_absolute = float(np.max(absolute))
        mean_absolute = float(np.mean(absolute))
        maximum_relative = float(np.max(relative))
    else:
        maximum_absolute = mean_absolute = maximum_relative = 0.0
    mismatches = np.not_equal(reference, candidate)
    return {
        "byte_identical": bool(reference.tobytes() == candidate.tobytes()),
        "value_identical": bool(not np.any(mismatches)),
        "mismatch_count": int(np.count_nonzero(mismatches)),
        "first_mismatches": _first_locations(mismatches),
        "nonfinite_mismatch_count": int(np.count_nonzero(nonfinite_mismatch)),
        "max_abs": maximum_absolute,
        "mean_abs": mean_absolute,
        "max_rel": maximum_relative,
    }


def _mask_difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.bool_)
    cand = np.asarray(candidate, dtype=np.bool_)
    mismatch = np.not_equal(ref, cand)
    return {
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "first_mismatches": _first_locations(mismatch),
        "reference_valid_count": int(np.count_nonzero(ref)),
        "candidate_valid_count": int(np.count_nonzero(cand)),
    }


def _normal_axis(array: np.ndarray) -> int | None:
    axes = [index for index, size in enumerate(array.shape) if size == 3]
    return axes[0] if len(axes) == 1 else None


def _normal_difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    axis = _normal_axis(reference)
    if axis is None:
        return {"normal_metrics_available": False}
    ref = np.moveaxis(np.asarray(reference, dtype=np.float64), axis, -1)
    cand = np.moveaxis(np.asarray(candidate, dtype=np.float64), axis, -1)
    ref_norm = np.linalg.norm(ref, axis=-1)
    cand_norm = np.linalg.norm(cand, axis=-1)
    ref_valid = np.all(np.isfinite(ref), axis=-1) & (ref_norm > 0.0)
    cand_valid = np.all(np.isfinite(cand), axis=-1) & (cand_norm > 0.0)
    common = ref_valid & cand_valid
    if np.any(common):
        dot = np.sum(ref[common] * cand[common], axis=-1)
        cosine = dot / (ref_norm[common] * cand_norm[common])
        angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        mean_angle = float(np.mean(angles))
        max_angle = float(np.max(angles))
    else:
        mean_angle = max_angle = 0.0
    return {
        "normal_metrics_available": True,
        "validity": _mask_difference(ref_valid, cand_valid),
        "common_valid_count": int(np.count_nonzero(common)),
        "mean_angle_degrees": mean_angle,
        "max_angle_degrees": max_angle,
    }


def _depth_difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref_valid = np.isfinite(reference) & (reference > 0.0)
    cand_valid = np.isfinite(candidate) & (candidate > 0.0)
    common = ref_valid & cand_valid
    metrics = _numeric_difference(reference[common], candidate[common])
    return {
        "validity": _mask_difference(ref_valid, cand_valid),
        "common_valid_count": int(np.count_nonzero(common)),
        "common_valid_metrics": metrics,
    }


def _coverage_difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    contour = (
        ((reference > 0.0) & (reference < 1.0))
        | ((candidate > 0.0) & (candidate < 1.0))
    )
    return {
        "contour_count": int(np.count_nonzero(contour)),
        "contour_metrics": _numeric_difference(reference[contour], candidate[contour]),
    }


def _broadcast_region_mask(mask: np.ndarray, array: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=np.bool_)
    if value.shape == array.shape:
        return value
    if value.ndim == array.ndim and value.shape[0] == 1:
        value = np.squeeze(value, axis=0)
    if value.shape == array.shape[1:]:
        return np.broadcast_to(value[np.newaxis, ...], array.shape)
    if value.shape == array.shape[:-1]:
        return np.broadcast_to(value[..., np.newaxis], array.shape)
    raise ReconstructionDeterminismError(
        f"region mask shape {mask.shape} is incompatible with array shape {array.shape}"
    )


def _raw_sibling(relative: str, field: str) -> str | None:
    parts = Path(relative).parts
    try:
        marker = parts.index("regression_raw")
    except ValueError:
        return None
    if marker + 2 >= len(parts):
        return None
    return Path(*parts[: marker + 1], field, parts[-1]).as_posix()


def _region_differences(
    relative: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    left_files: dict[str, Path],
    right_files: dict[str, Path],
) -> dict[str, Any]:
    regions: dict[str, Any] = {}
    for name, field in (
        ("sand", "sand_surface_valid"),
        ("water", "water_surface_valid"),
    ):
        sibling = _raw_sibling(relative, field)
        if sibling is None or sibling not in left_files or sibling not in right_files:
            continue
        ref_mask = _load_array(left_files[sibling])
        cand_mask = _load_array(right_files[sibling])
        validity = _mask_difference(ref_mask, cand_mask)
        ref_selection = _broadcast_region_mask(ref_mask, reference)
        cand_selection = _broadcast_region_mask(cand_mask, candidate)
        common = ref_selection & cand_selection
        regions[name] = {
            "mask_path": sibling,
            "validity": validity,
            "common_selected_count": int(np.count_nonzero(common)),
            "common_metrics": _numeric_difference(
                reference[common], candidate[common]
            ),
        }
    return regions


def _load_array(path: Path) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ReconstructionDeterminismError(f"cannot load NPY array {path}: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise ReconstructionDeterminismError(f"NPY did not contain an ndarray: {path}")
    return value


def _compare_array(
    reference_path: Path,
    candidate_path: Path,
    *,
    relative: str,
    left_files: dict[str, Path],
    right_files: dict[str, Path],
) -> dict[str, Any]:
    reference = _load_array(reference_path)
    candidate = _load_array(candidate_path)
    result: dict[str, Any] = {
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "same_dtype": bool(reference.dtype == candidate.dtype),
        "same_shape": bool(reference.shape == candidate.shape),
    }
    if reference.shape != candidate.shape:
        result.update({"exact": False, "classification": "shape_mismatch"})
        return result
    # Raw renderer arrays use the frame identifier as the file name and the
    # semantic field as the parent directory.  Solver arrays usually carry
    # the field in the file name, so inspect both forms.
    name = f"{reference_path.parent.name}/{reference_path.stem}".lower()
    exact_field = (
        reference.dtype.kind in "biu"
        or any(token in name for token in ("owner", "valid", "mask", "label"))
    )
    result["classification"] = "exact" if exact_field else "numeric"
    result["numeric"] = _numeric_difference(reference, candidate)
    result["exact"] = bool(
        result["same_dtype"] and result["numeric"]["value_identical"]
    )
    if "depth" in name:
        result["depth"] = _depth_difference(reference, candidate)
    if "normal" in name:
        result["normal"] = _normal_difference(reference, candidate)
    if "coverage" in name:
        result["coverage"] = _coverage_difference(reference, candidate)
    if (
        reference.dtype.kind == "f"
        and candidate.dtype.kind == "f"
        and reference.ndim >= 3
        and reference.shape == candidate.shape
    ):
        regions = _region_differences(
            relative,
            reference,
            candidate,
            left_files,
            right_files,
        )
        if regions:
            result["regions"] = regions
    return result


def _read_metrics(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ReconstructionDeterminismError(f"metrics CSV has no header: {path}")
            return list(reader.fieldnames), list(reader)
    except OSError as exc:
        raise ReconstructionDeterminismError(f"cannot read metrics CSV {path}: {exc}") from exc


def _compare_metrics(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    ref_fields, ref_rows = _read_metrics(reference_path)
    cand_fields, cand_rows = _read_metrics(candidate_path)
    result: dict[str, Any] = {
        "same_columns": ref_fields == cand_fields,
        "reference_row_count": len(ref_rows),
        "candidate_row_count": len(cand_rows),
        "same_row_count": len(ref_rows) == len(cand_rows),
        "columns": {},
    }
    if ref_fields != cand_fields or len(ref_rows) != len(cand_rows):
        result["exact"] = False
        return result
    exact = True
    for field in ref_fields:
        ref_text = [row[field] for row in ref_rows]
        cand_text = [row[field] for row in cand_rows]
        try:
            ref = np.asarray([float(value) for value in ref_text], dtype=np.float64)
            cand = np.asarray([float(value) for value in cand_text], dtype=np.float64)
        except ValueError:
            mismatch = np.not_equal(ref_text, cand_text)
            detail = {
                "kind": "text",
                "mismatch_count": int(np.count_nonzero(mismatch)),
                "first_mismatch_rows": np.flatnonzero(mismatch)[:16].astype(int).tolist(),
            }
            column_exact = detail["mismatch_count"] == 0
        else:
            detail = {"kind": "numeric", **_numeric_difference(ref, cand)}
            column_exact = detail["value_identical"]
        detail["exact"] = bool(column_exact)
        result["columns"][field] = detail
        exact = exact and bool(column_exact)
    result["exact"] = bool(exact)
    return result


def _selected_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or (path.suffix != ".npy" and path.name != "metrics.csv"):
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = path
    return files


def audit_repeated_runs(run_roots: list[Path]) -> dict[str, Any]:
    """Compare every pair of run trees without applying acceptance tolerances."""

    if len(run_roots) < 2:
        raise ReconstructionDeterminismError("at least two repeated run roots are required")
    roots = [path.expanduser().resolve() for path in run_roots]
    if len(set(roots)) != len(roots):
        raise ReconstructionDeterminismError("repeated run roots must be distinct")
    for root in roots:
        if not root.is_dir():
            raise ReconstructionDeterminismError(f"run root is not a directory: {root}")
    inventories = [_selected_files(root) for root in roots]
    if any(not inventory for inventory in inventories):
        raise ReconstructionDeterminismError("each run root must contain NPY or metrics.csv files")

    pair_reports = []
    all_exact = True
    for left, right in combinations(range(len(roots)), 2):
        left_files = inventories[left]
        right_files = inventories[right]
        common = sorted(set(left_files) & set(right_files))
        missing_left = sorted(set(right_files) - set(left_files))
        missing_right = sorted(set(left_files) - set(right_files))
        details: dict[str, Any] = {}
        pair_exact = not missing_left and not missing_right
        for relative in common:
            if relative.endswith("metrics.csv"):
                detail = _compare_metrics(left_files[relative], right_files[relative])
            else:
                detail = _compare_array(
                    left_files[relative],
                    right_files[relative],
                    relative=relative,
                    left_files=left_files,
                    right_files=right_files,
                )
            details[relative] = detail
            pair_exact = pair_exact and bool(detail["exact"])
        pair_reports.append(
            {
                "left_repeat": left + 1,
                "right_repeat": right + 1,
                "exact": bool(pair_exact),
                "missing_from_left": missing_left,
                "missing_from_right": missing_right,
                "files": details,
            }
        )
        all_exact = all_exact and pair_exact
    return {
        "schema_version": 1,
        "repeat_count": len(roots),
        "run_roots": [str(root) for root in roots],
        "selected_file_counts": [len(inventory) for inventory in inventories],
        "all_pairs_exact": bool(all_exact),
        "pairs": pair_reports,
    }


def write_determinism_report(path: Path, report: dict[str, Any]) -> None:
    """Write one new YAML report without overwriting prior evidence."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(report, stream, sort_keys=False, allow_unicode=True)
    except FileExistsError as exc:
        raise ReconstructionDeterminismError(
            f"refusing to overwrite determinism report: {destination}"
        ) from exc
