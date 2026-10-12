"""E3: review an E2 selection, then make an independent PGSR checkpoint.

The source PLY's complete vertex records are partitioned byte for byte.  The
unselected records keep their original order, including all unknown labels.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from PIL import Image
import yaml

from .e2 import _observation_views
from .workspace import (ExistingSandError, _load_yaml, _sha256, _stage_location,
                        record_stage, verify_run)


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)


def _selection(run_root: Path, selection_id: str, check_id: str):
    run = _load_yaml(run_root / "run.yaml", "run manifest")
    selection = _stage_location(run_root, "segmentation", selection_id)
    check = _stage_location(run_root, "segmentation", check_id)
    for stage in (selection, check):
        if not (stage / "manifest.yaml").is_file():
            raise ExistingSandError(f"unregistered E2 stage: {stage}")
    record = _load_yaml(selection / "record.yaml", "E2 selection record")
    if record.get("prompt", {}).get("type") not in {
        "multiview_sam_depth_seeds", "multiview_sam_frequency", "saga_affinity_expansion"
    }:
        raise ExistingSandError("E3 needs an E2 tri-state selection stage")
    check_record = _load_yaml(check / "record.yaml", "E2 check record")
    if check_record.get("prompt") != {
        "type": "pgsr_rasterized_holdout", "seed_id": selection_id
    }:
        raise ExistingSandError("E3 needs a PGSR holdout check for this exact selection")
    parents = check_record.get("parents", [])
    if {"stage": "segmentation", "stage_id": selection_id} not in parents:
        raise ExistingSandError("E2 holdout check does not name selection as parent")
    holdout = _load_yaml(check / "quality.yaml", "E2 holdout quality").get("holdout_views")
    if not isinstance(holdout, list) or not holdout:
        raise ExistingSandError("E2 holdout check has no scored view")
    mask_id = record["prompt"]["mask_id"]
    mask = _stage_location(run_root, "segmentation", mask_id)
    if not (mask / "manifest.yaml").is_file():
        raise ExistingSandError("E2 SAM mask stage is not registered")
    mask_record = _load_yaml(mask / "record.yaml", "E2 mask record")
    if mask_record.get("prompt", {}).get("type") != "manual_sam_points_box":
        raise ExistingSandError("E3 needs an E2 prompted SAM mask stage")
    capture_id = mask_record["parameters"]["capture_id"]
    observation, views = _observation_views(run_root, capture_id)
    labels_path = selection / "gaussian_labels.npy"
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    n = run["source"]["gaussian_count"]
    if labels.shape != (n,) or labels.dtype != np.uint8 or not np.isin(labels, [0, 1, 2]).all():
        raise ExistingSandError("E2 labels must be uint8 0/1/2 in source PLY order")
    positive = np.flatnonzero(labels == 1).astype(np.uint64)
    if not len(positive) or len(positive) == n:
        raise ExistingSandError("E3 needs a nonempty deletion set and retained Gaussian set")
    return run, selection, check, mask, observation, views, labels, positive


def _ply_layout(path: Path, count: int) -> tuple[bytes, int]:
    from plyfile import PlyData

    ply = PlyData.read(str(path), mmap="r")
    if ply.text or ply.byte_order != "<" or len(ply.elements) != 1 or ply.elements[0].name != "vertex":
        raise ExistingSandError("E3 byte-preserving split needs a vertex-only binary little-endian PLY")
    vertex = ply["vertex"].data
    if len(vertex) != count or vertex.dtype.hasobject:
        raise ExistingSandError("source PLY vertex count or scalar property layout differs")
    with path.open("rb") as stream:
        header = bytearray()
        for _ in range(4096):
            line = stream.readline()
            header.extend(line)
            if line.strip() == b"end_header":
                break
            if not line or len(header) > 1024 * 1024:
                raise ExistingSandError("source PLY header is missing or too large")
        else:
            raise ExistingSandError("source PLY header is too long")
    if path.stat().st_size != len(header) + count * vertex.dtype.itemsize:
        raise ExistingSandError("source PLY has trailing data or a non-scalar vertex property")
    return bytes(header), vertex.dtype.itemsize


def _header_with_count(header: bytes, count: int) -> bytes:
    lines = header.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.startswith(b"element vertex ")]
    if len(matches) != 1:
        raise ExistingSandError("source PLY needs exactly one vertex element declaration")
    i = matches[0]
    end = b"\r\n" if lines[i].endswith(b"\r\n") else b"\n"
    lines[i] = f"element vertex {count}".encode() + end
    return b"".join(lines)


def _split_ply(source: Path, selected: Path, retained: Path, labels: np.ndarray,
               header: bytes, stride: int, *, chunk_size: int = 100000) -> None:
    if chunk_size < 1:
        raise ExistingSandError("chunk_size must be positive")
    raw = np.memmap(source, dtype=np.dtype(f"V{stride}"), mode="r",
                    offset=len(header), shape=labels.shape)
    with selected.open("xb") as gone, retained.open("xb") as keep:
        gone.write(_header_with_count(header, int(np.count_nonzero(labels == 1))))
        keep.write(_header_with_count(header, int(np.count_nonzero(labels != 1))))
        for start in range(0, len(labels), chunk_size):
            end = min(start + chunk_size, len(labels))
            chosen = labels[start:end] == 1
            gone.write(raw[start:end][chosen].tobytes())
            keep.write(raw[start:end][~chosen].tobytes())


def _review_binding(run: dict, selection_id: str, check_id: str,
                    labels_path: Path, positive: np.ndarray) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run["run_id"],
        "source_checkpoint_sha256": run["source"]["checkpoint_sha256"],
        "source_point_cloud_sha256": run["source"]["point_cloud_sha256"],
        "selection_id": selection_id,
        "check_id": check_id,
        "labels_sha256": _sha256(labels_path),
        "selected_indices_sha256": hashlib.sha256(positive.astype("<u8").tobytes()).hexdigest(),
        "selected_count": len(positive),
    }


def plan_derivation(run_root: Path, selection_id: str, check_id: str,
                    output: Path) -> dict[str, Any]:
    """Write a bound review template; no checkpoint files are copied."""
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    verify_run(run_root)
    run, selection, check, _mask, _obs, _views, labels, positive = _selection(
        run_root, selection_id, check_id)
    if output.exists():
        raise ExistingSandError(f"refusing to overwrite review file: {output}")
    source_root = Path(run["source"]["checkpoint"]).resolve()
    if output.is_relative_to(source_root) or output.is_relative_to(run_root):
        raise ExistingSandError("review file must be outside source checkpoint and run directory")
    cloud = Path(run["source"]["checkpoint"]) / run["source"]["point_cloud"]
    header, stride = _ply_layout(cloud, len(labels))
    counts = {"positive": len(positive), "negative": int(np.count_nonzero(labels == 0)),
              "unknown": int(np.count_nonzero(labels == 2))}
    review = {**_review_binding(run, selection_id, check_id,
                               selection / "gaussian_labels.npy", positive),
              "approved": False,
              "counts": counts,
              "estimated_ply_output_bytes": int(len(labels) * stride + 2 * len(header)),
              "heldout_quality": str(check / "quality.yaml"),
              "heldout_overlays": [str(p) for p in sorted(check.glob("view_*/overlay.png"))],
              "review_note": "Inspect E2 masks and red leakage in held-out overlays; set approved: true only for the selected indices."}
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(output, review)
    return {"review_file": str(output), "counts": counts,
            "estimated_ply_output_bytes": review["estimated_ply_output_bytes"]}


def _patched_cfg_args(source: Path, destination: Path, dataset: Path, model: Path) -> None:
    """Repair only path fields in a PGSR Namespace, without evaluating source text."""
    value = ast.parse(source.read_text(encoding="utf-8"), mode="eval").body
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or value.func.id != "Namespace" or value.args:
        raise ExistingSandError("PGSR cfg_args is not a Namespace keyword expression")
    fields = {}
    for item in value.keywords:
        if item.arg is None or item.arg in fields:
            raise ExistingSandError("PGSR cfg_args has duplicate or unpacked fields")
        fields[item.arg] = ast.literal_eval(item.value)
    if "model_path" not in fields or "source_path" not in fields:
        raise ExistingSandError("PGSR cfg_args lacks model_path/source_path")
    fields["model_path"] = str(model)
    fields["source_path"] = str(dataset)
    destination.write_text("Namespace(" + ", ".join(
        f"{key}={value!r}" for key, value in fields.items()) + ")\n", encoding="utf-8")


def _references(stage: Path, mask: Path, views: dict) -> tuple[list[dict[str, str]], list[str]]:
    quality = _load_yaml(mask / "quality.yaml", "E2 mask quality")
    artifacts = []
    cameras = []
    entries = quality.get("views", [])
    if not isinstance(entries, list) or not entries:
        raise ExistingSandError("E2 masks need one or more views")
    for i, entry in enumerate(entries):
        camera_id = entry["camera_id"]
        if camera_id not in views:
            raise ExistingSandError(f"E2 mask camera missing from observation: {camera_id}")
        source_view, camera = views[camera_id]
        folder = stage / "references" / f"view_{i:04d}"
        folder.mkdir(parents=True)
        cameras.append(camera_id)
        files = {"camera.yaml": source_view / "camera.yaml",
                 "source_rgb.png": source_view / "source_rgb.png",
                 "render_rgb.png": source_view / "render_rgb.png",
                 "mask.png": mask / entry["directory"] / "mask.png"}
        for name, path in files.items():
            shutil.copy2(path, folder / name)
            artifacts.append({"path": f"references/{folder.name}/{name}", "role": "pre_deletion_appearance_reference"})
        depth = np.load(source_view / "plane_depth.npy", allow_pickle=False)
        valid = np.load(source_view / "depth_valid.npy", allow_pickle=False)
        with Image.open(files["mask.png"]) as image:
            selected = np.asarray(image.convert("L")) >= 128
        if depth.shape != valid.shape or selected.shape != depth.shape:
            raise ExistingSandError(f"E2 reference mask/depth shape mismatch: {camera_id}")
        surface = np.where(selected & valid & np.isfinite(depth) & (depth > 0), depth, np.nan).astype(np.float32)
        np.save(folder / "sam_region_pgsr_depth.npy", surface, allow_pickle=False)
        artifacts.append({"path": f"references/{folder.name}/sam_region_pgsr_depth.npy",
                          "role": "pre_deletion_pgsr_depth_inside_sam_mask"})
    return artifacts, cameras


def derive_scene(run_root: Path, selection_id: str, check_id: str, decision: Path,
                 stage_id: str, *, repo_root: Path, chunk_size: int = 100000,
                 command: list[str] | None = None) -> dict[str, Any]:
    """Copy checkpoint resources and partition the source PLY after explicit review."""
    run_root, repo_root = Path(run_root).resolve(), Path(repo_root).resolve()
    decision = Path(decision).resolve()
    verify_run(run_root)
    run, selection, check, mask, observation, views, labels, positive = _selection(
        run_root, selection_id, check_id)
    expected = _review_binding(run, selection_id, check_id,
                               selection / "gaussian_labels.npy", positive)
    review = _load_yaml(decision, "E3 deletion review")
    if review.get("approved") is not True or any(review.get(k) != v for k, v in expected.items()):
        raise ExistingSandError("E3 review is not approved or source/selection binding changed")
    if chunk_size < 1:
        raise ExistingSandError("chunk_size must be positive")
    if not (Path(run["source"]["checkpoint"]) / "cfg_args").is_file():
        raise ExistingSandError("E3 needs a full PGSR checkpoint with cfg_args")
    final_stage = _stage_location(run_root, "background", stage_id)
    if final_stage.exists():
        raise ExistingSandError(f"refusing to overwrite derived scene: {final_stage}")
    source = Path(run["source"]["checkpoint"])
    cloud_relative = Path(run["source"]["point_cloud"])
    cloud = source / cloud_relative
    header, stride = _ply_layout(cloud, len(labels))
    required = sum(item["size_bytes"] for item in run["source"]["checkpoint_files"])
    required += len(positive) * stride + len(labels) * 8 + 64 * 1024 * 1024
    final_stage.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(final_stage.parent).free
    if free < required * 1.1:
        raise ExistingSandError(f"insufficient free disk for E3: need about {required * 1.1 / 2**30:.2f} GiB, have {free / 2**30:.2f} GiB")
    stage = Path(tempfile.mkdtemp(prefix=f".e3_pending_{stage_id}_", dir=run_root))
    published = False
    sealed = False
    try:
        checkpoint = stage / "checkpoint"
        checkpoint.mkdir()
        artifacts = []
        for item in run["source"]["checkpoint_files"]:
            relative = Path(item["path"])
            original = source / relative
            target = checkpoint / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative == cloud_relative:
                continue
            if relative == Path("cfg_args"):
                shutil.copy2(original, stage / "original_cfg_args.txt")
                artifacts.append({"path": "original_cfg_args.txt", "role": "source_renderer_config_reference"})
                _patched_cfg_args(original, target, Path(run["source"]["dataset"]), final_stage / "checkpoint")
            else:
                shutil.copy2(original, target)
            artifacts.append({"path": f"checkpoint/{relative.as_posix()}", "role": "copied_pgsr_resource"})
        _split_ply(cloud, stage / "selected_sand.ply", checkpoint / cloud_relative,
                   labels, header, stride, chunk_size=chunk_size)
        artifacts.extend([{"path": "selected_sand.ply", "role": "all_original_selected_gaussian_attributes"},
                          {"path": f"checkpoint/{cloud_relative.as_posix()}", "role": "pgsr_checkpoint_without_selected_indices"}])
        np.save(stage / "selected_source_indices.npy", positive, allow_pickle=False)
        kept = np.flatnonzero(labels != 1).astype(np.uint64)
        np.save(stage / "retained_source_indices.npy", kept, allow_pickle=False)
        artifacts.extend([{"path": "selected_source_indices.npy", "role": "source_ply_indices_removed"},
                          {"path": "retained_source_indices.npy", "role": "derived_ply_to_source_index_map"}])
        shutil.copy2(decision, stage / "approved_review.yaml")
        artifacts.append({"path": "approved_review.yaml", "role": "human_review_and_exact_selection_binding"})
        refs, cameras = _references(stage, mask, views)
        artifacts.extend(refs)
        quality = {"source_count": len(labels), "selected_count": len(positive),
                   "retained_count": len(kept), "retained_unknown_count": int(np.count_nonzero(labels == 2)),
                   "source_point_cloud_sha256": run["source"]["point_cloud_sha256"],
                   "selected_ply_sha256": _sha256(stage / "selected_sand.ply"),
                   "derived_ply_sha256": _sha256(checkpoint / cloud_relative),
                   "appearance_reference": "original SH and other selected Gaussian fields plus pre-deletion observed RGB/masks/PGSR depth; mask-region depth is not verified sand-only geometry"}
        _write_yaml(stage / "quality.yaml", quality)
        artifacts.append({"path": "quality.yaml", "role": "partition_counts_and_hashes"})
        record = {"schema_version": 1, "stage": "background", "stage_id": stage_id,
                  "camera_ids": cameras, "prompt": {"type": "approved_gaussian_deletion",
                                                  "selection_id": selection_id, "check_id": check_id},
                  "seed": None, "models": [],
                  "parameters": {"point_cloud": cloud_relative.as_posix(),
                                 "source_checkpoint": str(source), "derived_checkpoint": str(final_stage / "checkpoint"),
                                 "chunk_size": chunk_size},
                  "commands": [command] if command else [], "command_cwd": str(repo_root),
                  "parents": [{"stage": "segmentation", "stage_id": selection_id},
                              {"stage": "segmentation", "stage_id": check_id},
                              {"stage": "segmentation", "stage_id": mask.name},
                              {"stage": "observations", "stage_id": observation.name}],
                  "artifacts": artifacts,
                  "code_files": ["sandygs/existing_sand/e3.py", "scripts/experiments/existing_sand_e3.py"]}
        _write_yaml(stage / "record.yaml", record)
        stage.rename(final_stage)
        published = True
        record_stage(run_root, final_stage / "record.yaml", repo_root=repo_root)
        sealed = True
        return {"stage_dir": str(final_stage), "derived_checkpoint": str(final_stage / "checkpoint"), **quality}
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        if published and not sealed and final_stage.exists():
            shutil.rmtree(final_stage)
        raise
