"""E2 prompted SAM masks and conservative multiview Gaussian seed labels."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image
import yaml

from .e2_projection import (NEGATIVE, POSITIVE, UNKNOWN, combine_votes,
                            prepare_vote_maps, source_gaussians, view_votes,
                            view_votes_frequency, frequency_labels)
from .workspace import (ExistingSandError, _load_yaml, _stage_location,
                        record_stage, verify_run)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)


def _observation_views(run_root: Path, capture_id: str) -> tuple[Path, dict[str, tuple[Path, dict]]]:
    stage = _stage_location(run_root, "observations", capture_id)
    manifest = stage / "manifest.yaml"
    if not manifest.is_file():
        raise ExistingSandError(f"unregistered E2 observation stage: {stage}")
    views = {}
    for path in sorted(stage.glob("view_*/camera.yaml")):
        camera = _load_yaml(path, "E2 observation camera")
        name = camera["camera_id"]
        if name in views:
            raise ExistingSandError(f"duplicate observation camera_id: {name}")
        views[name] = path.parent, camera
    return stage, views


def _prompts(path: Path, views: dict[str, tuple[Path, dict]]) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExistingSandError(f"cannot read E2 prompt JSON: {exc}") from exc
    entries = data.get("views") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ExistingSandError('prompt JSON needs a nonempty "views" list')
    used = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("camera_id") not in views:
            raise ExistingSandError("E2 prompt camera_id must identify a captured view")
        name = entry["camera_id"]
        if name in used or entry.get("role", "seed") not in {"seed", "holdout"}:
            raise ExistingSandError("E2 camera_id must be unique and role seed/holdout")
        used.add(name)
        w, h = views[name][1]["render_size_wh"]
        positive, negative, box = (entry.get(k, []) for k in ("positive", "negative", "box"))
        if not isinstance(positive, list) or not isinstance(negative, list):
            raise ExistingSandError("E2 points must be lists of [x,y]")
        for point in positive + negative:
            if (not isinstance(point, list) or len(point) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
                           for v in point)
                    or not (0 <= point[0] < w and 0 <= point[1] < h)):
                raise ExistingSandError("E2 points must lie on the captured image")
        if box:
            if (not isinstance(box, list) or len(box) != 4
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
                           for v in box)
                    or not (0 <= box[0] < box[2] < w and 0 <= box[1] < box[3] < h)):
                raise ExistingSandError("E2 box must be [x_min,y_min,x_max,y_max] within the view")
        if not positive and not negative and not box:
            raise ExistingSandError("each E2 view needs points or a box")
    return entries


def _sam_code(code: Path):
    if not (code / "segment_anything/__init__.py").is_file():
        raise ExistingSandError(f"missing SAM Python package: {code}")
    sys.path.insert(0, str(code.resolve()))
    try:
        from segment_anything import SamPredictor, sam_model_registry
    finally:
        sys.path.remove(str(code.resolve()))
    return SamPredictor, sam_model_registry


def create_masks(run_root: Path, capture_id: str, mask_id: str,
                 prompts: Path, sam_code: Path, sam_checkpoint: Path, *,
                 repo_root: Path, image_type: str = "render_rgb",
                 command: list[str] | None = None) -> dict:
    """Predict per-view prompted masks on captured PGSR pixel grids."""
    import torch

    run_root, repo_root = Path(run_root).resolve(), Path(repo_root).resolve()
    verify_run(run_root)
    capture, views = _observation_views(run_root, capture_id)
    entries = _prompts(Path(prompts), views)
    input_payload = json.loads(Path(prompts).read_text(encoding="utf-8"))
    generator = input_payload.get("generator") if isinstance(input_payload, dict) else None
    if generator is not None and (not isinstance(generator, dict) or
                                  generator.get("type") != "grounding_dino_swint_ogc"):
        raise ExistingSandError("unsupported E2 box generator metadata")
    if image_type not in {"render_rgb", "source_rgb"}:
        raise ExistingSandError("image_type must be render_rgb or source_rgb")
    checkpoint = Path(sam_checkpoint).resolve()
    if not checkpoint.is_file() or not torch.cuda.is_available():
        raise ExistingSandError("SAM checkpoint and CUDA device are required")
    stage = _stage_location(run_root, "segmentation", mask_id)
    if stage.exists():
        raise ExistingSandError(f"refusing to overwrite E2 masks: {stage}")
    SamPredictor, registry = _sam_code(Path(sam_code))
    try:
        predictor = SamPredictor(registry["vit_h"](checkpoint=str(checkpoint)).to("cuda"))
    except (KeyError, RuntimeError) as exc:
        raise ExistingSandError(f"cannot initialize SAM ViT-H: {exc}") from exc
    generated = []
    for index, entry in enumerate(entries):
        view, camera = views[entry["camera_id"]]
        with Image.open(view / f"{image_type}.png") as image:
            rgb = np.asarray(image.convert("RGB"))
        w, h = camera["render_size_wh"]
        if rgb.shape != (h, w, 3):
            raise ExistingSandError("SAM input differs from PGSR camera resolution")
        predictor.set_image(rgb)
        pos, neg = entry.get("positive", []), entry.get("negative", [])
        points = np.asarray(pos + neg, dtype=np.float32).reshape(-1, 2)
        values = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int32)
        with torch.no_grad():
            masks, scores, _ = predictor.predict(
                point_coords=points if len(points) else None,
                point_labels=values if len(points) else None,
                box=np.asarray(entry["box"], dtype=np.float32) if entry.get("box") else None,
                multimask_output=True,
            )
        best = int(np.argmax(scores))
        mask = masks[best].astype(bool)
        if not mask.any() or mask.all():
            raise ExistingSandError(f"SAM returned an empty/full mask for {entry['camera_id']}")
        overlay = rgb.copy()
        overlay[mask] = np.round(overlay[mask] * 0.55 +
                                  np.array([255, 50, 30]) * 0.45).astype(np.uint8)
        generated.append((index, entry, mask, overlay, float(scores[best])))
    del predictor
    torch.cuda.empty_cache()
    stage.mkdir(parents=True, exist_ok=False)
    artifacts = []
    (stage / "prompts.json").write_text(json.dumps({"views": entries, "generator": generator},
                                                indent=2) + "\n", encoding="utf-8")
    artifacts.append({"path": "prompts.json", "role": "per_view_prompt_and_holdout_roles"})
    summaries = []
    for index, entry, mask, overlay, score in generated:
        folder = stage / f"view_{index:04d}"
        folder.mkdir()
        Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "mask.png")
        Image.fromarray(overlay).save(folder / "overlay.png")
        artifacts.extend([{"path": f"{folder.name}/{file}", "role": role} for file, role in
                          (("mask.png", "sam_mask"), ("overlay.png", "sam_overlay"))])
        summaries.append({"camera_id": entry["camera_id"], "role": entry.get("role", "seed"),
                          "directory": folder.name, "mask_pixels": int(mask.sum()),
                          "sam_predicted_iou": score})
    _write_yaml(stage / "quality.yaml", {"views": summaries})
    artifacts.append({"path": "quality.yaml", "role": "sam_confidence_and_coverage"})
    models = [{"name": "segment-anything-vit-h", "version": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=sam_code, text=True).strip(),
        "checkpoint": str(checkpoint)}]
    if generator is not None:
        models += [{"name": "GroundingDINO-SwinT", "version": generator["code_revision"],
                    "checkpoint": generator["checkpoint"]},
                   {"name": "bert-base-uncased", "version": "grounding-dino-text-encoder",
                    "checkpoint": generator["bert_checkpoint"]}]
    record = {
        "schema_version": 1, "stage": "segmentation", "stage_id": mask_id,
        "camera_ids": [entry["camera_id"] for entry in entries],
        "prompt": {"type": "manual_sam_points_box", "file": "prompts.json"},
        "seed": None,
        "models": models,
        "parameters": {"capture_id": capture_id, "image_type": image_type,
                       "mask_convention": "white_is_selected", "pixel_grid": "PGSR_render_size"},
        "commands": [command] if command else [], "command_cwd": str(repo_root),
        "parents": [{"stage": "observations", "stage_id": capture_id}],
        "artifacts": artifacts, "code_files": ["sandygs/existing_sand/e2.py",
                                            "scripts/experiments/existing_sand_e2.py"],
    }
    _write_yaml(stage / "record.yaml", record)
    record_stage(run_root, stage / "record.yaml", repo_root=repo_root)
    return {"stage_dir": str(stage), "views": summaries}


def _mask_stage(run_root: Path, mask_id: str) -> tuple[Path, dict, list[dict]]:
    stage = _stage_location(run_root, "segmentation", mask_id)
    if not (stage / "manifest.yaml").is_file():
        raise ExistingSandError(f"unregistered E2 SAM stage: {stage}")
    record = _load_yaml(stage / "record.yaml", "E2 mask record")
    if record.get("prompt", {}).get("type") != "manual_sam_points_box":
        raise ExistingSandError("E2 selection requires a prompted SAM stage")
    quality = _load_yaml(stage / "quality.yaml", "E2 mask quality")
    return stage, record, quality["views"]


def select_seeds(run_root: Path, mask_id: str, stage_id: str, *, repo_root: Path,
                 min_views: int = 2, depth_absolute: float = 0.025,
                 depth_relative: float = 0.015, footprint_sigma: float = 2.0,
                 max_footprint_pixels: float = 12.0,
                 border_pixels: float = 1.5,
                 chunk_size: int = 100000, command: list[str] | None = None) -> dict:
    """Write positive/negative/unknown source-Gaussian seeds, with holdout checks."""
    run_root, repo_root = Path(run_root).resolve(), Path(repo_root).resolve()
    verify_run(run_root)
    mask_stage, mask_record, entries = _mask_stage(run_root, mask_id)
    capture_id = mask_record["parameters"]["capture_id"]
    _, views = _observation_views(run_root, capture_id)
    seeds = [entry for entry in entries if entry["role"] == "seed"]
    if (isinstance(min_views, bool) or min_views < 1 or min_views > len(seeds)
            or chunk_size < 1):
        raise ExistingSandError("min_views must be 1..number of seed views and chunk_size positive")
    stage = _stage_location(run_root, "segmentation", stage_id)
    if stage.exists():
        raise ExistingSandError(f"refusing to overwrite E2 seed selection: {stage}")
    run = _load_yaml(run_root / "run.yaml", "run manifest")
    source = run["source"]
    cloud = source_gaussians(Path(source["checkpoint"]) / source["point_cloud"],
                             source["gaussian_count"])
    positive = np.zeros(len(cloud), dtype=np.uint8)
    negative = np.zeros(len(cloud), dtype=np.uint8)
    params = {"depth_absolute": depth_absolute, "depth_relative": depth_relative,
              "footprint_sigma": footprint_sigma,
              "max_footprint_pixels": max_footprint_pixels, "border_pixels": border_pixels}
    def votes_for(entry):
        view, camera = views[entry["camera_id"]]
        path = mask_stage / entry["directory"] / "mask.png"
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L")) >= 128
        depth = np.load(view / "plane_depth.npy", allow_pickle=False)
        valid = np.load(view / "depth_valid.npy", allow_pickle=False)
        if mask.shape != depth.shape or depth.shape != valid.shape or not mask.any():
            raise ExistingSandError(f"E2 mask/depth mismatch: {entry['camera_id']}")
        maps = prepare_vote_maps(mask, depth, valid)
        k, w2c = np.asarray(camera["pgsr_K"]), np.asarray(camera["gs_w2c"])
        for start in range(0, len(cloud), chunk_size):
            end = min(start + chunk_size, len(cloud))
            block = cloud[start:end]
            xyz = np.stack([block[key] for key in ("x", "y", "z")], axis=-1)
            scale = np.stack([block[key] for key in ("scale_0", "scale_1")], axis=-1)
            yield start, end, view_votes(xyz, scale, mask, depth, valid, k, w2c,
                                          maps=maps, **params)
    for entry in seeds:
        for start, end, votes in votes_for(entry):
            positive[start:end] += votes == POSITIVE
            negative[start:end] += votes == NEGATIVE
    labels = combine_votes(positive, negative, min_views)
    reports = []
    for entry in entries:
        if entry["role"] != "holdout":
            continue
        # This checks independent SAM masks only where Gaussian centers are
        # confidently visible; it is not a rasterized IoU or ground truth.
        counts = {"positive_inside": 0, "positive_outside": 0,
                  "negative_inside": 0, "negative_outside": 0}
        for start, end, votes in votes_for(entry):
            labelled = labels[start:end]
            for key, label, vote in (("positive_inside", POSITIVE, POSITIVE),
                                     ("positive_outside", POSITIVE, NEGATIVE),
                                     ("negative_inside", NEGATIVE, POSITIVE),
                                     ("negative_outside", NEGATIVE, NEGATIVE)):
                counts[key] += int(np.count_nonzero((labelled == label) & (votes == vote)))
        reports.append({"camera_id": entry["camera_id"], **counts})
    stage.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for file, arr, role in (("gaussian_labels.npy", labels, "gaussian_tristate_index_aligned"),
                            ("positive_votes.npy", positive, "positive_view_votes"),
                            ("negative_votes.npy", negative, "negative_view_votes")):
        np.save(stage / file, arr, allow_pickle=False)
        artifacts.append({"path": file, "role": role})
    summary = {"labels": {"negative": 0, "positive": 1, "unknown": 2},
               "counts": {"negative": int(np.count_nonzero(labels == NEGATIVE)),
                          "positive": int(np.count_nonzero(labels == POSITIVE)),
                          "unknown": int(np.count_nonzero(labels == UNKNOWN))},
               "min_views": min_views, "holdout_center_checks": reports,
               "warning": "visibility and held-out checks use center/scale/depth proxies, not true PGSR rasterized mask IoU"}
    _write_yaml(stage / "quality.yaml", summary)
    artifacts.append({"path": "quality.yaml", "role": "selection_counts_and_holdout_checks"})
    record = {"schema_version": 1, "stage": "segmentation", "stage_id": stage_id,
              "camera_ids": [entry["camera_id"] for entry in entries],
              "prompt": {"type": "multiview_sam_depth_seeds", "mask_id": mask_id},
              "seed": None, "models": [],
              "parameters": {"capture_id": capture_id, "min_views": min_views,
                             "chunk_size": chunk_size, **params},
              "commands": [command] if command else [], "command_cwd": str(repo_root),
              "parents": [{"stage": "segmentation", "stage_id": mask_id},
                          {"stage": "observations", "stage_id": capture_id}],
              "artifacts": artifacts,
              "code_files": ["sandygs/existing_sand/e2.py", "sandygs/existing_sand/e2_projection.py",
                             "scripts/experiments/existing_sand_e2.py"]}
    _write_yaml(stage / "record.yaml", record)
    record_stage(run_root, stage / "record.yaml", repo_root=repo_root)
    return {"stage_dir": str(stage), **summary}


def select_frequency(run_root: Path, mask_id: str, stage_id: str, *, repo_root: Path,
                     sand_fraction: float = 0.65, depth_absolute: float = 0.025,
                     depth_relative: float = 0.03, chunk_size: int = 100000,
                     command: list[str] | None = None) -> dict:
    """Frequency votes on all prompted seed views with a loose depth occlusion test."""
    run_root, repo_root = Path(run_root).resolve(), Path(repo_root).resolve()
    verify_run(run_root)
    mask_stage, mask_record, entries = _mask_stage(run_root, mask_id)
    capture_id = mask_record["parameters"]["capture_id"]
    _, views = _observation_views(run_root, capture_id)
    seeds = [entry for entry in entries if entry["role"] == "seed"]
    if not seeds or len(seeds) > np.iinfo(np.uint16).max or chunk_size < 1:
        raise ExistingSandError("E2 frequency needs seed views and a positive chunk size")
    # Validate before writing a new stage, even if no Gaussian happens to project.
    if (not np.isfinite(sand_fraction) or not 0 < sand_fraction <= 1
            or not np.isfinite(depth_absolute) or depth_absolute < 0
            or not np.isfinite(depth_relative) or depth_relative < 0):
        raise ExistingSandError("invalid E2 frequency or depth thresholds")
    stage = _stage_location(run_root, "segmentation", stage_id)
    if stage.exists():
        raise ExistingSandError(f"refusing to overwrite E2 frequency selection: {stage}")
    run = _load_yaml(run_root / "run.yaml", "run manifest")
    source = run["source"]
    cloud = source_gaussians(Path(source["checkpoint"]) / source["point_cloud"],
                             source["gaussian_count"])
    positive = np.zeros(len(cloud), dtype=np.uint16)
    visible = np.zeros(len(cloud), dtype=np.uint16)

    def votes_for(entry):
        view, camera = views[entry["camera_id"]]
        with Image.open(mask_stage / entry["directory"] / "mask.png") as image:
            mask = np.asarray(image.convert("L")) >= 128
        depth = np.load(view / "plane_depth.npy", allow_pickle=False)
        valid = np.load(view / "depth_valid.npy", allow_pickle=False)
        if mask.shape != depth.shape or depth.shape != valid.shape or not mask.any():
            raise ExistingSandError(f"E2 mask/depth mismatch: {entry['camera_id']}")
        k, w2c = np.asarray(camera["pgsr_K"]), np.asarray(camera["gs_w2c"])
        for start in range(0, len(cloud), chunk_size):
            end = min(start + chunk_size, len(cloud))
            block = cloud[start:end]
            xyz = np.stack([block[key] for key in ("x", "y", "z")], axis=-1)
            yield start, end, view_votes_frequency(
                xyz, mask, depth, valid, k, w2c,
                depth_absolute=depth_absolute, depth_relative=depth_relative)

    seed_reports = []
    for entry in seeds:
        view_positive = view_visible = 0
        for start, end, votes in votes_for(entry):
            sand = votes == POSITIVE
            seen = votes != UNKNOWN
            positive[start:end] += sand
            visible[start:end] += seen
            view_positive += int(np.count_nonzero(sand))
            view_visible += int(np.count_nonzero(seen))
        seed_reports.append({"camera_id": entry["camera_id"],
                             "visible_gaussians": view_visible,
                             "sand_votes": view_positive})
    labels, fraction = frequency_labels(positive, visible, sand_fraction)
    negative = visible - positive
    holdout_reports = []
    for entry in entries:
        if entry["role"] != "holdout":
            continue
        counts = {"positive_inside": 0, "positive_outside": 0,
                  "negative_inside": 0, "negative_outside": 0}
        for start, end, votes in votes_for(entry):
            labelled = labels[start:end]
            for key, label, vote in (("positive_inside", POSITIVE, POSITIVE),
                                     ("positive_outside", POSITIVE, NEGATIVE),
                                     ("negative_inside", NEGATIVE, POSITIVE),
                                     ("negative_outside", NEGATIVE, NEGATIVE)):
                counts[key] += int(np.count_nonzero((labelled == label) & (votes == vote)))
        holdout_reports.append({"camera_id": entry["camera_id"], **counts})
    stage.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for filename, values, role in (
        ("gaussian_labels.npy", labels, "gaussian_tristate_index_aligned"),
        ("positive_votes.npy", positive, "sand_view_votes"),
        ("negative_votes.npy", negative, "nonsand_view_votes"),
        ("visible_votes.npy", visible, "visible_seed_view_count"),
        ("sand_fraction.npy", fraction, "sand_votes_divided_by_visible_votes"),
    ):
        np.save(stage / filename, values, allow_pickle=False)
        artifacts.append({"path": filename, "role": role})
    counts = {"negative": int(np.count_nonzero(labels == NEGATIVE)),
              "positive": int(np.count_nonzero(labels == POSITIVE)),
              "unknown": int(np.count_nonzero(labels == UNKNOWN))}
    support = {"one_visible_view": int(np.count_nonzero(visible == 1)),
               "one_view_positive": int(np.count_nonzero((visible == 1) & (labels == POSITIVE))),
               "two_or_more_view_positive": int(np.count_nonzero(
                   (visible >= 2) & (labels == POSITIVE)))}
    quality = {"labels": {"negative": 0, "positive": 1, "unknown": 2},
               "counts": counts, "support": support,
               "seed_view_reports": seed_reports,
               "holdout_center_checks": holdout_reports,
               "warning": "center/depth votes are visibility proxies; check held-out PGSR rasterization before E3 deletion"}
    _write_yaml(stage / "quality.yaml", quality)
    artifacts.append({"path": "quality.yaml", "role": "frequency_selection_and_support"})
    _write_yaml(stage / "record.yaml", {
        "schema_version": 1, "stage": "segmentation", "stage_id": stage_id,
        "camera_ids": [entry["camera_id"] for entry in entries],
        "prompt": {"type": "multiview_sam_frequency", "mask_id": mask_id},
        "seed": None, "models": [],
        "parameters": {"capture_id": capture_id, "sand_fraction": sand_fraction,
                       "depth_absolute": depth_absolute,
                       "depth_relative": depth_relative, "chunk_size": chunk_size,
                       "visibility_rule": "projected_center_in_frame_and_valid_PGSR_depth_and_z_le_depth_plus_margin",
                       "zero_vote_label": "unknown"},
        "commands": [command] if command else [], "command_cwd": str(repo_root),
        "parents": [{"stage": "segmentation", "stage_id": mask_id},
                    {"stage": "observations", "stage_id": capture_id}],
        "artifacts": artifacts,
        "code_files": ["sandygs/existing_sand/e2.py",
                       "sandygs/existing_sand/e2_projection.py",
                       "scripts/experiments/existing_sand_e2.py"]})
    record_stage(run_root, stage / "record.yaml", repo_root=repo_root)
    return {"stage_dir": str(stage), **quality}
