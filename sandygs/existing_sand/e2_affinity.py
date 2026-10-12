"""PGSR-aligned, SAGA-inspired scene affinity features and scale gating.

This is a SandyGS adaptation: SAM instance co-membership supervises fixed-index
32-D Gaussian embeddings and a learned 1->32 scale gate. It uses PGSR plane
camera-Z for pixel correspondence, rather than running SAGA's 3DGS depth
renderer or changing the source scene. It is not a verbatim SAGA reproduction.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import yaml

from .e2 import _mask_stage, _observation_views, _sam_code, _write_yaml
from .e2_projection import source_gaussians
from .workspace import ExistingSandError, _load_yaml, _stage_location, record_stage, verify_run


_FEATURE_DIM = 32


def _visible_samples(cloud, camera, view_dir: Path, *, downsample: int,
                     max_samples: int, rng: np.random.Generator,
                     chunk_size: int = 100000):
    """Subsample credible center-depth correspondences for affinity training."""
    from scipy.ndimage import maximum_filter, minimum_filter

    depth = np.load(view_dir / "plane_depth.npy", allow_pickle=False)
    valid = np.load(view_dir / "depth_valid.npy", allow_pickle=False)
    numeric = valid & np.isfinite(depth) & (depth > 0)
    nearby = minimum_filter(numeric.astype(np.uint8), size=3).astype(bool)
    high = maximum_filter(np.where(numeric, depth, -np.inf), size=3)
    low = minimum_filter(np.where(numeric, depth, np.inf), size=3)
    K = np.asarray(camera["pgsr_K"], dtype=np.float64)
    w2c = np.asarray(camera["gs_w2c"], dtype=np.float64)
    ids, us, vs, coords = [], [], [], []
    height, width = depth.shape
    for start in range(0, len(cloud), chunk_size):
        end = min(start + chunk_size, len(cloud))
        block = cloud[start:end]
        xyz = np.stack([block[k] for k in ("x", "y", "z")], axis=-1).astype(np.float64)
        camera_xyz = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        z = camera_xyz[:, 2]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            u = K[0, 0] * camera_xyz[:, 0] / z + K[0, 2]
            v = K[1, 1] * camera_xyz[:, 1] / z + K[1, 2]
            scales = np.stack([block["scale_0"], block["scale_1"]], axis=-1)
            radius = 2 * np.exp(np.clip(scales.max(axis=-1), -30, 30)) * max(K[0, 0], K[1, 1]) / z
        possible = np.flatnonzero(np.isfinite(z) & np.isfinite(u) & np.isfinite(v)
                                  & np.isfinite(radius) & (z > 0) & (radius <= 12)
                                  & (u >= 1) & (u < width - 1) & (v >= 1) & (v < height - 1))
        if not possible.size:
            continue
        px = np.rint(u[possible]).astype(int)
        py = np.rint(v[possible]).astype(int)
        observed = depth[py, px]
        tol = 0.025 + 0.015 * observed
        good = (nearby[py, px] & (np.abs(observed - z[possible]) <= tol)
                & (high[py, px] - low[py, px] <= 2 * tol))
        keep = possible[good]
        ids.append((keep + start).astype(np.int32))
        us.append(np.clip(((u[keep] + 0.5) / downsample - 0.5).round().astype(np.int32),
                          0, max(0, (width + downsample - 1) // downsample - 1)))
        vs.append(np.clip(((v[keep] + 0.5) / downsample - 0.5).round().astype(np.int32),
                          0, max(0, (height + downsample - 1) // downsample - 1)))
        coords.append(xyz[keep].astype(np.float32))
    if not ids:
        raise ExistingSandError("no PGSR depth-consistent Gaussians in affinity view")
    ids, us, vs, coords = [np.concatenate(parts) for parts in (ids, us, vs, coords)]
    if len(ids) > max_samples:
        subset = np.sort(rng.choice(len(ids), size=max_samples, replace=False))
        ids, us, vs, coords = (a[subset] for a in (ids, us, vs, coords))
    return ids, us, vs, coords


def _automatic_masks(sam, rgb: np.ndarray, *, points_per_side: int) -> list[np.ndarray]:
    from segment_anything import SamAutomaticMaskGenerator

    generator = SamAutomaticMaskGenerator(
        model=sam, points_per_side=points_per_side,
        pred_iou_thresh=0.88, stability_score_thresh=0.95,
        crop_n_layers=0, min_mask_region_area=80,
    )
    return [item["segmentation"].astype(bool) for item in generator.generate(rgb)
            if 80 <= item["area"] <= int(rgb.shape[0] * rgb.shape[1] * 0.9)]


def train_affinity(run_root: Path, capture_id: str, stage_id: str, camera_ids: list[str],
                   sam_code: Path, sam_checkpoint: Path, saga_code: Path, *,
                   repo_root: Path, downsample: int = 4, points_per_side: int = 16,
                   max_samples_per_view: int = 120000, iterations: int = 2000,
                   learning_rate: float = 0.01, seed: int = 17,
                   command: list[str] | None = None) -> dict:
    """Train fixed-index affinity vectors on scene-specific SAM co-membership."""
    import torch
    import torch.nn.functional as F

    run_root, repo_root = Path(run_root).resolve(), Path(repo_root).resolve()
    verify_run(run_root)
    _, views = _observation_views(run_root, capture_id)
    if not camera_ids or len(set(camera_ids)) != len(camera_ids) or any(v not in views for v in camera_ids):
        raise ExistingSandError("affinity training needs distinct captured camera IDs")
    if (downsample not in {1, 2, 4, 8} or not 4 <= points_per_side <= 32
            or max_samples_per_view < 512 or iterations <= 0 or not 0 < learning_rate <= 0.1):
        raise ExistingSandError("invalid affinity sampling or training parameters")
    saga_code = Path(saga_code).resolve()
    if not (saga_code / "train_contrastive_feature.py").is_file():
        raise ExistingSandError(f"missing SAGA reference source: {saga_code}")
    sam_checkpoint = Path(sam_checkpoint).resolve()
    if not sam_checkpoint.is_file() or not torch.cuda.is_available():
        raise ExistingSandError("affinity training requires SAM ViT-H and CUDA")
    stage = _stage_location(run_root, "segmentation", stage_id)
    if stage.exists():
        raise ExistingSandError(f"refusing to overwrite affinity stage: {stage}")
    source = _load_yaml(run_root / "run.yaml", "run manifest")["source"]
    cloud = source_gaussians(Path(source["checkpoint"]) / source["point_cloud"],
                             source["gaussian_count"])
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    SamPredictor, registry = _sam_code(Path(sam_code))
    sam = registry["vit_h"](checkpoint=str(sam_checkpoint)).to("cuda")
    sam.eval()
    training_views = []
    for camera_id in camera_ids:
        view_dir, camera = views[camera_id]
        with Image.open(view_dir / "render_rgb.png") as image:
            w, h = camera["render_size_wh"]
            rgb = np.asarray(image.convert("RGB").resize(
                (max(1, w // downsample), max(1, h // downsample)), Image.BILINEAR))
        with torch.no_grad():
            masks = _automatic_masks(sam, rgb, points_per_side=points_per_side)
        ids, us, vs, xyz = _visible_samples(cloud, camera, view_dir, downsample=downsample,
                                            max_samples=max_samples_per_view, rng=np_rng)
        membership = []
        scales = []
        usable_masks = []
        for mask in masks:
            selected = mask[vs, us]
            if 64 <= selected.sum() < len(ids) - 64:
                scale = float(2 * np.linalg.norm(xyz[selected].std(axis=0)))
                if np.isfinite(scale) and scale > 0:
                    membership.append(selected)
                    scales.append(scale)
                    usable_masks.append(mask)
        if not membership:
            raise ExistingSandError(f"no usable scene SAM instance masks in {camera_id}")
        training_views.append((camera_id, ids, np.stack(membership), np.asarray(scales),
                               np.stack(usable_masks), rgb))
    del sam
    torch.cuda.empty_cache()
    all_scales = np.concatenate([v[3] for v in training_views])
    bounds = np.percentile(all_scales, [5, 95]).astype(float)
    if bounds[1] <= bounds[0]:
        bounds[1] = bounds[0] + max(bounds[0], 1) * 1e-4
    table = torch.nn.Embedding(len(cloud), _FEATURE_DIM, sparse=True, device="cuda")
    with torch.no_grad():
        table.weight.normal_(std=0.07)
    gate = torch.nn.Sequential(torch.nn.Linear(1, _FEATURE_DIM), torch.nn.Sigmoid()).cuda()
    opt_features = torch.optim.SparseAdam(table.parameters(), lr=learning_rate)
    opt_gate = torch.optim.Adam(gate.parameters(), lr=learning_rate)
    touched = np.zeros(len(cloud), dtype=bool)
    final_loss = None
    for iteration in range(iterations):
        view = training_views[int(np_rng.integers(len(training_views)))]
        _camera_id, ids, membership, scales = view[:4]
        mask_idx = int(np_rng.integers(len(scales)))
        selected = membership[mask_idx]
        inside, outside = np.flatnonzero(selected), np.flatnonzero(~selected)
        if len(inside) < 64 or len(outside) < 64:
            continue
        chosen_in = np_rng.choice(inside, size=64, replace=False)
        chosen_out = np_rng.choice(outside, size=64, replace=False)
        indices = np.concatenate([ids[chosen_in], ids[chosen_out]])
        touched[indices] = True
        torch_idx = torch.as_tensor(indices.astype(np.int64), device="cuda")
        normalized_scale = float(np.clip((scales[mask_idx] - bounds[0]) /
                                         (bounds[1] - bounds[0]), 0, 1))
        scale_tensor = torch.tensor([[normalized_scale]], dtype=torch.float32, device="cuda")
        features = F.normalize(table(torch_idx) * gate(scale_tensor), dim=-1)
        scores = features @ features.T
        pos_pairs = scores[:64, :64][torch.triu_indices(64, 64, 1, device="cuda").unbind()]
        neg_pairs = scores[:64, 64:].reshape(-1)
        loss = (F.softplus(-8 * (pos_pairs - 0.5)).mean() +
                F.softplus(8 * (neg_pairs - 0.2)).mean())
        opt_features.zero_grad(set_to_none=True)
        opt_gate.zero_grad(set_to_none=True)
        loss.backward()
        opt_features.step()
        opt_gate.step()
        final_loss = float(loss.detach().item())
    stage.mkdir(parents=True, exist_ok=False)
    mask_artifacts = []
    mask_views = []
    for i, (camera_id, _ids, _membership, scales, masks, rgb) in enumerate(training_views):
        folder = stage / f"view_{i:04d}"
        folder.mkdir()
        np.savez_compressed(folder / "sam_masks.npz", masks=masks, scales=scales)
        # Show a few large instance regions on their actual training grid.
        preview = rgb.copy()
        for j, mask_index in enumerate(np.argsort(masks.sum(axis=(1, 2)))[-6:]):
            tint = np.array([(37 + 55 * j) % 256, (95 + 73 * j) % 256,
                             (180 + 41 * j) % 256])
            region = masks[mask_index]
            preview[region] = np.round(preview[region] * 0.65 + tint * 0.35).astype(np.uint8)
        Image.fromarray(preview).save(folder / "sam_preview.png")
        mask_views.append({"camera_id": camera_id, "directory": folder.name,
                           "usable_masks": len(masks)})
        mask_artifacts.extend({"path": f"{folder.name}/{file}", "role": role} for file, role in
                              (("sam_masks.npz", "training_sam_instance_masks_and_scales"),
                               ("sam_preview.png", "training_masks_visual_preview")))
    # Save half-precision per-source-Gaussian vectors without a full CPU copy.
    output = np.lib.format.open_memmap(stage / "features.npy", mode="w+",
                                       dtype=np.float16, shape=(len(cloud), _FEATURE_DIM))
    for start in range(0, len(cloud), 100000):
        output[start:start+100000] = table.weight.detach()[start:start+100000].cpu().numpy()
    output.flush()
    del output
    np.save(stage / "feature_valid.npy", touched, allow_pickle=False)
    linear = gate[0]
    np.savez(stage / "scale_gate.npz", weight=linear.weight.detach().cpu().numpy(),
             bias=linear.bias.detach().cpu().numpy(), bounds=bounds)
    summary = {"camera_ids": camera_ids,
               "sam_mask_counts": {v[0]: len(v[3]) for v in training_views},
               "mask_views": mask_views, "trained_gaussian_count": int(touched.sum()),
               "gaussian_count": len(cloud), "last_loss": final_loss,
               "reference": "SAGA-style SAM co-membership; PGSR center-depth training adapter"}
    _write_yaml(stage / "quality.yaml", summary)
    _write_yaml(stage / "record.yaml", {
        "schema_version": 1, "stage": "segmentation", "stage_id": stage_id,
        "camera_ids": camera_ids,
        "prompt": {"type": "scene_sam_affinity_training"}, "seed": seed,
        "models": [{"name": "segment-anything-vit-h", "version": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=sam_code, text=True).strip(),
            "checkpoint": str(sam_checkpoint)}],
        "parameters": {"capture_id": capture_id, "saga_reference_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=saga_code, text=True).strip(),
            "method": "PGSR-adapted SAM co-membership contrastive 32D scale gate",
            "downsample": downsample, "points_per_side": points_per_side,
            "max_samples_per_view": max_samples_per_view, "iterations": iterations,
            "learning_rate": learning_rate, "scale_bounds": bounds.tolist()},
        "commands": [command] if command else [], "command_cwd": str(repo_root),
        "parents": [{"stage": "observations", "stage_id": capture_id}],
        "artifacts": [{"path": file, "role": role} for file, role in
                      (("features.npy", "index_aligned_affinity_f16"),
                       ("feature_valid.npy", "trained_affinity_indices"),
                       ("scale_gate.npz", "learned_scale_gate"),
                       ("quality.yaml", "training_summary"))] + mask_artifacts,
        "code_files": ["sandygs/existing_sand/e2_affinity.py",
                       "sandygs/existing_sand/e2_projection.py",
                       "scripts/experiments/existing_sand_e2.py"]})
    record_stage(run_root, stage / "record.yaml", repo_root=repo_root)
    return {"stage_dir": str(stage), **summary}


def _prototypes(vectors: np.ndarray, *, limit: int, rng: np.random.Generator) -> np.ndarray:
    """Use several distinct exemplars so a sand pile need not be one centroid."""
    if len(vectors) > 4096:
        vectors = vectors[rng.choice(len(vectors), 4096, replace=False)]
    if not len(vectors):
        raise ExistingSandError("no trained feature vectors for reliable seed prototypes")
    prototypes = [vectors[0]]
    nearest = vectors @ prototypes[0]
    for _ in range(1, min(limit, len(vectors))):
        index = int(np.argmin(nearest))
        prototypes.append(vectors[index])
        nearest = np.maximum(nearest, vectors @ prototypes[-1])
    return np.stack(prototypes)


def expand_affinity(run_root: Path, seed_id: str, affinity_id: str,
                    stage_id: str, *, repo_root: Path,
                    scale: float, threshold: float = 0.8, margin: float = 0.05,
                    prototypes: int = 4, chunk_size: int = 100000,
                    command: list[str] | None = None) -> dict:
    """Expand unknowns supported by a positive vote; preserve negative evidence."""
    run_root, repo_root = Path(run_root).resolve(), Path(repo_root).resolve()
    verify_run(run_root)
    seed_stage = _stage_location(run_root, "segmentation", seed_id)
    affinity_stage = _stage_location(run_root, "segmentation", affinity_id)
    for stage in (seed_stage, affinity_stage):
        if not (stage / "manifest.yaml").is_file():
            raise ExistingSandError(f"unregistered E2 parent stage: {stage}")
    seed_record = _load_yaml(seed_stage / "record.yaml", "E2 seed record")
    affinity_record = _load_yaml(affinity_stage / "record.yaml", "E2 affinity record")
    if (seed_record.get("prompt", {}).get("type") != "multiview_sam_depth_seeds"
            or affinity_record.get("prompt", {}).get("type") != "scene_sam_affinity_training"
            or seed_record["parameters"]["capture_id"] != affinity_record["parameters"]["capture_id"]):
        raise ExistingSandError("affinity and seed stages must use the same PGSR capture")
    if (not np.isfinite(scale) or scale <= 0 or not 0 < threshold <= 1
            or not 0 <= margin <= 1 or not 1 <= prototypes <= 32 or chunk_size < 1):
        raise ExistingSandError("invalid affinity scale, score or prototype parameters")
    stage = _stage_location(run_root, "segmentation", stage_id)
    if stage.exists():
        raise ExistingSandError(f"refusing to overwrite E2 affinity expansion: {stage}")
    count = _load_yaml(run_root / "run.yaml", "run manifest")["source"]["gaussian_count"]
    base = np.load(seed_stage / "gaussian_labels.npy", mmap_mode="r", allow_pickle=False)
    positive = np.load(seed_stage / "positive_votes.npy", mmap_mode="r", allow_pickle=False)
    negative = np.load(seed_stage / "negative_votes.npy", mmap_mode="r", allow_pickle=False)
    feat = np.load(affinity_stage / "features.npy", mmap_mode="r", allow_pickle=False)
    trained = np.load(affinity_stage / "feature_valid.npy", mmap_mode="r", allow_pickle=False)
    if (base.shape != (count,) or positive.shape != (count,) or negative.shape != (count,)
            or trained.shape != (count,) or feat.shape != (count, _FEATURE_DIM)
            or not np.isin(base, [0, 1, 2]).all()):
        raise ExistingSandError("E2 features and tri-state labels must share PGSR Gaussian indices")
    with np.load(affinity_stage / "scale_gate.npz", allow_pickle=False) as gate_file:
        weights, bias, bounds = (gate_file[k] for k in ("weight", "bias", "bounds"))
    if weights.shape != (_FEATURE_DIM, 1) or bias.shape != (_FEATURE_DIM,) or bounds.shape != (2,):
        raise ExistingSandError("E2 scale gate must map one scale to 32 channels")
    normalized_scale = np.clip((scale - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)
    gate = 1 / (1 + np.exp(-np.clip(weights[:, 0] * normalized_scale + bias, -30, 30)))
    rng = np.random.default_rng(0)
    def take(indices):
        if len(indices) > 10000:
            indices = rng.choice(indices, 10000, replace=False)
        data = np.asarray(feat[indices], dtype=np.float32) * gate
        norms = np.linalg.norm(data, axis=1)
        return data[norms > 1e-8] / norms[norms > 1e-8, None]
    pos_proto = _prototypes(take(np.flatnonzero((base == 1) & trained)),
                            limit=prototypes, rng=rng)
    neg_proto = _prototypes(take(np.flatnonzero((base == 0) & trained)),
                            limit=prototypes, rng=rng)
    labels = np.asarray(base).copy()
    scores = np.full(count, np.nan, dtype=np.float16)
    eligible = (base == 2) & trained & (positive >= 1) & (negative == 0)
    added = 0
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        ids = np.flatnonzero(eligible[start:end]) + start
        if not len(ids):
            continue
        data = np.asarray(feat[ids], dtype=np.float32) * gate
        norms = np.linalg.norm(data, axis=1)
        good = np.isfinite(data).all(axis=1) & (norms > 1e-8)
        ids = ids[good]
        if not len(ids):
            continue
        vec = data[good] / norms[good, None]
        best_pos = (vec @ pos_proto.T).max(axis=1)
        best_neg = (vec @ neg_proto.T).max(axis=1)
        scores[ids] = (best_pos - best_neg).astype(np.float16)
        accepted = (best_pos >= threshold) & ((best_pos - best_neg) >= margin)
        labels[ids[accepted]] = 1
        added += int(accepted.sum())
    stage.mkdir(parents=True, exist_ok=False)
    np.save(stage / "gaussian_labels.npy", labels, allow_pickle=False)
    np.save(stage / "feature_margin.npy", scores, allow_pickle=False)
    quality = {"added_positive": added, "total_positive": int((labels == 1).sum()),
               "remaining_unknown": int((labels == 2).sum()),
               "source_negative_preserved": True, "scale_world_units": scale,
               "rule": "unknown + >=1 positive view + zero negative votes + trained affinity + positive prototype similarity"}
    _write_yaml(stage / "quality.yaml", quality)
    _write_yaml(stage / "record.yaml", {
        "schema_version": 1, "stage": "segmentation", "stage_id": stage_id,
        "camera_ids": seed_record["camera_ids"],
        "prompt": {"type": "saga_affinity_expansion", "seed_id": seed_id,
                   "mask_id": seed_record["prompt"]["mask_id"]},
        "seed": 0, "models": [],
        "parameters": {"affinity_id": affinity_id, "scale": scale,
                       "threshold": threshold, "margin": margin,
                       "prototype_count_per_class": prototypes},
        "commands": [command] if command else [], "command_cwd": str(repo_root),
        "parents": [{"stage": "segmentation", "stage_id": seed_id},
                    {"stage": "segmentation", "stage_id": affinity_id}],
        "artifacts": [{"path": "gaussian_labels.npy", "role": "expanded_source_gaussian_tristate"},
                      {"path": "feature_margin.npy", "role": "affinity_positive_minus_negative_similarity"},
                      {"path": "quality.yaml", "role": "feature_expansion_counts"}],
        "code_files": ["sandygs/existing_sand/e2_affinity.py",
                       "scripts/experiments/existing_sand_e2.py"]})
    record_stage(run_root, stage / "record.yaml", repo_root=repo_root)
    return {"stage_dir": str(stage), **quality}
