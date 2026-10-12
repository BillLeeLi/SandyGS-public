"""Render frozen source Gaussians with selected-label colors in held-out views."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import yaml

from .e2 import _mask_stage, _observation_views, _write_yaml
from .workspace import ExistingSandError, _load_yaml, _stage_location, record_stage, verify_run


def render_holdout(run_root: Path, seed_id: str, check_id: str, *,
                   repo_root: Path, coverage_threshold: float = 0.3,
                   command: list[str] | None = None) -> dict:
    """GPU rasterization check; held-out mask never participates in seed votes."""
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov

    run_root = Path(run_root).resolve()
    repo_root = Path(repo_root).resolve()
    verify_run(run_root)
    if not 0 < coverage_threshold < 1:
        raise ExistingSandError("coverage_threshold must be in (0,1)")
    seed_stage = _stage_location(run_root, "segmentation", seed_id)
    if not (seed_stage / "manifest.yaml").is_file():
        raise ExistingSandError(f"unregistered E2 seed stage: {seed_stage}")
    seed_record = _load_yaml(seed_stage / "record.yaml", "E2 seed record")
    if seed_record.get("prompt", {}).get("type") not in {"multiview_sam_depth_seeds", "multiview_sam_frequency", "saga_affinity_expansion"}:
        raise ExistingSandError("E2 holdout requires a tri-state selection stage")
    mask_id = seed_record["prompt"]["mask_id"]
    mask_stage, mask_record, mask_views = _mask_stage(run_root, mask_id)
    holdouts = [v for v in mask_views if v["role"] == "holdout"]
    if not holdouts:
        raise ExistingSandError("E2 holdout check needs one role=holdout prompt view")
    capture_id = mask_record["parameters"]["capture_id"]
    observation_stage, views = _observation_views(run_root, capture_id)
    observation_record = _load_yaml(observation_stage / "record.yaml", "PGSR capture record")
    sh_degree = observation_record.get("parameters", {}).get("sh_degree", 3)
    stage = _stage_location(run_root, "segmentation", check_id)
    if stage.exists():
        raise ExistingSandError(f"refusing to overwrite E2 holdout check: {stage}")
    run = _load_yaml(run_root / "run.yaml", "run manifest")
    source = run["source"]
    labels = np.load(seed_stage / "gaussian_labels.npy", allow_pickle=False)
    if labels.shape != (source["gaussian_count"],) or not np.isin(labels, [0, 1, 2]).all():
        raise ExistingSandError("source Gaussian label count or values mismatch")
    if not torch.cuda.is_available():
        raise ExistingSandError("E2 holdout PGSR rasterization requires CUDA")
    model = GaussianModel(sh_degree)
    model.load_ply(str(Path(source["checkpoint"]) / source["point_cloud"]))
    if model.get_xyz.shape[0] != labels.size:
        raise ExistingSandError("PGSR loaded point count differs from labels")
    color = torch.zeros((labels.size, 3), dtype=torch.float32, device="cuda")
    color[torch.from_numpy(labels == 1).to("cuda"), :] = 1.0
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device="cuda")
    previews = []
    with torch.no_grad():
        for entry in holdouts:
            view_dir, camera = views[entry["camera_id"]]
            intr = camera["colmap_intrinsics"]
            width, height = camera["render_size_wh"]
            w2c = np.asarray(camera["gs_w2c"])
            view = Camera(colmap_id=0, R=w2c[:3, :3].T.copy(),
                          T=w2c[:3, 3].copy(),
                          FoVx=focal2fov(intr["fx"], camera["source_size_wh"][0]),
                          FoVy=focal2fov(intr["fy"], camera["source_size_wh"][1]),
                          image_width=width, image_height=height,
                          image_path=str(view_dir / "source_rgb.png"),
                          image_name=entry["camera_id"], uid=0, preload_img=False)
            if not np.allclose(view.world_view_transform.T.detach().cpu().numpy(), w2c, atol=1e-5):
                raise ExistingSandError("held-out PGSR camera matrix mismatch")
            result = render(view, model, pipe, bg, override_color=color,
                            return_depth_normal=False, envmap=None)
            coverage = result["render"][0].detach().cpu().numpy()
            if coverage.shape != (height, width) or not np.isfinite(coverage).all():
                raise ExistingSandError("invalid held-out PGSR selection render")
            with Image.open(mask_stage / entry["directory"] / "mask.png") as img:
                expected = np.asarray(img.convert("L")) >= 128
            with Image.open(view_dir / "render_rgb.png") as img:
                rgb = np.asarray(img.convert("RGB"))
            predicted = coverage >= coverage_threshold
            intersection = int(np.count_nonzero(predicted & expected))
            union = int(np.count_nonzero(predicted | expected))
            overlay = rgb.copy()
            overlay[predicted & expected] = [30, 220, 80]
            overlay[predicted & ~expected] = [255, 35, 35]
            overlay[~predicted & expected] = [50, 100, 255]
            previews.append((entry, coverage.astype(np.float32), predicted,
                             overlay, {"camera_id": entry["camera_id"],
                                       "mask_iou_vs_sam": intersection / union if union else None,
                                       "predicted_pixels": int(predicted.sum()),
                                       "sam_pixels": int(expected.sum()),
                                       "false_positive_pixels": int(np.count_nonzero(predicted & ~expected)),
                                       "false_negative_pixels": int(np.count_nonzero(~predicted & expected))}))
            del view, result
    stage.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for i, (_entry, coverage, binary, overlay, _score) in enumerate(previews):
        name = f"view_{i:04d}"
        (stage / name).mkdir()
        np.save(stage / name / "selected_coverage.npy", coverage, allow_pickle=False)
        Image.fromarray(binary.astype(np.uint8) * 255).save(stage / name / "selected_mask.png")
        Image.fromarray(overlay).save(stage / name / "overlay.png")
        artifacts.extend({"path": f"{name}/{file}", "role": role} for file, role in
                         (("selected_coverage.npy", "pgsr_selected_contribution"),
                          ("selected_mask.png", "selected_threshold_mask"),
                          ("overlay.png", "holdout_error_overlay")))
    scores = [item[-1] for item in previews]
    _write_yaml(stage / "quality.yaml", {"coverage_threshold": coverage_threshold,
                                         "holdout_views": scores,
                                         "reference": "SAM holdout mask; not independent ground truth"})
    artifacts.append({"path": "quality.yaml", "role": "heldout_pgsr_render_metrics"})
    _write_yaml(stage / "record.yaml", {
        "schema_version": 1, "stage": "segmentation", "stage_id": check_id,
        "camera_ids": [entry["camera_id"] for entry in holdouts],
        "prompt": {"type": "pgsr_rasterized_holdout", "seed_id": seed_id},
        "seed": None, "models": [],
        "parameters": {"coverage_threshold": coverage_threshold,
                       "prediction": "positive label rendered with original PGSR geometry and opacity"},
        "commands": [command] if command else [], "command_cwd": str(repo_root),
        "parents": [{"stage": "segmentation", "stage_id": seed_id},
                    {"stage": "segmentation", "stage_id": mask_id}],
        "artifacts": artifacts,
        "code_files": ["sandygs/existing_sand/e2_render.py", "sandygs/existing_sand/e2.py",
                       "scripts/experiments/existing_sand_e2.py", "gaussian_renderer/__init__.py",
                       "scene/cameras.py", "scene/gaussian_model.py"]})
    record_stage(run_root, stage / "record.yaml", repo_root=repo_root)
    return {"stage_dir": str(stage), "holdout_views": scores}
