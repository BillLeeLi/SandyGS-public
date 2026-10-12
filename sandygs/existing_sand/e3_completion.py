"""E3 reference-view background completion from an approved derived PGSR scene.

The first stage renders the derived checkpoint and prepares a shared RGB/depth
edit mask. The second invokes local SDXL and InFusion workers. Neither stage
adds Gaussians; recovered depth/points are inputs to later local optimization.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion
import yaml

from .e1 import _crop, _read_binary_mask
from .e3_checkpoint import fitted_checkpoint
from .e1_run import (_INFUSION_CONFIG, _INFUSION_WEIGHTS, _SDXL_CONFIG,
                     _SDXL_WEIGHTS, _as_depth, _model_files, _model_revision,
                     _run_logged)
from .geometry import (backproject_camera_z, crop_resize_intrinsics,
                       depth_products, project_world, validate_pinhole)
from .workspace import (ExistingSandError, _load_yaml, _safe_relative,
                        _stage_location, record_stage, verify_run)


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)


def _image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def _derived_inputs(run_root: Path, derived_id: str, camera_id: str):
    derived = _stage_location(run_root, "background", derived_id)
    if not (derived / "manifest.yaml").is_file():
        raise ExistingSandError(f"unregistered E3 derived checkpoint: {derived}")
    record = _load_yaml(derived / "record.yaml", "E3 derived scene record")
    if record.get("prompt", {}).get("type") != "approved_gaussian_deletion":
        raise ExistingSandError("background stage is not an approved E3 deletion")
    selection_id = record["prompt"]["selection_id"]
    selection = _stage_location(run_root, "segmentation", selection_id)
    selection_record = _load_yaml(selection / "record.yaml", "E2 selection record")
    mask_id = selection_record["prompt"]["mask_id"]
    masks = _stage_location(run_root, "segmentation", mask_id)
    mask_quality = _load_yaml(masks / "quality.yaml", "E2 mask quality")
    entries = [item for item in mask_quality["views"] if item["camera_id"] == camera_id]
    if len(entries) != 1:
        raise ExistingSandError("reference camera must have exactly one E2 SAM mask")
    mask = masks / entries[0]["directory"] / "mask.png"
    capture = [item for item in record["parents"] if item["stage"] == "observations"]
    if len(capture) != 1:
        raise ExistingSandError("E3 derived scene needs one observation parent")
    observed = _stage_location(run_root, "observations", capture[0]["stage_id"])
    matches = []
    for camera_file in observed.glob("view_*/camera.yaml"):
        camera = _load_yaml(camera_file, "PGSR observation camera")
        if camera.get("camera_id") == camera_id:
            matches.append((camera_file.parent, camera))
    if len(matches) != 1:
        raise ExistingSandError("reference camera must have exactly one PGSR observation")
    return derived, masks, observed, mask, matches[0][0], matches[0][1]


def _render_derived(checkpoint: Path, cloud_relative: str, camera: dict,
                    observation_record: dict, image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render the same PGSR RGB/plane-depth/alpha contract as source capture."""
    import torch
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import focal2fov

    if not torch.cuda.is_available():
        raise ExistingSandError("derived PGSR reference rendering requires CUDA")
    params = observation_record.get("parameters", {})
    sh_degree = params.get("sh_degree")
    if not isinstance(sh_degree, int) or isinstance(sh_degree, bool):
        raise ExistingSandError("observation record needs its PGSR sh_degree")
    model = GaussianModel(sh_degree)
    model.load_ply(str(checkpoint / cloud_relative))
    width, height = camera["render_size_wh"]
    source_width, source_height = camera["source_size_wh"]
    intr = camera["colmap_intrinsics"]
    w2c = np.asarray(camera["gs_w2c"], dtype=np.float64)
    K = np.asarray(camera["pgsr_K"], dtype=np.float64)
    validate_pinhole(K, w2c)
    view = Camera(colmap_id=0, R=w2c[:3, :3].T.copy(), T=w2c[:3, 3].copy(),
                  FoVx=focal2fov(intr["fx"], source_width),
                  FoVy=focal2fov(intr["fy"], source_height),
                  image_width=width, image_height=height,
                  image_path=str(image_path),
                  image_name=camera["camera_id"], uid=0, preload_img=False)
    if not np.allclose(view.world_view_transform.T.detach().cpu().numpy(), w2c, atol=1e-5):
        raise ExistingSandError("derived PGSR camera differs from captured camera")
    rendered_K = np.array([[view.fx, 0, view.cx], [0, view.fy, view.cy], [0, 0, 1]])
    if not np.allclose(rendered_K, K, atol=1e-5, rtol=0):
        raise ExistingSandError("derived PGSR intrinsics differ from captured intrinsics")
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.ones(3, device="cuda") if params.get("white_background", False) else torch.zeros(3, device="cuda")
    with torch.no_grad():
        result = render(view, model, pipeline, bg, return_depth_normal=False, envmap=None)
    rgb = result["render"].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    depth = result["plane_depth"].detach().cpu().numpy().squeeze(0).astype(np.float32)
    alpha = result["rendered_alpha"].detach().cpu().numpy().squeeze(0).astype(np.float32)
    if rgb.shape != (height, width, 3) or depth.shape != (height, width) or not np.isfinite(rgb).all():
        raise ExistingSandError("derived PGSR render shape or RGB is invalid")
    return rgb, depth, alpha


def _masked_inverse(depth: np.ndarray, valid: np.ndarray, edit: np.ndarray,
                    *, ring_pixels: int = 16) -> tuple[np.ndarray, dict[str, Any]]:
    if depth.shape != valid.shape or valid.shape != edit.shape or valid.dtype != np.bool_ or edit.dtype != np.bool_:
        raise ExistingSandError("PGSR depth/valid/edit mask shape or dtype mismatch")
    known = ~edit
    if not known.any() or not np.all(valid[known]):
        raise ExistingSandError("crop has invalid PGSR depth outside edit mask; change crop or enlarge edit mask")
    ring = binary_dilation(edit, iterations=ring_pixels) & known
    if not ring.any():
        raise ExistingSandError("edit mask needs a valid known-depth context ring")
    inverse = np.empty(depth.shape, dtype=np.float32)
    inverse[known] = np.reciprocal(depth[known], dtype=np.float64).astype(np.float32)
    fill = float(np.median(inverse[ring]))
    inverse[edit] = fill
    if not np.isfinite(inverse).all() or np.any(inverse <= 0):
        raise ExistingSandError("masked inverse PGSR camera-Z is not finite and positive")
    return inverse, {"known_inverse_min": float(inverse[known].min()),
                     "known_inverse_max": float(inverse[known].max()),
                     "hole_fill_inverse": fill, "fill_ring_pixels": ring_pixels,
                     "infusion_resize_scale": 1.0,
                     "infusion_normalization_epsilon": 1e-5}


def prepare_background(run_root: Path, derived_id: str, camera_id: str,
                       stage_id: str, crop_xywh: tuple[int, int, int, int], *,
                       repo_root: Path, expand_pixels: int = 0,
                       protect_mask: Path | None = None, threshold: int = 128,
                       source_fit_id: str | None = None, edit_mask: Path | None = None,
                       command: list[str] | None = None) -> dict[str, Any]:
    """Render deleted scene and make inspectable RGB/depth conditions."""
    run_root, repo_root = Path(run_root).resolve(), Path(repo_root).resolve()
    verify_run(run_root)
    derived, masks, observed, sam_path, view, camera = _derived_inputs(run_root, derived_id, camera_id)
    if not isinstance(expand_pixels, int) or isinstance(expand_pixels, bool) or expand_pixels < 0:
        raise ExistingSandError("expand_pixels must be a nonnegative integer")
    if not isinstance(threshold, int) or not 1 <= threshold <= 255:
        raise ExistingSandError("mask threshold must be in [1, 255]")
    size_wh = tuple(camera["render_size_wh"])
    rows, cols = _crop(crop_xywh, size_wh)
    sam, ambiguous = _read_binary_mask(sam_path, size_wh, "E2 SAM mask", threshold)
    authored = sam
    if edit_mask is not None:
        authored, _ = _read_binary_mask(Path(edit_mask), size_wh, "authored edit mask", threshold)
    candidate = binary_dilation(authored, iterations=expand_pixels) if expand_pixels else authored
    if protect_mask is not None:
        protected, protect_ambiguous = _read_binary_mask(Path(protect_mask), size_wh, "protect mask", threshold)
    else:
        protected, protect_ambiguous = np.zeros_like(sam), 0
    edit_full = candidate & ~protected
    edit = edit_full[rows, cols]
    if not edit.any() or edit.all() or int(edit.sum()) != int(edit_full.sum()):
        raise ExistingSandError("crop must contain the entire nonempty edit mask and leave known context")
    stage = _stage_location(run_root, "segmentation", stage_id)
    if stage.exists():
        raise ExistingSandError(f"refusing to overwrite E3 background preparation: {stage}")
    derived_record = _load_yaml(derived / "record.yaml", "derived scene record")
    obs_record = _load_yaml(observed / "record.yaml", "PGSR observation record")
    point_cloud = derived_record["parameters"]["point_cloud"]
    cloud_relative, cloud = _safe_relative(derived / "checkpoint", point_cloud, "derived point cloud")
    derived_manifest = _load_yaml(derived / "manifest.yaml", "E3 derived manifest")
    if not cloud.is_file() or f"checkpoint/{cloud_relative}" not in {
        item["path"] for item in derived_manifest["artifacts"]
    }:
        raise ExistingSandError("derived point cloud is not a registered checkpoint artifact")
    render_stage = derived
    if source_fit_id is not None:
        render_stage, point_cloud, _, _ = fitted_checkpoint(run_root, source_fit_id, derived_id)
    rgb_full, depth_full, alpha_full = _render_derived(
        render_stage / "checkpoint", point_cloud, camera, obs_record, view / "source_rgb.png")
    alpha_min = obs_record.get("parameters", {}).get("alpha_min", 0.1)
    depth_full, valid_full, _ = depth_products(depth_full, alpha_full, alpha_min=alpha_min)
    rgb, depth, alpha, valid = (array[rows, cols] for array in
                                (rgb_full, depth_full, alpha_full, valid_full))
    K = crop_resize_intrinsics(np.asarray(camera["pgsr_K"]), crop_xywh,
                               (crop_xywh[2], crop_xywh[3]))
    w2c = np.asarray(camera["gs_w2c"], dtype=np.float64)
    validate_pinhole(K, w2c)
    inverse, normalization = _masked_inverse(depth, valid, edit)
    original_rgb = np.load(view / "render_rgb.npy", allow_pickle=False)[rows, cols]
    original_depth = np.load(view / "plane_depth.npy", allow_pickle=False)[rows, cols]
    if original_rgb.shape != rgb.shape or original_depth.shape != depth.shape:
        raise ExistingSandError("source and derived PGSR views differ in size")
    # The original scene may contain sand in the hole, so no hole metric is computed.
    known_delta = np.abs(rgb[~edit] - original_rgb[~edit])
    payload = Path(tempfile.mkdtemp(prefix=f".e3_prepare_{stage_id}_", dir=run_root))
    published = False
    sealed = False
    try:
        arrays = {"derived_rgb.npy": (rgb, "derived_pgsr_rgb"),
                  "derived_depth.npy": (depth, "derived_pgsr_camera_z"),
                  "derived_alpha.npy": (alpha, "derived_pgsr_alpha"),
                  "derived_depth_valid.npy": (valid, "derived_pgsr_validity"),
                  "source_rgb.npy": (original_rgb, "before_deletion_pgsr_rgb"),
                  "source_depth.npy": (original_depth, "before_deletion_pgsr_camera_z"),
                  "edit_mask.npy": (edit, "shared_rgb_depth_edit_mask"),
                  "input_inverse_depth.npy": (inverse, "masked_inverse_pgsr_camera_z"),
                  "camera_K.npy": (K, "cropped_pgsr_intrinsics"),
                  "camera_c2w.npy": (np.linalg.inv(w2c), "pgsr_camera_to_world")}
        artifacts = []
        for name, (array, role) in arrays.items():
            np.save(payload / name, array, allow_pickle=False)
            artifacts.append({"path": name, "role": role})
        condition = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        overlay = condition.copy()
        overlay[edit] = np.round(0.55 * overlay[edit] + 0.45 * np.array([255, 45, 30])).astype(np.uint8)
        _image(payload / "condition_rgb.png", condition)
        _image(payload / "mask_overlay.png", overlay)
        _image(payload / "edit_mask.png", edit.astype(np.uint8) * 255)
        artifacts.extend({"path": name, "role": role} for name, role in
                         (("condition_rgb.png", "derived_rgb_condition"),
                          ("mask_overlay.png", "edit_mask_overlay"),
                          ("edit_mask.png", "shared_rgb_depth_edit_mask_preview")))
        shutil.copy2(sam_path, payload / "sam_mask.png")
        artifacts.append({"path": "sam_mask.png", "role": "e2_sam_reference_mask"})
        if protect_mask is not None:
            shutil.copy2(protect_mask, payload / "protect_mask.png")
            artifacts.append({"path": "protect_mask.png", "role": "authored_protection_mask"})
        if edit_mask is not None:
            shutil.copy2(edit_mask, payload / "authored_edit_mask.png")
            artifacts.append({"path": "authored_edit_mask.png", "role": "authored_edit_mask"})
        crop = {"derived_id": derived_id, "camera_id": camera_id,
                "source_view": view.name, "source_size_wh": list(size_wh),
                "crop_xywh": list(crop_xywh),
                "model_size_wh": [crop_xywh[2], crop_xywh[3]],
                "pixel_centers": "integer_u_right_v_down",
                "depth": "positive_pgsr_camera_z",
                "sam_ambiguous_pixels": ambiguous,
                "protect_ambiguous_pixels": protect_ambiguous,
                "expand_pixels": expand_pixels, "threshold": threshold,
                "edit_pixels": int(edit.sum())}
        if source_fit_id is not None:
            crop["source_fit_id"] = source_fit_id
        _write_yaml(payload / "crop.yaml", crop)
        _write_yaml(payload / "normalization.yaml", normalization)
        quality = {"edit_pixels": int(edit.sum()),
                   "known_pixels": int((~edit).sum()),
                   "derived_invalid_inside_edit": int(np.count_nonzero(~valid & edit)),
                   "derived_invalid_outside_edit": int(np.count_nonzero(~valid & ~edit)),
                   "known_rgb_mae_vs_source_before_deletion": float(known_delta.mean()),
                   "known_rgb_p95_vs_source_before_deletion": float(np.percentile(known_delta, 95)),
                   "depth_inside_edit_is_not_reference_truth": True}
        _write_yaml(payload / "quality.yaml", quality)
        artifacts.extend({"path": name, "role": role} for name, role in
                         (("crop.yaml", "crop_and_camera_contract"),
                          ("normalization.yaml", "inverse_depth_normalization"),
                          ("quality.yaml", "deletion_and_mask_quality")))
        record = {"schema_version": 1, "stage": "segmentation", "stage_id": stage_id,
                  "camera_ids": [camera_id],
                  "prompt": {"type": "derived_sand_edit_mask", "sam_mask": "sam_mask.png"},
                  "seed": None, "models": [], "parameters": crop,
                  "commands": [command] if command else [], "command_cwd": str(repo_root),
                  "parents": [{"stage": "background", "stage_id": derived_id},
                              {"stage": "segmentation", "stage_id": masks.name},
                              {"stage": "observations", "stage_id": observed.name}],
                  "artifacts": artifacts,
                  "code_files": ["sandygs/existing_sand/e3_completion.py",
                                 "scripts/experiments/existing_sand_e3.py",
                                 "gaussian_renderer/__init__.py", "scene/gaussian_model.py"]}
        if source_fit_id is not None:
            record["parents"].append({"stage": "background", "stage_id": source_fit_id})
        record["code_files"].append("sandygs/existing_sand/e3_checkpoint.py")
        _write_yaml(payload / "record.yaml", record)
        stage.parent.mkdir(parents=True, exist_ok=True)
        payload.rename(stage)
        published = True
        record_stage(run_root, stage / "record.yaml", repo_root=repo_root)
        sealed = True
        return {"stage_dir": str(stage), **quality}
    except BaseException:
        if payload.exists():
            shutil.rmtree(payload)
        if published and not sealed and stage.exists():
            shutil.rmtree(stage)
        raise


def _depth_preview(depth: np.ndarray, scale: tuple[float, float]) -> Image.Image:
    low, high = scale
    good = np.isfinite(depth) & (depth > 0)
    t = np.clip((np.where(good, depth, low) - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.stack((t, np.sin(np.pi * t) * 0.85, 1 - t), axis=-1)
    rgb[~good] = 0
    return Image.fromarray(np.round(rgb * 255).astype(np.uint8))


def _report(payload: Path, known_depth: np.ndarray, predicted_depth: np.ndarray,
            completed_depth: np.ndarray, mask: np.ndarray, K: np.ndarray,
            condition_rgb: np.ndarray, completed_rgb: np.ndarray) -> dict[str, Any]:
    valid_prediction = mask & np.isfinite(predicted_depth) & (predicted_depth > 0)
    known = (~mask) & np.isfinite(known_depth) & (known_depth > 0)
    if not valid_prediction.any():
        raise ExistingSandError("InFusion produced no valid positive depth inside edit mask")
    ring = binary_dilation(mask, iterations=8) & ~mask
    inner = mask & ~binary_erosion(mask, iterations=8)
    adjacent = ring & known
    values = known_depth[known]
    scale = tuple(float(v) for v in np.percentile(values, [2, 98]))
    h_jump = np.abs(completed_depth[:, 1:] - completed_depth[:, :-1])
    v_jump = np.abs(completed_depth[1:, :] - completed_depth[:-1, :])
    h_edge = mask[:, 1:] != mask[:, :-1]
    v_edge = mask[1:, :] != mask[:-1, :]
    jumps = np.concatenate((h_jump[h_edge & np.isfinite(h_jump)],
                            v_jump[v_edge & np.isfinite(v_jump)]))
    quality = {"reference": "derived PGSR known region; original sand-covered hole has no ground truth",
               "edit_pixels": int(mask.sum()),
               "predicted_valid_pixels": int(valid_prediction.sum()),
               "predicted_valid_fraction": float(valid_prediction.sum() / mask.sum()),
               "predicted_depth_min": float(predicted_depth[valid_prediction].min()),
               "predicted_depth_max": float(predicted_depth[valid_prediction].max()),
               "known_depth_min": float(values.min()),
               "known_depth_max": float(values.max()),
               "known_depth_exactly_preserved": bool(np.array_equal(completed_depth[known], known_depth[known])),
               "known_rgb_exactly_preserved": bool(np.array_equal(completed_rgb[~mask], condition_rgb[~mask])),
               "inner_boundary_valid_pixels": int(np.count_nonzero(inner & valid_prediction)),
               "outer_known_ring_pixels": int(adjacent.sum()),
               "boundary_depth_jump_mean": float(jumps.mean()) if jumps.size else None,
               "boundary_depth_jump_p95": float(np.percentile(jumps, 95)) if jumps.size else None,
               "boundary_depth_jump_pairs": int(jumps.size)}
    report = payload / "report"
    report.mkdir()
    _write_yaml(report / "quality.yaml", quality)
    panels = [Image.fromarray(np.round(np.clip(condition_rgb, 0, 1) * 255).astype(np.uint8)),
              Image.fromarray(np.round(np.clip(completed_rgb, 0, 1) * 255).astype(np.uint8)),
              Image.fromarray(mask.astype(np.uint8) * 255).convert("RGB"),
              _depth_preview(known_depth, scale), _depth_preview(predicted_depth, scale),
              _depth_preview(completed_depth, scale)]
    labels = ["Derived PGSR RGB", "Completed RGB", "Shared edit mask",
              "Derived PGSR Z", "Predicted Z", "Completed Z"]
    w, h = panels[0].size
    canvas = Image.new("RGB", (3 * w, 2 * (h + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        x, y = (i % 3) * w, (i // 3) * (h + 24)
        draw.text((x + 4, y + 4), label, fill="black")
        canvas.paste(panel, (x, y + 24))
    canvas.save(report / "overview.png")
    return quality


def run_background_completion(run_root: Path, prepare_id: str, stage_id: str, *,
                              repo_root: Path, model_python: Path, infusion_code: Path,
                              infusion_model: Path, sdxl_model: Path, prompt: str,
                              seed: int, negative_prompt: str = "", rgb_steps: int = 20,
                              depth_steps: int = 20, guidance_scale: float = 8.0,
                              command: list[str] | None = None) -> dict[str, Any]:
    """Run local SDXL and InFusion, preserving derived known pixels exactly."""
    run_root, repo_root = Path(run_root).resolve(), Path(repo_root).resolve()
    # Resolving a venv Python symlink would bypass the venv.
    model_python = Path(model_python).expanduser().absolute()
    infusion_code = Path(infusion_code).resolve()
    infusion_model = Path(infusion_model).resolve()
    sdxl_model = Path(sdxl_model).resolve()
    verify_run(run_root)
    prepared = _stage_location(run_root, "segmentation", prepare_id)
    if not (prepared / "manifest.yaml").is_file():
        raise ExistingSandError(f"unregistered E3 preparation stage: {prepared}")
    prep_record = _load_yaml(prepared / "record.yaml", "E3 preparation record")
    if prep_record.get("prompt", {}).get("type") != "derived_sand_edit_mask":
        raise ExistingSandError("prepare_id must identify an E3 derived-scene edit mask")
    target = _stage_location(run_root, "background", stage_id)
    if target.exists():
        raise ExistingSandError(f"refusing to overwrite E3 completion: {target}")
    if not model_python.is_file():
        raise ExistingSandError(f"missing model Python: {model_python}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ExistingSandError("SDXL requires a nonempty background prompt")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ExistingSandError("seed must be a nonnegative integer")
    if min(rgb_steps, depth_steps) < 1 or not np.isfinite(guidance_scale) or guidance_scale < 0:
        raise ExistingSandError("steps must be positive and guidance_scale nonnegative")
    _model_files(infusion_model, _INFUSION_WEIGHTS + _INFUSION_CONFIG, "InFusion checkpoint")
    _model_files(sdxl_model, _SDXL_WEIGHTS + _SDXL_CONFIG, "SDXL checkpoint")
    upstream_worker = infusion_code / "depth_inpainting/run/run_inference_inpainting.py"
    if not upstream_worker.is_file():
        raise ExistingSandError(f"missing InFusion inference script: {upstream_worker}")
    mask = np.load(prepared / "edit_mask.npy", allow_pickle=False)
    known_depth = np.load(prepared / "derived_depth.npy", allow_pickle=False)
    K = np.load(prepared / "camera_K.npy", allow_pickle=False)
    c2w = np.load(prepared / "camera_c2w.npy", allow_pickle=False)
    if mask.dtype != np.bool_ or known_depth.shape != mask.shape or not mask.any():
        raise ExistingSandError("E3 preparation arrays are inconsistent")
    crop = _load_yaml(prepared / "crop.yaml", "E3 crop")
    if tuple(crop["model_size_wh"]) != (mask.shape[1], mask.shape[0]):
        raise ExistingSandError("E3 crop and mask size differ")
    with Image.open(prepared / "condition_rgb.png") as image:
        condition_byte = np.asarray(image.convert("RGB"), dtype=np.uint8)
    condition = condition_byte.astype(np.float32) / 255.0
    if condition.shape != mask.shape + (3,):
        raise ExistingSandError("E3 condition RGB and mask size differ")
    validate_pinhole(K, np.linalg.inv(c2w))
    commands = [command] if command else []
    with tempfile.TemporaryDirectory(prefix="e3_work_", dir=run_root) as temp:
        work = Path(temp)
        rgb_output = work / "sdxl_raw.png"
        rgb_command = [str(model_python), "-B", str(repo_root / "scripts/experiments/existing_sand_sdxl_worker.py"),
                       "--model", str(sdxl_model), "--image", str(prepared / "condition_rgb.png"),
                       "--mask", str(prepared / "edit_mask.png"), "--output", str(rgb_output),
                       "--prompt", prompt, "--negative-prompt", negative_prompt,
                       "--seed", str(seed), "--steps", str(rgb_steps),
                       "--guidance-scale", str(guidance_scale)]
        _run_logged(rgb_command, cwd=repo_root, log=work / "sdxl.log")
        commands.append(rgb_command)
        with Image.open(rgb_output) as image:
            if image.size != (mask.shape[1], mask.shape[0]):
                raise ExistingSandError("SDXL output size differs from E3 crop")
            raw_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        completed_byte = condition_byte.copy()
        completed_byte[mask] = raw_rgb[mask]
        completed_rgb = completed_byte.astype(np.float32) / 255.0
        completed_rgb_path = work / "completed_rgb.png"
        _image(completed_rgb_path, completed_byte)
        upstream_output = work / "infusion"
        upstream_output.mkdir()
        depth_command = [str(model_python), "-B", str(upstream_worker),
                         "--input_rgb_path", str(completed_rgb_path),
                         "--input_mask", str(prepared / "edit_mask.png"),
                         "--input_depth_path", str(prepared / "input_inverse_depth.npy"),
                         "--model_path", str(infusion_model), "--output_dir", str(upstream_output),
                         "--intri", str(prepared / "camera_K.npy"),
                         "--c2w", str(prepared / "camera_c2w.npy"),
                         "--processing_res", "768", "--denoise_steps", str(depth_steps),
                         "--seed", str(seed), "--use_mask", "--blend"]
        _run_logged(depth_command, cwd=upstream_worker.parent, log=work / "infusion.log")
        commands.append(depth_command)
        predicted_inverse = _as_depth(np.load(upstream_output / "completed_rgb_depth_dis.npy", allow_pickle=False),
                                      mask.shape, "InFusion inverse depth")
        good = np.isfinite(predicted_inverse) & (predicted_inverse > 0)
        predicted_depth = np.full(mask.shape, np.nan, dtype=np.float32)
        predicted_depth[good] = 1.0 / predicted_inverse[good]
        completed_depth = known_depth.astype(np.float32).copy()
        completed_depth[mask] = predicted_depth[mask]
        payload = work / "stage"
        payload.mkdir()
        np.save(payload / "predicted_inverse_depth.npy", predicted_inverse, allow_pickle=False)
        np.save(payload / "predicted_depth.npy", predicted_depth, allow_pickle=False)
        np.save(payload / "completed_depth.npy", completed_depth, allow_pickle=False)
        np.save(payload / "completed_rgb.npy", completed_rgb, allow_pickle=False)
        _image(payload / "sdxl_raw.png", raw_rgb)
        _image(payload / "completed_rgb.png", completed_byte)
        shutil.copy2(work / "sdxl.log", payload / "sdxl.log")
        shutil.copy2(work / "infusion.log", payload / "infusion.log")
        point_file = upstream_output / "completed_rgb_mask.ply"
        if point_file.is_file():
            shutil.copy2(point_file, payload / "inference_points.ply")
        quality = _report(payload, known_depth, predicted_depth, completed_depth,
                          mask, K, condition, completed_rgb)
        valid_points = mask & np.isfinite(completed_depth) & (completed_depth > 0)
        xyz = backproject_camera_z(completed_depth, K, np.linalg.inv(c2w), valid_points)
        v, u = np.nonzero(valid_points)
        points = xyz[valid_points].astype(np.float32)
        pixels = np.stack((u, v), axis=1).astype(np.int32)
        np.save(payload / "filled_points_world.npy", points, allow_pickle=False)
        np.save(payload / "filled_pixels_uv.npy", pixels, allow_pickle=False)
        np.save(payload / "filled_rgb.npy", completed_rgb[valid_points], allow_pickle=False)
        reproject, camera_z = project_world(points, K, np.linalg.inv(c2w))
        residual = np.abs(reproject - pixels)
        quality["filled_points"] = int(len(points))
        quality["reprojection_max_pixel_error"] = float(np.max(residual)) if len(points) else None
        quality["reprojection_max_depth_error"] = float(np.max(np.abs(camera_z - completed_depth[valid_points]))) if len(points) else None
        if len(points) and (quality["reprojection_max_pixel_error"] > 1e-3 or quality["reprojection_max_depth_error"] > 1e-3):
            raise ExistingSandError("E3 filled point reprojection check failed")
        (payload / "report/quality.yaml").write_text(yaml.safe_dump(quality, sort_keys=False), encoding="utf-8")
        model_entries = [{"name": "InFusion " + name.split("/")[0],
                          "version": _model_revision(infusion_model, _INFUSION_WEIGHTS[0]),
                          "checkpoint": str(infusion_model / name)} for name in _INFUSION_WEIGHTS]
        code_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=infusion_code,
                                       capture_output=True, text=True, check=False).stdout.strip() or "unknown"
        model_entries.extend({"name": "InFusion source " + name, "version": code_revision,
                              "checkpoint": str(infusion_code / name)}
                             for name in ("depth_inpainting/run/run_inference_inpainting.py",
                                          "depth_inpainting/inference/depth_inpainting_pipeline_half.py",
                                          "depth_inpainting/utils/image_util.py"))
        model_entries.extend({"name": "SDXL " + name.split("/")[0],
                              "version": _model_revision(sdxl_model, _SDXL_WEIGHTS[0]),
                              "checkpoint": str(sdxl_model / name)} for name in _SDXL_WEIGHTS)
        artifacts = [{"path": path.relative_to(payload).as_posix(), "role": "e3_" + path.stem}
                     for path in sorted(payload.rglob("*")) if path.is_file()]
        record = {"schema_version": 1, "stage": "background", "stage_id": stage_id,
                  "camera_ids": [crop["camera_id"]],
                  "prompt": {"type": "sdxl_background_completion", "text": prompt,
                             "negative_text": negative_prompt},
                  "seed": seed, "models": model_entries,
                  "parameters": {"mode": "derived_rgb_depth", "prepare_id": prepare_id,
                                 "derived_id": crop["derived_id"],
                                 "depth_source": "derived_PGSR_plane_depth_camera_z",
                                 "model_input": "masked_inverse_PGSR_camera_z",
                                 "known_region_composite": "derived_PGSR_rgb_and_camera_z",
                                 "rgb_steps": rgb_steps, "depth_steps": depth_steps,
                                 "guidance_scale": guidance_scale,
                                 "model_python": str(model_python),
                                 "infusion_code": str(infusion_code),
                                 "upstream_point_cloud": "inference_only_not_PGSR_Gaussians"},
                  "commands": commands, "command_cwd": str(repo_root),
                  "parents": [{"stage": "segmentation", "stage_id": prepare_id},
                              {"stage": "background", "stage_id": crop["derived_id"]}],
                  "artifacts": artifacts,
                  "code_files": ["sandygs/existing_sand/e3_completion.py",
                                 "sandygs/existing_sand/e1_run.py",
                                 "scripts/experiments/existing_sand_e3.py",
                                 "scripts/experiments/existing_sand_sdxl_worker.py"]}
        _write_yaml(payload / "record.yaml", record)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload.rename(target)
        try:
            record_stage(run_root, target / "record.yaml", repo_root=repo_root)
        except BaseException:
            shutil.rmtree(target)
            raise
    return {"stage_dir": str(target), "quality": quality}
