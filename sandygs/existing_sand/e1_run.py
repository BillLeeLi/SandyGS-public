"""Run the isolated InFusion/SDXL workers and register an E1 result stage."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
from PIL import Image
import yaml

from .e1_evaluate import save_depth_report
from .workspace import ExistingSandError, _load_yaml, _stage_location, record_stage, verify_run


_INFUSION_WEIGHTS = (
    "text_encoder/model.safetensors", "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
)
_INFUSION_CONFIG = (
    "scheduler/scheduler_config.json", "text_encoder/config.json",
    "tokenizer/merges.txt", "tokenizer/vocab.json", "unet/config.json",
    "vae/config.json",
)
_SDXL_WEIGHTS = (
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/model.fp16.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)
_SDXL_CONFIG = (
    "model_index.json", "scheduler/scheduler_config.json",
    "text_encoder/config.json", "text_encoder_2/config.json",
    "tokenizer/merges.txt", "tokenizer/vocab.json",
    "tokenizer_2/merges.txt", "tokenizer_2/vocab.json",
    "unet/config.json", "vae/config.json",
)


def _model_files(root: Path, files: tuple[str, ...], label: str) -> None:
    missing = [name for name in files if not (root / name).is_file()]
    if missing:
        raise ExistingSandError(f"{label} is incomplete at {root}: missing {missing}")


def _model_revision(root: Path, first_weight: str) -> str:
    metadata = root / ".cache/huggingface/download" / (first_weight + ".metadata")
    if metadata.is_file():
        revision = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
        if len(revision) == 40:
            return revision
    return "local_unpinned"


def _run_logged(argv: list[str], *, cwd: Path, log: Path) -> None:
    environment = os.environ.copy()
    # A repository PYTHONPATH can shadow InFusion's own ``utils`` package.
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(argv, cwd=cwd, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        raise ExistingSandError(
            f"model worker exited with status {result.returncode}; log: {log}\n{tail}"
        )


def _stage_inputs(run_root: Path, mask_id: str) -> tuple[Path, dict[str, Any]]:
    stage = _stage_location(run_root, "segmentation", mask_id)
    if not (stage / "manifest.yaml").is_file():
        raise ExistingSandError(f"unregistered E1 mask stage: {stage}")
    record = _load_yaml(stage / "record.yaml", "E1 mask stage")
    if record.get("prompt", {}).get("type") != "manual_synthetic_ground_hole":
        raise ExistingSandError("mask stage is not an E1 synthetic ground hole")
    return stage, _load_yaml(stage / "crop.yaml", "E1 crop")


def _as_depth(array: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.shape != shape or array.dtype.kind not in "fiu":
        raise ExistingSandError(f"{name} must be a numeric {shape[0]}x{shape[1]} array")
    return array.astype(np.float32)


def run_completion(run_root: Path, mask_id: str, stage_id: str, *,
                   repo_root: Path, model_python: Path, infusion_code: Path,
                   infusion_model: Path, seed: int, depth_steps: int = 20,
                   sdxl_model: Path | None = None, prompt: str | None = None,
                   negative_prompt: str = "", rgb_steps: int = 20,
                   guidance_scale: float = 8.0,
                   command: list[str] | None = None) -> dict[str, Any]:
    """Run depth-only or RGB+depth E1 without writing to source assets."""
    run_root = Path(run_root).resolve()
    repo_root = Path(repo_root).resolve()
    # Keep the venv executable path: resolving its symlink would bypass the venv.
    model_python = Path(model_python).expanduser().absolute()
    infusion_code = Path(infusion_code).resolve()
    infusion_model = Path(infusion_model).resolve()
    sdxl_model = Path(sdxl_model).resolve() if sdxl_model is not None else None
    verify_run(run_root)
    inputs, crop = _stage_inputs(run_root, mask_id)
    target = _stage_location(run_root, "background", stage_id)
    if target.exists():
        raise ExistingSandError(f"refusing to overwrite E1 completion: {target}")
    if not model_python.is_file():
        raise ExistingSandError(f"missing model Python: {model_python}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ExistingSandError("seed must be a nonnegative integer")
    if depth_steps < 1 or rgb_steps < 1 or not np.isfinite(guidance_scale) or guidance_scale < 0:
        raise ExistingSandError("steps must be positive and guidance_scale nonnegative")
    if sdxl_model is not None and (not isinstance(prompt, str) or not prompt.strip()):
        raise ExistingSandError("RGB+depth mode requires a nonempty prompt")
    _model_files(infusion_model, _INFUSION_WEIGHTS + _INFUSION_CONFIG, "InFusion checkpoint")
    upstream_worker = infusion_code / "depth_inpainting/run/run_inference_inpainting.py"
    if not upstream_worker.is_file():
        raise ExistingSandError(f"missing InFusion inference script: {upstream_worker}")
    if sdxl_model is not None:
        _model_files(sdxl_model, _SDXL_WEIGHTS + _SDXL_CONFIG, "SDXL checkpoint")
    mask = np.load(inputs / "edit_mask.npy", allow_pickle=False)
    reference = np.load(inputs / "reference_depth.npy", allow_pickle=False)
    with Image.open(inputs / "condition_rgb.png") as image:
        reference_rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    K = np.load(inputs / "camera_K.npy", allow_pickle=False)
    if mask.dtype != np.bool_ or reference.shape != mask.shape:
        raise ExistingSandError("E1 mask stage has inconsistent arrays")
    if tuple(crop["model_size_wh"]) != (mask.shape[1], mask.shape[0]):
        raise ExistingSandError("E1 crop metadata and mask shape disagree")

    commands: list[list[str]] = [command] if command else []
    with tempfile.TemporaryDirectory(prefix="e1_work_", dir=run_root) as temp_name:
        work = Path(temp_name)
        condition_rgb = inputs / "condition_rgb.png"
        raw_rgb: np.ndarray | None = None
        completed_rgb: np.ndarray | None = None
        if sdxl_model is not None:
            rgb_output = work / "sdxl_raw.png"
            rgb_command = [str(model_python), "-B",
                           str(repo_root / "scripts/experiments/existing_sand_sdxl_worker.py"),
                           "--model", str(sdxl_model),
                           "--image", str(inputs / "condition_rgb.png"),
                           "--mask", str(inputs / "edit_mask.png"),
                           "--output", str(rgb_output), "--prompt", prompt or "",
                           "--negative-prompt", negative_prompt,
                           "--seed", str(seed), "--steps", str(rgb_steps),
                           "--guidance-scale", str(guidance_scale)]
            _run_logged(rgb_command, cwd=repo_root, log=work / "sdxl.log")
            commands.append(rgb_command)
            with Image.open(rgb_output) as image:
                if image.size != (mask.shape[1], mask.shape[0]):
                    raise ExistingSandError("SDXL output size differs from E1 crop")
                raw_rgb_byte = np.asarray(image.convert("RGB"), dtype=np.uint8)
            raw_rgb = raw_rgb_byte.astype(np.float32) / 255.0
            completed_rgb = reference_rgb.astype(np.float32).copy()
            completed_rgb[mask] = raw_rgb[mask]
            condition_rgb = work / "completed_rgb.png"
            Image.fromarray(np.round(np.clip(completed_rgb, 0, 1) * 255).astype(np.uint8)).save(
                condition_rgb
            )

        upstream_output = work / "infusion"
        upstream_output.mkdir()
        depth_command = [str(model_python), "-B", str(upstream_worker),
                         "--input_rgb_path", str(condition_rgb),
                         "--input_mask", str(inputs / "edit_mask.png"),
                         "--input_depth_path", str(inputs / "input_inverse_depth.npy"),
                         "--model_path", str(infusion_model),
                         "--output_dir", str(upstream_output),
                         "--intri", str(inputs / "camera_K.npy"),
                         "--c2w", str(inputs / "camera_c2w.npy"),
                         "--processing_res", "768", "--denoise_steps", str(depth_steps),
                         "--seed", str(seed), "--use_mask", "--blend"]
        _run_logged(depth_command, cwd=upstream_worker.parent, log=work / "infusion.log")
        commands.append(depth_command)
        stem = condition_rgb.stem
        predicted_inv = _as_depth(np.load(upstream_output / f"{stem}_depth_dis.npy",
                                                  allow_pickle=False), mask.shape,
                                  "InFusion inverse depth")
        predicted = np.full(mask.shape, np.nan, dtype=np.float32)
        positive = np.isfinite(predicted_inv) & (predicted_inv > 0)
        predicted[positive] = 1.0 / predicted_inv[positive]
        completed = reference.astype(np.float32).copy()
        completed[mask] = predicted[mask]

        payload = work / "stage"
        payload.mkdir()
        np.save(payload / "predicted_inverse_depth.npy", predicted_inv, allow_pickle=False)
        np.save(payload / "predicted_depth.npy", predicted, allow_pickle=False)
        np.save(payload / "completed_depth.npy", completed, allow_pickle=False)
        shutil.copy2(work / "infusion.log", payload / "infusion.log")
        predicted_point_cloud = upstream_output / f"{stem}_mask.ply"
        if predicted_point_cloud.is_file():
            shutil.copy2(predicted_point_cloud, payload / "inference_points.ply")
        if sdxl_model is not None:
            shutil.copy2(work / "sdxl.log", payload / "sdxl.log")
            shutil.copy2(work / "sdxl_raw.png", payload / "sdxl_raw.png")
            shutil.copy2(condition_rgb, payload / "completed_rgb.png")
        report = save_depth_report(
            payload / "report", reference, predicted, completed, mask, K,
            reference_rgb, raw_predicted=predicted, completed_rgb=completed_rgb,
        )
        model_entries = [
            {"name": "InFusion " + name.split("/")[0],
             "version": _model_revision(infusion_model, _INFUSION_WEIGHTS[0]),
             "checkpoint": str(infusion_model / name)}
            for name in _INFUSION_WEIGHTS
        ]
        code_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=infusion_code,
            capture_output=True, text=True, check=False,
        ).stdout.strip() or "unknown"
        model_entries.extend([
            {"name": "InFusion source " + name,
             "version": code_revision, "checkpoint": str(infusion_code / name)}
            for name in ("depth_inpainting/run/run_inference_inpainting.py",
                         "depth_inpainting/inference/depth_inpainting_pipeline_half.py",
                         "depth_inpainting/utils/image_util.py")
        ])
        if sdxl_model is not None:
            model_entries.extend(
                {"name": "SDXL " + name.split("/")[0],
                 "version": _model_revision(sdxl_model, _SDXL_WEIGHTS[0]),
                 "checkpoint": str(sdxl_model / name)}
                for name in _SDXL_WEIGHTS
            )
        artifacts = [
            {"path": path.relative_to(payload).as_posix(),
             "role": "e1_" + path.stem}
            for path in sorted(payload.rglob("*")) if path.is_file()
        ]
        record = {
            "schema_version": 1, "stage": "background", "stage_id": stage_id,
            "camera_ids": [crop["camera_id"]],
            "prompt": {"type": "sdxl_text", "text": prompt,
                       "negative_text": negative_prompt} if sdxl_model else {},
            "seed": seed, "models": model_entries,
            "parameters": {
                "mode": "rgb_depth" if sdxl_model else "fixed_rgb_depth",
                "depth_steps": depth_steps, "rgb_steps": rgb_steps if sdxl_model else None,
                "guidance_scale": guidance_scale if sdxl_model else None,
                "model_python": str(model_python),
                "infusion_code": str(infusion_code),
                "mask_stage_id": mask_id,
                "depth_source": "PGSR_plane_depth_camera_z",
                "model_input": "masked_inverse_PGSR_camera_z",
                "known_region_composite": "original_PGSR_camera_z",
                "upstream_point_cloud": "inference_only_not_PGSR_Gaussians",
            },
            "commands": commands,
            "parents": [{"stage": "segmentation", "stage_id": mask_id}],
            "artifacts": artifacts,
            "code_files": [
                "sandygs/existing_sand/e1.py",
                "sandygs/existing_sand/e1_run.py",
                "sandygs/existing_sand/e1_evaluate.py",
                "scripts/experiments/existing_sand_e1.py",
                "scripts/experiments/existing_sand_sdxl_worker.py",
            ],
        }
        with (payload / "record.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(record, stream, sort_keys=False, allow_unicode=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload.rename(target)
        record_stage(run_root, target / "record.yaml", repo_root=repo_root)
    return {"stage_dir": str(target), "mode": record["parameters"]["mode"],
            "metrics": report}
