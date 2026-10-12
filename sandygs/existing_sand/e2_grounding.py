"""Grounding DINO text boxes in the same JSON prompt format consumed by E2 SAM."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from .e2 import _observation_views
from .workspace import ExistingSandError, verify_run


def normalized_cxcywh_to_xyxy(box, width: int, height: int):
    if len(box) != 4 or not all(0 <= float(value) <= 1 for value in box):
        raise ExistingSandError("invalid normalized Grounding DINO box")
    cx, cy, w, h = (float(value) for value in box)
    x0 = max(0., (cx - w / 2) * width)
    y0 = max(0., (cy - h / 2) * height)
    x1 = min(width - 1., (cx + w / 2) * width)
    y1 = min(height - 1., (cy + h / 2) * height)
    if x1 <= x0 or y1 <= y0:
        raise ExistingSandError("degenerate Grounding DINO box")
    return [x0, y0, x1, y1]


def propose_boxes(run_root: Path, capture_id: str, camera_ids: list[str], text: str,
                  output: Path, *, repo_root: Path, model_python: Path,
                  dino_code: Path, dino_checkpoint: Path, bert_dir: Path,
                  box_threshold: float = 0.35, text_threshold: float = 0.25,
                  max_candidates: int = 4) -> dict:
    """Emit reviewable candidates; the user picks a box before SAM inference."""
    run_root, repo_root, output = (Path(p).resolve() for p in (run_root, repo_root, output))
    verify_run(run_root)
    if not camera_ids or len(set(camera_ids)) != len(camera_ids) or not text.strip():
        raise ExistingSandError("specify distinct cameras and a nonempty text prompt")
    if not 0 < box_threshold < 1 or not 0 < text_threshold < 1 or max_candidates < 1:
        raise ExistingSandError("invalid Grounding DINO thresholds or max_candidates")
    _, views = _observation_views(run_root, capture_id)
    if any(camera_id not in views for camera_id in camera_ids):
        raise ExistingSandError("Grounding DINO camera_id must be a captured view")
    if output.exists():
        raise ExistingSandError(f"refusing to overwrite text candidate prompts: {output}")
    for path in (model_python, dino_checkpoint, bert_dir / "config.json"):
        if not Path(path).is_file():
            raise ExistingSandError(f"missing Grounding DINO runtime resource: {path}")
    if not any((bert_dir / item).is_file() for item in ("model.safetensors", "pytorch_model.bin")):
        raise ExistingSandError("missing local BERT model.safetensors or pytorch_model.bin")
    if not (dino_code / "groundingdino/config/GroundingDINO_SwinT_OGC.py").is_file():
        raise ExistingSandError(f"missing Grounding DINO config in {dino_code}")
    worker = repo_root / "scripts/experiments/existing_sand_dino_worker.py"
    result = []
    for camera_id in camera_ids:
        view, camera = views[camera_id]
        cmd = [str(model_python), str(worker), "--code", str(dino_code),
               "--checkpoint", str(dino_checkpoint), "--bert-dir", str(bert_dir),
               "--image", str(view / "render_rgb.png"), "--text", text,
               "--box-threshold", str(box_threshold),
               "--text-threshold", str(text_threshold)]
        proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)
        if proc.returncode:
            raise ExistingSandError(f"Grounding DINO failed on {camera_id}: {proc.stderr[-1500:]}")
        try:
            detections = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ExistingSandError(f"Grounding DINO returned invalid JSON for {camera_id}") from exc
        width, height = camera["render_size_wh"]
        candidates = sorted(({"box": normalized_cxcywh_to_xyxy(item["box"], width, height),
                              "score": float(item["score"]), "phrase": item["phrase"]}
                             for item in detections), key=lambda item: -item["score"])[:max_candidates]
        if candidates:
            result.append({"camera_id": camera_id, "role": "seed",
                           "box": candidates[0]["box"], "candidates": candidates,
                           "text": text, "source": "grounding_dino_swint_ogc"})
    if not result:
        raise ExistingSandError("no text candidates; adjust prompt or thresholds")
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = {"type": "grounding_dino_swint_ogc", "text": text,
                 "code_revision": subprocess.check_output(
                     ["git", "rev-parse", "HEAD"], cwd=dino_code, text=True).strip(),
                 "checkpoint": str(Path(dino_checkpoint).resolve()),
                 "bert_checkpoint": str((Path(bert_dir) / "model.safetensors" if
                                         (Path(bert_dir) / "model.safetensors").is_file() else
                                         Path(bert_dir) / "pytorch_model.bin").resolve()),
                 "box_threshold": box_threshold, "text_threshold": text_threshold}
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"generator": generator, "views": result}, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return {"prompts": str(output), "candidate_views": len(result),
            "instruction": "review/edit box and holdout role, then pass JSON to E2 masks"}
