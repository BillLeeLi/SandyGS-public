#!/usr/bin/env python3
"""Isolated Grounding DINO inference; emit normalized boxes as JSON on stdout."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bert-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--box-threshold", type=float, required=True)
    parser.add_argument("--text-threshold", type=float, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code.resolve()))
    import torch
    from groundingdino.models import build_model
    from groundingdino.util.inference import load_image, predict
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

    with redirect_stdout(sys.stderr):
        cfg = SLConfig.fromfile(str(args.code / "groundingdino/config/GroundingDINO_SwinT_OGC.py"))
        cfg.device = "cuda"
        cfg.text_encoder_type = str(args.bert_dir.resolve())
        model = build_model(cfg)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(clean_state_dict(state["model"]), strict=False)
        model.eval().to("cuda")
        _, image = load_image(str(args.image))
        with torch.no_grad():
            boxes, scores, phrases = predict(
                model, image, args.text, box_threshold=args.box_threshold,
                text_threshold=args.text_threshold, device="cuda")
    print(json.dumps([{"box": box.tolist(), "score": float(score), "phrase": phrase}
                      for box, score, phrase in zip(boxes.cpu(), scores.cpu(), phrases)]))


if __name__ == "__main__":
    main()
