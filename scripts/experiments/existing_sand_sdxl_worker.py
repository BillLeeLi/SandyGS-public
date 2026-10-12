#!/usr/bin/env python3
"""Isolated SDXL inpainting worker for E1; all model files must be local."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=8.0)
    args = parser.parse_args()

    import torch
    from diffusers import AutoPipelineForInpainting
    from PIL import Image

    if not torch.cuda.is_available():
        parser.error("SDXL E1 worker requires CUDA")
    with Image.open(args.image) as image:
        image = image.convert("RGB")
    with Image.open(args.mask) as mask:
        mask = mask.convert("L")
    if image.size != mask.size or any(size % 8 for size in image.size):
        parser.error("image and mask must match and have dimensions divisible by 8")
    pipeline = AutoPipelineForInpainting.from_pretrained(
        str(args.model), torch_dtype=torch.float16, variant="fp16",
        local_files_only=True,
    ).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with torch.inference_mode():
        result = pipeline(
            prompt=args.prompt, negative_prompt=args.negative_prompt,
            image=image, mask_image=mask, height=image.height, width=image.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            strength=1.0 if args.steps == 1 else 0.99, generator=generator,
        ).images[0]
    if result.size != image.size:
        raise RuntimeError("SDXL output resolution changed unexpectedly")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result.save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
