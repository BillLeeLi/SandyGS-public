#!/usr/bin/env python3
"""Select existing-scene objects with SAM and PGSR multiview depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.existing_sand.workspace import ExistingSandError  # noqa: E402
from sandygs.existing_sand.e2 import create_masks, select_seeds, select_frequency  # noqa: E402
from sandygs.existing_sand.e2_render import render_holdout  # noqa: E402
from sandygs.existing_sand.e2_affinity import train_affinity, expand_affinity  # noqa: E402
from sandygs.existing_sand.e2_grounding import propose_boxes  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    masks = sub.add_parser("masks", help="point/box prompts to SAM masks")
    masks.add_argument("--run", type=Path, required=True)
    masks.add_argument("--capture-id", required=True)
    masks.add_argument("--mask-id", required=True)
    masks.add_argument("--prompts", type=Path, required=True)
    masks.add_argument("--sam-code", type=Path, required=True)
    masks.add_argument("--sam-checkpoint", type=Path, required=True)
    masks.add_argument("--image-type", choices=("render_rgb", "source_rgb"),
                       default="render_rgb")

    select = sub.add_parser("select", help="index-aligned positive/negative/unknown seeds")
    select.add_argument("--run", type=Path, required=True)
    select.add_argument("--mask-id", required=True)
    select.add_argument("--stage-id", required=True)
    select.add_argument("--min-views", type=int, default=2)
    select.add_argument("--depth-absolute", type=float, default=0.025)
    select.add_argument("--depth-relative", type=float, default=0.015)
    select.add_argument("--footprint-sigma", type=float, default=2.)
    select.add_argument("--max-footprint-pixels", type=float, default=12.)
    select.add_argument("--border-pixels", type=float, default=1.5)
    select.add_argument("--chunk-size", type=int, default=100000)
    frequency = sub.add_parser("select-frequency", help="loose-depth multiview frequency votes")
    frequency.add_argument("--run", type=Path, required=True)
    frequency.add_argument("--mask-id", required=True)
    frequency.add_argument("--stage-id", required=True)
    frequency.add_argument("--sand-fraction", type=float, default=0.65)
    frequency.add_argument("--depth-absolute", type=float, default=0.025)
    frequency.add_argument("--depth-relative", type=float, default=0.03)
    frequency.add_argument("--chunk-size", type=int, default=100000)
    holdout = sub.add_parser("check", help="rasterize selection in held-out PGSR views")
    holdout.add_argument("--run", type=Path, required=True)
    holdout.add_argument("--seed-id", required=True)
    holdout.add_argument("--check-id", required=True)
    holdout.add_argument("--coverage-threshold", type=float, default=0.3)
    train = sub.add_parser("train-affinity", help="train SAGA-style PGSR-aligned 32D features")
    train.add_argument("--run", type=Path, required=True)
    train.add_argument("--capture-id", required=True)
    train.add_argument("--camera-id", action="append", required=True)
    train.add_argument("--stage-id", required=True)
    train.add_argument("--sam-code", type=Path, required=True)
    train.add_argument("--sam-checkpoint", type=Path, required=True)
    train.add_argument("--saga-code", type=Path, required=True)
    train.add_argument("--downsample", type=int, default=4)
    train.add_argument("--points-per-side", type=int, default=16)
    train.add_argument("--max-samples-per-view", type=int, default=120000)
    train.add_argument("--iterations", type=int, default=2000)
    train.add_argument("--learning-rate", type=float, default=0.01)
    train.add_argument("--seed", type=int, default=17)

    expand = sub.add_parser("expand", help="use scale-gated affinity to extend reliable seeds")
    expand.add_argument("--run", type=Path, required=True)
    expand.add_argument("--seed-id", required=True)
    expand.add_argument("--affinity-id", required=True)
    expand.add_argument("--stage-id", required=True)
    expand.add_argument("--scale", type=float, required=True, help="mask physical scale in PGSR world units")
    expand.add_argument("--threshold", type=float, default=0.8)
    expand.add_argument("--margin", type=float, default=0.05)
    expand.add_argument("--prototypes", type=int, default=4)
    text = sub.add_parser("propose-boxes", help="optional Grounding DINO text candidates")
    text.add_argument("--run", type=Path, required=True)
    text.add_argument("--capture-id", required=True)
    text.add_argument("--camera-id", action="append", required=True)
    text.add_argument("--text", required=True)
    text.add_argument("--output", type=Path, required=True)
    text.add_argument("--model-python", type=Path, required=True)
    text.add_argument("--dino-code", type=Path, required=True)
    text.add_argument("--dino-checkpoint", type=Path, required=True)
    text.add_argument("--bert-dir", type=Path, required=True)
    text.add_argument("--box-threshold", type=float, default=0.35)
    text.add_argument("--text-threshold", type=float, default=0.25)
    text.add_argument("--max-candidates", type=int, default=4)
    args = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()),
               *(argv if argv is not None else sys.argv[1:])]
    try:
        if args.action == "masks":
            result = create_masks(args.run, args.capture_id, args.mask_id, args.prompts,
                                  args.sam_code, args.sam_checkpoint, repo_root=REPO_ROOT,
                                  image_type=args.image_type, command=command)
        elif args.action == "train-affinity":
            result = train_affinity(args.run, args.capture_id, args.stage_id, args.camera_id,
                                    args.sam_code, args.sam_checkpoint, args.saga_code,
                                    repo_root=REPO_ROOT, downsample=args.downsample,
                                    points_per_side=args.points_per_side,
                                    max_samples_per_view=args.max_samples_per_view,
                                    iterations=args.iterations, learning_rate=args.learning_rate,
                                    seed=args.seed, command=command)
        elif args.action == "expand":
            result = expand_affinity(args.run, args.seed_id, args.affinity_id, args.stage_id,
                                     repo_root=REPO_ROOT, scale=args.scale,
                                     threshold=args.threshold, margin=args.margin,
                                     prototypes=args.prototypes, command=command)
        elif args.action == "propose-boxes":
            result = propose_boxes(args.run, args.capture_id, args.camera_id, args.text,
                                   args.output, repo_root=REPO_ROOT,
                                   model_python=args.model_python,
                                   dino_code=args.dino_code,
                                   dino_checkpoint=args.dino_checkpoint,
                                   bert_dir=args.bert_dir,
                                   box_threshold=args.box_threshold,
                                   text_threshold=args.text_threshold,
                                   max_candidates=args.max_candidates)
        elif args.action == "select-frequency":
            result = select_frequency(args.run, args.mask_id, args.stage_id,
                                      repo_root=REPO_ROOT,
                                      sand_fraction=args.sand_fraction,
                                      depth_absolute=args.depth_absolute,
                                      depth_relative=args.depth_relative,
                                      chunk_size=args.chunk_size, command=command)
        elif args.action == "check":
            result = render_holdout(args.run, args.seed_id, args.check_id,
                                    repo_root=REPO_ROOT,
                                    coverage_threshold=args.coverage_threshold,
                                    command=command)
        else:
            result = select_seeds(args.run, args.mask_id, args.stage_id, repo_root=REPO_ROOT,
                                  min_views=args.min_views, depth_absolute=args.depth_absolute,
                                  depth_relative=args.depth_relative,
                                  footprint_sigma=args.footprint_sigma,
                                  max_footprint_pixels=args.max_footprint_pixels,
                                  border_pixels=args.border_pixels, chunk_size=args.chunk_size,
                                  command=command)
    except (ExistingSandError, OSError, RuntimeError, ImportError, ValueError, KeyError) as exc:
        parser.exit(2, f"existing-sand E2: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
