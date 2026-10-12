#!/usr/bin/env python3
"""Prepare and run controlled E1 visible-ground depth completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.existing_sand.workspace import ExistingSandError  # noqa: E402
from sandygs.existing_sand.e1 import prepare_hole  # noqa: E402
from sandygs.existing_sand.e1_run import run_completion  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare", help="register a hand-painted synthetic hole")
    prepare.add_argument("--run", type=Path, required=True)
    prepare.add_argument("--capture-id", required=True)
    prepare.add_argument("--camera-id", required=True)
    prepare.add_argument("--annotation", type=Path, required=True,
                         help="full PGSR render-sized grayscale PNG; white is the hole")
    prepare.add_argument("--protect-mask", type=Path,
                         help="optional full-sized grayscale PNG; white must be kept")
    prepare.add_argument("--crop", type=int, nargs=4, required=True,
                         metavar=("X", "Y", "WIDTH", "HEIGHT"),
                         help="PGSR pixel rectangle; max edge 768, dimensions multiples of 8")
    prepare.add_argument("--threshold", type=int, default=128)
    prepare.add_argument("--mask-id", required=True)

    for name in ("run-depth", "run-rgb-depth"):
        command = sub.add_parser(name, help=(
            "run fixed-RGB InFusion depth completion" if name == "run-depth" else
            "run SDXL RGB followed by InFusion depth completion"
        ))
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--mask-id", required=True)
        command.add_argument("--stage-id", required=True)
        command.add_argument("--seed", type=int, required=True)
        command.add_argument("--model-python", type=Path, required=True,
                             help="Python executable with torch, diffusers and transformers")
        command.add_argument("--infusion-code", type=Path, required=True)
        command.add_argument("--infusion-model", type=Path, required=True)
        command.add_argument("--depth-steps", type=int, default=20)
        if name == "run-rgb-depth":
            command.add_argument("--sdxl-model", type=Path, required=True)
            command.add_argument("--prompt", required=True)
            command.add_argument("--negative-prompt", default="")
            command.add_argument("--rgb-steps", type=int, default=20)
            command.add_argument("--guidance-scale", type=float, default=8.0)

    args = parser.parse_args(argv)
    actual_argv = [sys.executable, str(Path(__file__).resolve()),
                   *(argv if argv is not None else sys.argv[1:])]
    try:
        if args.action == "prepare":
            result = prepare_hole(
                args.run, args.capture_id, args.camera_id, args.annotation,
                args.mask_id, tuple(args.crop), repo_root=REPO_ROOT,
                protect_mask=args.protect_mask, threshold=args.threshold,
                command=actual_argv,
            )
        else:
            result = run_completion(
                args.run, args.mask_id, args.stage_id, repo_root=REPO_ROOT,
                model_python=args.model_python, infusion_code=args.infusion_code,
                infusion_model=args.infusion_model, seed=args.seed,
                depth_steps=args.depth_steps,
                sdxl_model=args.sdxl_model if args.action == "run-rgb-depth" else None,
                prompt=args.prompt if args.action == "run-rgb-depth" else None,
                negative_prompt=args.negative_prompt if args.action == "run-rgb-depth" else "",
                rgb_steps=args.rgb_steps if args.action == "run-rgb-depth" else 20,
                guidance_scale=args.guidance_scale if args.action == "run-rgb-depth" else 8.0,
                command=actual_argv,
            )
    except (ExistingSandError, OSError, ImportError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"existing-sand E1: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
