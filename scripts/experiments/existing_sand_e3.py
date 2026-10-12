#!/usr/bin/env python3
"""Review E2 deletion, derive PGSR scene, and prepare E3 RGB/depth completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.existing_sand.e3 import derive_scene, plan_derivation  # noqa: E402
from sandygs.existing_sand.e3_completion import (  # noqa: E402
    prepare_background, run_background_completion,
)
from sandygs.existing_sand.e3_optimize import initialize, optimize  # noqa: E402
from sandygs.existing_sand.workspace import ExistingSandError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan", help="write a selection-bound review template")
    derive = sub.add_parser("derive", help="make the derived PGSR checkpoint")
    for item in (plan, derive):
        item.add_argument("--run", type=Path, required=True)
        item.add_argument("--selection-id", required=True)
        item.add_argument("--check-id", required=True)
    plan.add_argument("--output", type=Path, required=True)
    derive.add_argument("--decision", type=Path, required=True)
    derive.add_argument("--stage-id", required=True)
    derive.add_argument("--chunk-size", type=int, default=100000)

    prepare = sub.add_parser("prepare-background", help="render derived PGSR and prepare a shared edit mask")
    prepare.add_argument("--run", type=Path, required=True)
    prepare.add_argument("--derived-id", required=True)
    prepare.add_argument("--camera-id", required=True)
    prepare.add_argument("--stage-id", required=True)
    prepare.add_argument("--crop", type=int, nargs=4, required=True,
                         metavar=("X", "Y", "WIDTH", "HEIGHT"))
    prepare.add_argument("--expand-pixels", type=int, default=0)
    prepare.add_argument("--protect-mask", type=Path)
    prepare.add_argument("--threshold", type=int, default=128)
    prepare.add_argument("--source-fit-id", help="render an earlier optimized checkpoint")
    prepare.add_argument("--edit-mask", type=Path, help="full-view white edit mask replacing SAM")

    completion = sub.add_parser("run-rgb-depth", help="run SDXL then InFusion on a derived scene")
    completion.add_argument("--run", type=Path, required=True)
    completion.add_argument("--prepare-id", required=True)
    completion.add_argument("--stage-id", required=True)
    completion.add_argument("--seed", type=int, required=True)
    completion.add_argument("--prompt", required=True)
    completion.add_argument("--negative-prompt", default="")
    completion.add_argument("--model-python", type=Path, required=True)
    completion.add_argument("--infusion-code", type=Path, required=True)
    completion.add_argument("--infusion-model", type=Path, required=True)
    completion.add_argument("--sdxl-model", type=Path, required=True)
    completion.add_argument("--rgb-steps", type=int, default=20)
    completion.add_argument("--depth-steps", type=int, default=20)
    completion.add_argument("--guidance-scale", type=float, default=8.0)

    initialization = sub.add_parser("initialize-3d", help="initialize planar PGSR Gaussians from an E3 completion")
    initialization.add_argument("--run", type=Path, required=True)
    initialization.add_argument("--completion-id", required=True)
    initialization.add_argument("--stage-id", required=True)
    initialization.add_argument("--stride", type=int, default=4)
    initialization.add_argument("--opacity", type=float, default=0.8)
    initialization.add_argument("--cleanup-depth-margin", type=float)
    initialization.add_argument("--source-fit-id", help="inherit this fit; defaults to preparation source")
    initialization.add_argument("--add-mask", type=Path, help="cropped white add-point mask inside edit mask")
    initialization.add_argument("--point-depth", type=Path, help="cropped positive PGSR camera-Z for new point placement; completed depth stays the training target")
    initialization.add_argument("--duplicate-distance-factor", type=float, default=0.75)

    fitting = sub.add_parser("optimize-3d", help="fit new PGSR appearance, then constrained geometry")
    fitting.add_argument("--run", type=Path, required=True)
    source = fitting.add_mutually_exclusive_group(required=True)
    source.add_argument("--initialization-id")
    source.add_argument("--source-fit-id", help="continue this fit without adding points")
    fitting.add_argument("--completion-ids", nargs="+", help="ordered pseudo-targets; updates alternate views")
    fitting.add_argument("--snapshot-every", type=int, default=0)
    fitting.add_argument("--stage-id", required=True)
    fitting.add_argument("--appearance-iters", type=int, default=100)
    fitting.add_argument("--geometry-iters", type=int, default=50)
    fitting.add_argument("--lambda-ssim", type=float, default=0.1)
    fitting.add_argument("--lambda-depth", type=float, default=0.05)
    fitting.add_argument("--lambda-normal", type=float, default=0.02)
    fitting.add_argument("--position-proposal", type=Path,
                         help="audited E4 tabletop proposal NPZ for local correction")
    fitting.add_argument("--correction-strength", type=float, default=1.0,
                         help="fraction of candidate center-Z displacement applied before fitting")
    fitting.add_argument("--lambda-table", type=float, default=0.0,
                         help="world-Z tabletop anchor weight on proposed points")

    args = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()),
               *(argv if argv is not None else sys.argv[1:])]
    try:
        if args.action == "plan":
            result = plan_derivation(args.run, args.selection_id, args.check_id, args.output)
        elif args.action == "derive":
            result = derive_scene(args.run, args.selection_id, args.check_id, args.decision,
                                  args.stage_id, repo_root=REPO_ROOT,
                                  chunk_size=args.chunk_size, command=command)
        elif args.action == "prepare-background":
            result = prepare_background(args.run, args.derived_id, args.camera_id,
                                        args.stage_id, tuple(args.crop), repo_root=REPO_ROOT,
                                        expand_pixels=args.expand_pixels,
                                        protect_mask=args.protect_mask,
                                        threshold=args.threshold, source_fit_id=args.source_fit_id,
                                        edit_mask=args.edit_mask, command=command)
        elif args.action == "initialize-3d":
            result = initialize(args.run, args.completion_id, args.stage_id,
                                repo_root=REPO_ROOT, stride=args.stride,
                                opacity=args.opacity,
                                cleanup_depth_margin=args.cleanup_depth_margin,
                                source_fit_id=args.source_fit_id, add_mask=args.add_mask,
                                duplicate_distance_factor=args.duplicate_distance_factor,
                                point_depth=args.point_depth, command=command)
        elif args.action == "optimize-3d":
            result = optimize(args.run, args.initialization_id, args.stage_id,
                              repo_root=REPO_ROOT,
                              appearance_iters=args.appearance_iters,
                              geometry_iters=args.geometry_iters,
                              lambda_ssim=args.lambda_ssim,
                              lambda_depth=args.lambda_depth,
                              lambda_normal=args.lambda_normal, source_fit_id=args.source_fit_id,
                              completion_ids=args.completion_ids, snapshot_every=args.snapshot_every,
                              position_proposal=args.position_proposal,
                              correction_strength=args.correction_strength,
                              lambda_table=args.lambda_table,
                              command=command)
        else:
            result = run_background_completion(
                args.run, args.prepare_id, args.stage_id, repo_root=REPO_ROOT,
                model_python=args.model_python, infusion_code=args.infusion_code,
                infusion_model=args.infusion_model, sdxl_model=args.sdxl_model,
                prompt=args.prompt, negative_prompt=args.negative_prompt,
                seed=args.seed, rgb_steps=args.rgb_steps,
                depth_steps=args.depth_steps, guidance_scale=args.guidance_scale,
                command=command,
            )
    except (ExistingSandError, OSError, ValueError, KeyError, SyntaxError, ImportError) as exc:
        parser.exit(2, f"existing-sand E3: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
