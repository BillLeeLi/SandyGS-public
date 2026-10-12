#!/usr/bin/env python3
"""Reviewable E3 hole-mask A/B/C preparation and pretraining PGSR previews."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from sandygs.existing_sand.e3_holes import hole_masks,preview_initialization
from sandygs.existing_sand.workspace import ExistingSandError


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    masks=sub.add_parser('masks')
    masks.add_argument('--run',type=Path,required=True)
    masks.add_argument('--source-fit-id',required=True)
    masks.add_argument('--completion-id',required=True)
    masks.add_argument('--diagnostic-root',type=Path,required=True)
    masks.add_argument('--stage-id',required=True)
    masks.add_argument('--alpha-max',type=float,default=.9)
    masks.add_argument('--contribution-max',type=float,default=.2)
    preview=sub.add_parser('preview')
    preview.add_argument('--run',type=Path,required=True)
    preview.add_argument('--initialization-id',required=True)
    preview.add_argument('--stage-id',required=True)
    preview.add_argument('--camera-id',action='append',required=True)
    args=p.parse_args()
    command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]
    try:
        if args.action=='masks':
            result=hole_masks(args.run,args.source_fit_id,args.completion_id,args.diagnostic_root,args.stage_id,
                              repo_root=ROOT,alpha_max=args.alpha_max,contribution_max=args.contribution_max,command=command)
        else:
            result=preview_initialization(args.run,args.initialization_id,args.stage_id,args.camera_id,
                                          repo_root=ROOT,command=command)
    except (ExistingSandError,OSError,ValueError,KeyError) as exc:
        p.exit(2,f'existing-sand E3 holes: {exc}\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
