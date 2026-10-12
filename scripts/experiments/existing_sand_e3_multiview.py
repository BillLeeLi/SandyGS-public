#!/usr/bin/env python3
"""Run one inspectable step of progressive E3 or the final joint fit."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sandygs.existing_sand.e3_multiview import command_for, ids_for, load_config, validate_split
from sandygs.existing_sand.workspace import ExistingSandError


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--case', choices=('base', 'geometry100'), default='base')
    p.add_argument('--step', choices=('status', 'prepare', 'complete', 'initialize', 'fit', 'joint'), required=True)
    p.add_argument('--view', type=int, default=0, help='zero-based position in configured training order')
    p.add_argument('--dry-run', action='store_true', help='print argv; do not render, infer, or write stages')
    args = p.parse_args()
    try:
        config = load_config(args.config)
        validate_split(config)
        if args.step == 'status':
            result = {'run': config['run'], 'holdouts': config['holdout_camera_ids'], 'views': []}
            for i, view in enumerate(config['views']):
                names = ids_for(config, args.case, i)
                result['views'].append({'index': i, 'camera_id': view['camera_id'], **{
                    step: {'stage_id': name, 'registered': (Path(config['run']) / 'stages' /
                        ('segmentation' if step == 'prepare' else 'background') / name / 'manifest.yaml').is_file()}
                    for step, name in names.items()}})
            result['joint_id'] = f"{config['experiment_id']}_{args.case}_joint"
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        command, summary = command_for(config, args.case, args.step, args.view, ROOT)
        print(json.dumps({'command': command, **summary}, ensure_ascii=False, indent=2), flush=True)
        if command is not None and not args.dry_run:
            subprocess.run([sys.executable, str(ROOT / 'scripts/check_artifact_storage.py')], check=True, cwd=ROOT)
            subprocess.run(command, check=True, cwd=ROOT)
        return 0
    except (ExistingSandError, KeyError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        p.exit(2, f'existing-sand E3 multiview: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
