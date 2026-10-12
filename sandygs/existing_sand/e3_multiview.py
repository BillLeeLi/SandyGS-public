"""Build inspectable progressive E3 commands and enforce the E2 view split."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from .e3_completion import _derived_inputs
from .workspace import ExistingSandError, _identifier, _load_yaml, _stage_location


def load_config(path):
    config = _load_yaml(Path(path), 'E3 multiview config')
    if config.get('schema_version') != 1:
        raise ExistingSandError('multiview config schema_version must be 1')
    config['run'] = expand_path(config['run'])
    _identifier(config['experiment_id'], 'experiment_id')
    _identifier(config['target_id'], 'target_id')
    views = config['views']
    ids = [v['camera_id'] for v in views]
    if len(views) < 2 or len(set(ids)) != len(ids):
        raise ExistingSandError('multiview config needs at least two distinct cameras')
    holdouts = config['holdout_camera_ids']
    if not holdouts or len(set(holdouts)) != len(holdouts) or set(ids) & set(holdouts):
        raise ExistingSandError('training/holdout cameras must be distinct and disjoint')
    for case, params in config['cases'].items():
        _identifier(case, 'case')
        for name in ('appearance_iters', 'geometry_iters', 'joint_iters'):
            n = params[name]
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ExistingSandError(f'{name} must be a positive integer')
    return config


def expand_path(value, *, resolve_symlinks=True):
    result = os.path.expandvars(str(value))
    if '$' in result:
        raise ExistingSandError(f'unset environment variable in path: {value}')
    path = Path(result).expanduser()
    # Venv selection depends on the executable path, even when bin/python is
    # a symlink to the base interpreter. Resolving it bypasses pyvenv.cfg.
    return str(path.resolve() if resolve_symlinks else path.absolute())


def ids_for(config, case, index):
    prefix = f"{config['experiment_id']}_{case}_v{index}"
    target = f"{config['target_id']}_v{index}"
    view = config['views'][index]
    return {'prepare': target + '_edit',
            'complete': view.get('completion_id', target + '_rgbdepth'),
            'initialize': view.get('initialization_id', prefix + '_init'),
            'fit': prefix + '_fit'}


def validate_split(config):
    run = Path(config['run'])
    derived_id = config['derived_id']
    for view in config['views']:
        _, masks, _, _, _, _ = _derived_inputs(run, derived_id, view['camera_id'])
        entries = _load_yaml(masks / 'quality.yaml', 'E2 masks')['views']
        item = next(v for v in entries if v['camera_id'] == view['camera_id'])
        if item.get('role') != 'seed':
            raise ExistingSandError(f"training camera is not an E2 seed: {view['camera_id']}")
    derived = _stage_location(run, 'background', derived_id)
    rec = _load_yaml(derived / 'record.yaml', 'derived scene')
    check = _stage_location(run, 'segmentation', rec['prompt']['check_id'])
    actual = [v['camera_id'] for v in _load_yaml(check / 'quality.yaml', 'E2 holdouts')['holdout_views']]
    if actual != config['holdout_camera_ids']:
        raise ExistingSandError('configured holdouts differ from the approved E2 check')
    # Reused first-view assets must refer to this exact camera and deletion.
    from .e3_optimize import _context
    for view in config['views']:
        if 'completion_id' in view:
            ctx = _context(run, view['completion_id'])
            if ctx[6]['camera_id'] != view['camera_id'] or ctx[6]['derived_id'] != derived_id:
                raise ExistingSandError('reused completion has wrong camera or derived scene')
            if 'initialization_id' in view:
                init = _stage_location(run, 'background', view['initialization_id'])
                irec = _load_yaml(init / 'record.yaml', 'reused initialization')
                if irec['parameters']['completion_id'] != view['completion_id'] or not (init / 'manifest.yaml').is_file():
                    raise ExistingSandError('reused initialization is unregistered or belongs to another completion')


def command_for(config, case, step, index, repo):
    if case not in config['cases']:
        raise ExistingSandError(f'unknown case: {case}')
    if index < 0 or index >= len(config['views']):
        raise ExistingSandError('view index is out of range')
    run, view, counts = config['run'], config['views'][index], config['cases'][case]
    names = ids_for(config, case, index)
    base = [sys.executable, str(Path(repo) / 'scripts/experiments/existing_sand_e3.py')]
    common = ['--run', run]
    if step in ('prepare', 'complete') and 'completion_id' in view:
        return None, {'reuse_completion': names['complete']}
    if step == 'initialize' and 'initialization_id' in view:
        return None, {'reuse_initialization': names['initialize']}
    if step in ('prepare', 'complete') and case != 'base':
        raise ExistingSandError('generate shared targets using case base; geometry100 reuses them')
    if step == 'prepare':
        args = base + ['prepare-background', *common, '--derived-id', config['derived_id'],
                       '--camera-id', view['camera_id'], '--stage-id', names['prepare'],
                       '--crop', *map(str, view['crop']), '--expand-pixels', str(config['completion']['expand_pixels'])]
        if index:
            args += ['--source-fit-id', ids_for(config, 'base', index - 1)['fit']]
        if view.get('edit_mask'):
            args += ['--edit-mask', expand_path(view['edit_mask'])]
        return args, {'inspect': f"stages/segmentation/{names['prepare']}/mask_overlay.png"}
    if step == 'complete':
        args = base + ['run-rgb-depth', *common, '--prepare-id', names['prepare'], '--stage-id', names['complete'],
                       '--seed', str(config['seed'])]
        for key in ('prompt', 'negative_prompt', 'rgb_steps', 'depth_steps', 'guidance_scale'):
            args += ['--' + key.replace('_', '-'), str(config['completion'][key])]
        for key, value in config['models'].items():
            args += ['--' + key.replace('_', '-'), expand_path(value, resolve_symlinks=key != 'model_python')]
        return args, {'inspect': f"stages/background/{names['complete']}/report/overview.png"}
    if step == 'initialize':
        args = base + ['initialize-3d', *common, '--completion-id', names['complete'], '--stage-id', names['initialize']]
        for key in ('stride', 'opacity', 'duplicate_distance_factor'):
            args += ['--' + key.replace('_', '-'), str(config['initialization'][key])]
        if index:
            args += ['--source-fit-id', ids_for(config, case, index - 1)['fit']]
        if view.get('add_mask'):
            args += ['--add-mask', expand_path(view['add_mask'])]
        return args, {'stage': names['initialize']}
    if step == 'fit':
        args = base + ['optimize-3d', *common, '--initialization-id', names['initialize'], '--stage-id', names['fit'],
                       '--appearance-iters', str(counts['appearance_iters']), '--geometry-iters', str(counts['geometry_iters'])]
    elif step == 'joint':
        last = ids_for(config, case, len(config['views']) - 1)
        args = base + ['optimize-3d', *common, '--source-fit-id', last['fit'],
                       '--stage-id', f"{config['experiment_id']}_{case}_joint", '--appearance-iters', '0',
                       '--geometry-iters', str(counts['joint_iters']), '--completion-ids',
                       *[ids_for(config, case, i)['complete'] for i in range(len(config['views']))]]
    else:
        raise ExistingSandError(f'unknown progressive step: {step}')
    for key, value in config['loss'].items():
        args += ['--' + key.replace('_', '-'), str(value)]
    args += ['--snapshot-every', str(config.get('snapshot_every', 0))]
    return args, {'case': case, 'training_camera_ids': [v['camera_id'] for v in config['views']] if step == 'joint' else [view['camera_id']],
                  'holdout_camera_ids': config['holdout_camera_ids']}
