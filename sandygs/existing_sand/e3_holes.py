"""Registered, reviewable E3 hole masks and short before-fit PGSR previews."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image
import yaml

from .e3_checkpoint import fitted_checkpoint
from .e3_optimize import _context
from .workspace import ExistingSandError, _load_yaml, _sha256, _stage_location, record_stage, verify_run


def _write_yaml(path, value):
    with path.open('x', encoding='utf-8') as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)


def hole_masks(run_root, source_fit_id, completion_id, diagnostic_root, stage_id, *,
               repo_root, alpha_max=0.9, contribution_max=0.2, thresholds=(0.005, 0.01), command=None):
    """Create three cropped A/B/C masks; all point placements share one plane."""
    run, repo = Path(run_root).resolve(), Path(repo_root).resolve()
    verify_run(run)
    if not (0 < alpha_max < 1 and 0 < contribution_max < 1):
        raise ExistingSandError('alpha_max and contribution_max must be in (0,1)')
    if tuple(thresholds) != (0.005, 0.01):
        raise ExistingSandError('this registered comparison fixes depth gates to 0.5% and 1%')
    done, prep, _, _, _, (_, camera), crop, _ = _context(run, completion_id)
    fitted_checkpoint(run, source_fit_id, crop['derived_id'])
    derived_record = _load_yaml(_stage_location(run, 'background', crop['derived_id']) / 'record.yaml', 'derived')
    holdout_stage = _stage_location(run, 'segmentation', derived_record['prompt']['check_id'])
    holdout_ids = {item['camera_id'] for item in _load_yaml(holdout_stage / 'quality.yaml', 'E2 holdouts')['holdout_views']}
    if crop['camera_id'] in holdout_ids:
        raise ExistingSandError('hole point selection must use a training camera')
    root = Path(diagnostic_root).resolve()
    report = _load_yaml(root / 'report.yaml', 'depth diagnostic')
    if report.get('fit_id') != source_fit_id:
        raise ExistingSandError('depth diagnostic was rendered from a different fit')
    folder = root / crop['camera_id'].removesuffix('.jpg')
    sources = {name: folder / name for name in (
        'rendered_depth.npy', 'full_alpha.npy', 'restored_visible_contribution.npy',
        'edit_mask.npy', 'completed_depth.npy', 'local_plane_depth.npy', 'rgb.png')}
    width, height = camera['render_size_wh']
    arrays = {key: np.load(path, allow_pickle=False) for key, path in sources.items() if key.endswith('.npy')}
    shape = (height, width)
    if any(array.shape != shape for array in arrays.values()):
        raise ExistingSandError('depth diagnostic arrays must have full camera shape')
    rgb = np.asarray(Image.open(sources['rgb.png']).convert('RGB'))
    if rgb.shape != shape + (3,):
        raise ExistingSandError('depth diagnostic RGB size differs from camera')
    x, y, w, h = crop['crop_xywh']
    rows, cols = slice(y, y+h), slice(x, x+w)
    mask = np.load(prep / 'edit_mask.npy', allow_pickle=False)
    if mask.shape != (h, w) or mask.dtype != np.bool_ or not np.array_equal(arrays['edit_mask.npy'][rows, cols], mask):
        raise ExistingSandError('diagnostic edit region does not match registered completion')
    expected_full = np.zeros(shape, dtype=np.bool_)
    expected_full[rows, cols] = mask
    if not np.array_equal(arrays['edit_mask.npy'], expected_full):
        raise ExistingSandError('diagnostic edit region escapes or differs from registered crop')
    # The completion target is a pseudo-observation; ensure this diagnostic
    # uses the exact same saved target, not a later regenerated depth map.
    completed = np.load(done / 'completed_depth.npy', allow_pickle=False)
    if not np.allclose(arrays['completed_depth.npy'][rows, cols], completed, atol=1e-6, rtol=0):
        raise ExistingSandError('diagnostic completion depth differs from registered target')
    plane = arrays['local_plane_depth.npy'][rows, cols]
    if not np.isfinite(plane).all() or np.any(plane <= 0):
        raise ExistingSandError('local plane must give finite positive camera-Z throughout crop')
    alpha = arrays['full_alpha.npy'][rows, cols]
    contribution = arrays['restored_visible_contribution.npy'][rows, cols]
    rendered = arrays['rendered_depth.npy'][rows, cols]
    if not np.isfinite(alpha).all() or not np.isfinite(contribution).all() or not np.isfinite(rendered[mask]).all():
        raise ExistingSandError('diagnostic alpha, contribution or rendered edit depth is invalid')
    base = mask & (alpha < alpha_max) & (contribution < contribution_max)
    relative = (rendered - completed) / completed
    cases = {'candidate': base,
             'depth005': base & (relative > 0.005),
             'depth010': base & (relative > 0.01)}
    if not all(value.any() for value in cases.values()):
        raise ExistingSandError('at least one hole candidate case is empty')
    target = _stage_location(run, 'segmentation', stage_id)
    if target.exists():
        raise ExistingSandError(f'refusing to overwrite: {target}')
    payload = Path(tempfile.mkdtemp(prefix=f'.e3_holes_{stage_id}_', dir=run))
    try:
        np.save(payload / 'point_depth.npy', plane.astype(np.float32), allow_pickle=False)
        for name, selected in cases.items():
            np.save(payload / f'{name}.npy', selected, allow_pickle=False)
            Image.fromarray(selected.astype(np.uint8)*255).save(payload / f'{name}.png')
            overlay = rgb[rows, cols].copy()
            overlay[selected] = np.round(overlay[selected] * .45 + np.array([255, 25, 20]) * .55).astype(np.uint8)
            Image.fromarray(overlay).save(payload / f'{name}_overlay.png')
        quality = {'camera_id': crop['camera_id'], 'source_fit_id': source_fit_id,
                   'candidate_pixels': {name: int(value.sum()) for name, value in cases.items()},
                   'candidate_subset_order': ['depth010', 'depth005', 'candidate'],
                   'point_depth_source': 'local_plane_depth.npy',
                   'point_depth_vs_completion_edit_abs_p50_p95':
                       [float(v) for v in np.percentile(np.abs(plane[mask]-completed[mask]), [50, 95])],
                   'holdout_cameras_excluded': sorted(holdout_ids)}
        _write_yaml(payload / 'quality.yaml', quality)
        record = {'schema_version': 1, 'stage': 'segmentation', 'stage_id': stage_id,
                  'camera_ids': [crop['camera_id']],
                  'prompt': {'type': 'e3_covered_hole_mask_comparison'}, 'seed': None, 'models': [],
                  'parameters': {'source_fit_id': source_fit_id, 'completion_id': completion_id,
                                 'diagnostic_root': str(root), 'diagnostic_input_sha256':
                                     {name: _sha256(path) for name, path in sources.items()},
                                 'alpha_max': alpha_max, 'contribution_max': contribution_max,
                                 'relative_depth_thresholds': [0.005, 0.01],
                                 'point_depth': 'point_depth.npy', 'cases': list(cases)},
                  'commands': [command] if command else [], 'command_cwd': str(repo),
                  'parents': [{'stage': 'background', 'stage_id': source_fit_id},
                              {'stage': 'background', 'stage_id': completion_id},
                              {'stage': 'segmentation', 'stage_id': prep.name}],
                  'artifacts': [{'path': p.name, 'role': 'hole_ablation_' + p.stem} for p in sorted(payload.iterdir()) if p.is_file()],
                  'code_files': ['sandygs/existing_sand/e3_holes.py', 'sandygs/existing_sand/e3_optimize.py',
                                 'scripts/experiments/existing_sand_e3_holes.py']}
        _write_yaml(payload / 'record.yaml', record)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload.rename(target)
        try:
            record_stage(run, target / 'record.yaml', repo_root=repo)
        except BaseException:
            shutil.rmtree(target)
            raise
        return {'stage_dir': str(target), **quality}
    finally:
        if payload.exists():
            shutil.rmtree(payload)


def preview_initialization(run_root, initialization_id, stage_id, camera_ids, *, repo_root, command=None):
    """Render a candidate before training in listed seed views and holdouts."""
    import torch
    from .e3_completion import _render_derived
    run, repo = Path(run_root).resolve(), Path(repo_root).resolve()
    verify_run(run)
    init = _stage_location(run, 'background', initialization_id)
    if not (init / 'manifest.yaml').is_file():
        raise ExistingSandError('unregistered E3 initialization')
    rec = _load_yaml(init / 'record.yaml', 'E3 initialization')
    if rec['prompt']['type'] != 'pgsr_planar_gaussian_initialization' or not rec['parameters'].get('source_fit_id'):
        raise ExistingSandError('preview requires progressive E3 initialization')
    source_id = rec['parameters']['source_fit_id']
    source, cloud_rel, _, _ = fitted_checkpoint(run, source_id, rec['parameters']['derived_id'])
    completion_id = rec['parameters']['completion_id']
    _, _, _, _, observed, _, _, _ = _context(run, completion_id)
    derived = _stage_location(run, 'background', rec['parameters']['derived_id'])
    check_id = _load_yaml(derived / 'record.yaml', 'derived')['prompt']['check_id']
    holdout = {v['camera_id'] for v in _load_yaml(_stage_location(run,'segmentation',check_id)/'quality.yaml', 'holdouts')['holdout_views']}
    observations = _load_yaml(observed / 'record.yaml', 'observations')
    if not torch.cuda.is_available():
        raise ExistingSandError('PGSR initialization preview requires CUDA')
    if not camera_ids or len(set(camera_ids)) != len(camera_ids):
        raise ExistingSandError('preview needs distinct camera IDs')
    target = _stage_location(run, 'background', stage_id)
    if target.exists():
        raise ExistingSandError(f'refusing to overwrite: {target}')
    payload = Path(tempfile.mkdtemp(prefix=f'.e3_preview_{stage_id}_', dir=run))
    try:
        scores=[]
        for i,cid in enumerate(camera_ids):
            matches=[p for p in observed.glob('view_*/camera.yaml') if _load_yaml(p,'camera')['camera_id']==cid]
            if len(matches)!=1:
                raise ExistingSandError(f'preview camera missing or ambiguous: {cid}')
            camera=_load_yaml(matches[0],'camera')
            viewdir=matches[0].parent
            # Use the same PGSR renderer, camera, and color contract as E3 preparation.
            before_rgb,before_z,before_alpha=_render_derived(source/'checkpoint',cloud_rel,camera,observations,viewdir/'source_rgb.png')
            after_rgb,after_z,after_alpha=_render_derived(init,'point_cloud.ply',camera,observations,viewdir/'source_rgb.png')
            folder=payload/f'view_{i:04d}';folder.mkdir()
            for label,rgb,z,alpha in [('before',before_rgb,before_z,before_alpha),('after',after_rgb,after_z,after_alpha)]:
                Image.fromarray(np.round(np.clip(rgb,0,1)*255).astype(np.uint8)).save(folder/f'{label}_rgb.png')
                np.save(folder/f'{label}_depth.npy',z,allow_pickle=False)
                np.save(folder/f'{label}_alpha.npy',alpha,allow_pickle=False)
            delta=np.abs(after_rgb-before_rgb).mean(axis=-1)
            Image.fromarray(np.round(np.clip(delta*5,0,1)*255).astype(np.uint8)).save(folder/'rgb_change_x5.png')
            sheet=Image.new('RGB',(before_rgb.shape[1]*2,before_rgb.shape[0]))
            sheet.paste(Image.open(folder/'before_rgb.png'),(0,0));sheet.paste(Image.open(folder/'after_rgb.png'),(before_rgb.shape[1],0))
            sheet.save(folder/'comparison.png')
            scores.append({'camera_id':cid,'directory':folder.name,'role':'holdout' if cid in holdout else 'training',
                           'rgb_l1_change_full':float(delta.mean())})
        _write_yaml(payload/'quality.yaml',{'source_fit_id':source_id,'initialization_id':initialization_id,'views':scores,
                                            'initialization_quality':_load_yaml(init/'quality.yaml','init quality')})
        record={'schema_version':1,'stage':'background','stage_id':stage_id,
                'camera_ids':camera_ids,'prompt':{'type':'e3_initialization_pretraining_preview'},
                'seed':None,'models':[],'parameters':{'source_fit_id':source_id,'initialization_id':initialization_id},
                'commands':[command] if command else [],'command_cwd':str(repo),
                'parents':[{'stage':'background','stage_id':source_id},{'stage':'background','stage_id':initialization_id}],
                'artifacts':[{'path':p.relative_to(payload).as_posix(),'role':'e3_hole_preview'} for p in sorted(payload.rglob('*')) if p.is_file()],
                'code_files':['sandygs/existing_sand/e3_holes.py','scripts/experiments/existing_sand_e3_holes.py',
                              'sandygs/existing_sand/e3_completion.py']}
        _write_yaml(payload/'record.yaml',record)
        target.parent.mkdir(parents=True,exist_ok=True)
        payload.rename(target)
        try:record_stage(run,target/'record.yaml',repo_root=repo)
        except BaseException:
            shutil.rmtree(target)
            raise
        return {'stage_dir':str(target),'views':scores}
    finally:
        if payload.exists():shutil.rmtree(payload)
