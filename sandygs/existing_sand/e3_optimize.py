"""Initialize and locally fit PGSR planar Gaussians from E3 RGB/depth.

Only newly appended Gaussians may change. The synthetic reference view is a
pseudo-observation; original sand-covered photos are never used as hole truth.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

from .e3_checkpoint import fitted_checkpoint
import yaml

from .geometry import backproject_camera_z, project_world, validate_pinhole
from .workspace import ExistingSandError, _load_yaml, _safe_relative, _sha256, _stage_location, record_stage, verify_run


def _yaml(path, data):
    with path.open('x', encoding='utf-8') as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)


def _context(run, completion_id):
    done = _stage_location(run, 'background', completion_id)
    if not (done / 'manifest.yaml').is_file():
        raise ExistingSandError(f'unregistered E3 completion: {done}')
    rec = _load_yaml(done / 'record.yaml', 'E3 completion')
    if rec.get('prompt', {}).get('type') != 'sdxl_background_completion':
        raise ExistingSandError('completion_id must refer to an E3 SDXL/InFusion stage')
    prep = _stage_location(run, 'segmentation', rec['parameters']['prepare_id'])
    if not (prep / 'manifest.yaml').is_file():
        raise ExistingSandError('E3 preparation is unregistered')
    crop = _load_yaml(prep / 'crop.yaml', 'E3 crop')
    derived = _stage_location(run, 'background', crop['derived_id'])
    if not (derived / 'manifest.yaml').is_file():
        raise ExistingSandError('derived checkpoint is unregistered')
    drec = _load_yaml(derived / 'record.yaml', 'derived checkpoint')
    if drec.get('prompt', {}).get('type') != 'approved_gaussian_deletion':
        raise ExistingSandError('derived checkpoint is not an approved deletion')
    _, cloud = _safe_relative(derived / 'checkpoint', drec['parameters']['point_cloud'], 'derived point cloud')
    if not cloud.is_file():
        raise ExistingSandError(f'missing derived point cloud: {cloud}')
    parents = _load_yaml(prep / 'record.yaml', 'E3 preparation')['parents']
    observations = [p for p in parents if p['stage'] == 'observations']
    if len(observations) != 1:
        raise ExistingSandError('E3 preparation needs one observation parent')
    observed = _stage_location(run, 'observations', observations[0]['stage_id'])
    matches = []
    for path in observed.glob('view_*/camera.yaml'):
        camera = _load_yaml(path, 'PGSR camera')
        if camera.get('camera_id') == crop['camera_id']:
            matches.append((path.parent, camera))
    if len(matches) != 1:
        raise ExistingSandError('reference camera is not unique in observations')
    return done, prep, derived, cloud, observed, matches[0], crop, rec


def _dense_geometry(depth, K, c2w):
    if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ExistingSandError('completed depth must be finite positive camera-Z')
    validate_pinhole(K, np.linalg.inv(c2w))
    world = backproject_camera_z(depth, K, np.linalg.inv(c2w))
    smooth = backproject_camera_z(gaussian_filter(depth.astype(np.float64), 1.5), K, np.linalg.inv(c2w))
    dx, dy = np.gradient(smooth, axis=1), np.gradient(smooth, axis=0)
    tx = dx / np.maximum(np.linalg.norm(dx, axis=-1, keepdims=True), 1e-12)
    normal = np.cross(dx, dy)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
    facing = np.sum(normal * (c2w[:3, 3] - world), axis=-1) >= 0
    normal[~facing] *= -1
    ty = np.cross(normal, tx)
    ty /= np.maximum(np.linalg.norm(ty, axis=-1, keepdims=True), 1e-12)
    return world, np.stack((tx, ty, normal), axis=-1), normal


def _append_vertices(old, xyz, frames, rgb, z, K, stride, opacity):
    names = set(old.dtype.names or ())
    required = {'x', 'y', 'z', 'opacity', *(f'f_dc_{i}' for i in range(3)),
                *(f'scale_{i}' for i in range(3)), *(f'rot_{i}' for i in range(4))}
    if not required.issubset(names):
        raise ExistingSandError('PGSR PLY has missing Gaussian properties')
    out = np.zeros(len(xyz), dtype=old.dtype)
    if not len(xyz):
        return out
    for i, name in enumerate(('x', 'y', 'z')):
        out[name] = xyz[:, i].astype(np.float32)
    dc = (rgb - 0.5) / 0.28209479177387814
    for i in range(3):
        out[f'f_dc_{i}'] = dc[:, i].astype(np.float32)
    out['opacity'] = np.float32(np.log(opacity / (1 - opacity)))
    sx = np.maximum(z * stride * 0.65 / K[0, 0], 1e-5)
    sy = np.maximum(z * stride * 0.65 / K[1, 1], 1e-5)
    scales = np.stack((sx, sy, np.minimum(sx, sy) * 0.08), axis=-1)
    for i in range(3):
        out[f'scale_{i}'] = np.log(scales[:, i]).astype(np.float32)
    quaternion = Rotation.from_matrix(frames).as_quat()[:, [3, 0, 1, 2]]
    for i in range(4):
        out[f'rot_{i}'] = quaternion[:, i].astype(np.float32)
    return out


def _publish(run, target, payload, record, repo):
    _yaml(payload / 'record.yaml', record)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload.rename(target)
    try:
        record_stage(run, target / 'record.yaml', repo_root=repo)
    except BaseException:
        shutil.rmtree(target)
        raise


def initialize(run_root, completion_id, stage_id, *, repo_root, stride=4, opacity=0.8,
               cleanup_depth_margin=None, source_fit_id=None, add_mask=None,
               duplicate_distance_factor=0.75, point_depth=None, command=None):
    """CPU-only initialization. Existing PLY vertex records remain byte identical."""
    from plyfile import PlyData, PlyElement
    run, repo = Path(run_root).resolve(), Path(repo_root).resolve()
    verify_run(run)
    if not isinstance(stride, int) or not 2 <= stride <= 16 or not 0 < opacity < 1:
        raise ExistingSandError('stride must be 2..16 and opacity must be in (0,1)')
    if cleanup_depth_margin is not None and (not np.isfinite(cleanup_depth_margin) or not 0 <= cleanup_depth_margin <= 1):
        raise ExistingSandError('cleanup_depth_margin must be in [0,1] PGSR units')
    done, prep, derived, cloud, observed, (_, camera), crop, rec = _context(run, completion_id)
    source_fit_id = source_fit_id or crop.get('source_fit_id')
    fit_record = None
    if source_fit_id is not None:
        _, _, cloud, fit_record = fitted_checkpoint(run, source_fit_id, crop['derived_id'])
    if not np.isfinite(duplicate_distance_factor) or duplicate_distance_factor < 0:
        raise ExistingSandError('duplicate_distance_factor must be finite nonnegative')
    target = _stage_location(run, 'background', stage_id)
    if target.exists():
        raise ExistingSandError(f'refusing to overwrite: {target}')
    mask = np.load(prep / 'edit_mask.npy', allow_pickle=False)
    completed_z = np.load(done / 'completed_depth.npy', allow_pickle=False)
    z = completed_z
    if point_depth is not None:
        z = np.load(Path(point_depth), allow_pickle=False)
    rgb = np.load(done / 'completed_rgb.npy', allow_pickle=False)
    K = np.load(prep / 'camera_K.npy', allow_pickle=False)
    c2w = np.load(prep / 'camera_c2w.npy', allow_pickle=False)
    if (mask.dtype != np.bool_ or mask.shape != z.shape or completed_z.shape != z.shape
            or rgb.shape != z.shape + (3,) or not mask.any()
            or not np.isfinite(z).all() or np.any(z <= 0)):
        raise ExistingSandError('E3 completed arrays or point-depth override are inconsistent')
    world, frames, _ = _dense_geometry(z, K, c2w)
    add = mask
    if add_mask is not None:
        from .e1 import _read_binary_mask
        add, _ = _read_binary_mask(Path(add_mask), (mask.shape[1], mask.shape[0]), "cropped add mask", 128)
        if np.any(add & ~mask):
            raise ExistingSandError("add mask must be a subset of the edit mask")
    chosen = np.zeros_like(mask)
    chosen[::stride, ::stride] = add[::stride, ::stride]
    ply = PlyData.read(str(cloud), mmap='r')
    if ply.text or ply.byte_order != '<' or len(ply.elements) != 1 or ply.elements[0].name != 'vertex':
        raise ExistingSandError('derived PGSR PLY must be vertex-only, binary little endian')
    original = ply['vertex'].data
    frozen_count = int(fit_record['parameters']['new_start_index']) if fit_record else len(original)
    if not 0 < frozen_count <= len(original):
        raise ExistingSandError('invalid frozen Gaussian prefix')
    inherited_count = len(original) - frozen_count
    if inherited_count and duplicate_distance_factor:
        previous_xyz = np.stack([original[k][frozen_count:] for k in ('x', 'y', 'z')], axis=-1)
        ids = np.flatnonzero(chosen)
        distance, _ = cKDTree(previous_xyz).query(world.reshape(-1, 3)[ids])
        radius = duplicate_distance_factor * z.ravel()[ids] * stride / max(K[0, 0], K[1, 1])
        chosen.ravel()[ids[distance <= radius]] = False
    if not chosen.any() and not inherited_count:
        raise ExistingSandError('stride skipped all edit pixels')
    uv = np.stack(np.nonzero(chosen)[::-1], axis=-1).astype(np.int32)
    reproj, reproj_z = project_world(world[chosen], K, np.linalg.inv(c2w))
    pixel_error = float(np.max(np.abs(reproj - uv))) if len(uv) else 0.0
    depth_error = float(np.max(np.abs(reproj_z - z[chosen]))) if len(uv) else 0.0
    if pixel_error > 1e-3 or depth_error > 1e-3:
        raise ExistingSandError('initialized points fail PGSR reprojection')
    old = original
    removed = np.empty(0, dtype=np.int64)
    if cleanup_depth_margin is not None:
        # A center-in-mask test plus camera-Z gate removes local residual sand.
        # The selected source/derived checkpoint remains unchanged; indices
        # are saved so every extra deletion can be reviewed or rolled back.
        xyz = np.stack((original['x'], original['y'], original['z']), axis=-1)
        full_K = np.asarray(camera['pgsr_K'], dtype=np.float64)
        w2c = np.asarray(camera['gs_w2c'], dtype=np.float64)
        cam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = full_K[0, 0] * cam[:, 0] / cam[:, 2] + full_K[0, 2]
            v = full_K[1, 1] * cam[:, 1] / cam[:, 2] + full_K[1, 2]
        x0, y0, width, height = crop['crop_xywh']
        in_crop = np.isfinite(u) & np.isfinite(v) & (cam[:, 2] > 0) & (u >= x0) & (u < x0 + width) & (v >= y0) & (v < y0 + height)
        in_crop[frozen_count:] = False  # Never delete earlier completed background here.
        ids = np.flatnonzero(in_crop)
        ui = np.clip(np.rint(u[ids]).astype(np.int64) - x0, 0, width - 1)
        vi = np.clip(np.rint(v[ids]).astype(np.int64) - y0, 0, height - 1)
        ids, ui, vi = ids[mask[vi, ui]], ui[mask[vi, ui]], vi[mask[vi, ui]]
        removed = ids[cam[ids, 2] - z[vi, ui] < cleanup_depth_margin]
        keep = np.ones(len(original), dtype=np.bool_)
        keep[removed] = False
        old = original[keep]
    frozen_count -= len(removed)
    fresh = _append_vertices(old, world[chosen], frames[chosen], rgb[chosen], z[chosen], K, stride, opacity)
    all_vertices = np.concatenate((old, fresh))
    payload = Path(tempfile.mkdtemp(prefix=f'.e3_init_{stage_id}_', dir=run))
    try:
        PlyData([PlyElement.describe(all_vertices, 'vertex')], text=False, byte_order='<').write(str(payload / 'point_cloud.ply'))
        np.save(payload / 'new_pixels_uv.npy', uv, allow_pickle=False)
        np.save(payload / 'removed_derived_indices.npy', removed, allow_pickle=False)
        quality = {'retained_gaussians': frozen_count, 'new_gaussians': inherited_count + int(len(fresh)),
                   'inherited_background_gaussians': inherited_count, 'appended_gaussians': int(len(fresh)),
                   'total_gaussians': int(len(all_vertices)), 'stride_pixels': stride,
                   'derived_gaussians_before_cleanup': int(len(original)),
                   'cleanup_removed_gaussians': int(len(removed)),
                   'cleanup_depth_margin': cleanup_depth_margin,
                   'cleanup_index_space': 'source_fit_cloud' if source_fit_id else 'derived_checkpoint_cloud',
                   'initial_opacity': opacity, 'reprojection_max_pixel_error': pixel_error,
                   'reprojection_max_depth_error': depth_error,
                   'retained_vertex_records_identical': bool(np.array_equal(old, all_vertices[:len(old)])),
                   'point_depth_source': str(Path(point_depth).resolve()) if point_depth else 'completed_depth.npy',
                   'point_depth_vs_completion_edit_abs_p50_p95': [float(v) for v in
                       np.percentile(np.abs(z[mask] - completed_z[mask]), [50, 95])]}
        _yaml(payload / 'quality.yaml', quality)
        record = {'schema_version': 1, 'stage': 'background', 'stage_id': stage_id,
                  'camera_ids': [crop['camera_id']],
                  'prompt': {'type': 'pgsr_planar_gaussian_initialization'},
                  'seed': rec['seed'], 'models': [],
                  'parameters': {'completion_id': completion_id, 'derived_id': crop['derived_id'],
                                 'new_start_index': frozen_count, 'baseline_gaussians': int(len(old)),
                                 'source_fit_id': source_fit_id, 'condition_fit_id': crop.get('source_fit_id'),
                                 'duplicate_distance_factor': duplicate_distance_factor, 'stride_pixels': stride,
                                 'initial_opacity': opacity, 'cleanup_depth_margin': cleanup_depth_margin,
                                 'cleanup_index_space': 'source_fit_cloud' if source_fit_id else 'derived_checkpoint_cloud',
                                 'point_cloud': 'point_cloud.ply',
                                 'point_depth_source': str(Path(point_depth).resolve()) if point_depth else 'completed_depth.npy',
                                 'point_depth_sha256': _sha256(Path(point_depth)) if point_depth else None},
                  'commands': [command] if command else [], 'command_cwd': str(repo),
                  'parents': [{'stage': 'background', 'stage_id': completion_id},
                              {'stage': 'background', 'stage_id': crop['derived_id']}],
                  'artifacts': [{'path': 'point_cloud.ply', 'role': 'initial_pgsr_cloud'},
                                {'path': 'new_pixels_uv.npy', 'role': 'initial_sampled_pixels'},
                                {'path': 'removed_derived_indices.npy', 'role': 'local_residual_cleanup_indices'},
                                {'path': 'quality.yaml', 'role': 'initialization_quality'}],
                  'code_files': ['sandygs/existing_sand/e3_optimize.py',
                                 'scripts/experiments/existing_sand_e3.py',
                                 'scene/gaussian_model.py']}
        if source_fit_id is not None:
            record['parents'].append({'stage': 'background', 'stage_id': source_fit_id})
        if add_mask is not None:
            shutil.copy2(add_mask, payload / 'add_mask.png')
            record['artifacts'].append({'path': 'add_mask.png', 'role': 'authored_add_mask'})
        if point_depth is not None:
            shutil.copy2(point_depth, payload / 'point_depth.npy')
            record['artifacts'].append({'path': 'point_depth.npy', 'role': 'initial_point_camera_z'})
        record['code_files'].append('sandygs/existing_sand/e3_checkpoint.py')
        _publish(run, target, payload, record, repo)
        return {'stage_dir': str(target), **quality}
    finally:
        if payload.exists():
            shutil.rmtree(payload)


def _camera(camera, image_path):
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov
    width, height = camera['render_size_wh']
    source_width, source_height = camera['source_size_wh']
    intr = camera['colmap_intrinsics']
    w2c = np.asarray(camera['gs_w2c'], dtype=np.float64)
    view = Camera(colmap_id=0, R=w2c[:3, :3].T.copy(), T=w2c[:3, 3].copy(),
                  FoVx=focal2fov(intr['fx'], source_width),
                  FoVy=focal2fov(intr['fy'], source_height),
                  image_width=width, image_height=height, image_path=str(image_path),
                  image_name=camera['camera_id'], uid=0, preload_img=False)
    K = np.asarray(camera['pgsr_K'])
    actual = np.array([[view.fx, 0, view.cx], [0, view.fy, view.cy], [0, 0, 1]])
    if not np.allclose(actual, K, rtol=0, atol=1e-5):
        raise ExistingSandError('PGSR full-camera intrinsics differ from observation')
    return view


def _render(view, model, pipe, bg):
    from gaussian_renderer import render
    return render(view, model, pipe, bg, return_plane=True, return_depth_normal=False,
                  envmap=None, is_training=False)


def _png(path, tensor):
    rgb = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(path)


def _save_preserving(path, model, initial_path, old_count, mutable_indices=None):
    """Write trained suffix while retaining every original PLY field exactly."""
    from plyfile import PlyData, PlyElement
    scratch = path.with_suffix('.tmp.ply')
    model.save_ply(str(scratch))
    try:
        source = PlyData.read(str(initial_path), mmap='r')['vertex'].data
        trained = PlyData.read(str(scratch), mmap='r')['vertex'].data
        if source.dtype.names != trained.dtype.names or len(source) != len(trained) or old_count >= len(trained):
            raise ExistingSandError('trained PLY schema or Gaussian count changed')
        combined = np.empty(len(source), dtype=source.dtype)
        combined[:old_count] = source[:old_count]
        if mutable_indices is None:
            combined[old_count:] = trained[old_count:]
        else:
            combined[old_count:] = source[old_count:]
            combined[mutable_indices] = trained[mutable_indices]
        PlyData([PlyElement.describe(combined, 'vertex')], text=False, byte_order='<').write(str(path))
    finally:
        scratch.unlink(missing_ok=True)


def _trusted_geometry(mask, depth, K, c2w, w2c):
    if mask.dtype != np.bool_ or mask.shape != depth.shape or not mask.any() or mask.all():
        raise ExistingSandError('completion needs a nonempty edit mask and known context')
    _, _, normal_world = _dense_geometry(depth, K, c2w)
    normal_cam = normal_world @ np.asarray(w2c)[:3, :3].T
    gx, gy = np.abs(np.gradient(depth, axis=1)), np.abs(np.gradient(depth, axis=0))
    trusted = binary_erosion(mask, iterations=2)
    if not trusted.any():
        raise ExistingSandError('edit region has no trustworthy depth interior')
    jump_limit = max(0.03, float(np.percentile(np.maximum(gx[trusted], gy[trusted]), 90)))
    trusted &= (gx <= jump_limit) & (gy <= jump_limit)
    if not trusted.any():
        raise ExistingSandError('depth discontinuity mask removed all pixels')
    return normal_cam, trusted, jump_limit


def optimize(run_root, initialization_id, stage_id, *, repo_root,
             appearance_iters=100, geometry_iters=50, lambda_ssim=0.1,
             lambda_depth=0.05, lambda_normal=0.02, source_fit_id=None,
             completion_ids=None, snapshot_every=0, command=None,
             position_proposal=None, correction_strength=1.0, lambda_table=0.0):
    """Fit a new initialization or continue a fit with round-robin pseudo-views.

    Iteration counts are total optimizer updates, not counts per camera.
    A fit stores the full RGB targets so later joint fitting cannot silently
    replace earlier targets with its own final render.
    """
    import torch
    import torch.nn.functional as F
    from plyfile import PlyData, PlyElement
    from scene.gaussian_model import GaussianModel
    from utils.loss_utils import ssim
    run, repo = Path(run_root).resolve(), Path(repo_root).resolve()
    verify_run(run)
    if (initialization_id is None) == (source_fit_id is None):
        raise ExistingSandError('choose exactly one initialization_id or source_fit_id')
    target = _stage_location(run, 'background', stage_id)
    if target.exists():
        raise ExistingSandError(f'refusing to overwrite: {target}')
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0
           for n in (appearance_iters, geometry_iters, snapshot_every)) or appearance_iters + geometry_iters == 0:
        raise ExistingSandError('iterations must be nonnegative integers with at least one update')
    if not all(np.isfinite(v) and v >= 0 for v in (lambda_ssim, lambda_depth, lambda_normal)) or lambda_ssim > 1:
        raise ExistingSandError('loss weights must be finite nonnegative and lambda_ssim <= 1')
    if (not np.isfinite(correction_strength) or not 0 <= correction_strength <= 1
            or not np.isfinite(lambda_table) or lambda_table < 0):
        raise ExistingSandError('invalid E4 correction strength or tabletop weight')
    if position_proposal is None and (correction_strength != 1 or lambda_table):
        raise ExistingSandError('E4 correction options require position_proposal')
    previous = None
    previous_record = None
    if initialization_id is not None:
        initial = _stage_location(run, 'background', initialization_id)
        if not (initial / 'manifest.yaml').is_file():
            raise ExistingSandError('3D initialization is unregistered')
        irec = _load_yaml(initial / 'record.yaml', '3D initialization')
        if irec.get('prompt', {}).get('type') != 'pgsr_planar_gaussian_initialization':
            raise ExistingSandError('initialization_id does not identify PGSR planar initialization')
        initial_cloud = initial / 'point_cloud.ply'
        current_id = irec['parameters']['completion_id']
        inherited_fit_id = irec['parameters'].get('source_fit_id')
        if inherited_fit_id:
            previous, _, _, previous_record = fitted_checkpoint(run, inherited_fit_id, irec['parameters']['derived_id'])
    else:
        previous = _stage_location(run, 'background', source_fit_id)
        previous_record = _load_yaml(previous / 'record.yaml', 'source fit')
        previous, _, initial_cloud, previous_record = fitted_checkpoint(
            run, source_fit_id, previous_record['parameters']['derived_id'])
        irec = previous_record
        current_id = irec['parameters']['completion_id']
    ids = list(completion_ids) if completion_ids is not None else [current_id]
    if not ids or len(set(ids)) != len(ids):
        raise ExistingSandError('completion_ids must be nonempty and unique')
    contexts = [_context(run, cid) for cid in ids]
    done, prep, derived, cloud, observed, _, crop, crec = contexts[0]
    if any(ctx[6]['derived_id'] != crop['derived_id'] or ctx[4] != observed for ctx in contexts):
        raise ExistingSandError('all completion targets must share deletion and observation stages')
    if irec['parameters']['derived_id'] != crop['derived_id']:
        raise ExistingSandError('initialization/fit and targets have different derived checkpoints')
    derived_record = _load_yaml(derived / 'record.yaml', 'derived checkpoint')
    check_id = derived_record['prompt']['check_id']
    check = _stage_location(run, 'segmentation', check_id)
    holdouts = _load_yaml(check / 'quality.yaml', 'E2 holdout')['holdout_views']
    holdout_ids = [v['camera_id'] for v in holdouts]
    camera_ids = [ctx[6]['camera_id'] for ctx in contexts]
    if len(set(camera_ids)) != len(camera_ids) or set(camera_ids) & set(holdout_ids):
        raise ExistingSandError('training cameras must be unique and disjoint from E2 holdouts')
    if not torch.cuda.is_available():
        raise ExistingSandError('PGSR optimization requires CUDA')
    seed = crec['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    params = _load_yaml(observed / 'record.yaml', 'PGSR observation')['parameters']
    model = GaussianModel(params['sh_degree'])
    model.load_ply(str(initial_cloud))
    old_count = int(irec['parameters']['new_start_index'])
    source_vertices = PlyData.read(str(initial_cloud), mmap='r')['vertex'].data
    if len(source_vertices) != len(model.get_xyz) or not 0 < old_count < len(source_vertices):
        raise ExistingSandError('invalid frozen prefix or trainable Gaussian suffix')
    baseline_count = int(irec['parameters'].get('baseline_gaussians', old_count)) if initialization_id else len(source_vertices)
    if not old_count <= baseline_count <= len(source_vertices):
        raise ExistingSandError('invalid pre-addition baseline Gaussian count')
    correction_indices = None
    train_indices = None
    table_target = None
    if position_proposal is not None:
        from .e4_geometry_correction import load_position_proposal
        correction_indices, table_target, train_indices = load_position_proposal(
            Path(position_proposal), initial_cloud, old_count=old_count,
            point_count=len(source_vertices))
        with torch.no_grad():
            z = model._xyz[correction_indices, 2]
            target_z = torch.from_numpy(table_target).to(device=z.device, dtype=z.dtype)
            model._xyz[correction_indices, 2] = z + correction_strength * (target_z - z)
    fd, name = tempfile.mkstemp(prefix='e3_clean_reference_', suffix='.ply', dir=run)
    os.close(fd)
    reference_path = Path(name)
    try:
        PlyData([PlyElement.describe(source_vertices[:baseline_count], 'vertex')], text=False, byte_order='<').write(str(reference_path))
        reference = GaussianModel(params['sh_degree'])
        reference.load_ply(str(reference_path))
    finally:
        reference_path.unlink(missing_ok=True)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.ones(3, device='cuda') if params.get('white_background', False) else torch.zeros(3, device='cuda')
    inherited_targets = {}
    if previous_record:
        inherited_targets = {v['completion_id']: v for v in previous_record['parameters'].get('training_targets', [])}
    samples = []
    for cid, ctx in zip(ids, contexts):
        done, prep, _, _, _, (view_dir, camera), vcrop, vrec = ctx
        mask = np.load(prep / 'edit_mask.npy', allow_pickle=False)
        depth = np.load(done / 'completed_depth.npy', allow_pickle=False)
        rgb = np.load(done / 'completed_rgb.npy', allow_pickle=False)
        K = np.load(prep / 'camera_K.npy', allow_pickle=False)
        c2w = np.load(prep / 'camera_c2w.npy', allow_pickle=False)
        normal_cam, trusted, jump_limit = _trusted_geometry(mask, depth, K, c2w, camera['gs_w2c'])
        view = _camera(camera, view_dir / 'source_rgb.png')
        x, y, width, height = vcrop['crop_xywh']
        if rgb.shape != (height, width, 3) or mask.shape != (height, width) or not np.isfinite(rgb).all():
            raise ExistingSandError('completion and crop dimensions or RGB disagree')
        if x < 0 or y < 0 or x + width > view.image_width or y + height > view.image_height:
            raise ExistingSandError('completion crop escapes full camera')
        edit_np = np.zeros((view.image_height, view.image_width), dtype=np.bool_)
        edit_np[y:y + height, x:x + width] = mask
        with torch.no_grad():
            if cid in inherited_targets:
                _, target_path = _safe_relative(previous, inherited_targets[cid]['directory'] + '/target_rgb.npy', 'stored target')
                target_np = np.load(target_path, allow_pickle=False)
                if target_np.shape != (view.image_height, view.image_width, 3) or not np.isfinite(target_np).all():
                    raise ExistingSandError('stored full-camera RGB target is invalid')
                target_rgb = torch.from_numpy(target_np).to('cuda').permute(2, 0, 1)
            else:
                target_rgb = _render(view, reference, pipe, bg)['render'].detach().clone()
                patch = target_rgb[:, y:y + height, x:x + width]
                patch[:, torch.from_numpy(mask).to('cuda')] = torch.from_numpy(rgb[mask]).to('cuda', dtype=patch.dtype).T
        samples.append({'view': view, 'camera_id': vcrop['camera_id'], 'completion_id': cid,
                        'crop': (x, y, width, height), 'target': target_rgb,
                        'edit': torch.from_numpy(edit_np).to('cuda'),
                        'depth': torch.from_numpy(depth).to('cuda', dtype=torch.float32),
                        'normal': torch.from_numpy(normal_cam).to('cuda', dtype=torch.float32).permute(2, 0, 1),
                        'trust': torch.from_numpy(trusted).to('cuda'),
                        'trusted_pixels': int(trusted.sum()), 'jump_limit': jump_limit})
    xyz_initial = model._xyz.detach()[old_count:].clone()
    scale_initial = model._scaling.detach()[old_count:].clone()
    table_tensor = (torch.from_numpy(table_target).to(device='cuda')
                    if table_target is not None else None)
    payload = Path(tempfile.mkdtemp(prefix=f'.e3_fit_{stage_id}_', dir=run))
    rows = []
    target_entries = []

    def losses(sample):
        out = _render(sample['view'], model, pipe, bg)
        rendered, edit, target_rgb = out['render'], sample['edit'], sample['target']
        x, y, width, height = sample['crop']
        l_edit = (rendered[:, edit] - target_rgb[:, edit]).abs().mean()
        l_known = (rendered[:, ~edit] - target_rgb[:, ~edit]).abs().mean()
        dssim = 1 - ssim(rendered[None, :, y:y + height, x:x + width], target_rgb[None, :, y:y + height, x:x + width])
        rgb_loss = (1 - lambda_ssim) * (l_edit + 0.25 * l_known) + lambda_ssim * dssim
        trust = sample['trust']
        z_render = out['plane_depth'].squeeze()[y:y + height, x:x + width]
        n_render = F.normalize(out['rendered_normal'], dim=0)[:, y:y + height, x:x + width]
        dl = F.smooth_l1_loss(z_render[trust], sample['depth'][trust], beta=0.05)
        nl = (1 - (n_render[:, trust] * sample['normal'][:, trust]).sum(0).clamp(-1, 1)).mean()
        return out, rgb_loss, l_edit, l_known, dssim, dl, nl

    def save_views(label):
        with torch.no_grad():
            for i, sample in enumerate(samples):
                result = _render(sample['view'], model, pipe, bg)
                folder = payload / sample['directory']
                _png(folder / f'{label}_rgb.png', result['render'])
                np.save(folder / f'{label}_depth.npy', result['plane_depth'].squeeze().cpu().numpy(), allow_pickle=False)
                if i == 0:
                    _png(payload / f'{label}_rgb.png', result['render'])
                    if label != 'initial':
                        np.save(payload / f'{label}_depth.npy', result['plane_depth'].squeeze().cpu().numpy(), allow_pickle=False)

    try:
        save_base = initial_cloud
        if correction_indices is not None:
            corrected = source_vertices.copy()
            corrected['z'][correction_indices] = model._xyz[correction_indices, 2].detach().cpu().numpy()
            save_base = payload / 'corrected_initial.ply'
            PlyData([PlyElement.describe(corrected, 'vertex')], text=False,
                    byte_order='<').write(str(save_base))
            shutil.copy2(position_proposal, payload / 'position_proposal.npz')
        # Keep all previously established targets for the final joint pass,
        # including targets not used by this single-view training stage.
        for cid, entry in inherited_targets.items():
            if cid in ids:
                continue
            directory = f'training_views/view_{len(target_entries):04d}'
            folder = payload / directory
            folder.mkdir(parents=True)
            _, source = _safe_relative(previous, entry['directory'] + '/target_rgb.npy', 'stored target')
            shutil.copy2(source, folder / 'target_rgb.npy')
            arr = np.load(folder / 'target_rgb.npy', allow_pickle=False)
            Image.fromarray(np.round(np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(folder / 'target_rgb.png')
            target_entries.append({'completion_id': cid, 'camera_id': entry['camera_id'], 'directory': directory})
        for sample in samples:
            directory = f'training_views/view_{len(target_entries):04d}'
            sample['directory'] = directory
            folder = payload / directory
            folder.mkdir(parents=True)
            _png(folder / 'target_rgb.png', sample['target'])
            np.save(folder / 'target_rgb.npy', sample['target'].permute(1, 2, 0).cpu().numpy(), allow_pickle=False)
            target_entries.append({'completion_id': sample['completion_id'], 'camera_id': sample['camera_id'], 'directory': directory})
        _png(payload / 'target_rgb.png', samples[0]['target'])
        save_views('initial')
        initial_metrics = {}
        with torch.no_grad():
            for sample in samples:
                vals = losses(sample)
                initial_metrics[sample['camera_id']] = {'rgb_edit_l1': float(vals[2]), 'rgb_known_l1': float(vals[3]),
                                                        'depth': float(vals[5]), 'normal': float(vals[6])}
        for phase, count in (('appearance', appearance_iters), ('geometry', geometry_iters)):
            if not count:
                continue
            for name in ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation'):
                getattr(model, name).requires_grad_(name in ('_features_dc', '_features_rest') or phase == 'geometry')
            groups = [{'params': [model._features_dc], 'lr': 0.01}, {'params': [model._features_rest], 'lr': 0.0005}]
            if phase == 'geometry':
                groups += [{'params': [getattr(model, name)], 'lr': lr} for name, lr in
                           (('_xyz', 5e-5), ('_scaling', 5e-4), ('_rotation', 5e-4), ('_opacity', 5e-4))]
            optimizer = torch.optim.Adam(groups, eps=1e-15)
            for iteration in range(1, count + 1):
                sample = samples[(iteration - 1) % len(samples)]
                optimizer.zero_grad(set_to_none=True)
                _, rgb_loss, l_edit, l_known, dssim, dl, nl = losses(sample)
                depth_loss = dl if phase == 'geometry' else dl.detach() * 0
                normal_loss = nl if phase == 'geometry' else nl.detach() * 0
                anchor = torch.zeros((), device='cuda')
                table_loss = torch.zeros((), device='cuda')
                loss = rgb_loss
                if phase == 'geometry':
                    anchor = (model._xyz[old_count:] - xyz_initial).square().sum(-1).mean()
                    anchor += 0.01 * (model._scaling[old_count:] - scale_initial).square().mean()
                    if table_tensor is not None:
                        table_loss = (model._xyz[correction_indices, 2] - table_tensor).square().mean()
                    loss = (loss + lambda_depth * depth_loss + lambda_normal * normal_loss
                            + 0.01 * anchor + lambda_table * table_loss)
                if not torch.isfinite(loss):
                    raise ExistingSandError(f'nonfinite {phase} loss at iteration {iteration}')
                loss.backward()
                with torch.no_grad():
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            if param.grad is not None:
                                param.grad[:old_count] = 0
                                if train_indices is not None:
                                    active = torch.zeros(len(source_vertices), device=param.grad.device,
                                                         dtype=torch.bool)
                                    active[train_indices] = True
                                    param.grad[~active] = 0
                    optimizer.step()
                rows.append({'phase': phase, 'iteration': iteration, 'camera_id': sample['camera_id'],
                             'completion_id': sample['completion_id'], 'total': float(loss.detach()),
                             'rgb_edit_l1': float(l_edit.detach()), 'rgb_known_l1': float(l_known.detach()),
                             'dssim': float(dssim.detach()), 'depth': float(depth_loss.detach()),
                             'normal': float(normal_loss.detach()), 'anchor': float(anchor.detach()),
                             'table': float(table_loss.detach())})
                if snapshot_every and iteration % snapshot_every == 0 and iteration != count:
                    label = f'{phase}_{iteration:04d}'
                    _save_preserving(payload / f'{label}.ply', model, save_base, old_count,
                                     train_indices)
                    save_views(label)
            _save_preserving(payload / f'{phase}.ply', model, save_base, old_count,
                             train_indices)
            save_views(phase)
        cloud_relative, _ = _safe_relative(derived / 'checkpoint', derived_record['parameters']['point_cloud'], 'derived cloud')
        packaged = payload / 'checkpoint'
        packaged.mkdir()
        source_resources = derived / 'checkpoint'
        for name in ('cfg_args', 'cameras.json', 'multi_view.json', 'input.ply'):
            source = source_resources / name
            if source.is_file():
                shutil.copy2(source, packaged / name)
        if (source_resources / 'app_model').is_dir():
            shutil.copytree(source_resources / 'app_model', packaged / 'app_model')
        for resource in source_resources.iterdir():
            if resource.name.startswith(('env_light', 'envmap')):
                if resource.is_dir():
                    shutil.copytree(resource, packaged / resource.name)
                elif resource.is_file():
                    shutil.copy2(resource, packaged / resource.name)
        packaged_cloud = packaged / cloud_relative
        packaged_cloud.parent.mkdir(parents=True, exist_ok=True)
        final_ply = payload / ('geometry.ply' if geometry_iters else 'appearance.ply')
        shutil.copy2(final_ply, packaged_cloud)
        with (payload / 'losses.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        # These cameras only render under no_grad and never enter samples.
        for cid in holdout_ids:
            matches = [(p.parent, _load_yaml(p, 'holdout camera')) for p in observed.glob('view_*/camera.yaml')
                       if _load_yaml(p, 'holdout camera').get('camera_id') == cid]
            if len(matches) != 1:
                raise ExistingSandError(f'held-out camera unavailable: {cid}')
            hview = _camera(matches[0][1], matches[0][0] / 'source_rgb.png')
            with torch.no_grad():
                before = _render(hview, reference, pipe, bg)
                after = _render(hview, model, pipe, bg)
                _png(payload / f'holdout_{cid}_before_rgb.png', before['render'])
                _png(payload / f'holdout_{cid}_after_rgb.png', after['render'])
                difference = (after['render'] - before['render']).abs().mean(0)
                Image.fromarray(np.round(difference.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(payload / f'holdout_{cid}_rgb_change.png')
                for label, result in (('before', before), ('after', after)):
                    np.save(payload / f'holdout_{cid}_{label}_depth.npy', result['plane_depth'].squeeze().cpu().numpy(), allow_pickle=False)
        per_view = []
        with torch.no_grad():
            for sample in samples:
                vals = losses(sample)
                per_view.append({'camera_id': sample['camera_id'], 'completion_id': sample['completion_id'],
                                 'updates': sum(r['camera_id'] == sample['camera_id'] for r in rows),
                                 'trusted_depth_pixels': sample['trusted_pixels'], 'depth_jump_limit': sample['jump_limit'],
                                 'before': initial_metrics[sample['camera_id']],
                                 'after': {'rgb_edit_l1': float(vals[2]), 'rgb_known_l1': float(vals[3]),
                                           'depth': float(vals[5]), 'normal': float(vals[6])}})
        preserved = np.array_equal(PlyData.read(str(final_ply))['vertex'].data[:old_count], source_vertices[:old_count])
        if not preserved:
            raise ExistingSandError('retained Gaussian records changed')
        quality = {'retained_gaussians': old_count, 'new_gaussians': len(model.get_xyz) - old_count,
                   'appearance_iters': appearance_iters, 'geometry_iters': geometry_iters,
                   'trusted_depth_pixels': samples[0]['trusted_pixels'], 'depth_jump_limit': samples[0]['jump_limit'],
                   'first_total_loss': rows[0]['total'], 'last_total_loss': rows[-1]['total'],
                   'last_rgb_edit_l1': rows[-1]['rgb_edit_l1'], 'last_rgb_known_l1': rows[-1]['rgb_known_l1'],
                   'training_views': per_view, 'holdout_cameras': holdout_ids,
                   'retained_vertex_records_identical': preserved,
                   'reference': 'synthetic pseudo-observations; no hidden tabletop ground truth'}
        if train_indices is not None:
            final_vertices = PlyData.read(str(final_ply))['vertex'].data
            fixed = np.ones(len(source_vertices), dtype=np.bool_)
            fixed[train_indices] = False
            if not np.array_equal(final_vertices[fixed], source_vertices[fixed]):
                raise ExistingSandError('non-trainable Gaussian records changed')
            quality['e4_position_correction'] = {
                'target_gaussians': int(len(correction_indices)),
                'locally_trainable_gaussians': int(len(train_indices)),
                'other_gaussian_records_identical': True,
                'correction_strength': correction_strength,
                'lambda_table': lambda_table,
                'position_proposal_sha256': _sha256(Path(position_proposal)),
                'final_table_height_rmse': float(np.sqrt(np.mean(
                    (final_vertices['z'][correction_indices] - table_target) ** 2))),
            }
        _yaml(payload / 'quality.yaml', quality)
        artifacts = [{'path': p.relative_to(payload).as_posix(), 'role': 'e3_local_fit_' + p.stem}
                     for p in sorted(payload.rglob('*')) if p.is_file()]
        parents = [{'stage': 'background', 'stage_id': cid} for cid in ids]
        parents += [{'stage': 'segmentation', 'stage_id': check_id}]
        parent_id = initialization_id if initialization_id is not None else source_fit_id
        parents.append({'stage': 'background', 'stage_id': parent_id})
        record = {'schema_version': 1, 'stage': 'background', 'stage_id': stage_id,
                  'camera_ids': [*camera_ids, *holdout_ids],
                  'prompt': {'type': 'pgsr_local_background_optimization'}, 'seed': seed, 'models': [],
                  'parameters': {'initialization_id': initialization_id, 'source_fit_id': source_fit_id,
                                 'completion_id': ids[-1], 'completion_ids': ids, 'training_camera_ids': camera_ids,
                                 'training_targets': target_entries, 'derived_id': crop['derived_id'],
                                 'new_start_index': old_count, 'appearance_iters': appearance_iters,
                                 'geometry_iters': geometry_iters, 'iteration_counts': 'total_updates_round_robin',
                                 'lambda_ssim': lambda_ssim, 'lambda_depth': lambda_depth, 'lambda_normal': lambda_normal,
                                 'snapshot_every': snapshot_every, 'retained_gaussians_frozen': True,
                                 'position_proposal_sha256': _sha256(Path(position_proposal)) if position_proposal else None,
                                 'correction_strength': correction_strength if position_proposal else None,
                                 'lambda_table': lambda_table,
                                 'locally_trainable_gaussians': int(len(train_indices)) if train_indices is not None else None,
                                 'reference_camera_only_training': len(samples) == 1,
                                 'checkpoint_point_cloud': cloud_relative},
                  'commands': [command] if command else [], 'command_cwd': str(repo), 'parents': parents,
                  'artifacts': artifacts,
                  'code_files': ['sandygs/existing_sand/e3_optimize.py', 'sandygs/existing_sand/e3_checkpoint.py',
                                 'scripts/experiments/existing_sand_e3.py', 'scene/gaussian_model.py', 'gaussian_renderer/__init__.py']}
        if position_proposal is not None:
            record['code_files'].append('sandygs/existing_sand/e4_geometry_correction.py')
        _publish(run, target, payload, record, repo)
        return {'stage_dir': str(target), **quality}
    finally:
        if payload.exists():
            shutil.rmtree(payload)
