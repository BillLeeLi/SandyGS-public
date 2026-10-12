"""Progressive fits must preserve the frozen prefix and the held-out split."""
from pathlib import Path
import csv

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import pytest
import yaml

from sandygs.existing_sand.e3_optimize import initialize, optimize, _trusted_geometry
from sandygs.existing_sand.e3_multiview import command_for, load_config, validate_split
from sandygs.existing_sand.workspace import ExistingSandError, initialize_run, record_stage, verify_run

ROOT = Path(__file__).resolve().parents[1]
W, H = 128, 64


def register(run, kind, name, prompt, parameters, cameras, parents=()):
    stage = run / 'stages' / kind / name
    files = [p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()]
    record = {'schema_version': 1, 'stage': kind, 'stage_id': name, 'camera_ids': cameras,
              'prompt': prompt, 'seed': 23, 'models': [], 'parameters': parameters,
              'commands': [], 'parents': list(parents),
              'artifacts': [{'path': p, 'role': 'test'} for p in files]}
    (stage / 'record.yaml').write_text(yaml.safe_dump(record))
    record_stage(run, stage / 'record.yaml', repo_root=ROOT)


def scene(tmp_path, center_point=False):
    ds = tmp_path / 'dataset'; ds.mkdir(); (ds / 'cameras.txt').write_text('synthetic')
    source = tmp_path / 'source'; source.mkdir()
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
             'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    vertices = np.zeros(2, dtype=[(n, '<f4') for n in names])
    vertices['x'] = [-1, 1]; vertices['z'] = 3; vertices['rot_0'] = 1
    if center_point:
        vertices['x'][0] = 0; vertices['z'][0] = 2.9
    vertices['opacity'] = 3
    for n in ('scale_0', 'scale_1'): vertices[n] = np.log(0.1)
    vertices['scale_2'] = np.log(0.01)
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False, byte_order='<').write(str(source / 'point_cloud.ply'))
    config = tmp_path / 'source.yaml'
    config.write_text(yaml.safe_dump({'schema_version': 1, 'scene_id': 'synthetic',
        'source': {'dataset': str(ds), 'checkpoint': str(source), 'point_cloud': 'point_cloud.ply',
                   'dataset_files': ['cameras.txt']}, 'camera_ids': ['a.png', 'b.png', 'h.png'],
        'coordinate': {'sim_to_gs': np.eye(4).tolist()}}))
    run = tmp_path / 'run'; initialize_run(config, run, run_id='run', repo_root=ROOT)
    obs = run / 'stages/observations/obs'; obs.mkdir(parents=True)
    masks = run / 'stages/segmentation/masks'; masks.mkdir(parents=True)
    mask = np.zeros((H, W), bool); mask[20:44, 45:83] = True
    views = []
    for i, cid in enumerate(['a.png', 'b.png', 'h.png']):
        view = obs / f'view_{i:04d}'; view.mkdir()
        w2c = np.eye(4); w2c[0, 3] = i * .05
        camera = {'camera_id': cid, 'render_size_wh': [W, H], 'source_size_wh': [W, H],
                  'colmap_intrinsics': {'fx': 80., 'fy': 80.},
                  'pgsr_K': [[80., 0., W/2], [0., 80., H/2], [0., 0., 1.]], 'gs_w2c': w2c.tolist()}
        (view / 'camera.yaml').write_text(yaml.safe_dump(camera))
        Image.fromarray(np.full((H, W, 3), 64, np.uint8)).save(view / 'source_rgb.png')
        np.save(view / 'render_rgb.npy', np.full((H, W, 3), .25, np.float32))
        np.save(view / 'plane_depth.npy', np.full((H, W), 3., np.float32))
        mv = masks / f'view_{i:04d}'; mv.mkdir()
        Image.fromarray(mask.astype(np.uint8)*255).save(mv / 'mask.png')
        views.append({'camera_id': cid, 'role': 'holdout' if i == 2 else 'seed', 'directory': mv.name})
    (masks / 'quality.yaml').write_text(yaml.safe_dump({'views': views}))
    register(run, 'observations', 'obs', {}, {'sh_degree': 0, 'alpha_min': .1}, ['a.png', 'b.png', 'h.png'])
    register(run, 'segmentation', 'masks', {}, {}, ['a.png', 'b.png', 'h.png'])
    selected = run / 'stages/segmentation/selected'; selected.mkdir()
    np.save(selected / 'gaussian_labels.npy', np.zeros(2, np.uint8))
    register(run, 'segmentation', 'selected', {'mask_id': 'masks'}, {}, ['a.png', 'b.png'])
    check = run / 'stages/segmentation/check'; check.mkdir()
    (check / 'quality.yaml').write_text(yaml.safe_dump({'holdout_views': [{'camera_id': 'h.png'}]}))
    register(run, 'segmentation', 'check', {}, {}, ['h.png'])
    derived = run / 'stages/background/derived'; (derived / 'checkpoint').mkdir(parents=True)
    (derived / 'checkpoint/point_cloud.ply').write_bytes((source / 'point_cloud.ply').read_bytes())
    register(run, 'background', 'derived', {'type': 'approved_gaussian_deletion', 'selection_id': 'selected', 'check_id': 'check'},
             {'point_cloud': 'point_cloud.ply'}, ['a.png', 'b.png'], [{'stage': 'observations', 'stage_id': 'obs'}])
    return run, vertices, mask


def completion(run, cid, index, mask, *, fit=None):
    prep = run / 'stages/segmentation' / f'prep{index}'; prep.mkdir()
    cam = yaml.safe_load((run / f'stages/observations/obs/view_{index:04d}/camera.yaml').read_text())
    crop = {'camera_id': cid, 'derived_id': 'derived', 'crop_xywh': [0, 0, W, H]}
    if fit: crop['source_fit_id'] = fit
    (prep / 'crop.yaml').write_text(yaml.safe_dump(crop))
    np.save(prep / 'edit_mask.npy', mask)
    np.save(prep / 'camera_K.npy', np.array(cam['pgsr_K']))
    np.save(prep / 'camera_c2w.npy', np.linalg.inv(cam['gs_w2c']))
    register(run, 'segmentation', prep.name, {'type': 'derived_sand_edit_mask'}, crop, [cid],
             [{'stage': 'observations', 'stage_id': 'obs'}, {'stage': 'background', 'stage_id': 'derived'}])
    done = run / 'stages/background' / f'done{index}'; done.mkdir()
    np.save(done / 'completed_depth.npy', np.full((H, W), 3., np.float32))
    np.save(done / 'completed_rgb.npy', np.full((H, W, 3), .6 + index*.01, np.float32))
    register(run, 'background', done.name, {'type': 'sdxl_background_completion'}, {'prepare_id': prep.name}, [cid],
             [{'stage': 'segmentation', 'stage_id': prep.name}])
    return done.name


def fake_fit(run, init_id, fit_id):
    init = run / 'stages/background' / init_id
    rec = yaml.safe_load((init / 'record.yaml').read_text())
    fit = run / 'stages/background' / fit_id; (fit / 'checkpoint').mkdir(parents=True)
    (fit / 'checkpoint/point_cloud.ply').write_bytes((init / 'point_cloud.ply').read_bytes())
    register(run, 'background', fit_id, {'type': 'pgsr_local_background_optimization'},
             {'derived_id': 'derived', 'checkpoint_point_cloud': 'point_cloud.ply',
              'new_start_index': rec['parameters']['new_start_index'], 'completion_id': rec['parameters']['completion_id']}, ['a.png'])


def test_progressive_initialization_deduplicates_and_keeps_trainable_suffix(tmp_path):
    run, vertices, mask = scene(tmp_path)
    cid = completion(run, 'a.png', 0, mask)
    initialize(run, cid, 'init0', repo_root=ROOT, stride=4)
    first = PlyData.read(str(run / 'stages/background/init0/point_cloud.ply'))['vertex'].data.copy()
    fake_fit(run, 'init0', 'fit0')
    result = initialize(run, cid, 'init1', repo_root=ROOT, stride=4, source_fit_id='fit0')
    second = PlyData.read(str(Path(result['stage_dir']) / 'point_cloud.ply'))['vertex'].data
    np.testing.assert_array_equal(first, second)
    assert result['appended_gaussians'] == 0
    assert result['inherited_background_gaussians'] == len(first) - len(vertices)
    assert yaml.safe_load((Path(result['stage_dir']) / 'record.yaml').read_text())['parameters']['new_start_index'] == len(vertices)
    # Cleanup only touches the original frozen prefix, never the restored suffix.
    clean = initialize(run, cid, 'init2', repo_root=ROOT, source_fit_id='fit0', cleanup_depth_margin=.2)
    assert clean['cleanup_removed_gaussians'] == 0
    assert clean['new_gaussians'] == len(first) - len(vertices)
    verify_run(run)


def test_holdout_training_rejected_before_cuda(tmp_path):
    run, _, mask = scene(tmp_path)
    cid = completion(run, 'h.png', 2, mask)
    initialize(run, cid, 'init_h', repo_root=ROOT)
    with pytest.raises(ExistingSandError, match='disjoint from E2 holdouts'):
        optimize(run, 'init_h', 'bad_fit', repo_root=ROOT)
    assert not (run / 'stages/background/bad_fit').exists()


def test_command_budget_and_shared_targets(monkeypatch):
    for name in ('SANDYGS_ARTIFACT_ROOT', 'SANDYGS_CACHE_ROOT', 'SANDYGS_DATA_ROOT'):
        monkeypatch.setenv(name, '/tmp/resources')
    c = load_config(ROOT / 'configs/existing_sand/indoor02_multiview.yaml')
    for case, gi in [('base', '50'), ('geometry100', '100')]:
        argv, _ = command_for(c, case, 'fit', 1, ROOT)
        assert argv[argv.index('--geometry-iters')+1] == gi
        argv, summary = command_for(c, case, 'joint', 0, ROOT)
        assert argv[argv.index('--appearance-iters')+1] == '0'
        assert argv[argv.index('--geometry-iters')+1] == '150'
        assert summary['training_camera_ids'] == ['frame_00055.jpg', 'frame_00097.jpg', 'frame_00139.jpg']
        assert 'frame_00087.jpg' not in argv and 'frame_00126.jpg' not in argv
        argv, _ = command_for(c, case, 'initialize', 1, ROOT)
        assert f'e3_mv_001_{case}_v0_fit' in argv
    with pytest.raises(ExistingSandError, match='shared targets'):
        command_for(c, 'geometry100', 'prepare', 1, ROOT)


def test_flat_depth_normals_face_camera():
    z = np.full((H, W), 3., np.float32)
    mask = np.zeros((H, W), bool); mask[16:48, 32:96] = True
    K = np.array([[80, 0, W/2], [0, 80, H/2], [0, 0, 1]])
    normal, trusted, _ = _trusted_geometry(mask, z, K, np.eye(4), np.eye(4))
    np.testing.assert_allclose(normal[trusted], np.tile([0., 0., -1.], (trusted.sum(), 1)))
    assert trusted.sum() < mask.sum()


def test_progressive_and_joint_gpu(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    run, vertices, mask = scene(tmp_path)
    ca = completion(run, 'a.png', 0, mask)
    initialize(run, ca, 'init_a', repo_root=ROOT)
    optimize(run, 'init_a', 'fit_a', repo_root=ROOT, appearance_iters=1, geometry_iters=1)
    cb = completion(run, 'b.png', 1, mask, fit='fit_a')
    initialize(run, cb, 'init_b', repo_root=ROOT)
    from sandygs.existing_sand.e3_holes import preview_initialization
    preview = preview_initialization(run, 'init_b', 'preview_b', ['a.png', 'b.png', 'h.png'], repo_root=ROOT)
    assert (Path(preview['stage_dir']) / 'view_0000/comparison.png').is_file()
    assert preview['views'][-1]['role'] == 'holdout'
    b = optimize(run, 'init_b', 'fit_b', repo_root=ROOT, appearance_iters=1, geometry_iters=1)
    brec = yaml.safe_load((Path(b['stage_dir']) / 'record.yaml').read_text())
    assert len(brec['parameters']['training_targets']) == 2
    joint = optimize(run, None, 'joint', repo_root=ROOT, source_fit_id='fit_b',
                     completion_ids=[ca, cb], appearance_iters=0, geometry_iters=4, snapshot_every=2)
    out = Path(joint['stage_dir'])
    rows = list(csv.DictReader((out / 'losses.csv').open()))
    assert [r['camera_id'] for r in rows] == ['a.png', 'b.png', 'a.png', 'b.png']
    assert (out / 'geometry_0002.ply').is_file()
    assert (out / 'holdout_h.png_after_rgb.png').is_file()
    final = PlyData.read(str(out / 'geometry.ply'))['vertex'].data
    np.testing.assert_array_equal(final[:len(vertices)], vertices)
    jrec = yaml.safe_load((out / 'record.yaml').read_text())
    for item in jrec['parameters']['training_targets']:
        old = next(v for v in brec['parameters']['training_targets'] if v['completion_id'] == item['completion_id'])
        np.testing.assert_array_equal(np.load(out / item['directory'] / 'target_rgb.npy'),
                                     np.load(Path(b['stage_dir']) / old['directory'] / 'target_rgb.npy'))
    assert [v['updates'] for v in joint['training_views']] == [2, 2]
    assert all(np.isfinite(v['after']['depth']) for v in joint['training_views'])
    verify_run(run)


def test_progressive_cleanup_updates_frozen_boundary_only(tmp_path):
    run, vertices, mask = scene(tmp_path, center_point=True)
    ca = completion(run, 'a.png', 0, mask)
    initialize(run, ca, 'init0', repo_root=ROOT)
    first = PlyData.read(str(run / 'stages/background/init0/point_cloud.ply'))['vertex'].data.copy()
    fake_fit(run, 'init0', 'fit0')
    out = initialize(run, ca, 'clean', repo_root=ROOT, source_fit_id='fit0', cleanup_depth_margin=.2)
    final = PlyData.read(str(Path(out['stage_dir']) / 'point_cloud.ply'))['vertex'].data
    assert out['cleanup_removed_gaussians'] == 1 and out['retained_gaussians'] == 1
    np.testing.assert_array_equal(final[:1], vertices[1:])
    np.testing.assert_array_equal(final[1:], first[2:])


def test_completion_command_preserves_venv_python_symlink(tmp_path, monkeypatch):
    import sys
    for name in ('SANDYGS_ARTIFACT_ROOT', 'SANDYGS_CACHE_ROOT', 'SANDYGS_DATA_ROOT'):
        monkeypatch.setenv(name, str(tmp_path))
    venv = tmp_path / 'venv/bin/python'
    venv.parent.mkdir(parents=True)
    venv.symlink_to(sys.executable)
    c = load_config(ROOT / 'configs/existing_sand/indoor02_multiview.yaml')
    c['models']['model_python'] = str(venv)
    argv, _ = command_for(c, 'base', 'complete', 1, ROOT)
    executable = argv[argv.index('--model-python') + 1]
    assert executable == str(venv.absolute())
    assert executable != str(venv.resolve())


def test_point_depth_override_places_only_new_points_on_supplied_surface(tmp_path):
    run, source, mask = scene(tmp_path)
    cid = completion(run, 'a.png', 0, mask)
    z = np.full((H, W), 2.8, np.float32)
    path = tmp_path / 'surface.npy'; np.save(path, z)
    output = initialize(run, cid, 'plane_init', repo_root=ROOT, stride=4,
                        point_depth=path)
    stage = Path(output['stage_dir'])
    verts = PlyData.read(str(stage / 'point_cloud.ply'))['vertex'].data
    np.testing.assert_array_equal(verts[:len(source)], source)
    np.testing.assert_allclose(verts['z'][len(source):], 2.8, atol=1e-6)
    np.testing.assert_array_equal(np.load(stage / 'point_depth.npy'), z)
    assert output['point_depth_vs_completion_edit_abs_p50_p95'][0] == pytest.approx(.2)
    record = yaml.safe_load((stage / 'record.yaml').read_text())
    assert record['parameters']['point_depth_sha256']
    verify_run(run)


def test_hole_mask_ablation_uses_saved_coverage_and_completion_depth(tmp_path):
    from sandygs.existing_sand.e3_holes import hole_masks
    run, _, mask = scene(tmp_path)
    completion(run, 'a.png', 0, mask)
    initialize(run, 'done0', 'init0', repo_root=ROOT, stride=4)
    fake_fit(run, 'init0', 'fit0')
    cid = completion(run, 'b.png', 1, mask, fit='fit0')
    diag = tmp_path / 'diagnostic'; folder = diag / 'b.png'; folder.mkdir(parents=True)
    (diag / 'report.yaml').write_text(yaml.safe_dump({'fit_id': 'fit0'}))
    alpha = np.ones((H, W), np.float32)
    contribution = np.ones((H, W), np.float32)
    rendered = np.full((H, W), 3., np.float32)
    selected = np.zeros((H, W), bool); selected[25:40, 50:70] = True
    alpha[selected] = .8; contribution[selected] = .1
    rendered[25:30, 50:70] = 3.04  # above 1%
    rendered[30:35, 50:70] = 3.02  # above 0.5%
    rendered[35:40, 50:70] = 3.01  # only base candidate
    for name, data in [('rendered_depth.npy', rendered), ('full_alpha.npy', alpha),
                       ('restored_visible_contribution.npy', contribution),
                       ('edit_mask.npy', mask),
                       ('completed_depth.npy', np.full((H, W), 3., np.float32)),
                       ('local_plane_depth.npy', np.full((H, W), 2.9, np.float32))]:
        np.save(folder / name, data)
    Image.fromarray(np.full((H, W, 3), 128, np.uint8)).save(folder / 'rgb.png')
    result = hole_masks(run, 'fit0', cid, diag, 'holes', repo_root=ROOT)
    stage = Path(result['stage_dir'])
    counts = result['candidate_pixels']
    assert counts == {'candidate': 300, 'depth005': 200, 'depth010': 100}
    np.testing.assert_array_equal(np.load(stage / 'candidate.npy'), selected)
    np.testing.assert_array_equal(np.load(stage / 'depth005.npy'), selected & (rendered / 3 - 1 > .005))
    np.testing.assert_array_equal(np.load(stage / 'depth010.npy'), selected & (rendered / 3 - 1 > .01))
    assert (stage / 'point_depth.npy').is_file()
    verify_run(run)
    (diag / 'report.yaml').write_text(yaml.safe_dump({'fit_id': 'another_fit'}))
    with pytest.raises(ExistingSandError, match='different fit'):
        hole_masks(run, 'fit0', cid, diag, 'wrong', repo_root=ROOT)
