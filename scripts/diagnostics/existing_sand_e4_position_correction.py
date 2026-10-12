#!/usr/bin/env python3
"""Prepare and render XY-fixed E3 tabletop correction A/B cases."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sandygs.exchange import sha256_file
from sandygs.existing_sand.e4_geometry_correction import tabletop_proposal
from sandygs.pipeline import build_renderer_environment
from sandygs.storage_guard import validate_artifact_storage
from existing_sand_e4_source_ablation import _maps, _read_yaml, _save_yaml

SOURCE_ID = 'e3_hole_joint450_depth010_fit450_001'
SUBSET_ID = 'e4_source_subset_001'


def prepare(run: Path, stage: Path, neighborhood_radius: float) -> None:
    if stage.exists():
        raise ValueError(f'refusing to overwrite: {stage}')
    source = run / f'stages/background/{SOURCE_ID}/checkpoint'
    cloud = source / 'point_cloud/iteration_30000/point_cloud.ply'
    subset = run / f'diagnostics/{SUBSET_ID}'
    subset_manifest = _read_yaml(subset / 'manifest.yaml')
    if subset_manifest['source_cloud_sha256'] != sha256_file(cloud):
        raise ValueError('E4 subset diagnostic does not match the source checkpoint')
    vertices = PlyData.read(str(cloud), mmap='r')['vertex'].data
    if len(vertices) != 281999:
        raise ValueError('E3 point count changed; review the selected group bounds')
    indices = np.load(subset / 'footprint_above_top_final_indices.npy', allow_pickle=False)
    exchange = run / 'stages/modeling/e4_exchange_initial_v2_001/exchange'
    metadata = _read_yaml(exchange / 'metadata.yaml')
    if not np.allclose(metadata['coordinates']['sim_to_gs'], np.eye(4), rtol=0, atol=1e-9):
        raise ValueError('this E4 tabletop proposal requires identity sim_to_gs')
    terrain_path = exchange / 'frames/000000/terrain.npy'
    terrain = np.load(terrain_path, allow_pickle=False)
    target_z, train_indices = tabletop_proposal(
        vertices, indices, terrain, metadata['grid'], first_new=279190,
        last_new=281599, neighborhood_radius=neighborhood_radius)
    stage.mkdir(parents=True, exist_ok=False)
    np.savez(stage / 'position_proposal.npz', indices=indices,
             target_z=target_z, train_indices=train_indices,
             source_sha256=np.asarray(sha256_file(cloud)))
    delta = target_z - vertices['z'][indices]
    manifest = {
        'source_checkpoint': str(cloud), 'source_sha256': sha256_file(cloud),
        'subset_diagnostic': str(subset),
        'candidate_definition': '126 footprint_above_top 55-group points; world XY fixed',
        'terrain_exchange': str(exchange), 'terrain_sha256': sha256_file(terrain_path),
        'position_proposal_sha256': sha256_file(stage / 'position_proposal.npz'),
        'target_points': int(len(indices)), 'trainable_55_group_points': int(len(train_indices)),
        'neighborhood_radius_world': float(neighborhood_radius),
        'z_displacement_p05_p50_p95': [float(v) for v in np.quantile(delta, [.05, .5, .95])],
        'cases': {'half': 0.5, 'full': 1.0},
        'interpretation': 'E4 fitted tabletop is a geometric hypothesis, not hidden-scene ground truth',
    }
    _save_yaml(stage / 'manifest.yaml', manifest)
    print(f'Prepared {len(indices)} targets, {len(train_indices)} locally trainable points: {stage}')


def render(run: Path, stage: Path, case: str) -> None:
    manifest = _read_yaml(stage / 'manifest.yaml')
    source_cloud = Path(manifest['source_checkpoint'])
    if sha256_file(source_cloud) != manifest['source_sha256']:
        raise ValueError('source checkpoint changed')
    proposal = stage / 'position_proposal.npz'
    if sha256_file(proposal) != manifest['position_proposal_sha256']:
        raise ValueError('position proposal changed')
    output = stage / 'cases' / case
    if output.exists():
        raise ValueError(f'refusing to overwrite: {output}')
    parent = Path(manifest['subset_diagnostic']).parent / 'e4_source_ablation_001'
    renderer = _read_yaml(parent / 'cases/full/render.yaml')
    with np.load(proposal, allow_pickle=False) as data:
        indices = data['indices']
        target_z = data['target_z']
    vertices = PlyData.read(str(source_cloud), mmap='r')['vertex'].data.copy()
    strength = float(manifest['cases'][case])
    vertices['z'][indices] += strength * (target_z - vertices['z'][indices])
    checkpoint = output / 'checkpoint'
    checkpoint.mkdir(parents=True, exist_ok=False)
    for item in source_cloud.parents[2].iterdir():
        if item.name != 'point_cloud':
            (checkpoint / item.name).symlink_to(item)
    candidate = checkpoint / 'point_cloud/iteration_30000/point_cloud.ply'
    candidate.parent.mkdir(parents=True)
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False,
            byte_order='<').write(str(candidate))
    _save_yaml(output / 'render.yaml', renderer)
    command = [sys.executable, str(ROOT / 'render_sand_water.py'),
               '-s', _read_yaml(run / 'run.yaml')['source']['dataset'],
               '-m', str(checkpoint), '--iteration', '30000',
               '--config_file', str(output / 'render.yaml'),
               '--render_output_path', str(output), '--disable_preload_img']
    with (output / 'render.log').open('x', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=ROOT, env=build_renderer_environment(),
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'{case} render failed ({result.returncode}): {output / "render.log"}')
    _review(run, stage, case, parent)
    print(f'Completed {case}: {output}')


def _review(run: Path, stage: Path, case: str, parent: Path) -> None:
    output = stage / 'cases' / case
    views = ('frame_00055.jpg', 'frame_00139.jpg', 'frame_00126.jpg')
    metrics = {'case': case, 'views': []}
    for camera_index, camera_id in enumerate(views):
        full = _maps(parent / 'cases/full', camera_index)
        adjusted = _maps(output, camera_index)
        if (not np.array_equal(full['sand_depth'], adjusted['sand_depth'], equal_nan=True)
                or not np.array_equal(full['coverage'], adjusted['coverage'])):
            raise ValueError(f'{camera_id}: physical sand changed during position A/B')
        interior = binary_erosion(full['coverage'] >= .95, iterations=5)
        hidden = interior & ~full['owner'] & np.isfinite(full['delta']) & (full['delta'] < 0)
        introduced = interior & full['owner'] & ~adjusted['owner']
        before = np.asarray(Image.open(parent / 'cases/full/release_check/selected_views/time_0000/renders'
                                       / f'camera_{camera_index:03d}.png').convert('RGB'))
        after = np.asarray(Image.open(output / 'release_check/selected_views/time_0000/renders'
                                      / f'camera_{camera_index:03d}.png').convert('RGB'))
        marked = after.copy()
        marked[hidden & adjusted['owner']] = [30, 220, 50]
        marked[hidden & ~adjusted['owner']] = [240, 40, 40]
        marked[introduced] = [255, 220, 20]
        review = stage / 'review' / case / camera_id
        review.mkdir(parents=True)
        Image.fromarray(np.concatenate([before, after, marked], axis=1)).save(review / 'overview.png')
        metrics['views'].append({
            'camera_id': camera_id, 'baseline_hidden': int(hidden.sum()),
            'newly_sand_owned': int((hidden & adjusted['owner']).sum()),
            'newly_source_owned_interior': int(introduced.sum()),
            'rgb_mean_abs_change_255': float(np.abs(before.astype(np.float32)
                                                     - after.astype(np.float32)).mean()),
        })
    _save_yaml(output / 'position_review.yaml', metrics)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--diagnostic-id', default='e4_position_correction_001')
    sub = parser.add_subparsers(dest='action', required=True)
    init = sub.add_parser('prepare', help='CPU-only proposal and hash manifest')
    init.add_argument('--neighborhood-radius', type=float, default=.02)
    draw = sub.add_parser('render', help='GPU t=0 E4 render and source/sand depth comparison')
    draw.add_argument('--case', choices=('half', 'full'), required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    stage = run / 'diagnostics' / args.diagnostic_id
    validate_artifact_storage(ROOT / 'configs/paths.local.yaml')
    if args.action == 'prepare':
        prepare(run, stage, args.neighborhood_radius)
    else:
        render(run, stage, args.case)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
