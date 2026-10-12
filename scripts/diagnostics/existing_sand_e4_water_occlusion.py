#!/usr/bin/env python3
"""Read-only solver replay: water coverage, ownership and physical mesh references.

The worker hooks only this diagnostic process. Production rendering is unchanged.
Physical references triangulate cell centers (all three vertices wet for water),
so their shoreline is conservative; they are not a replacement renderer.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import runpy
import subprocess
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read(path):
    return yaml.safe_load(Path(path).read_text())


def save(path, obj):
    with Path(path).open('x') as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def mesh(adapter, height, active=None):
    from sandygs.exchange import apply_transform
    nx, ny = height.shape
    dx, dy = adapter.grid.spacing
    x = adapter.grid.domain_min[0] + (np.arange(nx) + .5) * dx
    y = adapter.grid.domain_min[1] + (np.arange(ny) + .5) * dy
    xx, yy = np.meshgrid(x, y, indexing='ij')
    v = apply_transform(np.stack([xx, yy, height], -1).reshape(-1, 3), adapter.sim_to_gs)
    idx = np.arange(nx * ny).reshape(nx, ny)
    a,b,c,d = [q.ravel() for q in (idx[:-1,:-1],idx[1:,:-1],idx[1:,1:],idx[:-1,1:])]
    faces = np.concatenate([np.stack([a,b,c],1), np.stack([a,c,d],1)])
    if active is not None:
        faces = faces[active.ravel()[faces].all(1)]
    return v[faces].astype(np.float32)


def worker(config, output, renderer_args):
    import torch
    import gaussian_renderer.sand_water_renderer as renderer
    from sandygs.rendering import rasterize_triangle_reference
    original = renderer.render_sand_water
    cfg = read(config)
    times = cfg['render_schedule']['sequences'][0]['times']
    cameras = cfg['render_schedule']['sequences'][0]['camera_indices']
    calls = 0
    raw = output / 'raw'
    raw.mkdir()

    def wrapped(view, pc, *args, **kwargs):
        nonlocal calls
        ti, ci = divmod(calls, len(cameras))
        calls += 1
        result = original(view, pc, *args, **kwargs)
        keys = ['water_raw_alpha','water_raw_coverage','water_surface_depth',
                'water_surface_plane_depth_valid','water_surface_coverage',
                'water_surface_thickness','water_depth_in_front','opaque_surface_depth',
                'opaque_surface_sand_owner','source_surface_depth','sand_surface_depth',
                'sand_surface_coverage']
        arrays = {}
        for key in keys:
            a = result[key].detach().cpu().numpy()
            arrays[key] = a[0] if a.ndim == 3 else a
        frame = pc.sand_water_adapter.sample(times[ti])
        # PGSR plane depth uses rays (pixel - W/2)/f, while nvdiffrast
        # samples pixel centers. Shift the mesh projection +0.5 pixels so
        # the reference evaluates precisely the same camera rays.
        depth_projection = view.get_full_proj_transform(orthographic=False).clone()
        depth_projection[:, 0] += depth_projection[:, 3] / int(view.image_width)
        depth_projection[:, 1] += depth_projection[:, 3] / int(view.image_height)
        for name, height, active in [
            ('terrain', frame.terrain, None),
            ('substrate', frame.sand_surface, None),
            ('water', frame.absolute_surface, frame.free_water_mask),
        ]:
            triangles = torch.as_tensor(mesh(pc.sand_water_adapter, height, active), device='cuda')
            ref = rasterize_triangle_reference(triangles, view.world_view_transform,
                depth_projection, int(view.image_height), int(view.image_width))
            for field, key in [('depth','sand_triangle_reference_depth_raw'),
                               ('coverage','sand_triangle_reference_coverage')]:
                arrays[f'{name}_mesh_{field}'] = ref[key][0].detach().cpu().numpy()
        np.savez_compressed(raw / f'time_{ti:04d}_camera_{cameras[ci]:03d}.npz', **arrays)
        return result

    renderer.render_sand_water = wrapped
    sys.argv = [str(ROOT / 'render_sand_water.py'), *renderer_args]
    runpy.run_module('sandygs.render_app', run_name='__main__')
    if calls != len(times) * len(cameras):
        raise RuntimeError('unexpected render call count')


def summarize(stage):
    from scipy.ndimage import binary_erosion
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest = read(stage / 'manifest.yaml')
    report = []
    review = stage / 'review'
    review.mkdir(exist_ok=False)
    for path in sorted((stage / 'baseline/raw').glob('*.npz')):
        stem = path.stem
        ti, ci = int(stem[5:9]), int(stem[-3:])
        a = dict(np.load(path)); b = dict(np.load(stage / 'zero_bias/raw' / path.name))
        # Erode away physical shoreline and mesh silhouette by two image pixels.
        physically_visible = (a['water_mesh_coverage'] & a['substrate_mesh_coverage'] &
            (a['water_mesh_depth'] < a['substrate_mesh_depth'] - 1e-5))
        roi = binary_erosion(physically_visible, iterations=2)
        alpha = a['water_raw_alpha']; depth = a['water_surface_depth']
        valid = (alpha > .02) & a['water_surface_plane_depth_valid'].astype(bool)
        front = a['water_depth_in_front'].astype(bool)
        hidden = roi & valid & ~front
        owner = a['opaque_surface_sand_owner'].astype(bool)
        source_hidden = hidden & ~owner
        margin = a['opaque_surface_depth'] - depth
        zero_front = b['water_depth_in_front'].astype(bool)
        metrics = {'time_seconds':manifest['times'][ti], 'camera_index':ci,
            'physical_water_interior_pixels':int(roi.sum()),
            'low_alpha_interior':int((roi & (alpha <= .02)).sum()),
            'raw_coverage_below_half_interior':int((roi & (a['water_raw_coverage'] < .5)).sum()),
            'invalid_depth_interior':int((roi & (alpha > .02) & ~valid).sum()),
            'hidden_interior':int(hidden.sum()), 'hidden_by_source':int(source_hidden.sum()),
            'hidden_by_sand':int((hidden & owner).sum()),
            'recovered_by_zero_bias':int((hidden & zero_front).sum()),
            'still_hidden_zero_bias':int((hidden & ~zero_front).sum()),
            'zero_bias_remaining_source_ahead_of_physical_water':int((source_hidden & ~zero_front & (a['source_surface_depth'] <= a['water_mesh_depth'])).sum()),
            'zero_bias_remaining_water_depth_error_only':int((source_hidden & ~zero_front & (a['source_surface_depth'] > a['water_mesh_depth'])).sum()),
            'raw_alpha_max_difference_between_bias_cases':float(np.max(abs(alpha-b['water_raw_alpha']))),
            'water_depth_max_difference_between_bias_cases':float(np.max(abs(depth-b['water_surface_depth'])))}
        for name, values, mask in [
            ('source_minus_terrain',a['source_surface_depth']-a['terrain_mesh_depth'],source_hidden & a['terrain_mesh_coverage']),
            ('remaining_source_minus_terrain',a['source_surface_depth']-a['terrain_mesh_depth'],source_hidden & ~zero_front & a['terrain_mesh_coverage']),
            ('remaining_water_minus_mesh',depth-a['water_mesh_depth'],hidden & ~zero_front),
            ('opaque_minus_water',margin,hidden),
            ('gaussian_water_minus_mesh',depth-a['water_mesh_depth'],roi & valid),
            ('free_water_thickness',a['water_surface_thickness'],roi & valid)]:
            metrics[name+'_p05_p50_p95'] = [float(v) for v in np.quantile(values[mask],[.05,.5,.95])] if mask.any() else None
        report.append(metrics)
        colors = np.zeros((*roi.shape,3),dtype=np.uint8)
        colors[roi] = [30,190,60]
        colors[roi & (alpha <= .02)] = [0,90,255]
        colors[roi & (alpha > .02) & ~valid] = [255,0,255]
        colors[hidden & owner] = [255,160,0]
        colors[source_hidden] = [255,35,35]
        colors[hidden & zero_front] = [0,220,220]
        Image.fromarray(colors).save(review / (stem+'_classification.png'))
        base = 'release_check/selected_views/'+f'time_{ti:04d}/renders/camera_{ci:03d}.png'
        rgb = np.array(Image.open(stage/'baseline'/base))
        rgb0 = np.array(Image.open(stage/'zero_bias'/base))
        ys,xs = np.where(a['water_mesh_coverage'])
        sl = (slice(max(0,ys.min()-25),min(roi.shape[0],ys.max()+26)),
              slice(max(0,xs.min()-25),min(roi.shape[1],xs.max()+26)))
        fig, axes = plt.subplots(2,3,figsize=(15,9))
        panels = [(rgb,'Baseline RGB'),(rgb0,'Zero bias RGB'),(colors,'Interior: red source, orange sand, cyan bias-only'),
                  (a['water_raw_coverage'],'Raw water coverage before occlusion'),
                  (np.where(roi,margin,np.nan),'Opaque Z - water Z'),
                  (np.where(roi,a['source_surface_depth']-a['terrain_mesh_depth'],np.nan),'Source Z - terrain mesh Z')]
        for ax,(img,title) in zip(axes.flat,panels):
            if img.ndim==3: ax.imshow(img[sl])
            elif 'coverage' in title: ax.imshow(img[sl],vmin=0,vmax=1,cmap='gray')
            else:
                m=ax.imshow(img[sl],vmin=-.04,vmax=.04,cmap='coolwarm');fig.colorbar(m,ax=ax,shrink=.7)
            ax.set_title(title,fontsize=10);ax.axis('off')
        fig.suptitle(f"t={manifest['times'][ti]} s, camera {ci}; blue=low alpha, magenta=invalid depth")
        fig.tight_layout();fig.savefig(review/(stem+'_overview.png'),dpi=120);plt.close(fig)
    save(stage/'metrics.yaml',{'frames':report})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',type=Path)
    p.add_argument('--summarize-only',action='store_true')
    args, rest=p.parse_known_args()
    if args.worker:
        worker(args.worker,args.output,rest);return
    if args.summarize_only:
        summarize(args.output);return
    from sandygs.pipeline import build_renderer_environment
    from sandygs.storage_guard import validate_artifact_storage
    from sandygs.exchange import sha256_file
    validate_artifact_storage(ROOT/'configs/paths.local.yaml')
    if args.output.exists():p.error('output already exists')
    cfg=read(args.baseline/'render.yaml');plan=read(args.baseline/'plan.yaml')
    cfg['render_schedule']['sequences'][0]['times']=[.5,1.,3.]
    cfg['render_schedule']['sequences'][0]['camera_indices']=[0,1,2]
    cfg['sand_debug_output_keys']=['water_raw_alpha','water_raw_coverage','water_coverage',
       'water_failure_classification','water_depth_in_front','sand_owner']
    args.output.mkdir(parents=True)
    save(args.output/'manifest.yaml',{'baseline':str(args.baseline),
         'baseline_config_sha256':sha256_file(args.baseline/'render.yaml'),
         'source_cloud_sha256':sha256_file(Path(plan['source_cloud'])),
         'times':[.5,1.,3.],'camera_indices':[0,1,2],
         'reference':'cell-center mesh; water triangles require all vertices wet; ROI eroded 2 pixels; projection shifted +0.5 pixel to match CUDA plane-depth rays'})
    for case,bias in [('baseline',cfg['water_depth_bias']),('zero_bias',0.)]:
        folder=args.output/case;folder.mkdir()
        config=copy.deepcopy(cfg);config['water_depth_bias']=bias
        save(folder/'render.yaml',config)
        command=list(plan['command'][2:])
        command[command.index('--config_file')+1]=str(folder/'render.yaml')
        command[command.index('--render_output_path')+1]=str(folder)
        with (folder/'render.log').open('x') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(folder/'render.yaml'),
                            '--output',str(folder),*command],cwd=ROOT,env=build_renderer_environment(),
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        print(case+' complete',flush=True)
    summarize(args.output)

if __name__=='__main__':main()
