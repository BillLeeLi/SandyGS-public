import os
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from utils.general_utils import build_rotation

try:
    # Prefer the extension built next to this source tree.  An older globally
    # installed wheel may not expose the geometry-only SSR entrypoint.
    _C = importlib.import_module(".bvh_tracing._C", __name__)
except Exception as e:
    try:
        from bvh_tracing import _C
    except Exception:
        _src_path = os.path.dirname(os.path.abspath(__file__))
        _C = load(
            name='_bvh_tracing',
            extra_cuda_cflags=["-O3", "--expt-extended-lambda"],
            extra_cflags=["-O3"],
            sources=[os.path.join(_src_path, 'src', f) for f in [
                'bvh.cu',
                'trace.cu',
                'construct.cu',
                'bindings.cpp',
            ]],
            extra_include_paths=[
                os.path.join(_src_path, 'include'),
            ],
            verbose=True)


class RayTracer:
    def __init__(self, means3D, scales, rotations):
        P = means3D.shape[0]
        rot = build_rotation(rotations)
        nodes = torch.full((2 * P - 1, 5), -1, device="cuda").int()
        nodes[:P - 1, 4] = 0
        nodes[P - 1:, 4] = 1
        aabbs = torch.zeros(2 * P - 1, 6, device="cuda").float()
        aabbs[:, :3] = 100000
        aabbs[:, 3:] = -100000

        a, b, c = rot[:, :, 0], rot[:, :, 1], rot[:, :, 2]
        m = 3
        sa, sb, sc = m * scales[:, 0], m * scales[:, 1], m * scales[:, 2]

        x111 = means3D + a * sa[:, None] + b * sb[:, None] + c * sc[:, None]
        x110 = means3D + a * sa[:, None] + b * sb[:, None] - c * sc[:, None]
        x101 = means3D + a * sa[:, None] - b * sb[:, None] + c * sc[:, None]
        x100 = means3D + a * sa[:, None] - b * sb[:, None] - c * sc[:, None]
        x011 = means3D - a * sa[:, None] + b * sb[:, None] + c * sc[:, None]
        x010 = means3D - a * sa[:, None] + b * sb[:, None] - c * sc[:, None]
        x001 = means3D - a * sa[:, None] - b * sb[:, None] + c * sc[:, None]
        x000 = means3D - a * sa[:, None] - b * sb[:, None] - c * sc[:, None]
        aabb_min = torch.minimum(torch.minimum(
            torch.minimum(torch.minimum(torch.minimum(torch.minimum(torch.minimum(x111, x110), x101), x100), x011),
                          x010), x001), x000)
        aabb_max = torch.maximum(torch.maximum(torch.maximum(torch.maximum(
            torch.maximum(torch.maximum(torch.maximum(x111, x110), x101), x100), x011), x010), x001), x000)

        aabbs[P - 1:] = torch.cat([aabb_min, aabb_max], dim=-1)

        self.tree, self.aabb, self.morton = _C.create_bvh(means3D, scales, rotations, nodes, aabbs)
        # self.tree [xxx,5], self.aabb [xxx,6]

    @torch.no_grad()
    def trace_visibility(self, rays_o, rays_d, means3D, symm_inv, opacity, normals, scaling):
        rays_o = rays_o + rays_d * 0.05
        """
        cotrib, opa = _C.trace_bvh_opacity(self.tree, self.aabb,
                                                 rays_o, rays_d,
                                                 means3D, symm_inv,
                                                 opacity, normals)
        return {
            "visibility": opa.unsqueeze(-1),
            "contribute": cotrib.unsqueeze(-1),
        }
        """
        reflvec_gs_ids, reflvec_gs_alphas, reflvec_gs_Ts = _C.trace_bvh_opacity(self.tree, self.aabb,
                                                                             rays_o, rays_d,
                                                                             means3D, symm_inv,
                                                                             opacity, normals, scaling)
        return {
            "reflvec_gs_ids": reflvec_gs_ids,
            "reflvec_gs_alphas": reflvec_gs_alphas,
            "reflvec_gs_Ts": reflvec_gs_Ts
        }


    @torch.no_grad()
    def calc_refl_ray_colors(self, reflvec_gs_colors, reflvec_gs_alphas, reflvec_gs_Ts):
        #print(reflvec_gs_colors.shape, reflvec_gs_alphas.shape, reflvec_gs_Ts.shape)
        refl_ray_colors = _C.accum_refl_ray_colors(reflvec_gs_colors, reflvec_gs_alphas, reflvec_gs_Ts)

        return {
            "refl_ray_colors": refl_ray_colors[0]
        }


class ScreenRayTracer:
    def __init__(self, refra_k, thickness):
        self.refra_k = refra_k
        self.thickness = thickness

    @torch.no_grad()
    def calc_screen_refl_colors(self, _rgb_fresnel, _rgb_src, _depth,
                                start_coords, end_coords,
                                start_3dpos, end_3dpos,
                                _fluid_mask, normal_fiuld_2d):
        screen_refl_colors, screen_refra_colors = _C.get_screen_refl_colors(_rgb_fresnel, _rgb_src, _depth,
                                                        start_coords, end_coords,
                                                        start_3dpos, end_3dpos,
                                                       _fluid_mask, normal_fiuld_2d, self.refra_k, self.thickness)
        return {
            "screen_refl_colors": screen_refl_colors,
            "screen_refra_colors": screen_refra_colors
            #"nohit_indices": nohit_indices
        }

    @torch.no_grad()
    def trace(self, depth, normal_camera, start_coords, end_coords,
              start_3dpos, end_3dpos, *, start_step=2,
              frontface_cos_threshold=0.08, backface_cos_threshold=0.08,
              max_depth_jump_absolute=0.25, max_depth_jump_relative=0.05):
        """Return face-aware geometry-only SSR classifications for SandyGS P4.

        ``depth`` and camera-facing ``normal_camera`` describe the composed
        source+sand G-buffer. Status 1 is a usable front-face hit, status 2 saw
        only a back face and carries ``proxy_coords``, status 3 is geometrically
        ambiguous, and status 0 found no intersection. Color, Fresnel and all
        fallbacks deliberately stay outside CUDA.
        """
        if not 0.0 <= frontface_cos_threshold <= 1.0:
            raise ValueError("frontface_cos_threshold must lie in [0,1]")
        if not 0.0 <= backface_cos_threshold <= 1.0:
            raise ValueError("backface_cos_threshold must lie in [0,1]")
        if max_depth_jump_absolute < 0.0 or max_depth_jump_relative < 0.0:
            raise ValueError("SSR depth-jump thresholds must be non-negative")
        hit_coords, hit_status, proxy_coords, hit_normal_dot = (
            _C.trace_screen_reflections(
                depth, normal_camera, start_coords, end_coords,
                start_3dpos, end_3dpos, float(self.thickness), int(start_step),
                float(frontface_cos_threshold), float(backface_cos_threshold),
                float(max_depth_jump_absolute), float(max_depth_jump_relative),
            )
        )
        return {
            "hit_coords": hit_coords,
            "hit_status": hit_status,
            "hit_valid": hit_status == 1,
            "proxy_coords": proxy_coords,
            "hit_normal_dot": hit_normal_dot,
        }
