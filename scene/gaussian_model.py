#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from dataclasses import dataclass
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_scaling
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from pytorch3d.transforms import quaternion_to_matrix
from utils.my_utils import transform_point_cloud_2, _logit, _log
from scene.NVDIFFREC import create_trainable_env_rnd
from sandygs.simulation import SandWaterRunAdapter
from sandygs.rendering import linear_to_srgb, srgb_to_linear


@dataclass(frozen=True)
class GaussianLayerSpan:
    """Contiguous material range inside the shared Gaussian attribute arrays."""

    name: str
    start: int
    end: int
    opaque: bool

    @property
    def count(self):
        return self.end - self.start

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, args=None, is_render_sand_water=False):
        # 所有层共享相同的连续属性数组；layer_spans 记录显式区间。
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._knn_f = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.max_weight = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.denom_abs = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.knn_dists = None
        self.knn_idx = None
        # Dynamic material ranges are registered explicitly.
        self.layer_spans = {}
        # Renderer-only scalar arrays are aligned with their named dynamic
        # layer rather than concatenated into the trained PGSR attributes.
        self.water_thickness_t = torch.empty(0)
        self.setup_functions()
        self.use_app = False
        self.is_render_sand_water = is_render_sand_water

        ## qy
        if args != None:
            self.args = args
            # 为积水的环境光、高光与反射着色提供可训练的环境贴图。
            self.brdf_mlp_fluid = create_trainable_env_rnd(args.brdf_envmap_res, scale=0.0, bias=0.8)
            self.simulation_backend = getattr(
                args, "simulation_backend", "sand_water"
            )
            if self.simulation_backend != "sand_water":
                raise ValueError(
                    "SandyGS requires simulation_backend=sand_water; use the "
                    "pre-refactor-reconstruction-v1 tag for SimRain"
                )
            # Read solver snapshots directly so their physical time, ghost-cell
            # and axis semantics remain owned by the exchange boundary.
            self.sand_water_adapter = SandWaterRunAdapter(
                args.simulation_exchange_path,
                args.sand_water_run_path,
                require_pure_water=not getattr(args, "render_sand", False),
                free_water_epsilon=getattr(args, "free_water_epsilon", None),
                sand_epsilon=getattr(args, "sand_epsilon", None),
                water_normal_gradient_mode=getattr(
                    args,
                    "water_normal_gradient_mode",
                    "legacy_full",
                ),
            )
            # Water lighting is a renderer property, not a side effect of
            # constructing a sand material. Pure-water P4 must therefore own
            # a valid light direction even when no sand layer exists.
            water_light_sim = np.asarray(
                getattr(
                    args,
                    "water_light_direction_sim",
                    [-0.60, -0.50, 0.45],
                ),
                dtype=np.float64,
            )
            if (
                water_light_sim.shape != (3,)
                or not np.isfinite(water_light_sim).all()
            ):
                raise ValueError(
                    "water_light_direction_sim must contain three numbers"
                )
            water_light_gs = (
                water_light_sim @ self.sand_water_adapter.sim_to_gs[:3, :3].T
            )
            water_light_norm = np.linalg.norm(water_light_gs)
            if water_light_norm <= np.finfo(np.float64).eps:
                raise ValueError("water_light_direction_sim must be non-zero")
            self.water_light_direction_gs_t = torch.tensor(
                water_light_gs / water_light_norm,
                dtype=torch.float32,
                device="cuda",
            )
            self.num_src_gaussian = 0
            self.num_sand_gaussian = 0
            # Populated only by opt-in T2.1--T2.3 diagnostics. Keeping the
            # reference mesh and primitive labels outside the shared Gaussian
            # arrays guarantees they cannot affect production rasterization or
            # checkpoints.
            self.sand_reference_triangles_gs_t = torch.empty(
                (0, 3, 3), dtype=torch.float32, device="cuda"
            )
            self.water_terrain_reference_triangles_gs_t = torch.empty(
                (0, 3, 3), dtype=torch.float32, device="cuda"
            )
            self.water_surface_reference_triangles_gs_t = torch.empty(
                (0, 3, 3), dtype=torch.float32, device="cuda"
            )
            self.sand_reference_vertex_heights_t = torch.empty(
                (0, 3), dtype=torch.float32, device="cuda"
            )
            self.sand_reference_face_wetness_t = torch.empty(
                0, dtype=torch.float32, device="cuda"
            )
            self.sand_primitive_boundary_t = torch.empty(
                0, dtype=torch.bool, device="cuda"
            )
            self.sand_primitive_slope_t = torch.empty(
                0, dtype=torch.float32, device="cuda"
            )
            self.sand_reference_sand_epsilon = 0.0
            self.sand_reference_full_coverage_height = 1.0
            self.num_fluid_gaussian = 0
            self.align_rotation_angles = list(args.align_rotation_angles)
            self.align_translation_vector = list(args.align_translation_vector)
        ##

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._knn_f,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.max_weight,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.denom_abs,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._knn_f,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        self.max_weight,
        xyz_gradient_accum, 
        xyz_gradient_accum_abs,
        denom,
        denom_abs,
        opt_dict, 
        self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.denom_abs = denom_abs
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
        
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    
    def get_normal(self, view_cam):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = view_cam.camera_center - self._xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global
    
    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist = torch.sqrt(torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001))
        # print(f"new scale {torch.quantile(dist, 0.1)}")
        scales = torch.log(dist)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        knn_f = torch.randn((fused_point_cloud.shape[0], 6)).float().cuda()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._knn_f = nn.Parameter(knn_f.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.abs_split_radii2D_threshold = training_args.abs_split_radii2D_threshold
        self.max_abs_split_points = training_args.max_abs_split_points
        self.max_all_points = training_args.max_all_points
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._knn_f], 'lr': 0.01, "name": "knn_f"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def clip_grad(self, norm=1.0):
        for group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"][0], norm)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, mask=None):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        ## qy
        if self.is_render_sand_water:
            # 保留一份原始静态场景；每个物理时刻在其后重建沙水 Gaussian。
            self.features_extra_np = features_extra
            self.rots_np = rots
            self.num_src_gaussian = xyz.shape[0]  # 固定原始场景的边界
            self.layer_spans = {
                "source": GaussianLayerSpan(
                    "source", 0, self.num_src_gaussian, True
                )
            }
            # 新增的物理 Gaussian 使用配置给定的 DC 颜色，高阶 SH/旋转则复用一个原始点作为模板。
            self.index = np.random.choice(self.num_src_gaussian, 1)
        ##

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        ## qy
        if self.is_render_sand_water:
            self._normal = torch.zeros((self.num_src_gaussian, 3), dtype=torch.float, device="cuda")
        ##

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.denom_abs = self.denom_abs[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.max_weight = self.max_weight[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "knn_f": new_knn_f,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, max_radii2D, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_grads_abs = torch.zeros((n_init_points), device="cuda")
        padded_grads_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        padded_max_radii2D = torch.zeros((n_init_points), device="cuda")
        padded_max_radii2D[:max_radii2D.shape[0]] = max_radii2D.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            padded_grad[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(padded_grad, (1.0-ratio))
            selected_pts_mask = torch.where(padded_grad > threshold, True, False)
            # print(f"split {selected_pts_mask.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")
        else:
            padded_grads_abs[selected_pts_mask] = 0
            mask = (torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent) & (padded_max_radii2D > self.abs_split_radii2D_threshold)
            padded_grads_abs[~mask] = 0
            selected_pts_mask_abs = torch.where(padded_grads_abs >= grad_abs_threshold, True, False)
            limited_num = min(self.max_all_points - n_init_points - selected_pts_mask.sum(), self.max_abs_split_points)
            if selected_pts_mask_abs.sum() > limited_num:
                ratio = limited_num / float(n_init_points)
                threshold = torch.quantile(padded_grads_abs, (1.0-ratio))
                selected_pts_mask_abs = torch.where(padded_grads_abs > threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
            # print(f"split {selected_pts_mask.sum()}, abs {selected_pts_mask_abs.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_knn_f = self._knn_f[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            grads_tmp = grads.squeeze().clone()
            grads_tmp[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(grads_tmp, (1.0-ratio))
            selected_pts_mask = torch.where(grads_tmp > threshold, True, False)

        if selected_pts_mask.sum() > 0:
            # print(f"clone {selected_pts_mask.sum()}")
            new_xyz = self._xyz[selected_pts_mask]

            stds = self.get_scaling[selected_pts_mask]
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
            
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_knn_f = self._knn_f[selected_pts_mask]

            self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, abs_max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads_abs = self.xyz_gradient_accum_abs / self.denom_abs
        grads[grads.isnan()] = 0.0
        grads_abs[grads_abs.isnan()] = 0.0
        max_radii2D = self.max_radii2D.clone()

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, grads_abs, abs_max_grad, extent, max_radii2D)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        # print(f"all points {self._xyz.shape[0]}")
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, viewspace_point_tensor_abs, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor_abs.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.denom_abs[update_filter] += 1

    def get_points_depth_in_depth_map(self, fov_camera, depth, points_in_camera_space, scale=1):
        st = max(int(scale/2)-1,0)
        depth_view = depth[None,:,st::scale,st::scale]
        W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
        depth_view = depth_view[:H, :W]
        pts_projections = torch.stack(
                        [points_in_camera_space[:,0] * fov_camera.Fx / points_in_camera_space[:,2] + fov_camera.Cx,
                         points_in_camera_space[:,1] * fov_camera.Fy / points_in_camera_space[:,2] + fov_camera.Cy], -1).float()/scale
        mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) &\
               (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:,2] > 0.1)

        pts_projections[..., 0] /= ((W - 1) / 2)
        pts_projections[..., 1] /= ((H - 1) / 2)
        pts_projections -= 1
        pts_projections = pts_projections.view(1, -1, 1, 2)
        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
        return map_z, mask
    
    def get_points_from_depth(self, fov_camera, depth, scale=1):
        st = int(max(int(scale/2)-1,0))
        depth_view = depth.squeeze()[st::scale,st::scale]
        rays_d = fov_camera.get_rays(scale=scale)
        depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
        pts = (rays_d * depth_view[..., None]).reshape(-1,3)
        R = torch.tensor(fov_camera.R).float().cuda()
        T = torch.tensor(fov_camera.T).float().cuda()
        pts = (pts-T)@R.transpose(-1,-2)
        return pts

    def get_layer_span(self, name):
        """Return an explicit material span, or an empty span when absent."""

        span = self.layer_spans.get(name)
        if span is not None:
            return span
        return GaussianLayerSpan(name, self._xyz.shape[0], self._xyz.shape[0], False)

    def _reset_dynamic_gaussians(self):
        """Restore the immutable PGSR scene before constructing a new frame."""

        if self._xyz.shape[0] != self.num_src_gaussian:
            self._xyz = self._xyz[:self.num_src_gaussian, ...]
            self._features_dc = self._features_dc[:self.num_src_gaussian, ...]
            self._features_rest = self._features_rest[:self.num_src_gaussian, ...]
            self._opacity = self._opacity[:self.num_src_gaussian, ...]
            self._scaling = self._scaling[:self.num_src_gaussian, ...]
            self._rotation = self._rotation[:self.num_src_gaussian, ...]
            self._normal = self._normal[:self.num_src_gaussian, ...]
        self.num_sand_gaussian = 0
        self.num_fluid_gaussian = 0
        self.water_thickness_t = torch.empty(
            0, dtype=torch.float32, device=self._xyz.device
        )
        self.sand_reference_triangles_gs_t = torch.empty(
            (0, 3, 3), dtype=torch.float32, device=self._xyz.device
        )
        self.water_terrain_reference_triangles_gs_t = torch.empty(
            (0, 3, 3), dtype=torch.float32, device=self._xyz.device
        )
        self.water_surface_reference_triangles_gs_t = torch.empty(
            (0, 3, 3), dtype=torch.float32, device=self._xyz.device
        )
        self.sand_reference_vertex_heights_t = torch.empty(
            (0, 3), dtype=torch.float32, device=self._xyz.device
        )
        self.sand_reference_face_wetness_t = torch.empty(
            0, dtype=torch.float32, device=self._xyz.device
        )
        self.sand_primitive_boundary_t = torch.empty(
            0, dtype=torch.bool, device=self._xyz.device
        )
        self.sand_primitive_slope_t = torch.empty(
            0, dtype=torch.float32, device=self._xyz.device
        )
        self.sand_reference_sand_epsilon = 0.0
        self.sand_reference_full_coverage_height = 1.0
        self.layer_spans = {
            "source": GaussianLayerSpan(
                "source", 0, self.num_src_gaussian, True
            )
        }

    def _append_dynamic_layer(
        self,
        name,
        *,
        xyz,
        features_dc,
        features_rest,
        opacity,
        scaling,
        rotation,
        normal,
        opaque,
    ):
        """Append one fully specified layer and register its exact array range."""

        count = xyz.shape[0]
        attributes = (
            features_dc,
            features_rest,
            opacity,
            scaling,
            rotation,
            normal,
        )
        if any(attribute.shape[0] != count for attribute in attributes):
            raise ValueError(f"Gaussian layer {name} has inconsistent attribute counts")
        if name in self.layer_spans:
            raise ValueError(f"Gaussian layer {name} was appended twice")
        start = self._xyz.shape[0]
        self._xyz = torch.cat([self._xyz, xyz], dim=0)
        self._features_dc = torch.cat([self._features_dc, features_dc], dim=0)
        self._features_rest = torch.cat(
            [self._features_rest, features_rest], dim=0
        )
        self._opacity = torch.cat([self._opacity, opacity], dim=0)
        self._scaling = torch.cat([self._scaling, scaling], dim=0)
        self._rotation = torch.cat([self._rotation, rotation], dim=0)
        self._normal = torch.cat([self._normal, normal], dim=0)
        self.layer_spans[name] = GaussianLayerSpan(
            name, start, start + count, opaque
        )

    ## qy
    def get_normal_with_fluid(self, view_cam):
        """Return camera-facing normals in the exact shared-array order."""
        # PGSR 场景的 Gaussian 近似局部平面，因此用最短半轴作为表面法线。
        normal_global_src = self.get_smallest_axis()[:self.num_src_gaussian, ...]
        gaussian_to_cam_global_src = view_cam.camera_center - self._xyz[:self.num_src_gaussian, ...]
        neg_mask_src = (normal_global_src * gaussian_to_cam_global_src).sum(-1) < 0.0
        normal_global_src[neg_mask_src] = -normal_global_src[neg_mask_src]

        # Dynamic normals are written next to their Gaussian attributes by the
        # layer builder. Flipping a clone avoids making the stored normal depend
        # on which camera happened to render first.
        normal_dynamic = self._normal[self.num_src_gaussian:, ...].clone()
        if normal_dynamic.shape[0] > 0:
            to_camera = (
                view_cam.camera_center
                - self._xyz[self.num_src_gaussian:, ...]
            )
            neg_mask = (normal_dynamic * to_camera).sum(-1) < 0.0
            normal_dynamic[neg_mask] = -normal_dynamic[neg_mask]
        return torch.cat([normal_global_src, normal_dynamic], dim=0)

    def get_stored_layer_normals(self, name):
        """Return stable world-space normals for one explicit dynamic layer.

        ``get_normal_with_fluid`` intentionally flips normals toward each
        camera because rasterized plane surfaces are treated as two-sided.
        Material shading and P4's reflection/refraction contract
        instead require a camera-independent physical orientation.  Sand and
        water builders already store that orientation in ``_normal``; exposing
        it by named span prevents the renderer from reaching into private
        arrays or accidentally applying the visibility flip.
        """

        span = self.get_layer_span(name)
        if span.count == 0:
            return self._normal[0:0, ...]
        if span.start < self.num_src_gaussian:
            raise ValueError(
                f"layer {name!r} has no stored physical normal contract"
            )
        return self._normal[span.start:span.end, ...]

    def prune_for_height_map(self, dataset_name, height_map_z):
        """Keep the lower scene layer used to rasterize the ground height map."""
        mask = torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")

        #if dataset_name in ["garden","bicycle"]:
        xyz_transform = transform_point_cloud_2(self.get_xyz, self.align_rotation_angles, self.align_translation_vector)
        # save_tensor_point_cloud_as_ply(xyz_transform, "./debug/xyz_transform.ply")
        # exit()
        # else:
        #     xyz_transform = self.get_xyz

        valid = xyz_transform[:,2] <= height_map_z
        mask[valid] = False

        valid_points_mask = ~mask
        self._xyz = self._xyz[valid_points_mask]
        self._features_dc = self._features_dc[valid_points_mask]
        self._features_rest = self._features_rest[valid_points_mask]
        self._opacity = self._opacity[valid_points_mask]
        self._scaling = self._scaling[valid_points_mask]
        self._rotation = self._rotation[valid_points_mask]


    def update_dynamic_layers(self, *, time_seconds):
        """Replace sand/water layers at one explicit physical simulation time."""

        if time_seconds is None:
            raise ValueError("sand-water rendering requires an explicit physical time")
        if getattr(self.args, "render_sand", False):
            return self._add_sand_water_gaussians(time_seconds)
        return self._add_sand_water_fluid_gaussian(time_seconds)

    def _fluid_layer_attributes(self, xyz_fluid_t, normal_fluid_t):
        """Build free-water material arrays without mutating the model."""

        count = xyz_fluid_t.shape[0]
        features_dc_value = (
            torch.tensor(
                self.args.diffuse_fluid, dtype=torch.float, device="cuda"
            )
            / 255
        )
        features_dc_fluid_t = features_dc_value.repeat(
            count, 1, 1
        )
        features_extra_fluid = self.features_extra_np[self.index, ...]
        features_extra_fluid_t = (
            torch.tensor(
                features_extra_fluid, dtype=torch.float, device="cuda"
            )
            .repeat(count, 1, 1)
            .transpose(1, 2)
        )
        opacities_value = _logit(
            torch.tensor(
                self.args.opacity_fluid, dtype=torch.float, device="cuda"
            )
        )
        opacities_fluid_t = opacities_value.repeat(
            count, 1
        )
        scales_value = _log(
            torch.tensor(
                self.args.scale_fluid, dtype=torch.float, device="cuda"
            )
        )
        scales_fluid_t = scales_value.repeat(count, 1)
        rots_fluid = self.rots_np[self.index, ...]
        rots_fluid_t = torch.tensor(
            rots_fluid, dtype=torch.float, device="cuda"
        ).repeat(count, 1)
        return {
            "xyz": xyz_fluid_t,
            "features_dc": features_dc_fluid_t,
            "features_rest": features_extra_fluid_t,
            "opacity": opacities_fluid_t,
            "scaling": scales_fluid_t,
            "rotation": rots_fluid_t,
            "normal": normal_fluid_t,
            "opaque": False,
        }

    def _replace_fluid_gaussians(self, xyz_fluid_t, normal_fluid_t):
        """Install water geometry while preserving configured material parameters."""

        attributes = self._fluid_layer_attributes(
            xyz_fluid_t, normal_fluid_t
        )
        self._reset_dynamic_gaussians()
        self.num_fluid_gaussian = xyz_fluid_t.shape[0]
        self.normal_fluid_t = normal_fluid_t
        self._append_dynamic_layer("fluid", **attributes)

    def _sand_layer_attributes(self, geometry):
        """Convert CPU surfels and mixed-water wetness into Gaussian tensors."""

        xyz = torch.from_numpy(geometry.xyz_gs).to(
            device="cuda", dtype=torch.float32
        )
        normal = torch.from_numpy(geometry.normals_gs).to(
            device="cuda", dtype=torch.float32
        )
        rotation = torch.from_numpy(geometry.rotations_wxyz).to(
            device="cuda", dtype=torch.float32
        )
        scaling = torch.log(
            torch.from_numpy(geometry.scales_gs).to(
                device="cuda", dtype=torch.float32
            )
        )
        mixed_water_wetness = torch.from_numpy(
            geometry.mixed_water_wetness
        ).to(
            device="cuda", dtype=torch.float32
        )
        # New profiles may name these by their actual P4 meaning.  Fall back to
        # the P3 option names so committed experiments remain reproducible.
        dry_threshold = float(
            getattr(
                self.args,
                "sand_mixed_water_wetness_dry",
                getattr(self.args, "sand_saturation_dry", 0.0),
            )
        )
        wet_threshold = float(
            getattr(
                self.args,
                "sand_mixed_water_wetness_wet",
                getattr(self.args, "sand_saturation_wet", 0.5),
            )
        )
        if wet_threshold <= dry_threshold:
            raise ValueError(
                "sand wetness upper threshold must exceed the dry threshold"
            )
        wetness = torch.clamp(
            (mixed_water_wetness - dry_threshold)
            / (wet_threshold - dry_threshold),
            0.0,
            1.0,
        )
        wetness = wetness * wetness * (3.0 - 2.0 * wetness)

        dry_color = torch.tensor(
            getattr(self.args, "sand_dry_color", [194, 154, 96]),
            dtype=torch.float32,
            device="cuda",
        ) / 255.0
        wet_color = torch.tensor(
            getattr(self.args, "sand_wet_color", [112, 82, 52]),
            dtype=torch.float32,
            device="cuda",
        ) / 255.0
        exposure = float(getattr(self.args, "sand_exposure", 1.0))
        # Material interpolation uses linear RGB so intermediate wet sand
        # preserves the intended intensity during blending.
        rgb_linear = (
            srgb_to_linear(dry_color)[None, :] * (1.0 - wetness[:, None])
            + srgb_to_linear(wet_color)[None, :] * wetness[:, None]
        )
        rgb = torch.clamp(
            linear_to_srgb(rgb_linear * exposure), 0.0, 1.0
        )
        # Unlike the legacy water path, P3 stores a mathematically valid
        # view-independent SH DC coefficient and zeroes all higher bands.
        features_dc = RGB2SH(rgb).unsqueeze(1)
        features_rest = torch.zeros(
            (
                xyz.shape[0],
                self._features_rest.shape[1],
                self._features_rest.shape[2],
            ),
            dtype=torch.float32,
            device="cuda",
        )

        coverage = torch.from_numpy(geometry.coverage).to(
            device="cuda", dtype=torch.float32
        )
        maximum_opacity = float(getattr(self.args, "sand_opacity", 0.995))
        if not 0.0 < maximum_opacity < 1.0:
            raise ValueError("sand_opacity must lie strictly between zero and one")
        activated_opacity = torch.clamp(
            maximum_opacity * coverage, min=1.0e-4, max=maximum_opacity
        )
        opacity = inverse_sigmoid(activated_opacity[:, None])

        dry_roughness = float(getattr(self.args, "sand_dry_roughness", 0.9))
        wet_roughness = float(getattr(self.args, "sand_wet_roughness", 0.45))
        dry_specular = float(getattr(self.args, "sand_dry_specular", 0.04))
        wet_specular = float(getattr(self.args, "sand_wet_specular", 0.18))
        self.sand_roughness_t = (
            dry_roughness * (1.0 - wetness) + wet_roughness * wetness
        )
        self.sand_specular_t = (
            dry_specular * (1.0 - wetness) + wet_specular * wetness
        )
        self.sand_wetness_t = wetness
        light_sim = np.asarray(
            getattr(self.args, "sand_light_direction_sim", [-0.35, -0.25, 1.0]),
            dtype=np.float64,
        )
        if light_sim.shape != (3,) or not np.isfinite(light_sim).all():
            raise ValueError("sand_light_direction_sim must contain three numbers")
        light_gs = light_sim @ self.sand_water_adapter.sim_to_gs[:3, :3].T
        light_norm = np.linalg.norm(light_gs)
        if light_norm <= np.finfo(np.float64).eps:
            raise ValueError("sand_light_direction_sim must be non-zero")
        self.sand_light_direction_gs_t = torch.tensor(
            light_gs / light_norm, dtype=torch.float32, device="cuda"
        )
        return {
            "xyz": xyz,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacity": opacity,
            "scaling": scaling,
            "rotation": rotation,
            "normal": normal,
            "opaque": True,
        }

    def _add_sand_water_gaussians(self, time_seconds):
        """Install opaque sand first and optional free water second."""

        frame = self.sand_water_adapter.sample(float(time_seconds))
        full_height = getattr(self.args, "sand_full_coverage_height", None)
        geometry_backend = getattr(
            self.args, "sand_geometry_backend", "cell_tangent"
        )
        # T1 keeps both representations behind one renderer-facing contract.
        # Material, rasterization and P4 source/sand depth composition below
        # are intentionally identical for a controlled geometry-only A/B.
        sand_geometry = self.sand_water_adapter.build_sand_geometry(
            frame,
            backend=geometry_backend,
            coverage_factor=float(
                getattr(self.args, "sand_coverage_factor", 0.55)
            ),
            normal_thickness_ratio=float(
                getattr(self.args, "sand_normal_thickness_ratio", 0.10)
            ),
            max_tangent_scale_ratio=float(
                getattr(self.args, "sand_max_tangent_scale_ratio", 4.0)
            ),
            full_coverage_height=(
                None if full_height is None else float(full_height)
            ),
            triangle_coverage_gamma=float(
                getattr(self.args, "sand_triangle_coverage_gamma", 1.0)
            ),
            triangle_terrain_max_edge_ratio=(
                None
                if getattr(
                    self.args,
                    "sand_triangle_terrain_max_edge_ratio",
                    None,
                )
                is None
                else float(self.args.sand_triangle_terrain_max_edge_ratio)
            ),
            triangle_contact_tolerance=(
                None
                if getattr(
                    self.args, "sand_triangle_contact_tolerance", None
                )
                is None
                else float(self.args.sand_triangle_contact_tolerance)
            ),
            triangle_diagonal_mode=getattr(
                self.args, "sand_triangle_diagonal_mode", "checkerboard"
            ),
            triangle_boundary_mode=getattr(
                self.args, "sand_triangle_boundary_mode", "clipped"
            ),
            triangle_subdivision_level=getattr(
                self.args, "sand_triangle_subdivision_level", 0
            ),
        )
        water_geometry = self.sand_water_adapter.geometry(frame)
        sand_attributes = self._sand_layer_attributes(sand_geometry)
        xyz_fluid_t = torch.from_numpy(water_geometry.xyz_gs).to(
            device="cuda", dtype=torch.float32
        )
        normal_fluid_t = torch.from_numpy(water_geometry.normals_gs).to(
            device="cuda", dtype=torch.float32
        )
        fluid_attributes = self._fluid_layer_attributes(
            xyz_fluid_t, normal_fluid_t
        )

        self._reset_dynamic_gaussians()
        if bool(getattr(self.args, "water_terrain_depth_proxy", False)):
            self.water_terrain_reference_triangles_gs_t = torch.from_numpy(
                self.sand_water_adapter.terrain_reference_triangles(frame)
            ).to(device="cuda", dtype=torch.float32)
        if bool(getattr(
            self.args, "water_physical_reference_depth_recovery", False
        )):
            self.water_surface_reference_triangles_gs_t = torch.from_numpy(
                self.sand_water_adapter.water_reference_triangles(frame)
            ).to(device="cuda", dtype=torch.float32)
        retain_reference_mesh = bool(getattr(
            self.args, "sand_triangle_reference_diagnostics", False
        )) or bool(getattr(
            self.args, "sand_soft_boundary_diagnostics", False
        )) or bool(getattr(
            self.args, "sand_primitive_attribution_diagnostics", False
        )) or bool(getattr(
            self.args, "sand_soft_boundary_guidance", False
        )) or bool(getattr(
            self.args,
            "sand_center_trusted_invalid_attribute_diagnostics",
            False,
        )) or bool(getattr(
            self.args,
            "sand_center_trusted_invalid_normal_recovery",
            False,
        )) or bool(getattr(
            self.args,
            "sand_physical_reference_normal_recovery",
            False,
        ))
        if retain_reference_mesh:
            self.sand_reference_triangles_gs_t = torch.from_numpy(
                sand_geometry.triangle_vertices_gs
            ).to(device="cuda", dtype=torch.float32)
            self.sand_reference_vertex_heights_t = torch.from_numpy(
                sand_geometry.triangle_vertex_sand_height
            ).to(device="cuda", dtype=torch.float32)
            # This is the primitive material value before screen-space alpha
            # blending. triangle_games has one Gaussian per retained triangle,
            # so it is also the independent face-wetness counterfactual.
            self.sand_reference_face_wetness_t = (
                self.sand_wetness_t.detach().clone()
            )
            self.sand_reference_sand_epsilon = float(
                sand_geometry.sand_epsilon
            )
            self.sand_reference_full_coverage_height = float(
                sand_geometry.full_coverage_height
            )
        if bool(getattr(
            self.args, "sand_primitive_attribution_diagnostics", False
        )):
            if sand_geometry.geometry_backend != "triangle_games":
                raise ValueError(
                    "sand primitive attribution requires triangle_games"
                )
            # Each final triangle creates exactly one Gaussian. Retain only
            # the compact labels needed by the T2.3 diagnostic passes.
            self.sand_primitive_boundary_t = torch.from_numpy(
                sand_geometry.boundary_class == 1
            ).to(device="cuda", dtype=torch.bool)
            self.sand_primitive_slope_t = torch.from_numpy(
                sand_geometry.triangle_slope
            ).to(device="cuda", dtype=torch.float32)
        self.num_sand_gaussian = sand_geometry.xyz_gs.shape[0]
        self._append_dynamic_layer("sand", **sand_attributes)
        self.num_fluid_gaussian = water_geometry.xyz_gs.shape[0]
        self.normal_fluid_t = normal_fluid_t
        self.water_thickness_t = torch.from_numpy(
            water_geometry.pure_water_thickness
        ).to(device="cuda", dtype=torch.float32)
        self._append_dynamic_layer("fluid", **fluid_attributes)
        self.last_sand_water_sample = {
            "time_seconds": frame.requested_time_seconds,
            "lower_step": frame.lower_step,
            "upper_step": frame.upper_step,
            "interpolation_weight": frame.interpolation_weight,
            "sand_geometry_backend": sand_geometry.geometry_backend,
            "active_sand_cells": int(np.count_nonzero(frame.sand_mask)),
            "sand_gaussians": self.num_sand_gaussian,
            "base_sand_triangles": sand_geometry.base_triangle_count,
            "candidate_sand_triangles": sand_geometry.candidate_triangle_count,
            "rejected_sand_triangles": sand_geometry.rejected_triangle_count,
            "rejected_degenerate_sand_triangles": (
                sand_geometry.rejected_degenerate_count
            ),
            "rejected_terrain_cliff_sand_triangles": (
                sand_geometry.rejected_terrain_cliff_count
            ),
            "rejected_high_terrain_contact_sand_triangles": (
                sand_geometry.rejected_high_terrain_contact_count
            ),
            "accepted_buried_step_sand_triangles": (
                sand_geometry.accepted_buried_step_count
            ),
            "steep_sand_triangles": sand_geometry.steep_sand_triangle_count,
            "sand_subdivision_level": sand_geometry.subdivision_level,
            "pre_subdivision_sand_triangles": (
                sand_geometry.pre_subdivision_triangle_count
            ),
            "active_water_cells": self.num_fluid_gaussian,
        }
        print(
            "sand-water frame: "
            f"t={frame.requested_time_seconds:.9g}, "
            f"steps={frame.lower_step}->{frame.upper_step}, "
            f"alpha={frame.interpolation_weight:.6g}, "
            f"sand_backend={sand_geometry.geometry_backend}, "
            f"active_sand_cells={np.count_nonzero(frame.sand_mask)}, "
            f"sand_gaussians={self.num_sand_gaussian}, "
            f"rejected_triangles={sand_geometry.rejected_triangle_count}, "
            "rejected_by_contact="
            f"{sand_geometry.rejected_high_terrain_contact_count}, "
            "rejected_by_terrain_cliff="
            f"{sand_geometry.rejected_terrain_cliff_count}, "
            f"buried_step={sand_geometry.accepted_buried_step_count}, "
            f"steep_sand={sand_geometry.steep_sand_triangle_count}, "
            f"subdivision={sand_geometry.subdivision_level}, "
            "pre_subdivision="
            f"{sand_geometry.pre_subdivision_triangle_count}, "
            f"active_water_cells={self.num_fluid_gaussian}"
        )

    def _add_sand_water_fluid_gaussian(self, time_seconds):
        """Create water Gaussians from one time-sampled solver snapshot."""
        frame = self.sand_water_adapter.sample(float(time_seconds))
        geometry = self.sand_water_adapter.geometry(frame)
        xyz_fluid_t = torch.from_numpy(geometry.xyz_gs).to(
            device="cuda", dtype=torch.float32
        )
        normal_fluid_t = torch.from_numpy(geometry.normals_gs).to(
            device="cuda", dtype=torch.float32
        )
        self.last_water_sample = {
            "time_seconds": frame.requested_time_seconds,
            "lower_step": frame.lower_step,
            "upper_step": frame.upper_step,
            "interpolation_weight": frame.interpolation_weight,
            "active_cells": int(geometry.xyz_gs.shape[0]),
        }
        print(
            "sand-water frame: "
            f"t={frame.requested_time_seconds:.9g}, "
            f"steps={frame.lower_step}->{frame.upper_step}, "
            f"alpha={frame.interpolation_weight:.6g}, "
            f"active_water_cells={geometry.xyz_gs.shape[0]}"
        )
        self._replace_fluid_gaussians(xyz_fluid_t, normal_fluid_t)
        if bool(getattr(self.args, "water_terrain_depth_proxy", False)):
            self.water_terrain_reference_triangles_gs_t = torch.from_numpy(
                self.sand_water_adapter.terrain_reference_triangles(frame)
            ).to(device="cuda", dtype=torch.float32)
        if bool(getattr(
            self.args, "water_physical_reference_depth_recovery", False
        )):
            self.water_surface_reference_triangles_gs_t = torch.from_numpy(
                self.sand_water_adapter.water_reference_triangles(frame)
            ).to(device="cuda", dtype=torch.float32)
        # P2 historically needed geometry only. The SandyGS P4 optics also
        # consumes local free-water thickness for refraction and absorption, so
        # preserve it for pure-water runs exactly as in the coupled P4 path.
        self.water_thickness_t = torch.from_numpy(
            geometry.pure_water_thickness
        ).to(device="cuda", dtype=torch.float32)
