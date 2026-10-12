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
import math
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.app_model import AppModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image
from gaussian_renderer.opaque_gbuffer import compose_opaque_gbuffer
from gaussian_renderer.water_gbuffer import build_water_gbuffer
from gaussian_renderer.water_optics import compose_water_optics


def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def render_sand_water(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,
           app_model: AppModel=None, return_depth_normal = True, envmap=None, is_training=True, is_render_height_map=False, args=None, cur_frame=-1):
    """
    渲染源场景与 SandyGS 沙水层；bg_color 必须位于 GPU 上。

    与经典 3DGS（以及本仓库 gaussian_renderer.render 中的 PDSR render）不同，本函数不是
    对全部 Gaussian 做一次光栅化后直接返回结果，而是按以下顺序执行多阶段合成：
      1. 按具名 span 划分静态场景、沙和自由水；
      2. 分别生成 source/sand 不透明 G-buffer 与自由水 G-buffer；
      3. 对自由水执行 Fresnel、折射、吸收、SSR 和最终合成。

    注意：各具名层仍须在 pc 的共享属性数组中连续存储；实际路径直接切片
    shs/scales/rotations，因此通常要求 convert_SHs_python=False、
    compute_cov3D_python=False。
    """

    """
    ## bicycle
    viewpoint_camera.world_view_transform = torch.tensor([[-0.9615, 0.0476, -0.2707, 0.0000],
            [0.2744, 0.2219, -0.9357, 0.0000],
            [0.0156, -0.9739, -0.2264, 0.0000],
            [0.5187, -0.9378, 3.3023, 1.0000]], device='cuda:0')
    viewpoint_camera.full_proj_transform = torch.tensor([[-1.2530, 0.1072, -0.2707, -0.2707],
            [0.3576, 0.5001, -0.9358, -0.9357],
            [0.0203, -2.1954, -0.2264, -0.2264],
            [0.6760, -2.1139, 3.2926, 3.3023]], device='cuda:0')
    viewpoint_camera.camera_center = torch.tensor([1.4372, 3.1556, -0.1738], device='cuda:0')
    viewpoint_camera.fx = torch.tensor(834.0642, device='cuda:0')
    viewpoint_camera.fy = torch.tensor(834.0642, device='cuda:0')
    viewpoint_camera.cx = torch.tensor(640, device='cuda:0', dtype=torch.int32)
    viewpoint_camera.cy = torch.tensor(370, device='cuda:0', dtype=torch.int32)
    viewpoint_camera.FoVx = torch.tensor(1.3090, device='cuda:0')
    viewpoint_camera.FoVy = torch.tensor(0.8351, device='cuda:0')
    viewpoint_camera.image_height = torch.tensor(740, device='cuda:0', dtype=torch.int32)
    viewpoint_camera.image_width = torch.tensor(1280, device='cuda:0', dtype=torch.int32)
    """
    #################################
    # 1. 准备经典 3DGS/PDSR 共有的光栅化输入
    #################################
    # 伪造可求导的屏幕空间均值，使 rasterizer 的反向传播能返回 2D mean 梯度。
    # 与 PDSR render 相同；本函数虽然主要用于渲染，仍保留了训练路径所需的张量。
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # 沙水渲染只支持透视投影；高度图仍由普通 PGSR 渲染入口生成。
    if not is_render_height_map:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        full_proj_transform = viewpoint_camera.get_full_proj_transform(orthographic=False)
        orthographic = False
        ortho_x_min = -1
        ortho_x_max = -1
        ortho_y_min = -1
        ortho_y_max = -1
    else:
        raise NotImplementedError("Not support is_render_height_map")

    # Dynamic material ranges are described by explicit named spans.
    num_all_gaussians = means3D.shape[0]
    num_src_gaussian = pc.num_src_gaussian
    num_sand_gaussian = getattr(pc, "num_sand_gaussian", 0)
    num_fluid_gaussian = pc.num_fluid_gaussian
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)

    num_gausssian_all = num_all_gaussians
    print(
        "num_src_gaussian, num_sand_gaussian, num_fluid_gaussian",
        num_src_gaussian,
        num_sand_gaussian,
        num_fluid_gaussian,
    )

    source_span = pc.get_layer_span("source")
    sand_span = pc.get_layer_span("sand")
    fluid_span = pc.get_layer_span("fluid")
    if getattr(args, "simulation_backend", "sand_water") != "sand_water":
        raise ValueError("SandyGS rendering requires simulation_backend=sand_water")
    if getattr(args, "water_rendering_backend", "sandygs_p4") != "sandygs_p4":
        raise ValueError("SandyGS rendering requires water_rendering_backend=sandygs_p4")

    # Source depth/normal pass uses the plane rasterizer's geometry output.
    return_plane = True
    raster_settings_img_dep_nor = PlaneGaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        render_geo=return_plane,
        debug=pipe.debug,
        orthographic=orthographic,
        ortho_x_min=ortho_x_min,
        ortho_x_max=ortho_x_max,
        ortho_y_min=ortho_y_min,
        ortho_y_max=ortho_y_max,
    )
    rasterizer_img_dep_nor = PlaneGaussianRasterizer(raster_settings=raster_settings_img_dep_nor)

    #################################
    # 2. 仅渲染原始/静态场景的 RGB 与深度
    #################################
    print("rendering src rgb")

    # Sand is composed in its own opaque depth pass, so this pass stops at the
    # immutable source span.
    source_pass_end = source_span.end
    means3D_src = means3D[:source_pass_end, ...]
    means2D_src = means2D[:source_pass_end, ...]
    means2D_abs_src = means2D_abs[:source_pass_end, ...]
    shs_src = None if shs is None else shs[:source_pass_end, ...]
    colors_precomp_src = (
        None
        if colors_precomp is None
        else colors_precomp[:source_pass_end, ...]
    )
    opacity_src = opacity[:source_pass_end, ...]
    scales_src = scales[:source_pass_end, ...]
    rotations_src = rotations[:source_pass_end, ...]

    """
    rendered_rgb_src, _, _, _, _ = rasterizer_only_img(
        means3D=means3D_src,
        means2D=means2D_src,
        means2D_abs=means2D_abs_src,
        shs=shs_src,
        colors_precomp=None,
        opacities=opacity_src,
        scales=scales_src,
        rotations=rotations_src,
        cov3D_precomp=None)    # [3,h,w]
    """

    # PDSR 会对全场景调用 get_normal；这里使用能处理新增流体 Gaussian 的法线接口，
    # 但此 pass 只截取源场景部分，作为后续折射采样的背景颜色和背景深度。
    global_normal_all = pc.get_normal_with_fluid(viewpoint_camera)
    global_normal_src = global_normal_all[:source_pass_end, ...]
    local_normal_src = global_normal_src @ viewpoint_camera.world_view_transform[:3, :3] # 相机空间法线

    pts_in_cam_src = means3D_src @ viewpoint_camera.world_view_transform[:3, :3] + viewpoint_camera.world_view_transform[3, :3]
    local_distance_src = (local_normal_src * pts_in_cam_src).sum(-1).abs()

    # 与 PDSR 的 plane rasterizer 约定一致：all_map = [normal.xyz, alpha常量, 平面到相机原点距离]。
    input_all_map_src = torch.zeros((means3D_src.shape[0], 5)).cuda().float()
    input_all_map_src[:, :3] = local_normal_src
    input_all_map_src[:, 3] = 1.0
    input_all_map_src[:, 4] = local_distance_src

    rendered_rgb_src, _, _, out_all_map_src, rendered_depth_src = rasterizer_img_dep_nor(
        means3D=means3D_src,
        means2D=means2D_src,
        means2D_abs=means2D_abs_src,
        shs=shs_src,
        colors_precomp=colors_precomp_src,
        opacities=opacity_src,
        scales=scales_src,
        rotations=rotations_src,
        all_map=input_all_map_src,
        cov3D_precomp=None)
    rendered_depth_src = rendered_depth_src.repeat(3,1,1)

    result = compose_opaque_gbuffer(
        args=args,
        bg_color=bg_color,
        colors_precomp=colors_precomp,
        full_proj_transform=full_proj_transform,
        image_height=image_height,
        image_width=image_width,
        means2D=means2D,
        means2D_abs=means2D_abs,
        means3D=means3D,
        num_gausssian_all=num_gausssian_all,
        num_sand_gaussian=num_sand_gaussian,
        opacity=opacity,
        orthographic=orthographic,
        ortho_x_min=ortho_x_min,
        ortho_x_max=ortho_x_max,
        ortho_y_min=ortho_y_min,
        ortho_y_max=ortho_y_max,
        out_all_map_src=out_all_map_src,
        pc=pc,
        pipe=pipe,
        rendered_depth_src=rendered_depth_src,
        rendered_rgb_src=rendered_rgb_src,
        rotations=rotations,
        sand_span=sand_span,
        scales=scales,
        scaling_modifier=scaling_modifier,
        shs=shs,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        viewpoint_camera=viewpoint_camera,
    )
    if num_fluid_gaussian == 0:
        return result
    rendered_rgb_composite = result["opaque_surface_rgb"]
    opaque_surface_depth = result["opaque_surface_depth"]
    opaque_surface_normal_camera_facing = result[
        "opaque_surface_normal_camera_facing"
    ]

    #################################
    # Independently rasterize the transparent free-water surface.
    #################################
    # Keep the already composed opaque source (or source+sand) G-buffer
    # immutable and rasterize only the fluid span; this makes both SSR hits
    # and submerged-sand occlusion well-defined.
    water_gbuffer = build_water_gbuffer(
        args=args,
        bg_color=bg_color,
        fluid_span=fluid_span,
        full_proj_transform=full_proj_transform,
        image_height=image_height,
        image_width=image_width,
        means2D=means2D,
        means2D_abs=means2D_abs,
        means3D=means3D,
        num_fluid_gaussian=num_fluid_gaussian,
        opacity=opacity,
        opaque_surface_depth=opaque_surface_depth,
        opaque_surface_sand_owner=result["opaque_surface_sand_owner"],
        orthographic=orthographic,
        ortho_x_min=ortho_x_min,
        ortho_x_max=ortho_x_max,
        ortho_y_min=ortho_y_min,
        ortho_y_max=ortho_y_max,
        pc=pc,
        pipe=pipe,
        rotations=rotations,
        scales=scales,
        scaling_modifier=scaling_modifier,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        viewpoint_camera=viewpoint_camera,
    )
    return compose_water_optics(
        args=args,
        image_height=image_height,
        image_width=image_width,
        opaque_surface_depth=opaque_surface_depth,
        opaque_surface_normal_camera_facing=(
            opaque_surface_normal_camera_facing
        ),
        pc=pc,
        rendered_rgb_composite=rendered_rgb_composite,
        result=result,
        viewpoint_camera=viewpoint_camera,
        water_gbuffer=water_gbuffer,
    )
