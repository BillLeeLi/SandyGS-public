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
from scene import Scene, EnvLight
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import copy
from collections import deque
from utils.graphics_utils import fov2focal
from utils.my_utils import depth_to_xyz_map, transform_c2w, transform_w2c, save_tensor_point_cloud_as_ply, transform_point_cloud, transform_cam2pixel
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import yaml
import torch.nn.functional as F


'''
def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0
'''


def post_process_hmap(src_height_map_path, args, cam_for_hmap_filter,
                      gaussians, pipeline, background, app_model, envmap, is_render_height_map):
    ## read height map
    src_height_map = torch.from_numpy(np.load(src_height_map_path)).cuda()

    ## turn to xyz
    hmap_h, hmap_w = src_height_map.shape[:2]

    height_map_xmin, height_map_xmax, height_map_ymin, height_map_ymax = args.height_map_range
    x = torch.linspace(height_map_xmin, height_map_xmax, hmap_w, device="cuda")
    y = torch.linspace(height_map_ymin, height_map_ymax, hmap_h, device="cuda")
    xx, yy = torch.meshgrid(x, y, indexing='xy')  # 注意这里使用indexing='xy'

    # 使用torch.vstack或stack，然后转置
    xyz = torch.stack([
        xx.flatten(),  # (height*width,)
        yy.flatten(),  # (height*width,)
        src_height_map.flatten()  # (height*width,)
    ], dim=0).T  # Flatten and stack x, y, z coordinates

    # transform to align the coord sys
    xyz_transform = transform_point_cloud(xyz, list(args.align_rotation_angles), list(args.align_translation_vector))   # [height*width, 3]

    # if args.dataset_name == "garden":
    #     cam_for_hmap_filter = cam_for_hmap_filter[0:30]
    # elif args.dataset_name == "treehill":
    #     cam_for_hmap_filter = cam_for_hmap_filter[30:60]

    for idx, view in enumerate(tqdm(cam_for_hmap_filter, desc="Rendering progress")):
        out = render(view, gaussians, pipeline, background, app_model=app_model, envmap=envmap, is_render_height_map=is_render_height_map, args=args)
        rendered_depth = out["plane_depth"]     # [1,h,w]
        rendering = out["render"].clamp(0.0, 1.0)
        #torchvision.utils.save_image(rendering, "debug/refine_hmap/garden_render.png")

        # height points world to camera
        _, rend_depth_h, rend_depth_w = rendered_depth.shape
        fx = view.fx
        fy = view.fy
        cx = view.cx
        cy = view.cy
        K = torch.tensor([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]]).float().cuda()

        w2c_mat = view.world_view_transform.transpose(1, 0)
        xyz_transform_c = transform_w2c(xyz_transform, w2c_mat)    # [M,3]

        # height points camera to pixel
        pixel_coords = transform_cam2pixel(xyz_transform_c, K)     # [M,2]
        xyz_transform_c_zdepth = xyz_transform_c[:,2]              # [M]
        u, v = pixel_coords[:, 0], pixel_coords[:, 1]              # [M]
        u_int, v_int = u.long(), v.long()          # [M]

        def filter_min_C_optimized_cuda(A, B, C, org_device='cuda', cur_device='cpu'):
            # 将张量移至GPU
            A, B, C = A.to(cur_device), B.to(cur_device), C.to(cur_device)

            # 初始化一个空字典来存储(A, B)和对应最小C及其索引
            AB_to_min_C_and_index = {}

            # 使用 tqdm 添加进度条，遍历所有的 (A, B, C) 组合
            for i in tqdm(range(len(A)), desc="Processing", ncols=100):
                ab_pair = (A[i].item(), B[i].item())  # 将A和B配对成一个元组作为键
                if ab_pair not in AB_to_min_C_and_index:
                    AB_to_min_C_and_index[ab_pair] = (C[i].item(), i)  # 新配对，直接存储C值和索引
                else:
                    current_min_C, current_index = AB_to_min_C_and_index[ab_pair]
                    if C[i].item() < current_min_C:  # 如果当前C更小，更新最小C和索引
                        AB_to_min_C_and_index[ab_pair] = (C[i].item(), i)

            # 提取出最终的A, B, C和对应的索引
            final_A = []
            final_B = []
            final_C = []
            final_indices = []  # 用来存储原始输入中的索引
            for (a, b), (c, idx) in AB_to_min_C_and_index.items():
                final_A.append(a)
                final_B.append(b)
                final_C.append(c)
                final_indices.append(idx)  # 存储对应的原始索引

            # 转换成张量返回
            final_A = torch.tensor(final_A, dtype=torch.int, device=org_device)
            final_B = torch.tensor(final_B, dtype=torch.int, device=org_device)
            final_C = torch.tensor(final_C, dtype=torch.float, device=org_device)
            final_indices = torch.tensor(final_indices, dtype=torch.int, device=org_device)

            return final_A, final_B, final_C, final_indices

        u_int_new, v_int_new, xyz_transform_c_zdepth_new, indices = filter_min_C_optimized_cuda(u_int, v_int, xyz_transform_c_zdepth)   # [M2]

        # 筛选投影在图像内的点
        valid_mask = (u_int_new >= 0) & (u_int_new < rend_depth_w) & (v_int_new >= 0) & (v_int_new < rend_depth_h)
        u_valid = u_int_new[valid_mask]      # [M3]
        v_valid = v_int_new[valid_mask]
        xyz_transform_c_zdepth_valid = xyz_transform_c_zdepth_new[valid_mask]
        # 获取 valid_mask 为 True 的索引
        valid_indices = torch.nonzero(valid_mask).squeeze(1)  # [M3]，包含u_int_new和v_int_new中满足条件的索引

        # 查询深度图，获取
        depth_mask = (rendered_depth[0, v_valid, u_valid] - xyz_transform_c_zdepth_valid) > 0.1
        valid_depth_indices = depth_mask.nonzero(as_tuple=True)     # [M4]，包含depth_mask为1的所有索引的张量

        if True:
            # debug
            rendering[:, v_valid[depth_mask], u_valid[depth_mask]] = 1.0
            os.makedirs("debug/refine_hmap/%s"%args.dataset_name, exist_ok=True)
            torchvision.utils.save_image(rendering, "debug/refine_hmap/%s/render_selected_%d.png"%(args.dataset_name, idx))

        # final index
        hmap_refine_indices = indices[valid_indices[valid_depth_indices]]

        if False:
            # debug
            rendering[:, v_int[hmap_refine_indices], u_int[hmap_refine_indices]] = 1.0
            torchvision.utils.save_image(rendering, "debug/refine_hmap/treehill/render_selected2_%d.png"%idx)

        # # todo: debug
        # hmap_refine_indices = 603 * hmap_w + 726

        # 这里我们先将 xyz 转换回原始的像素坐标
        # 从 xyz 中获取 x, y 坐标（即 xx 和 yy）
        xyz_x = xyz[hmap_refine_indices, 0]  # (M,)
        xyz_y = xyz[hmap_refine_indices, 1]  # (M,)

        ## todo: debug
        # xyz_x = 0.5161
        # xyz_y = -0.9267

        #np.savetxt('debug/xyz[hmap_refine_indices, :].txt', xyz[hmap_refine_indices, :].cpu().numpy(), fmt='%.6f', delimiter=' ')

        # 将 3D 坐标映射回原始的 height map 像素坐标
        # 通过线性映射的关系，将坐标从 [height_map_xmin, height_map_xmax] 映射到 [0, hmap_w-1]，并且将坐标从 [height_map_ymin, height_map_ymax] 映射到 [0, hmap_h-1]
        src_x = ((xyz_x - height_map_xmin) / (height_map_xmax - height_map_xmin) * (hmap_w - 1)).round().long()
        src_y = ((xyz_y - height_map_ymin) / (height_map_ymax - height_map_ymin) * (hmap_h - 1)).round().long()

        # print(src_height_map[603, 725], src_height_map[603, 726], xyz[603 * hmap_w + 726,:], pixel_coords[603 * hmap_w + 726,:])
        # print("src_x, src_y", xyz_x, xyz_y, src_x, src_y)
        # print((xyz_x - height_map_xmin), (xyz_x - height_map_xmin) / (height_map_xmax - height_map_xmin), (xyz_x - height_map_xmin) / (height_map_xmax - height_map_xmin) * (hmap_w - 1))
        # print((xyz_y - height_map_ymin) / (height_map_ymax - height_map_ymin) * (hmap_h - 1))
        # exit()

        #np.savetxt('debug/src_xy.txt', torch.cat([src_x.unsqueeze(1), src_y.unsqueeze(1)], dim=1).cpu().numpy(), fmt='%d', delimiter=' ')

        # import os
        # os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        # cv2.imwrite("debug/src_height_map_before.exr", src_height_map.cpu().numpy())

        src_height_map[src_y, src_x] = 0

        # print(src_height_map_path)
        # cv2.imwrite("debug/src_height_map_after.exr", src_height_map.cpu().numpy())
        # exit()


        ## turn to xyz
        hmap_h, hmap_w = src_height_map.shape[:2]

        height_map_xmin, height_map_xmax, height_map_ymin, height_map_ymax = args.height_map_range
        x = torch.linspace(height_map_xmin, height_map_xmax, hmap_w, device="cuda")
        y = torch.linspace(height_map_ymin, height_map_ymax, hmap_h, device="cuda")
        xx, yy = torch.meshgrid(x, y, indexing='xy')  # 注意这里使用indexing='xy'

        # 使用torch.vstack或stack，然后转置
        xyz = torch.stack([
            xx.flatten(),  # (height*width,)
            yy.flatten(),  # (height*width,)
            src_height_map.flatten()  # (height*width,)
        ], dim=0).T  # Flatten and stack x, y, z coordinates

        # transform to align the coord sys
        xyz_transform = transform_point_cloud(xyz, list(args.align_rotation_angles), list(args.align_translation_vector))  # [height*width, 3]


    src_height_map_np = src_height_map.cpu().numpy()
    np.save(src_height_map_path[:-4] + "_refined.npy", src_height_map_np)

    from utils.my_utils import save_tensor_point_cloud_as_ply
    save_tensor_point_cloud_as_ply(xyz, "debug/refine_hmap/%s/hmap_refine.ply"%args.dataset_name)
    #exit()


def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background, 
               app_model=None, max_depth=5.0, volume=None, use_depth_filter=False,
               is_render_360=False, is_render_height_map=False, envmap=None, args=None, height_map_z=-1):
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    if is_render_height_map:
        render_height_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_height")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")
    render_distance_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_distance")
    if app_model:
        render_app_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_app_image")

    if not is_render_360 and not is_render_height_map:
        makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    if is_render_height_map:
        makedirs(render_height_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)
    makedirs(render_distance_path, exist_ok=True)
    if app_model:
        makedirs(render_app_path, exist_ok=True)


    depths_tsdf_fusion = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt, _ = view.get_image()
        out = render(view, gaussians, pipeline, background, app_model=app_model, envmap=envmap, is_render_height_map=is_render_height_map, args=args)
        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape

        depth = out["plane_depth"].squeeze()    # [h,w]
        # # todo: debug
        depth[depth > 10] = 10
        depth[depth < 0] = 0
        print("depth.min(), depth.max(): ", depth.min(), depth.max(), depth.shape)
        #depth_tsdf = depth.clone()
        depth_np = depth.detach().cpu().numpy()
        depth_i = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

        if is_render_height_map:
            # height map
            depth = out["plane_depth"].squeeze()  # [h,w]
            # todo: debug
            if args.dataset_name in ["waymo181", "waymo362"]:
                pass
            else:
                depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(H//2, W//2), mode='bicubic', align_corners=False)
                depth = depth.squeeze(0).squeeze(0)
            height_map = args.ortho_cam_z - depth
            if args.dataset_name in ["waymo362"]:
                height_map[height_map > height_map_z] = 1.72 #height_map_z
            elif args.dataset_name in ["garden"]:
                height_map[height_map > height_map_z] = height_map_z
                height_map[height_map < 1.173] = height_map_z
            else:
                height_map[height_map > height_map_z] = 0 #height_map_z
            height_map[height_map < 0.1] = 0.1

            print("height_map.min(), height_map.max(): ", height_map.min(), height_map.max(), height_map.shape)
            height_np = height_map.detach().cpu().numpy()
            height_i = (height_np - height_np.min()) / (height_np.max() - height_np.min() + 1e-20)
            height_i = (height_i * 255).clip(0, 255).astype(np.uint8)
            height_color = cv2.applyColorMap(height_i, cv2.COLORMAP_JET)

        # distance map
        distance = out["rendered_distance"].squeeze()
        distance_np = distance.detach().cpu().numpy()
        distance_i = (distance_np - distance_np.min()) / (distance_np.max() - distance_np.min() + 1e-20)
        distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
        distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)

        normal = out["rendered_normal"].permute(1,2,0)
        normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
        normal = normal.detach().cpu().numpy()
        normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)

        if app_model:
            app_image = out["app_image"].clamp(0.0, 1.0)

        if is_render_360 or is_render_height_map:
            image_name = '{0:05d}'.format(idx)
        else:
            image_name = view.image_name

        #if name == 'test':
        if not is_render_360 and not is_render_height_map:
            torchvision.utils.save_image(gt.clamp(0.0, 1.0), os.path.join(gts_path, image_name + ".png"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, image_name + ".png"))
        cv2.imwrite(os.path.join(render_depth_path, image_name + ".png"), depth_color)

        if is_render_height_map:
            cv2.imwrite(os.path.join(render_height_path, image_name + ".png"), height_color)
            cv2.imwrite(os.path.join(render_height_path, image_name + ".exr"), height_np)
            np.save(os.path.join(render_height_path, image_name + ".npy"), height_np)

        cv2.imwrite(os.path.join(render_normal_path, image_name + ".png"), normal)
        cv2.imwrite(os.path.join(render_distance_path, image_name + ".png"), distance_color)
        if app_model:
            torchvision.utils.save_image(app_image, os.path.join(render_app_path, image_name + ".png"))

        if is_render_height_map:
            if True:
                ## turn to xyz
                height, width = depth.shape[:2]

                height_map_xmin, height_map_xmax, height_map_ymin, height_map_ymax = args.height_map_range
                x = torch.linspace(height_map_xmin, height_map_xmax, width, device="cuda")
                y = torch.linspace(height_map_ymin, height_map_ymax, height, device="cuda")
                xx, yy = torch.meshgrid(x, y, indexing='xy')  # 注意这里使用indexing='xy'

                # 使用torch.vstack或stack，然后转置
                xyz = torch.stack([
                    xx.flatten(),  # (height*width,)
                    yy.flatten(),  # (height*width,)
                    height_map.flatten()  # (height*width,)
                ], dim=0).T  # Flatten and stack x, y, z coordinates

                # sample_size = 100000
                # indices = torch.randint(0, xyz.shape[0], (sample_size,))
                # xyz = xyz[indices]

                save_tensor_point_cloud_as_ply(xyz, os.path.join(model_path, name, "ours_{}".format(iteration), "height_map_pc.ply"))

            if False:
                # get point cloud
                H, W = depth.shape
                fx = fov2focal(view.FoVx, W)
                fy = fov2focal(view.FoVy, H)
                cx = W / 2 - 0.5
                cy = H / 2 - 0.5
                K = torch.tensor([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]]).float().cuda()
                xyz_map, nonzero_depth_mask = depth_to_xyz_map(depth, H, W, K)  # [h,w,3]
                point_cloud_c = xyz_map.view(-1, 3)  # [h*w, 3]
                # np.savetxt(os.path.join(model_path, name, "ours_{}".format(iteration), "point_cloud_c.txt"), point_cloud_c.cpu().numpy())

                # transform cam2world
                w2c_mat = view.world_view_transform.transpose(1, 0)
                point_cloud_w = transform_c2w(point_cloud_c, w2c_mat)
                save_tensor_point_cloud_as_ply(point_cloud_w, os.path.join(model_path, name, "ours_{}".format(iteration), "point_cloud_w.ply"))

    """
        if use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = out["depth_normal"].permute(1,2,0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir*depth_normal, dim=-1).abs()
            angle = torch.acos(dot)
            mask = angle > (80.0 / 180 * 3.14159)
            depth_tsdf[mask] = 0
        depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())
        
    if volume is not None:
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()

            if view.mask is not None:
                ref_depth[view.mask.squeeze() < 0.5] = 0
            ref_depth[ref_depth>max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()
            
            pose = np.identity(4)
            pose[:3,:3] = view.R.transpose(-1,-2)
            pose[:3, 3] = view.T
            color = o3d.io.read_image(os.path.join(render_path, view.image_name + ".jpg"))
            depth = o3d.geometry.Image((ref_depth*1000).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                pose)
    """

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, num_cluster: int, use_depth_filter : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, args)
        #gaussians.use_app = True
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, dataset_name=args.dataset_name, all_args=args)
        app_model = None
        # app_model = AppModel()
        # app_model.load_weights(scene.model_path)
        # app_model.eval()
        # app_model.cuda()

        if args.is_sky_model:
            env_map = EnvLight(resolution=1024).cuda()
            env_map.training_setup()
            env_checkpoint = os.path.join(dataset.model_path, "env_light_chkpnt30000.pth")
            (light_params, _) = torch.load(env_checkpoint)
            env_map.restore(light_params)
        else:
            env_map = None

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0*voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        # post process to remove floaters
        if args.is_render_height_map_post:

            from utils.camera_utils import generate_cam_path

            if args.dataset_name in ["waymo181","waymo362"]:
                num_frame_selected = 10
                selected_cam = scene.getTrainCameras()[::num_frame_selected]
            elif args.dataset_name == "bicycle":
                n_frames = 60 #30
                cam_traj = generate_cam_path(scene.getTrainCameras(), args.dataset_name + "_hmap_refine",
                                             n_frames=n_frames)
                selected_cam = cam_traj  # scene.getTrainCameras()
            else:
                n_frames = 60
                cam_traj = generate_cam_path(scene.getTrainCameras(), args.dataset_name + "_hmap_refine", n_frames=n_frames)
                selected_cam = cam_traj     # scene.getTrainCameras()

            #for src_height_map_path in [args.src_height_map_path, args.src_height_map_high_path]:
            for src_height_map_path in [args.src_height_map_high_path]:
                post_process_hmap(src_height_map_path, args, selected_cam, gaussians, pipeline, background,
                                  app_model, env_map, is_render_height_map=False)

        elif args.is_render_height_map:
            #for height_map_z in [args.low_height_map_z]:
            for height_map_z in [args.high_height_map_z, args.low_height_map_z]:
                print("is_render_height_map, z: ", height_map_z)

                gaussians.prune_for_height_map(args.dataset_name, height_map_z)

                from utils.camera_utils import generate_cam_path
                from utils.my_utils import transform_w_to_w2

                # n_frames = 1
                # cam_traj = generate_cam_path(scene.getTrainCameras(), args.dataset_name + "_hmap", n_frames=n_frames)
                # selected_cam = cam_traj

                c2w = torch.tensor([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, -1, args.ortho_cam_z],
                    [0, 0, 0, 1]
                ], dtype=torch.float, device="cuda")
                #if args.dataset_name in ["garden","bicycle"]:
                c2w = transform_w_to_w2(c2w, list(args.align_rotation_angles), list(args.align_translation_vector))
                c2w = c2w.cpu().numpy()

                cam = copy.deepcopy(scene.getTrainCameras()[0])
                cam.image_height = int(cam.image_height / 2) * 2
                cam.image_width = int(cam.image_width / 2) * 2
                cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
                cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
                cam.camera_center = cam.world_view_transform.inverse()[3, :3]
                selected_cam = [cam]

                #elected_cam = scene.getTrainCameras()[:1]
                render_set(dataset.model_path, "hmap_%f"%height_map_z, scene.loaded_iter, selected_cam, scene, gaussians, pipeline,
                           background, app_model=app_model,
                           max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter,
                           is_render_height_map=True, envmap=env_map, args=args, height_map_z=height_map_z)

        elif args.is_render_360:
            print("is_render_360")

            from utils.camera_utils import generate_cam_path

            n_frames = 200 #200,400
            cam_traj = generate_cam_path(scene.getTrainCameras(), args.dataset_name, n_frames=n_frames)
            selected_cam = cam_traj[:1]

            #selected_cam = scene.getTrainCameras()[:1]
            render_set(dataset.model_path, "360", scene.loaded_iter, selected_cam, scene, gaussians, pipeline, background, app_model=app_model,
                           max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter, is_render_360=True, envmap=env_map)

        else:

            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene, gaussians, pipeline, background, app_model=app_model,
                           max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter, envmap=env_map)
                """
                print(f"extract_triangle_mesh")
                mesh = volume.extract_triangle_mesh()
    
                path = os.path.join(dataset.model_path, "mesh")
                os.makedirs(path, exist_ok=True)
                
                o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion.ply"), mesh, 
                                           write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                
                mesh = post_process_mesh(mesh, num_cluster)
                o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion_post.ply"), mesh, 
                                           write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                """

            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, gaussians, pipeline, background, app_model=app_model, envmap=env_map)


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def update_args_with_yaml(args, yaml_config):
    for key, value in yaml_config.items():
        if not hasattr(args, key):
            setattr(args, key, value)

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")
    ## qy
    parser.add_argument('--load_depth', action='store_true')
    parser.add_argument('--load_normal', action='store_true')
    parser.add_argument("--config_file", type=str, help="Path to the YAML configuration file")

    args = get_combined_args(parser)

    if args.config_file:
        yaml_config = load_yaml_config(args.config_file)
        update_args_with_yaml(args, yaml_config)
    ##

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter, args)