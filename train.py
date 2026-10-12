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

import os
from datetime import datetime
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight
from utils.graphics_utils import patch_offsets, patch_warp
from gaussian_renderer import render, network_gui
import sys, time
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import cv2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, erode
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel
from scene.cameras import Camera
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F

from utils.loss_utils import ScaleAndShiftInvariantLoss, depth_rank_loss
from scene.envlight import EnvLight

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(22)

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # backup main code
    cmd = f'cp ./train.py {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./arguments {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scene {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./utils {dataset.model_path}/'
    os.system(cmd)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, all_args=args)
    gaussians.training_setup(opt)

    if args.is_sky_model:
        print('make envmap.')
        env_map = EnvLight(resolution=1024).cuda()
        env_map.training_setup()
    else:
        env_map = None

    app_model = AppModel()
    app_model.train()
    app_model.cuda()
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    normal_loss, geo_loss, ncc_loss = None, None, None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    mono_depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)

    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image, gt_image_gray = viewpoint_cam.get_image()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        ## qy
        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
        #                     return_plane=iteration>opt.single_view_weight_from_iter, return_depth_normal=iteration>opt.single_view_weight_from_iter)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            return_plane=iteration>opt.single_view_weight_from_iter, return_depth_normal=iteration>opt.single_view_weight_from_iter,
                            envmap=env_map, is_training=True)
        ##
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        loss_extra = {}

        ssim_loss = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()

        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += opt.scale_loss_weight * min_scale_loss.mean()
            loss_extra["scale_loss"] = min_scale_loss.mean()

        ## qy
        if True:
            if viewpoint_cam.original_normal_image is not None and iteration>opt.single_view_weight_from_iter:
                gt_normal = viewpoint_cam.original_normal_image.cuda()
                gt_normal = - gt_normal
                render_normal = torch.nn.functional.normalize(render_pkg["rendered_normal"], p=2, dim=0)
                # print(render_normal.shape, render_normal.max(), render_normal.min())
                # print(gt_normal.shape, gt_normal.max(), gt_normal.min())

                loss_prior_normal = l1_loss(gt_normal, render_normal) + (1.0 - torch.sum(render_normal * gt_normal, dim=0)).mean()
                loss += loss_prior_normal * 0.03
                loss_extra["prior_normal"] = loss_prior_normal

        if False:
            if viewpoint_cam.original_depth_image is not None and iteration > opt.prior_depth_loss_from_iter:
                gt_depth = viewpoint_cam.original_depth_image.cuda()
                render_depth = render_pkg['plane_depth']
                #mask = gt_depth < 0.90
                #mask = None
                mask = gt_depth > 0.90
                loss_prior_depth = mono_depth_loss(render_depth, gt_depth, mask)
                #loss_prior_depth = depth_rank_loss(render_depth, gt_depth)
                loss += loss_prior_depth * 0.01 #1,0.03,0.01
                loss_extra["prior_depth"] = loss_prior_depth


        if args.is_sky_model:
            #sky_mask = viewpoint_cam.sky_mask.to(torch.bool)
            gt_depth = viewpoint_cam.original_depth_image.cuda()
            sky_mask = gt_depth > 0.90
            alpha = render_pkg['rendered_alpha'].cuda()
            o = alpha.clamp(1e-6, 1- 1e-6)
            sky = sky_mask.float().cuda()
            # waymo019: (1-sky) else: (sky)
            loss_sky_opac = (-(sky) * torch.log(1 - o)).mean()
            loss += 0.05 * loss_sky_opac
            loss_extra["sky_opacity"] = loss_sky_opac
        ##

        if True:
            normal_loss = None
            geo_loss = None
            ncc_loss = None


        # single-view loss
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]
            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0, 1).detach() ** 2

            if True:
                if not opt.wo_image_weight:
                    # image_weight = erode(image_weight[None,None]).squeeze()
                    normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
                else:
                    normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
                loss += (normal_loss)
                loss_extra["single_view_loss"] = normal_loss



        if True:
            # multi-view loss
            if iteration > opt.multi_view_weight_from_iter:
                nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
                use_virtul_cam = False
                if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                    nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                    use_virtul_cam = True
                if nearest_cam is not None:
                    patch_size = opt.multi_view_patch_size
                    sample_num = opt.multi_view_sample_num
                    pixel_noise_th = opt.multi_view_pixel_noise_th
                    total_patch_size = (patch_size * 2 + 1) ** 2
                    ncc_weight = opt.multi_view_ncc_weight
                    geo_weight = opt.multi_view_geo_weight
                    ## compute geometry consistency mask and loss
                    H, W = render_pkg['plane_depth'].squeeze().shape
                    ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
                    pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                    nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, app_model=app_model,
                                                return_plane=True, return_depth_normal=False)

                    pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                    pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                    map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)

                    pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                    pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                    R = torch.tensor(nearest_cam.R).float().cuda()
                    T = torch.tensor(nearest_cam.T).float().cuda()
                    pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                    pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                    pts_projections = torch.stack(
                                [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                                pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                    pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                    if not opt.wo_use_geo_occ_aware:
                        d_mask = d_mask & (pixel_noise < pixel_noise_th)
                        weights = (1.0 / torch.exp(pixel_noise)).detach()
                        weights[~d_mask] = 0
                    else:
                        d_mask = d_mask
                        weights = torch.ones_like(pixel_noise)
                        weights[~d_mask] = 0

                    ## qianfan
                    if iteration % 200 == 0:
                        gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)

                        if 'app_image' in render_pkg:
                            img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                        else:
                            img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)

                        normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)

                        depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)

                        d_mask_show = (weights.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                        d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)

                        depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

                        distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
                        distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                        distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                        distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)

                        image_weight = image_weight.detach().cpu().numpy()
                        image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
                        image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)

                        row0 = np.concatenate([gt_img_show, img_show, normal_show, distance_color], axis=1)
                        row1 = np.concatenate([d_mask_show_color, depth_color, depth_normal_show, image_weight_color], axis=1)

                        image_to_show = np.concatenate([row0, row1], axis=0)
                        cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)
                    ##

                    if d_mask.sum() > 0:
                        geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                        loss += geo_loss
                        if use_virtul_cam is False:
                            with torch.no_grad():
                                ## sample mask
                                d_mask = d_mask.reshape(-1)
                                valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                                if d_mask.sum() > sample_num:
                                    index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                    valid_indices = valid_indices[index]

                                weights = weights.reshape(-1)[valid_indices]
                                ## sample ref frame patch
                                pixels = pixels.reshape(-1,2)[valid_indices]
                                offsets = patch_offsets(patch_size, pixels.device)
                                ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()

                                H, W = gt_image_gray.squeeze().shape
                                pixels_patch = ori_pixels_patch.clone()
                                pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                                pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                                ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                                ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                                ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                                ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                            ## compute Homography
                            ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                            ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                            ref_local_d = render_pkg['rendered_distance'].squeeze()
                            # rays_d = viewpoint_cam.get_rays()
                            # rendered_normal2 = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3)
                            # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                            # ref_local_d = ref_local_d.reshape(*render_pkg['plane_depth'].shape)

                            ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                            H_ref_to_neareast = ref_to_neareast_r[None] - \
                                torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1),
                                            ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                            H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                            H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)

                            ## compute neareast frame patch
                            grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                            grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                            grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                            _, nearest_image_gray = nearest_cam.get_image()
                            sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                            sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

                            ## compute loss
                            ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                            mask = ncc_mask.reshape(-1)
                            ncc = ncc.reshape(-1) * weights
                            ncc = ncc[mask].squeeze()

                            if mask.sum() > 0:
                                ncc_loss = ncc_weight * ncc.mean()
                                loss += ncc_loss
                                loss_extra["multi_view_loss"] = ncc_loss


        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Single": f"{ema_single_view_for_log:.{5}f}",
                    "Geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                    "Pho": f"{ema_multi_view_pho_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, loss_extra, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), app_model, args, env_map=env_map)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if env_map is not None:
                    torch.save((env_map.capture(), iteration), scene.model_path + "/env_light_chkpnt" + str(iteration) + ".pth")

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold,
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)

            # multi-view observe trim
            if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(gaussians.get_opacity)
                for view in scene.getTrainCameras():
                    render_pkg_tmp = render(view, gaussians, pipe, bg, app_model=app_model, return_plane=False, return_depth_normal=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    gaussians.prune_points(prune_mask)

            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if env_map is not None and iteration < env_map.env_optimize_until:
                env_map.optimizer.step()
                env_map.optimizer.zero_grad(set_to_none = True)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)

    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, loss_extra, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, app_model, args, env_map=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        for k in loss_extra.keys():
            tb_writer.add_scalar(f'train_loss_patches/{k}_loss', loss_extra[k], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations or iteration == 0 or iteration == 10:
        torch.cuda.empty_cache()
        ## qy
        # validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
        #                       {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        validation_train_cams = []
        for view in scene.getTrainCameras():
            if (args.dataset_name == "truck" and view.image_name in ["000046", "000074", "000088"]) or (args.dataset_name == "family" and view.image_name in ["00001","00002","00003","00051"]):   # todo
                validation_train_cams.append(view)
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': validation_train_cams})
        ##
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model, envmap=env_map)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)

                    ## rendered_normal
                    rendered_normal = out["rendered_normal"]
                    # rendering_np = rendered_normal.permute(1, 2, 0).detach().cpu().numpy()
                    # os.makedirs("./debug", exist_ok=True)
                    # cv2.imwrite("debug/%05d_rendered_normal.exr" % iteration, rendering_np)

                    render_normal = 0.5 + (0.5 * rendered_normal)  # (-1, 1) -> (0, 1)
                    normal = torch.clamp(render_normal, 0.0, 1.0)

                    # rendering_np = normal.permute(1, 2, 0).detach().cpu().numpy()
                    # os.makedirs("./debug", exist_ok=True)
                    # cv2.imwrite("debug/%05d_rendered_normal.exr" % iteration, rendering_np)


                    gt_normal, gt_depth = viewpoint.get_gt_depth_and_normal_image()
                    ## gt normal
                    if viewpoint.original_normal_image is not None:

                        gt_normal = - gt_normal

                        #rendering_np = gt_normal.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                        #cv2.imwrite("debug/0-%05d_gt_normal.exr" % iteration, rendering_np)

                        gt_normal = 0.5 + (0.5 * gt_normal)  # (-1, 1) -> (0, 1)
                        gt_normal = torch.clamp(gt_normal, 0.0, 1.0)

                        # rendering_np = gt_normal.permute(1, 2, 0).detach().cpu().numpy()
                        # cv2.imwrite("debug/%05d_gt_normal.exr" % iteration, rendering_np)

                    ## gt depth
                    if viewpoint.original_depth_image is not None:

                        valid_depth_mask = gt_depth < 0.90
                        valid_depth_mask = valid_depth_mask.detach().cpu().numpy()

                        gt_depth = gt_depth.detach().cpu().numpy()
                        gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-20)
                        gt_depth = (gt_depth * 255).clip(0, 255).astype(np.uint8)
                        gt_depth = cv2.applyColorMap(gt_depth[0], cv2.COLORMAP_JET)
                        gt_depth = torch.tensor(gt_depth).cuda() / 255
                        gt_depth = gt_depth.permute(2,0,1)

                    ## rendered depth
                    render_depth = out['plane_depth']
                    render_depth = render_depth.detach().cpu().numpy()
                    render_depth = (render_depth - render_depth.min()) / (render_depth.max() - render_depth.min() + 1e-20)
                    render_depth = (render_depth * 255).clip(0, 255).astype(np.uint8)
                    render_depth = cv2.applyColorMap(render_depth[0], cv2.COLORMAP_JET)
                    render_depth = torch.tensor(render_depth).cuda() / 255
                    depth = render_depth.permute(2,0,1)

                    ## rendered alpha
                    alpha = out['rendered_alpha']

                    if tb_writer:
                        if config["name"] == "train" or (config["name"] == "test" and idx < 5):

                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if args.load_normal:
                                tb_writer.add_images(config['name'] + "_view_{}/normal".format(viewpoint.image_name), normal[None], global_step=iteration)
                            if args.load_depth:
                                tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/alpha".format(viewpoint.image_name), alpha[None], global_step=iteration)

                            if iteration == 10: #testing_iterations[0]:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                                if args.load_normal:
                                    tb_writer.add_images(config['name'] + "_view_{}/normal_ground_truth".format(viewpoint.image_name), gt_normal[None], global_step=iteration)
                                if args.load_depth:
                                    tb_writer.add_images(config['name'] + "_view_{}/depth_ground_truth".format(viewpoint.image_name), gt_depth[None], global_step=iteration)
                                    tb_writer.add_images(config['name'] + "_view_{}/valid_depth_mask".format(viewpoint.image_name), valid_depth_mask[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

        #exit()

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i for i in range(1, 30001) if i % 1000 ==0])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    ## qy
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument('--load_depth', action='store_true')
    parser.add_argument('--load_normal', action='store_true')
    parser.add_argument('--is_sky_model', action='store_true')
    parser.add_argument("--is_render_360", action="store_true")
    parser.add_argument("--is_render_height_map", action="store_true")
    parser.add_argument("--is_render_height_map_post", action="store_true")
    ##
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
