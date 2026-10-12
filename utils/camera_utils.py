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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
import sys

## qy
import torchvision.transforms.functional as torchvision_F
WARNED = False

def NumpytoTorch(image, resolution):
    image = torch.from_numpy(image)
    if image.ndim == 4: image = image.squeeze(0)
    if image.shape[-1] == 3 or image.shape[-1] == 1:
        image = image.permute(2, 0, 1)
    _, orig_h, orig_w = image.shape
    if resolution == [orig_h, orig_w]:
        resized_image = image
    else:
        resized_image = torchvision_F.resize(image, resolution, antialias=True)

    return resized_image
##

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, all_args):
    orig_w, orig_h = cam_info.width, cam_info.height
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    print(f"scale {float(global_down) * float(resolution_scale)}")
                    WARNED = True
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    sys.stdout.write('\r')
    sys.stdout.write("load camera {}".format(id))
    sys.stdout.flush()

    # # todo: debug
    resolution = (1280, 768)

    ## qy
    if cam_info.normal_image is not None:
        size = list(resolution)[::-1]
        gt_normal_image = NumpytoTorch(cam_info.normal_image, size).to(args.data_device)
    else:
        gt_normal_image = None

    if cam_info.depth_image is not None:
        size = list(resolution)[::-1]
        gt_depth_image = NumpytoTorch(cam_info.depth_image, size).to(args.data_device)
    else:
        gt_depth_image = None

    # resized_normal_image_rgb = PILtoTorch(cam_info.normal_image, resolution)
    # gt_normal_image = resized_normal_image_rgb[:3, ...]
    # gt_normal_image = torch.tensor(cv2.resize(np.transpose(cam_info.normal_image, (1,2,0)), resolution, interpolation=cv2.INTER_NEAREST)).to(args.data_device)

    ##

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image_width=resolution[0], image_height=resolution[1],
                  image_path=cam_info.image_path,
                  image_name=cam_info.image_name, uid=cam_info.global_id, 
                  preload_img=args.preload_img, 
                  ncc_scale=args.ncc_scale,
                  data_device=args.data_device,
                  ## qy
                  normal_image=gt_normal_image,
                  depth_image=gt_depth_image,
                  all_args=all_args
                  ##
                  )

def cameraList_from_camInfos(cam_infos, resolution_scale, args, all_args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, all_args))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

## qy
import torch
from typing import List, Mapping, Optional, Text, Tuple, Union
import copy


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms poses so principal components lie on XYZ axes.

    Args:
      poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

    Returns:
      A tuple (poses, transform), with the transformed poses and the applied
      camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    return poses_recentered, transform


def generate_camera_poses_on_hemisphere(
    dataset_name,
    n_frames,
    ustudy_id=-1,
    azimuth_range_degrees=None,
):
    """
    Generate camera poses looking at the sphere's center from a hemisphere.

    Args:
        radius (float): Radius of the hemisphere.
        azimuth_angles (torch.Tensor): 1D tensor of azimuth angles in radians.
        elevation_angles (torch.Tensor): 1D tensor of elevation angles in radians.

    Returns:
        torch.Tensor: A tensor of shape [N, 4, 4], where N is the number of camera poses.
    """

    ########################
    # general
    #######################
    if dataset_name == "family":
        #n_frames = 30
        # radius = 3.8
        # azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(-30)), torch.deg2rad(torch.tensor(170)), steps=n_frames)  # 8 azimuth angles
        # elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(10))])
        # x_offset = 0.0
        # y_offset = -1.3
        # z_offset = 0.0
        # look_up_point = [0.0, -1.3, -0.8]

        radius = 4.0
        azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(-20)), torch.deg2rad(torch.tensor(80)), steps=n_frames)
        #azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(0)), torch.deg2rad(torch.tensor(100)), steps=n_frames)
        #azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(0)), torch.deg2rad(torch.tensor(360)), steps=n_frames)
        elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(7))])
        x_offset = 0.0
        y_offset = -1.3
        z_offset = 0.0
        look_up_point = [0.0, -1.3, -0.8]

        # radius = 4.0
        # azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(0)), torch.deg2rad(torch.tensor(360)), steps=n_frames)
        # elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(7))])
        # x_offset = 0.0
        # y_offset = 1.3
        # z_offset = 0.0
        # look_up_point = [0.0, 1.3, -0.8]

    elif dataset_name == "truck":
        # radius = 4
        # azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(90)), torch.deg2rad(torch.tensor(270)), steps=n_frames)
        # elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(10))])
        # x_offset = 0.0
        # y_offset = 0.0
        # z_offset = -0.5
        # look_up_point = [0.0, 0.0, -0.5]

        radius = 3.8
        azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(130)), torch.deg2rad(torch.tensor(280)), steps=n_frames)
        # azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(240)), torch.deg2rad(torch.tensor(240)), steps=n_frames)
        elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(10))])
        x_offset = 0.0
        y_offset = 0.0
        z_offset = -1.0
        look_up_point = [0.0, 0.0, -1.3]

    elif dataset_name in ["bicycle","bicycle_rebuttal"]:

        radius = 4.5  # Hemisphere radius
        azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(-100)), torch.deg2rad(torch.tensor(100)), steps=n_frames)
        # azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(0)), torch.deg2rad(torch.tensor(360)), steps=n_frames)
        elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(15))])
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        look_up_point = [0.0, 1.0, -1.5]

        # # todo: fixV
        # radius = 2.6  # Hemisphere radius
        # azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(250)), torch.deg2rad(torch.tensor(250)), steps=n_frames)
        # elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(30))])
        # x_offset = 0.0
        # y_offset = 0.0
        # z_offset = 0.0
        # look_up_point = [0.0, 1.0, -1.0]

    elif dataset_name == "treehill":
        radius = 4  # Hemisphere radius
        azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(60)), torch.deg2rad(torch.tensor(150)), steps=n_frames)
        #azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(60)), torch.deg2rad(torch.tensor(240)), steps=n_frames)
        #azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(240)), torch.deg2rad(torch.tensor(420)), steps=n_frames)
        #azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(0)), torch.deg2rad(torch.tensor(360)), steps=n_frames)
        elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(15))])
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        look_up_point = [0.0, 0.0, -1.0]

    elif dataset_name in ["garden","garden_igs2gs"]:
        radius = 5.3  # Hemisphere radius
        #azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(140)), torch.deg2rad(torch.tensor(460)), steps=n_frames)
        #azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(200)), torch.deg2rad(torch.tensor(440)), steps=n_frames)
        #azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(400)), torch.deg2rad(torch.tensor(400)), steps=n_frames)
        azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(260)), torch.deg2rad(torch.tensor(400)), steps=n_frames)
        #azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(340)), torch.deg2rad(torch.tensor(340)), steps=n_frames)
        # azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(220)), torch.deg2rad(torch.tensor(220)), steps=n_frames)
        elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(0))])
        x_offset = 0.0
        y_offset = 0.0
        z_offset = -0.7
        look_up_point = [0.0, 0.5, -2.0]

    elif dataset_name in ["garden_rebuttal"]:
        radius = 5.3  # Hemisphere radius
        #azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(140)), torch.deg2rad(torch.tensor(460)), steps=n_frames)
        #azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(200)), torch.deg2rad(torch.tensor(440)), steps=n_frames)
        #azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(260)), torch.deg2rad(torch.tensor(400)), steps=n_frames)
        azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(400)), torch.deg2rad(torch.tensor(400)), steps=n_frames)
        # azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(220)), torch.deg2rad(torch.tensor(220)), steps=n_frames)
        elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(0))])
        x_offset = 0.0
        y_offset = 0.0
        z_offset = -0.7
        look_up_point = [0.0, 0.5, -2.0]

    elif dataset_name == "siat_city":
        radius = 6  # Hemisphere radius
        #azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(0)), torch.deg2rad(torch.tensor(360)), steps=n_frames)
        azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(-120)), torch.deg2rad(torch.tensor(-150)), steps=n_frames)
        elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(6))])
        x_offset = 0.0
        y_offset = 0.0
        z_offset = -0.5
        look_up_point = [0.0, 0.0, -3.5]

    ########################
    # height map refinement
    #######################
    elif dataset_name == "family_hmap_refine":
        radius = 4.0
        azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(0)), torch.deg2rad(torch.tensor(360)),
                                        steps=n_frames)
        elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(7))])
        x_offset = 0.0
        y_offset = -1.3
        z_offset = 0.0
        look_up_point = [0.0, -1.3, -0.8]

    elif dataset_name == "truck_hmap_refine":
        radius = 3.8
        azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(0)), torch.deg2rad(torch.tensor(360)),
                                        steps=n_frames)
        elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(10))])
        x_offset = 0.0
        y_offset = 0.0
        z_offset = -1.0
        look_up_point = [0.0, 0.0, -1.3]

    elif dataset_name == "bicycle_hmap_refine":
        radius = 4.5
        # azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(0)), torch.deg2rad(torch.tensor(360)), steps=n_frames)
        azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(40)), torch.deg2rad(torch.tensor(100)),
                                        steps=n_frames)
        elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(15))])
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        look_up_point = [0.0, 1.0, -1.5]

    ########################
    # user study
    ########################
    elif dataset_name in ["garden_ustudy", "garden_igs2gs_ustudy"]:
        if ustudy_id == 0:
            radius = 5.3
            azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(270)), torch.deg2rad(torch.tensor(400)), steps=n_frames)
            elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(0))])
        elif ustudy_id == 1:
            radius = 5.3
            azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(60)), torch.deg2rad(torch.tensor(270)), steps=n_frames)
            elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(0))])
        else:
            raise NotImplementedError
        x_offset = 0.0
        y_offset = 0.0
        z_offset = -0.7
        look_up_point = [0.0, 0.5, -2.0]

    elif dataset_name == "treehill_ustudy":
        if ustudy_id == 0:
            radius = 4
            azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(60)), torch.deg2rad(torch.tensor(150)), steps=n_frames)
            elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(15))])
        elif ustudy_id == 1:
            radius = 4
            azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(150)), torch.deg2rad(torch.tensor(420)), steps=n_frames)
            elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(15))])
        else:
            raise NotImplementedError
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        look_up_point = [0.0, 0.0, -1.0]

    elif dataset_name == "bicycle_ustudy":
        if ustudy_id == 0:
            radius = 4.5
            azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(-100)), torch.deg2rad(torch.tensor(100)), steps=n_frames)
            elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(15))])
        elif ustudy_id == 1:
            radius = 4.5
            azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(100)), torch.deg2rad(torch.tensor(260)), steps=n_frames)
            elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(15))])
        else:
            raise NotImplementedError
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        look_up_point = [0.0, 1.0, -1.5]

    elif dataset_name == "family_ustudy":
        if ustudy_id == 0:
            radius = 4.0
            azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(-20)), torch.deg2rad(torch.tensor(80)), steps=n_frames)
            elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(7))])
        elif ustudy_id == 1:
            radius = 4.0
            azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(80)), torch.deg2rad(torch.tensor(340)), steps=n_frames)
            elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(7))])
        else:
            raise NotImplementedError
        x_offset = 0.0
        y_offset = -1.3
        z_offset = 0.0
        look_up_point = [0.0, -1.3, -0.8]

    elif dataset_name == "truck_ustudy":
        if ustudy_id == 0:
            radius = 3.8
            azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(130)), torch.deg2rad(torch.tensor(280)), steps=n_frames)
            elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(10))])
        elif ustudy_id == 1:
            radius = 3.8
            azimuth_angles = torch.linspace(torch.deg2rad(torch.tensor(280)), torch.deg2rad(torch.tensor(490)), steps=n_frames)
            elevation_angles = torch.tensor([torch.deg2rad(torch.tensor(10))])
        else:
            raise NotImplementedError
        x_offset = 0.0
        y_offset = 0.0
        z_offset = -1.0
        look_up_point = [0.0, 0.0, -1.3]



    else:
        raise NotImplementedError

    # Some trajectory presets cover only a preferred demo sector
    # (Garden uses 260--400 degrees). Diagnostics may need the opposite side
    # of an off-centre sand deposit. A render schedule may override only
    # the azimuth sweep while retaining the
    # dataset-specific radius, elevation, offset and look-at target above.
    if azimuth_range_degrees is not None:
        if (
            not isinstance(azimuth_range_degrees, (list, tuple))
            or len(azimuth_range_degrees) != 2
        ):
            raise ValueError("azimuth_range_degrees must contain [start, end]")
        start_degrees = float(azimuth_range_degrees[0])
        end_degrees = float(azimuth_range_degrees[1])
        if (
            not np.isfinite(start_degrees)
            or not np.isfinite(end_degrees)
            or end_degrees <= start_degrees
        ):
            raise ValueError(
                "azimuth_range_degrees must contain finite increasing values"
            )
        azimuth_angles = torch.linspace(
            torch.deg2rad(torch.tensor(start_degrees)),
            torch.deg2rad(torch.tensor(end_degrees)),
            steps=n_frames,
        )

    # Create a grid of azimuth and elevation angles
    azimuth_grid, elevation_grid = torch.meshgrid(azimuth_angles, elevation_angles, indexing="ij")
    azimuth_grid = azimuth_grid.flatten()
    elevation_grid = elevation_grid.flatten()

    # Compute the camera positions in Cartesian coordinates
    x = radius * torch.cos(elevation_grid) * torch.cos(azimuth_grid) + x_offset
    y = radius * torch.cos(elevation_grid) * torch.sin(azimuth_grid) + y_offset
    z = radius * torch.sin(elevation_grid) + z_offset
    camera_positions = torch.stack([x, y, z], dim=-1)  # Shape: [N, 3]

    # Define the "look-at" center (sphere center is at origin)
    look_at = torch.tensor(look_up_point, device=camera_positions.device)  # torch.zeros_like(camera_positions)

    # Define the world up direction (for hemisphere, use z-up convention)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=camera_positions.device)

    # Compute camera rotation matrices (using a simple look-at mechanism)
    forward = (look_at - camera_positions)  # Look-at direction
    forward = -forward / forward.norm(dim=-1, keepdim=True)  # Normalize
    right = torch.cross(world_up.expand_as(forward), forward)  # Compute right vector
    right = right / right.norm(dim=-1, keepdim=True)  # Normalize
    up = torch.cross(forward, right)  # Compute up vector (orthogonal to forward and right)

    # Assemble rotation matrices
    rotation_matrices = torch.stack([right, up, forward], dim=-1)  # Shape: [N, 3, 3]

    # Combine rotation and translation into [N, 3, 4]
    poses = torch.cat([rotation_matrices, camera_positions.unsqueeze(-1)], dim=-1)  # Shape: [N, 3, 4]

    return poses


def generate_cam_path(
    viewpoint_cameras,
    dataset_name,
    n_frames=480,
    ustudy_id=-1,
    azimuth_range_degrees=None,
):
    c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_cameras])
    pose = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
    pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

    # generate new poses
    # new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames)   # [n_frames,3,4]
    new_poses = generate_camera_poses_on_hemisphere(
        dataset_name,
        n_frames,
        ustudy_id,
        azimuth_range_degrees=azimuth_range_degrees,
    )
    #print(new_poses.shape)

    # warp back to orignal scale
    new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

    traj = []
    for c2w in new_poses:
        c2w = c2w @ np.diag([1, -1, -1, 1]) # (4,4)
        cam = copy.deepcopy(viewpoint_cameras[0])
        cam.image_height = int(cam.image_height / 2) * 2
        cam.image_width = int(cam.image_width / 2) * 2
        cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
        traj.append(cam)

    return traj
##
