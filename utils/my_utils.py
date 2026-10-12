import numpy as np
import plyfile
import torch
from utils.graphics_utils import depth_pcd2normal
import torch.nn.functional as F
import math

########################################
#
# save
#
########################################
def save_tensor_point_cloud_as_ply(tensor, filename):
    # Ensure the input tensor has shape [N, 3]
    assert tensor.ndim == 2 and tensor.shape[1] == 3, "Input tensor must have shape [N, 3]"

    # Convert the PyTorch tensor to a numpy array
    points = tensor.cpu().numpy()

    # Define the elements of the PLY file
    vertex = np.array(
        [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )

    # Create a list of elements for the PLY file
    element = plyfile.PlyElement.describe(vertex, 'vertex')

    # Save as a PLY file
    plyfile.PlyData([element], text=True).write(filename)
    print(f"Point cloud saved as {filename}")


########################################
#
# coordinate system transformation
#
########################################
## calculate end 3d position in cam sys
'''
def transform_cam2pixel(P_cam, K):
    """
        Project 3D points in camera coordinates to pixel coordinates using camera intrinsic matrix.

        Args:
            K (torch.Tensor): Camera intrinsic matrix of shape [3, 3].
            P_cam (torch.Tensor): 3D points in camera coordinates of shape [N, 3].

        Returns:
            torch.Tensor: Pixel coordinates of shape [N, 2].
        """
    # Extract intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Decompose camera coordinates
    Xc = P_cam[:, 0]
    Yc = P_cam[:, 1]
    Zc = P_cam[:, 2]
    # Prevent division by zero
    epsilon = 1e-6
    Zc = Zc + epsilon
    # Compute normalized image coordinates
    x = Xc / Zc
    y = Yc / Zc
    # Apply intrinsic parameters to get pixel coordinates
    u = fx * x + cx
    v = fy * y + cy
    # Stack and return pixel coordinates
    pixels = torch.stack((u, v), dim=1).round() #.ceil() #.floor() #round()

    return pixels
'''


def transform_cam2pixel(camera_coords_batch, intrinsic_matrix):
    """
    使用PyTorch批量将相机坐标系中的点投影到成像平面并获取像素坐标
    :param camera_coords_batch: 批量相机坐标系中的点，形状为 (batch_size, 3)，每一行是一个 (X_c, Y_c, Z_c)
    :param intrinsic_matrix: 相机的内参矩阵 K，形状为 (3, 3)
    :return: 像素坐标 (u, v)，形状为 (batch_size, 2)
    """

    # 处理背后点，将 Z_c < 0 的深度取反
    Z_c = camera_coords_batch[:, 2]
    camera_coords_batch[:, 2] = torch.abs(Z_c)  # 对 Z_c < 0 的点取绝对值

    # 归一化相机坐标 (X_c / Z_c, Y_c / Z_c)
    normalized_coords_batch = camera_coords_batch[:, :2] / camera_coords_batch[:, 2:3]  # 保持形状 (batch_size, 2)

    # 添加 1 到归一化的坐标，得到 [X_c / Z_c, Y_c / Z_c, 1]
    ones = torch.ones(camera_coords_batch.shape[0], 1, dtype=torch.float32, device=camera_coords_batch.device)  # 形状 (batch_size, 1)
    normalized_coords_batch = torch.cat([normalized_coords_batch, ones], dim=1)  # 形状 (batch_size, 3)

    # 批量投影到成像平面
    pixel_coords_batch = torch.matmul(normalized_coords_batch, intrinsic_matrix.T)  # 形状 (batch_size, 3)

    # 提取像素坐标 u 和 v
    u_v_coords = pixel_coords_batch[:, :2]  # 取前两列，形状 (batch_size, 2)

    return u_v_coords


'''
## transform to 2d screen
def transform_ray_direction_cam2pixel(rays_camera, K):
    """
    将相机坐标系下的光线方向投影到屏幕空间
    参数:
    rays_camera: torch.Tensor, shape [N,3] - 相机坐标系下的光线方向
    K: torch.Tensor, shape [3,3] - 相机内参矩阵
    返回:
    screen_coords: torch.Tensor, shape [N,2] - 屏幕空间坐标
    """

    print(K)

    # # 1. 将方向向量归一化到z=1平面
    # rays_normalized = rays_camera / rays_camera[..., 2:3]  # [N,3]
    # # 2. 应用内参矩阵投影
    # screen_coords = torch.matmul(rays_normalized, K.transpose(-2, -1))  # [N,3]
    # # 3. 只保留x,y坐标
    # screen_coords = screen_coords[..., :2]  # [N,2]

    pixel_coords = rays_camera @ K.T  # 使用矩阵乘法，输出为 [N, 3]
    screen_coords = pixel_coords[:, :2] / pixel_coords[:, 2:3] # 将齐次坐标归一化，得到屏幕坐标


    return screen_coords
'''

def depth_to_xyz_map(depth_map, H, W, K):
    u_coords = torch.arange(0, W, dtype=torch.float32, device=depth_map.device)
    v_coords = torch.arange(0, H, dtype=torch.float32, device=depth_map.device)
    u, v = torch.meshgrid(u_coords, v_coords, indexing='xy')

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) / fx
    y = (v - cy) / fy
    z = depth_map

    x = x * z
    y = y * z

    # Create mask for valid depth values
    nonzero_depth_mask = depth_map > 0
    x = x * nonzero_depth_mask
    y = y * nonzero_depth_mask
    z = z * nonzero_depth_mask

    xyz_map = torch.stack((x, y, z), dim=-1)    # [h,w,3]
    #point_cloud = point3d.view(-1, 3)          # [N, 3]

    return xyz_map, nonzero_depth_mask


def transform_c2w(point_cloud, W2C):
    # 计算C2W矩阵
    C2W = torch.inverse(W2C)
    # 转换点云为齐次坐标
    ones = torch.ones(point_cloud.shape[0], 1, dtype=point_cloud.dtype, device=point_cloud.device)
    point_cloud_homogeneous = torch.cat([point_cloud, ones], dim=1)  # [N, 4]
    # 应用C2W变换
    point_cloud_world = (C2W @ point_cloud_homogeneous.T).T  # [N, 4]
    # 去掉齐次分量
    point_cloud_world = point_cloud_world[:, :3]  # [N, 3]

    return point_cloud_world


def transform_w2c(point_cloud, W2C):
    #point_cloud: [N,3]
    # 转换点云为齐次坐标
    ones = torch.ones(point_cloud.shape[0], 1, dtype=point_cloud.dtype, device=point_cloud.device)
    point_cloud_homogeneous = torch.cat([point_cloud, ones], dim=1)  # [N, 4]
    # 应用W2C变换
    point_cloud_cam = (W2C @ point_cloud_homogeneous.T).T  # [N, 4]
    # 去掉齐次分量
    point_cloud_cam = point_cloud_cam[:, :3]  # [N, 3]

    return point_cloud_cam


def transform_ray_direction_w2c(ray_directions_world, W2C):
    # 去掉平移部分，只保留旋转部分
    R = W2C[:3, :3]  # 提取3x3旋转矩阵
    # 将光线方向从世界坐标系转换到相机坐标系
    ray_directions_camera = (R @ ray_directions_world.T).T  # [N, 3]

    return ray_directions_camera


def rotation_matrix_x(theta):
    return torch.tensor([[1, 0, 0],
                         [0, torch.cos(theta), -torch.sin(theta)],
                         [0, torch.sin(theta), torch.cos(theta)]], dtype=torch.float32, device="cuda")


def rotation_matrix_y(theta):
    return torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                         [0, 1, 0],
                         [-torch.sin(theta), 0, torch.cos(theta)]], dtype=torch.float32, device="cuda")


def rotation_matrix_z(theta):
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0],
                         [0, 0, 1]], dtype=torch.float32, device="cuda")


def transform_point_cloud(point_cloud, rotation_angles, translation_vector):
    # 将旋转角度从度数转换为弧度
    rx, ry, rz = [torch.deg2rad(torch.tensor(angle, dtype=torch.float32, device="cuda")) for angle in rotation_angles]

    # 分别计算每个轴的旋转矩阵
    rot_x = rotation_matrix_x(rx)
    rot_y = rotation_matrix_y(ry)
    rot_z = rotation_matrix_z(rz)

    # 平移向量转换为 PyTorch 张量并取负值以实现逆平移
    translation_vector = torch.tensor([-x for x in translation_vector], dtype=torch.float32, device="cuda")
    point_cloud = point_cloud + translation_vector  # 平移点云

    # 组合旋转矩阵并计算逆矩阵
    rotation_matrix = rot_z @ rot_y @ rot_x
    rotation_matrix = torch.linalg.inv(rotation_matrix)

    # 旋转点云
    transformed_points = (rotation_matrix @ point_cloud.T).T  # 旋转点云

    return transformed_points


def transform_point_cloud_2(point_cloud, rotation_angles, translation_vector):
    # 将旋转角度从度数转换为弧度
    rx, ry, rz = [torch.deg2rad(torch.tensor(angle, dtype=torch.float32, device="cuda")) for angle in rotation_angles]
    # 分别计算每个轴的旋转矩阵
    rot_x = rotation_matrix_x(rx)
    rot_y = rotation_matrix_y(ry)
    rot_z = rotation_matrix_z(rz)
    # 组合旋转矩阵并计算逆矩阵
    rotation_matrix = rot_z @ rot_y @ rot_x
    # # 旋转点云
    # transformed_points = point_cloud @ rotation_matrix.T

    def homogeneous_transform(rotation_matrix, translation):
        # 构造 4x4 齐次矩阵
        transform = torch.eye(4, device="cuda")
        transform[:3, :3] = rotation_matrix  # 放入旋转矩阵
        transform[:3, 3] = translation  # 放入平移向量
        return transform

    # 计算齐次变换矩阵
    translation = torch.tensor(translation_vector, device="cuda")
    transform_matrix = homogeneous_transform(rotation_matrix, translation)

    # 将点云 A 扩展为齐次坐标（每个点增加一个维度，最后一列为 1）
    points_A_homogeneous = torch.cat([point_cloud, torch.ones(point_cloud.size(0), 1, device="cuda")], dim=1)  # [N, 4]

    # 应用齐次变换矩阵
    points_B_homogeneous = points_A_homogeneous @ transform_matrix.T

    # 提取转换后的点云（去掉齐次坐标最后一列）
    transformed_points = points_B_homogeneous[:, :3]

    return transformed_points




def euler_xyz_to_rotation_matrix(euler_deg):
    """
    将Blender的 'XYZ Euler' (rx, ry, rz) 转为 4x4 旋转矩阵（右乘列向量）。
    euler_deg: (rx_deg, ry_deg, rz_deg)，单位为度。
    """
    # 转为弧度
    rx_deg, ry_deg, rz_deg = euler_deg
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    # 分别构造绕X/Y/Z轴的旋转(4x4)
    Rx = torch.tensor([
        [1, 0, 0, 0],
        [0, math.cos(rx), -math.sin(rx), 0],
        [0, math.sin(rx), math.cos(rx), 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device="cuda")

    Ry = torch.tensor([
        [math.cos(ry), 0, math.sin(ry), 0],
        [0, 1, 0, 0],
        [-math.sin(ry), 0, math.cos(ry), 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device="cuda")

    Rz = torch.tensor([
        [math.cos(rz), -math.sin(rz), 0, 0],
        [math.sin(rz), math.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32, device="cuda")

    # 对于 “XYZ Euler”，表示先X，再Y，再Z，
    # 以矩阵右乘向量的写法，对应：R = Rz @ Ry @ Rx
    # （若你平时习惯左乘行向量，可自行调整顺序）
    R = Rz @ Ry @ Rx
    return R

def make_transform_matrix(location, euler_deg, scale):
    """
    构造 4x4 变换矩阵 M = T * R * S，用于右乘列向量。

    location: (tx, ty, tz)
    euler_deg: (rx_deg, ry_deg, rz_deg) in degrees
    scale: (sx, sy, sz)
    """
    # 平移矩阵 T
    T = torch.eye(4, dtype=torch.float32, device="cuda")
    T[0, 3] = location[0]
    T[1, 3] = location[1]
    T[2, 3] = location[2]

    # 旋转矩阵 R
    R = euler_xyz_to_rotation_matrix(euler_deg)

    # 缩放矩阵 S
    S = torch.eye(4, dtype=torch.float32, device="cuda")
    S[0, 0] = scale[0]
    S[1, 1] = scale[1]
    S[2, 2] = scale[2]

    # 最终 M = T * R * S
    M = T @ R @ S
    return M

def invert_transform(location, euler_deg, scale):
    """
    返回逆变换矩阵 M^{-1} = (T * R * S)^{-1}。
    """
    M = make_transform_matrix(location, euler_deg, scale)
    M_inv = torch.inverse(M)
    return M_inv


def transform_point_cloud_3(points, euler_deg, location, scale):
    """
    对一批3D点 (N,3) 应用 4x4 矩阵变换，约定 points 作为列向量右乘M。
    """
    # # 1. 原变换矩阵
    # M = make_transform_matrix(location, euler_deg, scale)
    # print("Transform Matrix M:\n", M)

    # 2. 逆变换矩阵
    M_inv = invert_transform(location, euler_deg, scale)
    #print("Inverse Transform Matrix M_inv:\n", M_inv)

    # 先将 (N,3) 扩展为 (N,4)，最后一列是1
    N = points.shape[0]
    ones = torch.ones((N, 1), dtype=torch.float32, device="cuda")
    points_h = torch.cat([points, ones], dim=1)  # (N,4)

    # 右乘 M，需要先转置再转置回来，或者直接点乘也可
    # 这里演示一种常见做法：把 points_h 视作 (N,4)，需要 M (4x4) 对每一行进行右乘：
    # points_out = points_h @ M.T  (如果习惯行向量在左)
    # 但为了和上面“右乘列向量”的思路对应，可以写成：
    points_out_h = points_h @ M_inv.t()  # (N,4)

    # 最终只取前三维
    return points_out_h[:, :3]



def transform_w_to_w2(c2w, rotation_angles, translation_vector):
    # 将旋转角度从度数转换为弧度
    rx, ry, rz = [torch.deg2rad(torch.tensor(angle, dtype=torch.float32, device="cuda")) for angle in rotation_angles]
    # 分别计算每个轴的旋转矩阵
    rot_x = rotation_matrix_x(rx)
    rot_y = rotation_matrix_y(ry)
    rot_z = rotation_matrix_z(rz)
    # 组合旋转矩阵并计算逆矩阵
    rotation_matrix = rot_z @ rot_y @ rot_x
    # # 旋转点云
    # transformed_points = point_cloud @ rotation_matrix.T

    def homogeneous_transform(rotation_matrix, translation):
        # 构造 4x4 齐次矩阵
        transform = torch.eye(4, device="cuda")
        transform[:3, :3] = rotation_matrix  # 放入旋转矩阵
        transform[:3, 3] = translation  # 放入平移向量
        return transform

    # 计算齐次变换矩阵
    translation = torch.tensor(translation_vector, device="cuda")
    transform_matrix = homogeneous_transform(rotation_matrix, translation)

    transform_matrix = torch.linalg.inv(transform_matrix)

    # # 将点云 A 扩展为齐次坐标（每个点增加一个维度，最后一列为 1）
    # points_A_homogeneous = torch.cat([point_cloud, torch.ones(point_cloud.size(0), 1, device="cuda")], dim=1)  # [N, 4]

    # 应用齐次变换矩阵
    new_c2w = transform_matrix @ c2w

    # # 提取转换后的点云（去掉齐次坐标最后一列）
    # transformed_points = points_B_homogeneous[:, :3]

    return new_c2w

########################################
#
# utils
#
########################################
def compute_normal(height_map, rotation_angles, height_map_xmin, height_map_xmax, height_map_ymin, height_map_ymax):
        # Convert height map to xyz coordinates
        height, width = height_map.shape[:2]

        x = torch.linspace(height_map_xmin, height_map_xmax, width).cuda()
        y = torch.linspace(height_map_ymin, height_map_ymax, height).cuda()
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        xyz_image = torch.zeros((3, height, width), dtype=torch.float32).cuda()
        xyz_image[0, :, :] = xx
        xyz_image[1, :, :] = yy
        xyz_image[2, :, :] = height_map #torch.tensor(height_map, dtype=torch.float32).cuda()
        xyz_t = xyz_image.permute(1, 2, 0).view(-1, 3)  # Reshape to [h*w, 3]

        # Transform xyz to normal coordinates in world space
        xyz_world = xyz_t
        xyz_world = xyz_world.view(height, width, 3)#torch.tensor(xyz_world, dtype=torch.float32).view(height, width, 3)

        # Compute normals from depth
        xyz_normal = depth_pcd2normal(xyz_world)  # Assuming depth_pcd2normal is a function that computes normals

        # Correct normals with [0, 0, 0] values
        mask = (xyz_normal == torch.tensor([0.0, 0.0, 0.0]).cuda()).all(dim=-1)
        xyz_normal[mask] = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32).cuda()

        # Invert normals for consistency todo:debug
        xyz_normal = -xyz_normal

        # Apply rotation transformation to normals
        normal = transform_point_cloud(xyz_normal.view(-1, 3), rotation_angles, [0.0, 0.0, 0.0])
        normal = normal.view(height, width, 3)
        #normal = normal.reshape(height, width, 3).astype(np.float32)

        # Save normal map for debugging
        # normal_save = (normal + 1) * 127.5
        # normal_save = normal_save.clip(0, 255).astype(np.uint8)
        # normal_bgr = cv2.cvtColor(normal_save, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("normal_map.png", normal_bgr)

        return normal

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    # 创建高斯核
    x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    x = -0.5 * (x ** 2) / (sigma ** 2)
    kernel_1d = torch.exp(x)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d

def apply_gaussian_smoothing(image: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    # image: 输入的 [3, h, w] 大小的图像 tensor
    kernel = gaussian_kernel(kernel_size, sigma).cuda()
    kernel = kernel.expand(3, 1, kernel_size, kernel_size)  # 扩展到 [3, 1, k, k] 大小
    padding = kernel_size // 2  # 填充大小，以保持图像尺寸不变

    # 对每个通道分别进行卷积
    smoothed_image = F.conv2d(image.unsqueeze(0), kernel, padding=padding, groups=3)
    return smoothed_image.squeeze(0)


def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    调整亮度和对比度，模拟PS的效果。

    image: 输入图像张量，形状为 [3, h, w]，值范围为 [0, 1]
    brightness: 亮度调节参数 (-150 到 150)，默认0为中性值
    contrast: 对比度调节参数 (-50 到 100)，默认0为中性值
    """
    # 1. 亮度调整 (brightness ∈ [-150, 150])
    brightness_offset = brightness / 255.0  # 转换到 [0, 1] 的范围
    image = image + brightness_offset

    # 2. 对比度调整 (contrast ∈ [-50, 100])
    contrast_factor = 1 + (contrast / 100.0)  # 转换对比度增益 (1 为原始对比度)
    mean_gray = torch.mean(image, dim=[1, 2], keepdim=True)  # 计算每通道灰度均值
    image = (image - mean_gray) * contrast_factor + mean_gray

    # 3. 限制输出范围到 [0, 1]
    image = torch.clamp(image, 0, 1)

    return image

########################################
#
# math
#
########################################
def _logit(y):
    return torch.log(y / (1 - y))

def _log(y):
    return torch.log(y)

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)
