"""Read-only PGSR observation capture for a small, explicit camera selection.

GPU imports and work happen only inside ``capture_source_views``. In particular
this module never constructs ``Scene``: its constructor and dataset reader write
files into the source checkpoint and dataset directories.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from .geometry import (
    backproject_camera_z,
    depth_products,
    project_world,
    validate_pinhole,
)
from .workspace import (
    ExistingSandError, _load_yaml, _safe_relative, _stage_location,
    record_stage, verify_run,
)


def _selected_colmap_cameras(dataset: Path, requested: list[str],
                             dataset_name: str) -> list[dict[str, Any]]:
    """Match COLMAP views and reproduce PGSR's optional pose normalization."""
    from scene.colmap_loader import (  # GPU packages stay out of CPU geometry tests
        qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary,
    )
    from scene.dataset_readers import transform_poses_pca, rotate_poses_and_transform

    sparse = dataset / "sparse/0"
    extrinsics = read_extrinsics_binary(str(sparse / "images.bin"))
    intrinsics = read_intrinsics_binary(str(sparse / "cameras.bin"))
    images = sorted(extrinsics.values(), key=lambda image: Path(image.name).stem)
    if not images:
        raise ExistingSandError("COLMAP images.bin contains no cameras")
    w2cs = []
    for image in images:
        rotation = qvec2rotmat(image.qvec)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = rotation
        w2c[:3, 3] = image.tvec
        w2cs.append(w2c)
    w2cs = np.stack(w2cs)
    mode = "colmap_original"
    transform = np.eye(4, dtype=np.float64)
    if dataset_name not in {"garden", "garden_rebuttal"}:
        poses, transform, _ = transform_poses_pca(
            np.linalg.inv(w2cs)[:, :3, :4], dataset_name=dataset_name
        )
        if dataset_name == "waymo362":
            poses, transform = rotate_poses_and_transform(poses, transform, 187.5)
        padded = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
        padded[:, :3, :4] = poses
        w2cs = np.linalg.inv(padded)
        mode = "pgsr_pca" if dataset_name != "waymo362" else "pgsr_pca_waymo362"

    matches: dict[str, list[int]] = {camera_id: [] for camera_id in requested}
    for index, image in enumerate(images):
        basename = Path(image.name).name
        stem = Path(image.name).stem
        for camera_id in requested:
            if camera_id in {basename, stem}:
                matches[camera_id].append(index)
    selected = []
    used_indices = set()
    for camera_id in requested:
        indices = matches[camera_id]
        if len(indices) != 1:
            raise ExistingSandError(
                f"camera_id {camera_id!r} matches {len(indices)} COLMAP images; "
                "use an unambiguous basename"
            )
        index = indices[0]
        if index in used_indices:
            raise ExistingSandError("multiple camera_ids refer to the same COLMAP image")
        used_indices.add(index)
        image = images[index]
        intr = intrinsics[image.camera_id]
        if intr.model == "SIMPLE_PINHOLE":
            fx = fy = float(intr.params[0])
            cx, cy = (float(value) for value in intr.params[1:3])
        elif intr.model == "PINHOLE":
            fx, fy, cx, cy = (float(value) for value in intr.params[:4])
        else:
            raise ExistingSandError(
                f"unsupported COLMAP camera model {intr.model} for {camera_id}"
            )
        selected.append({
            "camera_id": camera_id,
            "image_name": Path(image.name).name,
            "image_width": int(intr.width),
            "image_height": int(intr.height),
            "colmap_intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "w2c": w2cs[index],
            "pose_mode": mode,
            "colmap_to_gs": transform,
        })
    return selected


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)


def capture_source_views(run_root: Path, capture_id: str, camera_ids: list[str],
                         *, dataset_name: str, width: int, height: int,
                         sh_degree: int, alpha_min: float,
                         white_background: bool, repo_root: Path,
                         images_dir: str = "images",
                         command: list[str] | None = None) -> dict[str, Any]:
    """Export unclipped PGSR RGB, camera-Z depth, alpha, and source photos.

    This is a short, selected-camera GPU operation. No source file is written.
    The output is registered as an ``observations`` stage for later segmentation.
    """
    run_root = Path(run_root).resolve()
    repo_root = Path(repo_root).resolve()
    if not camera_ids or any(not isinstance(item, str) or not item
                             for item in camera_ids) or len(set(camera_ids)) != len(camera_ids):
        raise ExistingSandError("provide distinct non-empty --camera-id values")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ExistingSandError("dataset_name must be the PGSR training dataset_name")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (width, height)):
        raise ExistingSandError("width and height must be positive integers")
    if isinstance(sh_degree, bool) or not isinstance(sh_degree, int) or sh_degree < 0 or sh_degree > 4:
        raise ExistingSandError("sh_degree must be between 0 and 4")
    if not np.isfinite(alpha_min) or not 0 < alpha_min <= 1:
        raise ExistingSandError("alpha_min must be in (0, 1]")
    verify_run(run_root)
    run = _load_yaml(run_root / "run.yaml", "run manifest")
    if any(camera_id not in run["camera_ids"] for camera_id in camera_ids):
        raise ExistingSandError("capture camera_id must be listed in run.yaml")
    source = run["source"]
    dataset = Path(source["dataset"])
    images_relative, images_root = _safe_relative(dataset, images_dir, "images_dir")
    if not images_root.is_dir():
        raise ExistingSandError(f"missing images directory: {images_root}")
    checkpoint = Path(source["checkpoint"])
    declared = {item["path"] for item in source["dataset_files"]}
    required = {"sparse/0/cameras.bin", "sparse/0/images.bin"}
    if not required.issubset(declared):
        raise ExistingSandError(
            "run source.dataset_files must include sparse/0/cameras.bin and images.bin"
        )
    selected = _selected_colmap_cameras(dataset, camera_ids, dataset_name)
    for camera in selected:
        image_relative = images_relative + "/" + camera["image_name"]
        if image_relative not in declared:
            raise ExistingSandError(
                f"source.dataset_files must include selected image: {image_relative}"
            )
        if not (dataset / image_relative).is_file():
            raise ExistingSandError(f"missing source image: {dataset / image_relative}")
    stage_dir = _stage_location(run_root, "observations", capture_id)
    if stage_dir.exists():
        raise ExistingSandError(f"refusing to overwrite capture: {stage_dir}")

    import torch
    from PIL import Image
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import focal2fov

    if not torch.cuda.is_available():
        raise ExistingSandError("PGSR source capture requires CUDA")
    point_cloud = checkpoint / source["point_cloud"]
    gaussians = GaussianModel(sh_degree)
    try:
        gaussians.load_ply(str(point_cloud))
    except AssertionError as exc:
        raise ExistingSandError(
            f"source PLY SH coefficients do not match --sh-degree {sh_degree}"
        ) from exc
    if gaussians.get_xyz.shape[0] != source["gaussian_count"]:
        raise ExistingSandError("loaded Gaussian count differs from run manifest")
    pipeline = SimpleNamespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False
    )
    background = torch.ones(3, device="cuda") if white_background else torch.zeros(
        3, device="cuda"
    )
    stage_dir.mkdir(parents=True, exist_ok=False)
    artifacts = []
    summaries = []
    for index, camera in enumerate(selected):
        camera_id = camera["camera_id"]
        view_dir = f"view_{index:04d}"
        intr = camera["colmap_intrinsics"]
        fovx = focal2fov(intr["fx"], camera["image_width"])
        fovy = focal2fov(intr["fy"], camera["image_height"])
        view = Camera(
            colmap_id=0, R=camera["w2c"][:3, :3].T.copy(),
            T=camera["w2c"][:3, 3].copy(), FoVx=fovx, FoVy=fovy,
            image_width=width, image_height=height,
            image_path=str(images_root / camera["image_name"]),
            image_name=camera_id, uid=0, preload_img=False,
        )
        K = np.array([[view.fx, 0, view.cx], [0, view.fy, view.cy], [0, 0, 1]],
                     dtype=np.float64)
        actual_w2c = view.world_view_transform.T.detach().cpu().numpy().astype(np.float64)
        validate_pinhole(K, actual_w2c)
        with torch.no_grad():
            result = render(view, gaussians, pipeline, background,
                            return_depth_normal=False, envmap=None)
        rgb = result["render"].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        depth_raw = result["plane_depth"].detach().cpu().numpy().squeeze(0)
        alpha = result["rendered_alpha"].detach().cpu().numpy().squeeze(0).astype(np.float32)
        if rgb.shape != (height, width, 3) or not np.isfinite(rgb).all():
            raise ExistingSandError(f"invalid PGSR RGB for {camera_id}")
        depth, valid, inverse = depth_products(depth_raw, alpha, alpha_min=alpha_min)
        if not valid.any():
            raise ExistingSandError(f"no valid PGSR camera-Z depth for {camera_id}")
        sample_index = np.flatnonzero(valid.ravel())[:1024]
        if sample_index.size:
            sampled_depth = np.full(depth.shape, np.nan, dtype=np.float64)
            sampled_depth.ravel()[sample_index] = depth.ravel()[sample_index]
            sampled_valid = np.isfinite(sampled_depth)
            points = backproject_camera_z(sampled_depth, K, actual_w2c, sampled_valid)
            pixels, recovered_z = project_world(points, K, actual_w2c)
            expected_v, expected_u = np.unravel_index(sample_index, depth.shape)
            residual = pixels.reshape(-1, 2)[sample_index] - np.stack(
                [expected_u, expected_v], axis=-1
            )
            max_pixel_error = float(np.max(np.abs(residual)))
            max_z_error = float(np.max(np.abs(recovered_z.ravel()[sample_index] -
                                              depth.ravel()[sample_index])))
            if max_pixel_error > 1e-4 or max_z_error > 1e-4:
                raise ExistingSandError(f"camera-Z reprojection failed for {camera_id}")
        else:
            max_pixel_error = None
            max_z_error = None
        directory = stage_dir / view_dir
        directory.mkdir()
        with Image.open(images_root / camera["image_name"]) as image:
            if image.size != (camera["image_width"], camera["image_height"]):
                raise ExistingSandError(f"COLMAP and image size differ for {camera_id}")
            source_image = image.convert("RGB").resize(
                (width, height), resample=Image.BILINEAR
            )
        source_rgb = np.asarray(source_image, dtype=np.float32) / 255.0
        arrays = {
            "source_rgb.npy": (source_rgb, "source_rgb"),
            "render_rgb.npy": (rgb, "pgsr_render_rgb_raw"),
            "plane_depth.npy": (depth, "pgsr_camera_z_raw"),
            "rendered_alpha.npy": (alpha, "pgsr_rendered_alpha"),
            "depth_valid.npy": (valid, "numeric_depth_valid"),
            "inverse_depth.npy": (inverse, "inverse_camera_z_valid_only"),
        }
        for filename, (array, role) in arrays.items():
            np.save(directory / filename, array, allow_pickle=False)
            artifacts.append({"path": f"{view_dir}/{filename}", "role": role})
        source_image.save(directory / "source_rgb.png")
        artifacts.append({"path": f"{view_dir}/source_rgb.png", "role": "source_rgb_preview"})
        Image.fromarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(
            directory / "render_rgb.png"
        )
        artifacts.append({"path": f"{view_dir}/render_rgb.png", "role": "render_rgb_preview"})
        camera_payload = {
            "schema_version": 1,
            "camera_id": camera_id,
            "projection": "pinhole",
            "pixel_centers": "integer_u_right_v_down",
            "depth": "positive_camera_z_in_gs_units",
            "source_image": camera["image_name"],
            "source_size_wh": [camera["image_width"], camera["image_height"]],
            "render_size_wh": [width, height],
            "colmap_intrinsics": intr,
            "pgsr_K": K.tolist(),
            "gs_w2c": actual_w2c.tolist(),
            "gs_c2w": np.linalg.inv(actual_w2c).tolist(),
            "sim_to_gs": run["coordinate"]["sim_to_gs"],
            "gs_to_sim": run["coordinate"]["gs_to_sim"],
            "pose_mode": camera["pose_mode"],
            "colmap_to_gs": camera["colmap_to_gs"].tolist(),
        }
        _write_yaml(directory / "camera.yaml", camera_payload)
        artifacts.append({"path": f"{view_dir}/camera.yaml", "role": "camera"})
        summaries.append({
            "camera_id": camera_id,
            "directory": view_dir,
            "valid_pixels": int(valid.sum()),
            "total_pixels": int(valid.size),
            "reprojection_max_pixel_error": max_pixel_error,
            "reprojection_max_z_error": max_z_error,
            "valid_depth_min": float(depth[valid].min()) if valid.any() else None,
            "valid_depth_max": float(depth[valid].max()) if valid.any() else None,
        })
        del result, view
        torch.cuda.empty_cache()
    _write_yaml(stage_dir / "quality.yaml", {"cameras": summaries})
    artifacts.append({"path": "quality.yaml", "role": "capture_quality"})
    record = {
        "schema_version": 1,
        "stage": "observations", "stage_id": capture_id,
        "camera_ids": camera_ids,
        "prompt": {}, "seed": None,
        "models": [{"name": "PGSR source Gaussian",
                    "version": "point_cloud_sha256:" + source["point_cloud_sha256"],
                    "checkpoint": str(point_cloud)}],
        "parameters": {
            "dataset_name": dataset_name, "images_dir": images_relative,
            "width": width, "height": height,
            "sh_degree": sh_degree, "alpha_min": alpha_min,
            "white_background": white_background,
            "rendered_rgb_uses_app_model": False,
            "rendered_rgb_uses_envmap": False,
        },
        "commands": [command or [sys.executable, *sys.argv]],
        "command_cwd": str(Path.cwd()),
        "parents": [],
        "artifacts": artifacts,
        "code_files": [
            "sandygs/existing_sand/geometry.py",
            "sandygs/existing_sand/capture.py",
            "scripts/experiments/existing_sand_capture.py",
            "scene/cameras.py", "scene/dataset_readers.py",
            "scene/colmap_loader.py", "scene/gaussian_model.py",
            "gaussian_renderer/__init__.py",
        ],
    }
    record_path = stage_dir / "record.yaml"
    _write_yaml(record_path, record)
    record_stage(run_root, record_path, repo_root=repo_root)
    return {"stage_dir": str(stage_dir), "cameras": summaries}
