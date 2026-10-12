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
from pathlib import Path
from scene import Scene, EnvLight
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer.sand_water_renderer import render_sand_water
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from sandygs.render_schedule import expand_render_schedule
from sandygs.regression_capture import (
    RegressionCaptureError,
    save_regression_raw_arrays,
    validate_regression_output_keys,
)
from sandygs.storage_guard import validate_artifact_output
from sandygs.materials import resolve_material_root
from sandygs.configuration import ConfigError, validate_current_runtime_config
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import yaml
import time

REPO_ROOT = Path(__file__).resolve().parents[1]


from sandygs.diagnostics.render_outputs import (
    save_render_intermediates,
    save_sand_debug_maps,
)


def render_set_360(model_path, name, iteration, views, scene, gaussians, pipeline, background,
               app_model=None, is_render_height_map=False, envmap=None, args=None):
    render_path = os.path.join(model_path, name, "renders")
    makedirs(render_path, exist_ok=True)

    num_frame_per_view = 1
    rendering_time_sum = 0
    simulation_time_sum = 0
    num_gs_sum = 0
    video_frame_id = 0
    for _, view in enumerate(tqdm(views, desc="Rendering progress")):

        for _ in range(num_frame_per_view):
            t0 = time.time()
            # Camera/video frames map to physical solver time. The adapter
            # interpolates through metrics.csv; snapshot numbers are not time.
            target_time = (
                args.simulation_start_time
                + video_frame_id / args.simulation_render_fps
            )
            gaussians.update_dynamic_layers(time_seconds=target_time)
            t1 = time.time()
            simulation_time_sum += (t1 - t0)

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t0 = time.time()
            render_pkg = render_sand_water(view, gaussians, pipeline, background, app_model=app_model, envmap=envmap,
                               is_render_height_map=is_render_height_map, args=args, cur_frame=video_frame_id+1)
            t1 = time.time()
            rendering_time_sum += (t1-t0)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            num_gausssian_all = render_pkg["num_gausssian_all"]
            num_gs_sum += num_gausssian_all
            print("######################", t1-t0, num_gausssian_all)

            rendering = render_pkg["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(video_frame_id) + ".png"))
            save_sand_debug_maps(
                render_pkg, model_path, name, video_frame_id
            )

            rendered_src = render_pkg["rendered_src_save"]
            if rendered_src != None:
                render_src_path = os.path.join(model_path, name, "src")
                makedirs(render_src_path, exist_ok=True)
                cv2.imwrite(os.path.join(render_src_path, '{0:05d}'.format(video_frame_id) + ".png"), rendered_src.permute(1,2,0)[:,:,[2,1,0]].cpu().numpy() * 255)


            rendered_src_refl = render_pkg["rendered_image_src_refl_save"]
            if rendered_src_refl != None:
                render_src_refl_path = os.path.join(model_path, name, "src_refl")
                makedirs(render_src_refl_path, exist_ok=True)
                cv2.imwrite(os.path.join(render_src_refl_path, '{0:05d}'.format(video_frame_id) + ".png"), rendered_src_refl.permute(1,2,0)[:,:,[2,1,0]].cpu().numpy() * 255)


            rendered_src_refl_refra = render_pkg["rendered_image_src_refl_refra_save"]
            if rendered_src_refl_refra != None:
                render_src_refl_refra_path = os.path.join(model_path, name, "src_refl_refra")
                makedirs(render_src_refl_refra_path, exist_ok=True)
                cv2.imwrite(os.path.join(render_src_refl_refra_path, '{0:05d}'.format(video_frame_id) + ".png"),rendered_src_refl_refra.permute(1, 2, 0)[:, :, [2, 1, 0]].cpu().numpy() * 255)


            video_frame_id += 1

    print("simulation_time_sum: ", simulation_time_sum / len(views))
    print("rendering_time_sum: ", rendering_time_sum / len(views))
    print("num_gs_sum: ", num_gs_sum / len(views))


def render_scheduled(
    model_path,
    name,
    camera_views,
    schedule_frames,
    gaussians,
    pipeline,
    background,
    app_model=None,
    is_render_height_map=False,
    envmap=None,
    args=None,
):
    """Render an explicit physical-time/camera table.

    Every frame owns a physical solver time and a camera index. This avoids the
    old accidental coupling where advancing to another camera also advanced
    simulation time, and permits synchronized multi-view validation.
    """

    schedule_root = os.path.join(model_path, name)
    makedirs(schedule_root, exist_ok=True)
    with open(
        os.path.join(schedule_root, "render_schedule.yaml"),
        "w",
        encoding="utf-8",
    ) as stream:
        yaml.safe_dump(
            {
                "schema_version": 1,
                "frames": [frame.to_mapping() for frame in schedule_frames],
            },
            stream,
            sort_keys=False,
        )

    rendering_seconds = 0.0
    simulation_seconds = 0.0
    gaussian_total = 0
    sand_gaussian_total = 0
    peak_sand_gaussian_count = 0
    pre_subdivision_triangle_total = 0
    peak_pre_subdivision_triangle_count = 0
    last_time = None
    # Reset after the static PGSR scene is resident so both A/B branches report
    # the peak allocation of the complete scheduled-render workload.  These
    # additive metrics make the fourfold T2 sand-primitive cost auditable even
    # when the much larger immutable source layer dominates the total count.
    torch.cuda.reset_peak_memory_stats()
    for global_index, frame in enumerate(
        tqdm(schedule_frames, desc="Scheduled rendering")
    ):
        view = camera_views[frame.camera_index]
        if last_time is None or not np.isclose(
            frame.time_seconds, last_time, rtol=0.0, atol=1.0e-12
        ):
            start = time.time()
            gaussians.update_dynamic_layers(
                time_seconds=frame.time_seconds,
            )
            simulation_seconds += time.time() - start
            last_time = frame.time_seconds

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start = time.time()
        render_pkg = render_sand_water(
            view,
            gaussians,
            pipeline,
            background,
            app_model=app_model,
            envmap=envmap,
            is_render_height_map=is_render_height_map,
            args=args,
            cur_frame=global_index + 1,
        )
        torch.cuda.synchronize()
        rendering_seconds += time.time() - start
        gaussian_total += int(render_pkg["num_gausssian_all"])
        sand_gaussian_count = int(getattr(gaussians, "num_sand_gaussian", 0))
        sand_gaussian_total += sand_gaussian_count
        peak_sand_gaussian_count = max(
            peak_sand_gaussian_count, sand_gaussian_count
        )
        sample_audit = getattr(gaussians, "last_sand_water_sample", {})
        pre_subdivision_count = int(
            sample_audit.get("pre_subdivision_sand_triangles", 0)
        )
        pre_subdivision_triangle_total += pre_subdivision_count
        peak_pre_subdivision_triangle_count = max(
            peak_pre_subdivision_triangle_count, pre_subdivision_count
        )

        group = os.path.join(name, frame.output_group)
        render_path = os.path.join(model_path, group, "renders")
        makedirs(render_path, exist_ok=True)
        torchvision.utils.save_image(
            render_pkg["render"].clamp(0.0, 1.0),
            os.path.join(render_path, f"{frame.output_name}.png"),
        )
        if frame.save_debug:
            save_sand_debug_maps(
                render_pkg,
                model_path,
                group,
                frame.output_name,
                output_keys=getattr(args, "sand_debug_output_keys", None),
            )
        if frame.save_intermediates:
            save_render_intermediates(
                render_pkg, model_path, group, frame.output_name
            )
        regression_keys = getattr(args, "regression_raw_output_keys", None)
        if regression_keys is not None:
            save_regression_raw_arrays(
                render_pkg,
                Path(model_path) / group,
                frame.output_name,
                regression_keys,
            )

        # Scheduled videos can change the dynamic Gaussian count by millions
        # of primitives over time.  Drop the previous frame's full-resolution
        # RGB/G-buffer dictionary before constructing the next dynamic state;
        # otherwise those live tensors overlap with the next frame's geometry
        # even though empty_cache() is called at the top of the loop.
        del render_pkg
        torch.cuda.empty_cache()

    frame_count = len(schedule_frames)
    metrics = {
        "schema_version": 1,
        "frame_count": frame_count,
        "unique_physical_times": len(
            {frame.time_seconds for frame in schedule_frames}
        ),
        "simulation_update_seconds": simulation_seconds,
        "render_seconds": rendering_seconds,
        "mean_render_seconds": rendering_seconds / frame_count,
        "mean_gaussian_count": gaussian_total / frame_count,
        "mean_sand_gaussian_count": sand_gaussian_total / frame_count,
        "peak_sand_gaussian_count": peak_sand_gaussian_count,
        "mean_pre_subdivision_triangle_count": (
            pre_subdivision_triangle_total / frame_count
        ),
        "peak_pre_subdivision_triangle_count": (
            peak_pre_subdivision_triangle_count
        ),
        "peak_cuda_memory_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024.0**2)
        ),
        "peak_cuda_memory_reserved_mib": (
            torch.cuda.max_memory_reserved() / (1024.0**2)
        ),
    }
    with open(
        os.path.join(schedule_root, "render_metrics.yaml"),
        "w",
        encoding="utf-8",
    ) as stream:
        yaml.safe_dump(metrics, stream, sort_keys=False)



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        render_output_path = args.render_output_path or dataset.model_path
        gaussians = GaussianModel(
            dataset.sh_degree, args, is_render_sand_water=True
        )
        observed_camera_files = getattr(args, "observed_camera_files", None)
        if observed_camera_files is not None:
            # Existing-sand E4 uses the captured PGSR cameras and a sealed E3
            # checkpoint. Scene() writes camera metadata into model_path and
            # expects an environment file that this derived checkpoint lacks.
            # Load both inputs directly so this render remains read-only.
            from scene.cameras import Camera
            from utils.graphics_utils import focal2fov
            if not isinstance(observed_camera_files, list) or not observed_camera_files:
                raise ValueError("observed_camera_files must be a non-empty list")
            schedule_spec = getattr(args, "render_schedule", None)
            if not isinstance(schedule_spec, dict) or schedule_spec.get("camera_path") != "observed_cameras":
                raise ValueError("observed_camera_files requires an observed_cameras render_schedule")
            schedule_frames, camera_count = expand_render_schedule(schedule_spec)
            if camera_count != len(observed_camera_files):
                raise ValueError("observed camera count differs from render_schedule.camera_path_frames")
            if iteration <= 0:
                raise ValueError("observed camera rendering needs an explicit checkpoint iteration")
            cloud = Path(dataset.model_path) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
            if not cloud.is_file():
                raise FileNotFoundError(f"derived PGSR point cloud is missing: {cloud}")
            gaussians.load_ply(str(cloud))
            environment_file = (Path(dataset.model_path) / "brdf_mlp" /
                                f"iteration_{iteration}" / "overcast_soil_puresky_1k_3.png")
            if environment_file.is_file():
                from scene.NVDIFFREC import load_env
                gaussians.brdf_mlp_fluid = load_env(str(environment_file), scale=1.0)
            else:
                # GaussianModel creates a constant neutral cubemap. Its
                # diffuse/specular mip levels must be prepared before the P4
                # shader can sample it, just as load_env() prepares a file.
                gaussians.brdf_mlp_fluid.build_mips()
                print("E4 observed-camera render: using the initialized neutral environment")
            view_list = []
            seen_ids = set()
            for index, camera_file in enumerate(observed_camera_files):
                camera = load_yaml_config(camera_file)
                camera_id = camera.get("camera_id") if isinstance(camera, dict) else None
                if not isinstance(camera_id, str) or not camera_id or camera_id in seen_ids:
                    raise ValueError(f"invalid or duplicate observed camera ID: {camera_file}")
                seen_ids.add(camera_id)
                w2c = np.asarray(camera["gs_w2c"], dtype=np.float64)
                sim_to_gs = np.asarray(camera["sim_to_gs"], dtype=np.float64)
                if (w2c.shape != (4, 4) or not np.isfinite(w2c).all()
                        or sim_to_gs.shape != (4, 4)
                        or not np.allclose(sim_to_gs, gaussians.sand_water_adapter.sim_to_gs, rtol=0, atol=1e-6)):
                    raise ValueError(f"observed camera pose or simulation transform is invalid: {camera_file}")
                source_w, source_h = camera["source_size_wh"]
                width, height = camera["render_size_wh"]
                intr = camera["colmap_intrinsics"]
                source_image = Path(camera["source_image"])
                if source_image.name != str(source_image):
                    raise ValueError(f"observed camera source image is not a basename: {camera_file}")
                view = Camera(
                    colmap_id=index, R=w2c[:3, :3].T.copy(), T=w2c[:3, 3].copy(),
                    FoVx=focal2fov(float(intr["fx"]), int(source_w)),
                    FoVy=focal2fov(float(intr["fy"]), int(source_h)),
                    image_width=int(width), image_height=int(height),
                    image_path=str(Path(dataset.source_path) / dataset.images / source_image),
                    image_name=camera_id, uid=index, preload_img=False,
                )
                actual_w2c = view.world_view_transform.T.detach().cpu().numpy()
                actual_K = np.array([[view.fx, 0, view.cx], [0, view.fy, view.cy], [0, 0, 1]])
                if (not np.allclose(actual_w2c, w2c, rtol=0, atol=1e-5)
                        or not np.allclose(actual_K, camera["pgsr_K"], rtol=0, atol=1e-4)):
                    raise ValueError(f"observed camera differs from captured PGSR view: {camera_file}")
                view_list.append(view)
            if args.is_sky_model:
                raise ValueError("E4 observed-camera pilot requires is_sky_model=false")
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            render_scheduled(render_output_path, "release_check", view_list,
                             schedule_frames, gaussians, pipeline, background,
                             app_model=None, envmap=None, args=args)
            return
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

        if args.dataset_name in {"waymo181", "waymo362"}:
            raise ValueError(
                "legacy Waymo rain rendering was removed; use the "
                "pre-refactor-reconstruction-v1 tag for historical runs"
            )

        if args.is_render_360:
            print("is_render_360")

            from utils.camera_utils import generate_cam_path

            render_schedule = getattr(args, "render_schedule", None)
            if render_schedule is not None:
                schedule_frames, n_frames = expand_render_schedule(
                    render_schedule
                )
            elif args.dataset_name in ["garden","garden_rebuttal"]:
                n_frames = 180
            # elif args.dataset_name in ["garden","garden_rebuttal"]:
            #     n_frames = 4 #180
            elif args.dataset_name in ["treehill"]:
                n_frames = 200 #400
            elif args.dataset_name in ["family"]:
                n_frames = 345 #400
            else:
                n_frames = 200
            # A schedule may request a wider azimuth sweep for synchronized
            # front/back geometry acceptance.
            camera_path_azimuth_degrees = None
            if render_schedule is not None:
                camera_path_azimuth_degrees = render_schedule.get(
                    "camera_path_azimuth_degrees"
                )
            cam_traj = generate_cam_path(
                scene.getTrainCameras(),
                args.dataset_name,
                n_frames=n_frames,
                azimuth_range_degrees=camera_path_azimuth_degrees,
            )
            if render_schedule is not None:
                render_scheduled(
                    render_output_path,
                    "release_check",
                    cam_traj,
                    schedule_frames,
                    gaussians,
                    pipeline,
                    background,
                    app_model=app_model,
                    envmap=env_map,
                    args=args,
                )
                return
            selected_cam = cam_traj
            if args.max_views is not None:
                selected_cam = selected_cam[:args.max_views]

            # selected_cam = scene.getTrainCameras()[:1]
            render_set_360(render_output_path, "release_check", scene.loaded_iter, selected_cam, scene, gaussians, pipeline,
                       background, app_model=app_model, envmap=env_map, args=args)

        else:
            raise ValueError("sand-water rendering requires is_render_360=true")


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def update_args_with_yaml(args, yaml_config):
    for key, value in yaml_config.items():
        # A None parser default means "not explicitly supplied on the CLI", so
        # a generated YAML may fill it. Non-None CLI values retain precedence.
        if not hasattr(args, key) or getattr(args, key) is None:
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
    ## qy
    parser.add_argument('--load_depth', action='store_true')
    parser.add_argument('--load_normal', action='store_true')
    parser.add_argument("--config_file", type=str, help="Path to the YAML configuration file")
    parser.add_argument(
        "--render_output_path",
        type=str,
        default=None,
        help="Write rendered artifacts outside the checkpoint directory",
    )
    parser.add_argument(
        "--max_views",
        type=int,
        default=None,
        help="Render only the first N selected views (useful for smoke tests)",
    )
    parser.add_argument(
        "--disable_preload_img",
        action="store_true",
        help="Do not preload every source image (recommended for bounded 360 smoke tests)",
    )
    parser.add_argument(
        "--simulation_backend",
        choices=("simrain", "sand_water"),
        default=None,
        help="Simulation source; only sand_water is supported by current mainline",
    )
    parser.add_argument(
        "--simulation_exchange_path",
        type=str,
        default=None,
        help="P1 exchange bundle used by the sand_water backend",
    )
    parser.add_argument(
        "--sand_water_run_path",
        type=str,
        default=None,
        help="sand-water-mixtures run directory used by the sand_water backend",
    )
    parser.add_argument(
        "--simulation_start_time",
        type=float,
        default=None,
        help="Physical solver time assigned to the first rendered video frame",
    )
    parser.add_argument(
        "--simulation_render_fps",
        type=float,
        default=None,
        help="Video-frame rate used for camera-frame to solver-time mapping",
    )
    parser.add_argument(
        "--free_water_epsilon",
        type=float,
        default=None,
        help="Minimum pure-water thickness that creates a visible surface",
    )
    parser.add_argument(
        "--water_normal_gradient_mode",
        choices=("legacy_full", "free_water_masked"),
        default=None,
        help=(
            "Water height-field normal derivative: preserve the original "
            "full-grid gradient or restrict differences to free-water cells"
        ),
    )
    parser.add_argument(
        "--render_sand",
        action="store_true",
        default=None,
        help="Construct the P3 opaque sand layer for the sand_water backend",
    )
    parser.add_argument(
        "--sand_geometry_backend",
        choices=("cell_tangent", "triangle_games"),
        default=None,
        help=(
            "Dynamic sand geometry representation. The legacy cell_tangent "
            "backend remains the default for old configs."
        ),
    )
    parser.add_argument(
        "--sand_triangle_coverage_gamma",
        type=float,
        default=None,
        help="T1 tangent footprint multiplier for triangle_games",
    )
    parser.add_argument(
        "--sand_triangle_terrain_max_edge_ratio",
        type=float,
        default=None,
        help=(
            "Maximum 3D/horizontal edge ratio used to classify regular "
            "terrain; defaults to sand_max_tangent_scale_ratio"
        ),
    )
    parser.add_argument(
        "--sand_triangle_contact_tolerance",
        type=float,
        default=None,
        help=(
            "Optional high-terrain contact tolerance; by default reuse the "
            "value recorded by the physics solver"
        ),
    )
    parser.add_argument(
        "--sand_triangle_diagonal_mode",
        choices=("checkerboard",),
        default=None,
        help="Fixed base-grid diagonal rule for triangle_games",
    )
    parser.add_argument(
        "--sand_triangle_boundary_mode",
        choices=("clipped", "fully_active"),
        default=None,
        help="Sand/no-sand boundary treatment for triangle_games",
    )
    parser.add_argument(
        "--sand_triangle_subdivision_level",
        type=int,
        default=None,
        choices=(0, 1),
        help="Fixed triangle subdivision level: zero=T1.1, one=uniform T2",
    )
    parser.add_argument(
        "--sand_texture_enabled",
        action="store_true",
        default=None,
        help="Enable the T4.1 per-pixel sand PBR microtexture",
    )
    parser.add_argument(
        "--sand_texture_diagnostics",
        action="store_true",
        default=None,
        help="Save T4.1 UV, LOD and sampled material diagnostics",
    )
    parser.add_argument(
        "--sand_texture_material",
        type=str,
        default=None,
        help="Auditable label for the selected sand material",
    )
    parser.add_argument(
        "--sand_texture_material_mode",
        choices=("perturbation", "texture_primary"),
        default=None,
        help=(
            "Sand texture composition rule: retain the authored dry/wet base "
            "or use sampled maps as the dry-sand endpoint"
        ),
    )
    parser.add_argument(
        "--sand_texture_root",
        type=str,
        default=None,
        help="Directory containing albedo.jpg, roughness.jpg and normal_gl.png",
    )
    parser.add_argument(
        "--sand_texture_projection",
        choices=("simulation_xy", "simulation_triplanar"),
        default=None,
        help="Coordinate projection used by the sand texture",
    )
    parser.add_argument(
        "--sand_texture_triplanar_sharpness", type=float, default=None
    )
    parser.add_argument(
        "--sand_texture_phase_mode",
        choices=("world_locked",),
        default=None,
        help="Temporal phase rule for the first T4.1 implementation",
    )
    parser.add_argument("--sand_texture_period_sim", type=float, default=None)
    parser.add_argument(
        "--sand_texture_rotation_degrees", type=float, default=None
    )
    parser.add_argument(
        "--sand_texture_offset", type=float, nargs=2, default=None
    )
    parser.add_argument(
        "--sand_texture_albedo_strength", type=float, default=None
    )
    parser.add_argument(
        "--sand_texture_roughness_strength", type=float, default=None
    )
    parser.add_argument(
        "--sand_texture_normal_strength", type=float, default=None
    )
    parser.add_argument(
        "--sand_texture_normal_highpass_enabled",
        action="store_true",
        default=None,
        help=(
            "Remove low-frequency tangent-normal slopes before building the "
            "material mip pyramid"
        ),
    )
    parser.add_argument(
        "--sand_texture_normal_highpass_sigma_texels", type=float, default=None
    )
    parser.add_argument(
        "--sand_texture_normal_highpass_strength", type=float, default=None
    )
    parser.add_argument(
        "--sand_texture_ao_enabled",
        action="store_true",
        default=None,
        help=(
            "Apply dark high-frequency ambient occlusion only to indirect "
            "sand diffuse lighting"
        ),
    )
    parser.add_argument(
        "--sand_texture_ao_highpass_sigma_texels", type=float, default=None
    )
    parser.add_argument(
        "--sand_texture_ao_response",
        choices=("linear", "sparse_smoothstep"),
        default=None,
        help=(
            "Map the dark AO residual with the T4.1i linear response or an "
            "opt-in sparse smoothstep response"
        ),
    )
    parser.add_argument("--sand_texture_ao_strength", type=float, default=None)
    parser.add_argument("--sand_texture_ao_min", type=float, default=None)
    parser.add_argument("--sand_texture_ao_sparse_low", type=float, default=None)
    parser.add_argument("--sand_texture_ao_sparse_high", type=float, default=None)
    parser.add_argument(
        "--sand_texture_ao_sparse_contrast", type=float, default=None
    )
    parser.add_argument("--sand_texture_lod_bias", type=float, default=None)
    parser.add_argument(
        "--sand_texture_albedo_detail_min", type=float, default=None
    )
    parser.add_argument(
        "--sand_texture_albedo_detail_max", type=float, default=None
    )
    parser.add_argument(
        "--sand_texture_roughness_min", type=float, default=None
    )
    parser.add_argument(
        "--sand_texture_roughness_max", type=float, default=None
    )
    parser.add_argument(
        "--sand_texture_primary_albedo_gain",
        type=float,
        nargs=3,
        default=None,
        help="Linear-RGB gain for texture_primary Albedo",
    )
    parser.add_argument(
        "--sand_texture_primary_roughness_scale", type=float, default=None
    )
    parser.add_argument(
        "--sand_texture_wet_adjustment_enabled",
        action="store_true",
        default=None,
        help=(
            "Apply the texture-primary pore-water Albedo brightness response; "
            "Roughness, Normal and specular remain unchanged"
        ),
    )
    parser.add_argument(
        "--sand_texture_wet_albedo_scale",
        type=float,
        default=None,
        help="Fully wet linear-RGB Albedo brightness multiplier in [0,1]",
    )
    parser.add_argument(
        "--sand_texture_wet_response_rate",
        type=float,
        default=None,
        help="Positive rate of the normalized saturating wetness response",
    )
    parser.add_argument(
        "--sand_specular_environment_enabled",
        action="store_true",
        default=None,
        help=(
            "Add roughness-prefiltered split-sum environment specular to the "
            "opaque sand shader"
        ),
    )
    parser.add_argument(
        "--sand_specular_environment_intensity", type=float, default=None
    )
    parser.add_argument(
        "--sand_specular_environment_min", type=float, default=None
    )
    parser.add_argument(
        "--sand_specular_environment_max", type=float, default=None
    )
    parser.add_argument(
        "--water_rendering_backend",
        choices=("rainygs_legacy", "sandygs_p4"),
        default=None,
        help=(
            "Water optics implementation; only sandygs_p4 is supported by "
            "current mainline"
        ),
    )

    args = get_combined_args(parser)
    if getattr(args, "max_views", None) is not None and args.max_views <= 0:
        parser.error("--max_views must be greater than zero")
    if getattr(args, "config_file", None):
        yaml_config = load_yaml_config(args.config_file)
        try:
            validate_current_runtime_config(
                yaml_config, source=str(args.config_file)
            )
        except ConfigError as exc:
            parser.error(str(exc))
        update_args_with_yaml(args, yaml_config)
    # get_combined_args drops None-valued command-line fields while merging
    # checkpoint cfg_args. Restore optional runtime controls before downstream
    # code accesses them as normal Namespace attributes.
    args.simulation_backend = (
        getattr(args, "simulation_backend", None) or "sand_water"
    )
    args.render_sand = bool(getattr(args, "render_sand", False))
    # Old P3--P5 profiles predate the selector and must remain pixel-compatible
    # unless they explicitly opt into the T1 triangle representation.
    args.sand_geometry_backend = (
        getattr(args, "sand_geometry_backend", None) or "cell_tangent"
    )
    args.sand_triangle_coverage_gamma = (
        1.0
        if getattr(args, "sand_triangle_coverage_gamma", None) is None
        else float(args.sand_triangle_coverage_gamma)
    )
    args.sand_triangle_terrain_max_edge_ratio = (
        None
        if getattr(args, "sand_triangle_terrain_max_edge_ratio", None) is None
        else float(args.sand_triangle_terrain_max_edge_ratio)
    )
    args.sand_triangle_contact_tolerance = (
        None
        if getattr(args, "sand_triangle_contact_tolerance", None) is None
        else float(args.sand_triangle_contact_tolerance)
    )
    args.sand_triangle_diagonal_mode = (
        getattr(args, "sand_triangle_diagonal_mode", None) or "checkerboard"
    )
    args.sand_triangle_boundary_mode = (
        getattr(args, "sand_triangle_boundary_mode", None) or "clipped"
    )
    args.sand_triangle_subdivision_level = (
        0
        if getattr(args, "sand_triangle_subdivision_level", None) is None
        else int(args.sand_triangle_subdivision_level)
    )
    args.sand_texture_enabled = bool(getattr(
        args, "sand_texture_enabled", False
    ))
    args.sand_texture_diagnostics = bool(getattr(
        args, "sand_texture_diagnostics", False
    ))
    args.sand_texture_normal_highpass_enabled = bool(getattr(
        args, "sand_texture_normal_highpass_enabled", False
    ))
    args.sand_texture_ao_enabled = bool(getattr(
        args, "sand_texture_ao_enabled", False
    ))
    args.sand_texture_ao_response = (
        getattr(args, "sand_texture_ao_response", None) or "linear"
    )
    args.sand_specular_environment_enabled = bool(getattr(
        args, "sand_specular_environment_enabled", False
    ))
    args.sand_texture_material = (
        getattr(args, "sand_texture_material", None) or "unnamed"
    )
    # Preserve every existing T4.1 render unless a profile explicitly opts
    # into the T4.1e texture-primary dry endpoint.
    args.sand_texture_material_mode = (
        getattr(args, "sand_texture_material_mode", None) or "perturbation"
    )
    args.sand_texture_projection = (
        getattr(args, "sand_texture_projection", None) or "simulation_xy"
    )
    args.sand_texture_phase_mode = (
        getattr(args, "sand_texture_phase_mode", None) or "world_locked"
    )
    texture_defaults = {
        "sand_texture_period_sim": 0.10,
        "sand_texture_triplanar_sharpness": 4.0,
        "sand_texture_rotation_degrees": 0.0,
        "sand_texture_albedo_strength": 0.35,
        "sand_texture_roughness_strength": 0.20,
        # Texture-primary treats the normal map as a material endpoint, while
        # the compatibility perturbation path keeps its former half strength.
        "sand_texture_normal_strength": (
            1.0
            if args.sand_texture_material_mode == "texture_primary"
            else 0.50
        ),
        "sand_texture_normal_highpass_sigma_texels": 8.0,
        "sand_texture_normal_highpass_strength": 1.0,
        "sand_texture_ao_highpass_sigma_texels": 8.0,
        "sand_texture_ao_strength": 1.0,
        "sand_texture_ao_min": 0.4,
        "sand_texture_ao_sparse_low": 0.04,
        "sand_texture_ao_sparse_high": 0.08,
        "sand_texture_ao_sparse_contrast": 0.40,
        "sand_texture_lod_bias": 0.0,
        "sand_texture_albedo_detail_min": 0.50,
        "sand_texture_albedo_detail_max": 1.50,
        "sand_texture_roughness_min": 0.08,
        "sand_texture_roughness_max": 1.0,
        "sand_texture_wet_albedo_scale": 0.65,
        "sand_texture_wet_response_rate": 4.0,
        "sand_specular_environment_intensity": 1.0,
        "sand_specular_environment_min": 0.0,
        "sand_specular_environment_max": 4.0,
    }
    for texture_key, texture_default in texture_defaults.items():
        texture_value = getattr(args, texture_key, None)
        setattr(
            args,
            texture_key,
            texture_default if texture_value is None else float(texture_value),
        )
    args.sand_texture_offset = (
        [0.0, 0.0]
        if getattr(args, "sand_texture_offset", None) is None
        else [float(value) for value in args.sand_texture_offset]
    )
    args.sand_texture_primary_albedo_gain = (
        [1.0, 1.0, 1.0]
        if getattr(args, "sand_texture_primary_albedo_gain", None) is None
        else [
            float(value) for value in args.sand_texture_primary_albedo_gain
        ]
    )
    args.sand_texture_primary_roughness_scale = (
        1.0
        if getattr(args, "sand_texture_primary_roughness_scale", None) is None
        else float(args.sand_texture_primary_roughness_scale)
    )
    args.sand_texture_wet_adjustment_enabled = bool(getattr(
        args, "sand_texture_wet_adjustment_enabled", False
    ))
    args.water_normal_gradient_mode = (
        getattr(args, "water_normal_gradient_mode", None) or "legacy_full"
    )
    args.water_depth_bias = (
        0.002
        if getattr(args, "water_depth_bias", None) is None
        else float(args.water_depth_bias)
    )
    args.water_terrain_depth_proxy = bool(getattr(
        args, "water_terrain_depth_proxy", False
    ))
    args.water_terrain_proxy_source_tolerance = (
        0.10
        if getattr(args, "water_terrain_proxy_source_tolerance", None) is None
        else float(args.water_terrain_proxy_source_tolerance)
    )
    args.water_terrain_proxy_depth_bias = (
        0.0
        if getattr(args, "water_terrain_proxy_depth_bias", None) is None
        else float(args.water_terrain_proxy_depth_bias)
    )
    args.water_physical_reference_depth_recovery = bool(getattr(
        args, "water_physical_reference_depth_recovery", False
    ))
    args.water_physical_reference_erosion_pixels = (
        2
        if getattr(
            args, "water_physical_reference_erosion_pixels", None
        ) is None
        else args.water_physical_reference_erosion_pixels
    )
    args.water_physical_reference_gaussian_coverage_min = (
        0.5
        if getattr(
            args,
            "water_physical_reference_gaussian_coverage_min",
            None,
        ) is None
        else float(args.water_physical_reference_gaussian_coverage_min)
    )
    args.water_rendering_backend = (
        getattr(args, "water_rendering_backend", None) or "sandygs_p4"
    )
    try:
        validate_current_runtime_config(vars(args), source="render arguments")
    except ConfigError as exc:
        parser.error(str(exc))
    if args.sand_geometry_backend not in {"cell_tangent", "triangle_games"}:
        parser.error(
            "sand_geometry_backend must be cell_tangent or triangle_games"
        )
    if not np.isfinite(args.water_depth_bias) or args.water_depth_bias < 0.0:
        parser.error("water_depth_bias must be finite and non-negative")
    if (
        not np.isfinite(args.water_terrain_proxy_source_tolerance)
        or args.water_terrain_proxy_source_tolerance <= 0.0
    ):
        parser.error(
            "water_terrain_proxy_source_tolerance must be finite and positive"
        )
    if (
        not np.isfinite(args.water_terrain_proxy_depth_bias)
        or args.water_terrain_proxy_depth_bias < 0.0
    ):
        parser.error(
            "water_terrain_proxy_depth_bias must be finite and non-negative"
        )
    if (
        isinstance(args.water_physical_reference_erosion_pixels, bool)
        or not isinstance(args.water_physical_reference_erosion_pixels, int)
        or args.water_physical_reference_erosion_pixels < 0
    ):
        parser.error(
            "water_physical_reference_erosion_pixels must be a "
            "non-negative integer"
        )
    if (
        not np.isfinite(args.water_physical_reference_gaussian_coverage_min)
        or not 0.0
        <= args.water_physical_reference_gaussian_coverage_min
        <= 1.0
    ):
        parser.error(
            "water_physical_reference_gaussian_coverage_min must be in [0,1]"
        )
    if args.sand_triangle_coverage_gamma <= 0.0:
        parser.error("sand_triangle_coverage_gamma must be positive")
    if (
        args.sand_triangle_terrain_max_edge_ratio is not None
        and args.sand_triangle_terrain_max_edge_ratio < 1.0
    ):
        parser.error("sand_triangle_terrain_max_edge_ratio must be at least one")
    if (
        args.sand_triangle_contact_tolerance is not None
        and args.sand_triangle_contact_tolerance < 0.0
    ):
        parser.error("sand_triangle_contact_tolerance must be non-negative")
    if args.sand_triangle_subdivision_level not in {0, 1}:
        parser.error("sand_triangle_subdivision_level must be zero or one")
    if args.sand_texture_projection not in {
        "simulation_xy",
        "simulation_triplanar",
    }:
        parser.error(
            "sand_texture_projection must be simulation_xy or "
            "simulation_triplanar"
        )
    if args.sand_texture_phase_mode != "world_locked":
        parser.error("sand_texture_phase_mode must be world_locked")
    if args.sand_texture_material_mode not in {
        "perturbation",
        "texture_primary",
    }:
        parser.error(
            "sand_texture_material_mode must be perturbation or texture_primary"
        )
    if (
        not np.isfinite(args.sand_texture_period_sim)
        or args.sand_texture_period_sim <= 0.0
    ):
        parser.error("sand_texture_period_sim must be finite and positive")
    if (
        not np.isfinite(args.sand_texture_triplanar_sharpness)
        or args.sand_texture_triplanar_sharpness <= 0.0
    ):
        parser.error(
            "sand_texture_triplanar_sharpness must be finite and positive"
        )
    if not np.isfinite(args.sand_texture_rotation_degrees):
        parser.error("sand_texture_rotation_degrees must be finite")
    if len(args.sand_texture_offset) != 2 or not np.isfinite(
        args.sand_texture_offset
    ).all():
        parser.error("sand_texture_offset must contain two finite values")
    if not 0.0 <= args.sand_texture_albedo_strength <= 1.0:
        parser.error("sand_texture_albedo_strength must lie in [0,1]")
    if not 0.0 <= args.sand_texture_roughness_strength <= 1.0:
        parser.error("sand_texture_roughness_strength must lie in [0,1]")
    if (
        not np.isfinite(args.sand_texture_normal_strength)
        or args.sand_texture_normal_strength < 0.0
    ):
        parser.error("sand_texture_normal_strength must be finite and non-negative")
    if (
        not np.isfinite(args.sand_texture_normal_highpass_sigma_texels)
        or args.sand_texture_normal_highpass_sigma_texels <= 0.0
    ):
        parser.error(
            "sand_texture_normal_highpass_sigma_texels must be finite and positive"
        )
    if (
        not np.isfinite(args.sand_texture_normal_highpass_strength)
        or args.sand_texture_normal_highpass_strength < 0.0
    ):
        parser.error(
            "sand_texture_normal_highpass_strength must be finite and non-negative"
        )
    if (
        not np.isfinite(args.sand_texture_ao_highpass_sigma_texels)
        or args.sand_texture_ao_highpass_sigma_texels <= 0.0
    ):
        parser.error(
            "sand_texture_ao_highpass_sigma_texels must be finite and positive"
        )
    if args.sand_texture_ao_response not in {"linear", "sparse_smoothstep"}:
        parser.error(
            "sand_texture_ao_response must be linear or sparse_smoothstep"
        )
    if (
        not np.isfinite(args.sand_texture_ao_strength)
        or args.sand_texture_ao_strength < 0.0
    ):
        parser.error("sand_texture_ao_strength must be finite and non-negative")
    if not np.isfinite(args.sand_texture_ao_min) or not (
        0.0 <= args.sand_texture_ao_min <= 1.0
    ):
        parser.error("sand_texture_ao_min must lie in [0,1]")
    if not (
        np.isfinite(args.sand_texture_ao_sparse_low)
        and np.isfinite(args.sand_texture_ao_sparse_high)
        and 0.0
        <= args.sand_texture_ao_sparse_low
        < args.sand_texture_ao_sparse_high
    ):
        parser.error(
            "sand AO sparse thresholds must satisfy finite 0 <= low < high"
        )
    if not np.isfinite(args.sand_texture_ao_sparse_contrast) or not (
        0.0 <= args.sand_texture_ao_sparse_contrast <= 1.0
    ):
        parser.error("sand_texture_ao_sparse_contrast must lie in [0,1]")
    if not np.isfinite(args.sand_texture_lod_bias):
        parser.error("sand_texture_lod_bias must be finite")
    if (
        not np.isfinite(args.sand_specular_environment_intensity)
        or args.sand_specular_environment_intensity < 0.0
    ):
        parser.error(
            "sand_specular_environment_intensity must be finite and non-negative"
        )
    if not (
        np.isfinite(args.sand_specular_environment_min)
        and np.isfinite(args.sand_specular_environment_max)
        and 0.0 <= args.sand_specular_environment_min
        < args.sand_specular_environment_max
    ):
        parser.error(
            "sand specular environment clamp must satisfy finite 0 <= min < max"
        )
    if not (
        0.0 < args.sand_texture_albedo_detail_min
        <= args.sand_texture_albedo_detail_max
    ):
        parser.error("sand texture albedo detail bounds must be positive and ordered")
    if not (
        0.0 <= args.sand_texture_roughness_min
        <= args.sand_texture_roughness_max
        <= 1.0
    ):
        parser.error("sand texture roughness bounds must lie in [0,1]")
    if (
        len(args.sand_texture_primary_albedo_gain) != 3
        or not np.isfinite(args.sand_texture_primary_albedo_gain).all()
        or np.any(np.asarray(args.sand_texture_primary_albedo_gain) < 0.0)
    ):
        parser.error(
            "sand_texture_primary_albedo_gain must contain three finite "
            "non-negative values"
        )
    if (
        not np.isfinite(args.sand_texture_primary_roughness_scale)
        or args.sand_texture_primary_roughness_scale <= 0.0
    ):
        parser.error(
            "sand_texture_primary_roughness_scale must be finite and positive"
        )
    if not (
        np.isfinite(args.sand_texture_wet_albedo_scale)
        and 0.0 <= args.sand_texture_wet_albedo_scale <= 1.0
    ):
        parser.error(
            "sand_texture_wet_albedo_scale must be finite and lie in [0,1]"
        )
    if (
        not np.isfinite(args.sand_texture_wet_response_rate)
        or args.sand_texture_wet_response_rate <= 0.0
    ):
        parser.error(
            "sand_texture_wet_response_rate must be finite and positive"
        )
    if (
        args.sand_texture_wet_adjustment_enabled
        and args.sand_texture_material_mode != "texture_primary"
    ):
        parser.error(
            "sand_texture_wet_adjustment_enabled requires "
            "sand_texture_material_mode=texture_primary"
        )
    if args.sand_texture_material_mode == "texture_primary":
        dry_specular = float(getattr(args, "sand_dry_specular", 0.04))
        if not np.isfinite(dry_specular) or not 0.0 <= dry_specular <= 1.0:
            parser.error("sand_dry_specular must lie in [0,1]")
    if args.sand_texture_enabled:
        if not args.render_sand:
            parser.error("sand_texture_enabled requires render_sand=true")
        if not getattr(args, "sand_texture_root", None):
            parser.error("sand_texture_enabled requires sand_texture_root")
        try:
            args.sand_texture_root = str(resolve_material_root(
                args.sand_texture_root
            ))
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
    if args.sand_texture_diagnostics:
        if not args.sand_texture_enabled:
            parser.error("sand_texture_diagnostics requires sand_texture_enabled")
        if not bool(getattr(args, "sand_save_debug_maps", False)):
            parser.error(
                "sand_texture_diagnostics requires sand_save_debug_maps=true"
            )
    if bool(getattr(
        args, "sand_center_trusted_invalid_normal_recovery", False
    )) and not bool(getattr(
        args, "sand_center_trusted_invalid_depth_recovery", False
    )):
        parser.error(
            "sand_center_trusted_invalid_normal_recovery requires "
            "sand_center_trusted_invalid_depth_recovery=true"
        )
    if bool(getattr(
        args, "sand_physical_reference_normal_recovery", False
    )) and not bool(getattr(
        args, "sand_physical_reference_depth_recovery", False
    )):
        parser.error(
            "sand_physical_reference_normal_recovery requires "
            "sand_physical_reference_depth_recovery=true"
        )
    if bool(getattr(
        args, "sand_center_trusted_invalid_attribute_diagnostics", False
    )):
        if not bool(getattr(
            args, "sand_center_trusted_invalid_depth_recovery", False
        )):
            parser.error(
                "sand_center_trusted_invalid_attribute_diagnostics requires "
                "sand_center_trusted_invalid_depth_recovery=true"
            )
        if not args.sand_texture_enabled:
            parser.error(
                "sand_center_trusted_invalid_attribute_diagnostics requires "
                "sand_texture_enabled=true"
            )
        if args.sand_texture_material_mode != "texture_primary":
            parser.error(
                "sand_center_trusted_invalid_attribute_diagnostics requires "
                "sand_texture_material_mode=texture_primary"
            )
        if args.sand_texture_projection != "simulation_triplanar":
            parser.error(
                "sand_center_trusted_invalid_attribute_diagnostics requires "
                "sand_texture_projection=simulation_triplanar"
            )
        if not bool(getattr(args, "sand_save_debug_maps", False)):
            parser.error(
                "sand_center_trusted_invalid_attribute_diagnostics requires "
                "sand_save_debug_maps=true"
            )
    if bool(getattr(
        args, "sand_physical_reference_attribute_diagnostics", False
    )):
        if bool(getattr(
            args, "sand_center_trusted_invalid_attribute_diagnostics", False
        )):
            parser.error(
                "physical and center recovery attribute diagnostics are "
                "mutually exclusive"
            )
        if not bool(getattr(
            args, "sand_physical_reference_depth_recovery", False
        )):
            parser.error(
                "sand_physical_reference_attribute_diagnostics requires "
                "sand_physical_reference_depth_recovery=true"
            )
        if not args.sand_texture_enabled:
            parser.error(
                "sand_physical_reference_attribute_diagnostics requires "
                "sand_texture_enabled=true"
            )
        if args.sand_texture_material_mode != "texture_primary":
            parser.error(
                "sand_physical_reference_attribute_diagnostics requires "
                "sand_texture_material_mode=texture_primary"
            )
        if args.sand_texture_projection != "simulation_triplanar":
            parser.error(
                "sand_physical_reference_attribute_diagnostics requires "
                "sand_texture_projection=simulation_triplanar"
            )
        if not bool(getattr(args, "sand_save_debug_maps", False)):
            parser.error(
                "sand_physical_reference_attribute_diagnostics requires "
                "sand_save_debug_maps=true"
            )
    render_schedule = getattr(args, "render_schedule", None)
    regression_raw_output_keys = getattr(
        args, "regression_raw_output_keys", None
    )
    if regression_raw_output_keys is not None:
        if render_schedule is None:
            parser.error(
                "regression_raw_output_keys requires an explicit render_schedule"
            )
        try:
            validate_regression_output_keys(regression_raw_output_keys)
        except RegressionCaptureError as exc:
            parser.error(str(exc))
    if render_schedule is not None:
        if args.simulation_backend != "sand_water":
            parser.error("render_schedule is only supported by sand_water")
        if not bool(getattr(args, "is_render_360", False)):
            parser.error("render_schedule requires is_render_360=true")
        try:
            # Validate the declarative table before loading the heavy PGSR
            # scene so server jobs fail fast on camera/time schema mistakes.
            expand_render_schedule(render_schedule)
        except ValueError as exc:
            parser.error(str(exc))
    if not args.simulation_exchange_path or not args.sand_water_run_path:
        parser.error(
            "sand_water requires simulation_exchange_path and sand_water_run_path"
        )
    if getattr(args, "simulation_start_time", None) is None:
        args.simulation_start_time = 0.0
    if getattr(args, "simulation_render_fps", None) is None:
        args.simulation_render_fps = 30.0
    if args.simulation_render_fps <= 0.0:
        parser.error("simulation_render_fps must be positive")
    if args.free_water_epsilon is not None and args.free_water_epsilon < 0.0:
        parser.error("free_water_epsilon must be non-negative")
    if getattr(args, "disable_preload_img", False):
        args.preload_img = False
    ##

    # Direct re-render commands bypass the P2--P5 drivers. Apply the same
    # machine-local volume guard whenever their explicit destination belongs
    # to the configured artifact root, before loading the PGSR checkpoint.
    if args.render_output_path is not None:
        validate_artifact_output(
            REPO_ROOT / "configs/paths.local.yaml",
            Path(args.render_output_path),
        )

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)
