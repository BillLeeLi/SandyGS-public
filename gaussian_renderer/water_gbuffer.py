"""Transparent free-water G-buffer construction for SandyGS rendering."""

from dataclasses import dataclass
import math
from typing import Any, Dict

import torch
from diff_plane_rasterization import (
    GaussianRasterizationSettings as PlaneGaussianRasterizationSettings,
)
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer

from sandygs.rendering import (
    camera_normals_to_world,
    classify_water_gbuffer_stages,
    fallback_invalid_plane_depth,
    opaque_surface_coverage,
    rasterize_triangle_reference,
    valid_visible_coverage,
    validate_surface_gbuffer,
)


@dataclass(frozen=True)
class WaterGBuffer:
    """Geometry, visibility and diagnostics consumed by water optics."""

    depth: torch.Tensor
    plane_depth_valid: torch.Tensor
    depth_fallback_used: torch.Tensor
    depth_fallback_rejected_by_opaque: torch.Tensor
    surface_valid: torch.Tensor
    visible_coverage: torch.Tensor
    invalid_visible_candidate: torch.Tensor
    stage_diagnostics: Dict[str, torch.Tensor]
    validation: Any
    normal_camera: torch.Tensor
    normal_world: torch.Tensor
    thickness: torch.Tensor
    raw_alpha: torch.Tensor
    raw_coverage: torch.Tensor
    depth_in_front: torch.Tensor
    depth_in_front_without_proxy: torch.Tensor
    depth_bias: float
    occlusion_depth: torch.Tensor
    terrain_proxy_depth: torch.Tensor
    terrain_proxy_coverage: torch.Tensor
    terrain_proxy_used: torch.Tensor
    terrain_proxy_rejected_foreground: torch.Tensor
    terrain_proxy_source_delta: torch.Tensor
    occlusion_bias: torch.Tensor
    depth_in_front_without_physical_reference: torch.Tensor
    physical_reference_depth: torch.Tensor
    physical_reference_coverage: torch.Tensor
    physical_reference_trusted: torch.Tensor
    physical_reference_recovered: torch.Tensor
    physical_reference_removed: torch.Tensor
    physical_reference_rejected_opaque: torch.Tensor


def resolve_terrain_depth_proxy(
    *,
    opaque_surface_depth: torch.Tensor,
    opaque_surface_sand_owner: torch.Tensor,
    terrain_depth: torch.Tensor,
    terrain_coverage: torch.Tensor,
    water_valid: torch.Tensor,
    source_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use physical terrain depth only for near-terrain source pixels.

    The absolute source/terrain tolerance preserves clear foreground objects;
    dynamic sand ownership always wins independently of this threshold.
    """

    if not math.isfinite(source_tolerance) or source_tolerance <= 0:
        raise ValueError(
            "water_terrain_proxy_source_tolerance must be finite and positive"
        )
    reference_valid = (
        terrain_coverage.to(torch.bool)
        & torch.isfinite(terrain_depth)
        & (terrain_depth > 0.0)
        & torch.isfinite(opaque_surface_depth)
        & (opaque_surface_depth > 0.0)
        & (~opaque_surface_sand_owner.to(torch.bool))
    )
    source_delta = opaque_surface_depth - terrain_depth
    used = (
        reference_valid
        & water_valid
        & (torch.abs(source_delta) <= source_tolerance)
    )
    rejected_foreground = (
        reference_valid
        & water_valid
        & (source_delta < -source_tolerance)
    )
    resolved = torch.where(used, terrain_depth, opaque_surface_depth)
    return resolved, used, rejected_foreground, source_delta


def resolve_water_physical_reference_visibility(
    *,
    water_depth: torch.Tensor,
    water_valid: torch.Tensor,
    water_coverage: torch.Tensor,
    physical_depth: torch.Tensor,
    physical_coverage: torch.Tensor,
    occlusion_depth: torch.Tensor,
    occlusion_bias: torch.Tensor,
    erosion_pixels: int,
    gaussian_coverage_min: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Resolve water visibility from a conservative physical surface mesh.

    Only the camera-Z value used for the front/behind decision is replaced.
    Gaussian coverage, normals, thickness and optical shading remain intact.
    Eroding the all-wet triangle coverage keeps the reference away from an
    uncertain wet/dry shoreline.
    """

    if isinstance(erosion_pixels, bool) or erosion_pixels < 0:
        raise ValueError(
            "water_physical_reference_erosion_pixels must be non-negative"
        )
    if (
        not math.isfinite(gaussian_coverage_min)
        or gaussian_coverage_min < 0.0
        or gaussian_coverage_min > 1.0
    ):
        raise ValueError(
            "water_physical_reference_gaussian_coverage_min must be in [0,1]"
        )
    trusted_coverage = physical_coverage.to(torch.bool)
    if erosion_pixels > 0:
        kernel = 2 * erosion_pixels + 1
        invalid = torch.nn.functional.pad(
            (~trusted_coverage).to(torch.float32),
            (
                erosion_pixels,
                erosion_pixels,
                erosion_pixels,
                erosion_pixels,
            ),
            mode="constant",
            value=1.0,
        )
        nearby_invalid = torch.nn.functional.max_pool2d(
            invalid,
            kernel_size=kernel,
            stride=1,
        ) > 0.0
        trusted_coverage = trusted_coverage & (~nearby_invalid)
    trusted = (
        trusted_coverage
        & torch.isfinite(physical_depth)
        & (physical_depth > 0.0)
        & water_valid
        & (water_coverage >= gaussian_coverage_min)
    )
    occlusion_valid = (
        torch.isfinite(occlusion_depth) & (occlusion_depth > 0.0)
    )
    gaussian_in_front = torch.where(
        occlusion_valid,
        water_depth < (occlusion_depth - occlusion_bias),
        torch.ones_like(water_valid),
    )
    physical_in_front = torch.where(
        occlusion_valid,
        physical_depth < (occlusion_depth - occlusion_bias),
        torch.ones_like(water_valid),
    )
    visibility_depth = torch.where(trusted, physical_depth, water_depth)
    resolved_in_front = torch.where(
        trusted, physical_in_front, gaussian_in_front
    )
    recovered = trusted & (~gaussian_in_front) & physical_in_front
    removed = trusted & gaussian_in_front & (~physical_in_front)
    rejected_opaque = trusted & (~physical_in_front)
    return (
        visibility_depth,
        resolved_in_front,
        trusted,
        recovered,
        removed,
        rejected_opaque,
    )


def build_water_gbuffer(
    *,
    args,
    bg_color,
    fluid_span,
    full_proj_transform,
    image_height,
    image_width,
    means2D,
    means2D_abs,
    means3D,
    num_fluid_gaussian,
    opacity,
    opaque_surface_depth,
    opaque_surface_sand_owner,
    orthographic,
    ortho_x_min,
    ortho_x_max,
    ortho_y_min,
    ortho_y_max,
    pc,
    pipe,
    rotations,
    scales,
    scaling_modifier,
    tanfovx,
    tanfovy,
    viewpoint_camera,
) -> WaterGBuffer:
    """Rasterize only the fluid span and resolve its visible surface."""

    fluid_slice = slice(fluid_span.start, fluid_span.end)
    means3D_water = means3D[fluid_slice]
    means2D_water = means2D[fluid_slice]
    means2D_abs_water = means2D_abs[fluid_slice]
    opacity_water = opacity[fluid_slice]
    scales_water = scales[fluid_slice]
    rotations_water = rotations[fluid_slice]
    normal_world_water_gaussian = pc.get_stored_layer_normals("fluid")
    normal_camera_water_gaussian = (
        normal_world_water_gaussian
        @ viewpoint_camera.world_view_transform[:3, :3]
    )
    points_camera_water = (
        means3D_water @ viewpoint_camera.world_view_transform[:3, :3]
        + viewpoint_camera.world_view_transform[3, :3]
    )
    distance_water = (
        normal_camera_water_gaussian * points_camera_water
    ).sum(-1).abs()
    input_all_map_water = torch.zeros(
        (num_fluid_gaussian, 5),
        dtype=torch.float32,
        device="cuda",
    )
    input_all_map_water[:, :3] = normal_camera_water_gaussian
    input_all_map_water[:, 3] = 1.0
    input_all_map_water[:, 4] = distance_water
    water_settings = PlaneGaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        render_geo=True,
        debug=pipe.debug,
        orthographic=orthographic,
        ortho_x_min=ortho_x_min,
        ortho_x_max=ortho_x_max,
        ortho_y_min=ortho_y_min,
        ortho_y_max=ortho_y_max,
    )
    water_rasterizer = PlaneGaussianRasterizer(
        raster_settings=water_settings
    )
    water_geometry_color = torch.ones(
        (num_fluid_gaussian, 3), device="cuda", dtype=torch.float32
    )
    (
        _, _, _, water_all_map, water_depth
    ) = water_rasterizer(
        means3D=means3D_water,
        means2D=means2D_water,
        means2D_abs=means2D_abs_water,
        shs=None,
        colors_precomp=water_geometry_color,
        opacities=opacity_water,
        scales=scales_water,
        rotations=rotations_water,
        all_map=input_all_map_water,
        cov3D_precomp=None,
    )
    water_alpha = torch.clamp(water_all_map[3:4], 0.0, 1.0)
    water_normal_camera = torch.nn.functional.normalize(
        water_all_map[:3], dim=0, eps=1.0e-6
    )
    water_normal_world = camera_normals_to_world(
        water_normal_camera,
        viewpoint_camera.world_view_transform[:3, :3],
    )

    def rasterize_water_scalar(values):
        scalar_color = values[:, None].repeat(1, 3)
        rendered_scalar, _, _, _, _ = water_rasterizer(
            means3D=means3D_water,
            means2D=means2D_water,
            means2D_abs=means2D_abs_water,
            shs=None,
            colors_precomp=scalar_color,
            opacities=opacity_water,
            scales=scales_water,
            rotations=rotations_water,
            all_map=input_all_map_water,
            cov3D_precomp=None,
        )
        return rendered_scalar.mean(0, keepdim=True) / torch.clamp(
            water_alpha, min=1.0e-4
        )

    water_thickness = torch.clamp_min(
        rasterize_water_scalar(pc.water_thickness_t), 0.0
    )
    water_coverage = opaque_surface_coverage(
        water_alpha,
        float(getattr(args, "water_coverage_alpha_low", 0.02)),
        float(getattr(args, "water_coverage_alpha_high", 0.80)),
    )
    fallback_mode = str(
        getattr(args, "water_depth_fallback_mode", "none")
    ).lower()
    if fallback_mode not in {"none", "center"}:
        raise ValueError(
            "water_depth_fallback_mode must be 'none' or 'center'"
        )
    if fallback_mode == "center":
        # Rasterizing camera-space Gaussian center z avoids the tangent-
        # plane division that becomes singular around steep dynamic
        # normals.  It is used only for invalid plane depths below; valid
        # plane intersections retain their more accurate surface depth.
        water_center_depth = rasterize_water_scalar(
            points_camera_water[:, 2]
        )
    else:
        # Keep the default path free of another rasterization and preserve
        # its output bit-for-bit when the experimental fallback is off.
        water_center_depth = torch.zeros_like(water_depth)
    water_depth, water_plane_valid, water_depth_fallback_used = (
        fallback_invalid_plane_depth(
            water_depth,
            water_center_depth,
            water_alpha,
            water_coverage,
            enabled=fallback_mode == "center",
            alpha_min=float(getattr(
                args, "water_depth_fallback_alpha_min", 0.10
            )),
            coverage_min=float(getattr(
                args, "water_depth_fallback_coverage_min", 0.50
            )),
        )
    )
    opaque_valid = (
        torch.isfinite(opaque_surface_depth)
        & (opaque_surface_depth > 0.0)
    )
    water_alpha_min = float(getattr(args, "water_alpha_min", 0.02))
    water_valid = (
        (water_plane_valid | water_depth_fallback_used)
        & (water_alpha > water_alpha_min)
    )
    water_depth_bias = float(getattr(args, "water_depth_bias", 0.002))
    if not math.isfinite(water_depth_bias) or water_depth_bias < 0:
        raise ValueError("water_depth_bias must be finite and non-negative")
    proxy_enabled = bool(getattr(args, "water_terrain_depth_proxy", False))
    terrain_proxy_depth = torch.zeros_like(opaque_surface_depth)
    terrain_proxy_coverage = torch.zeros_like(
        opaque_surface_depth, dtype=torch.bool
    )
    terrain_proxy_used = torch.zeros_like(
        opaque_surface_depth, dtype=torch.bool
    )
    terrain_proxy_rejected_foreground = torch.zeros_like(
        opaque_surface_depth, dtype=torch.bool
    )
    terrain_proxy_source_delta = torch.zeros_like(opaque_surface_depth)
    occlusion_depth = opaque_surface_depth
    occlusion_bias = torch.full_like(opaque_surface_depth, water_depth_bias)
    if proxy_enabled:
        terrain_triangles = getattr(
            pc,
            "water_terrain_reference_triangles_gs_t",
            torch.empty(
                (0, 3, 3), dtype=water_depth.dtype, device=water_depth.device
            ),
        )
        if terrain_triangles.shape[0] == 0:
            raise ValueError(
                "water terrain depth proxy requires a physical terrain mesh"
            )
        # The plane Gaussian CUDA kernel evaluates rays through integer pixel
        # coordinates, whereas nvdiffrast samples pixel centres. Shift the
        # reference projection by +0.5 pixel so both depths share one ray.
        terrain_projection = full_proj_transform.clone()
        terrain_projection[:, 0] += (
            terrain_projection[:, 3] / float(image_width)
        )
        terrain_projection[:, 1] += (
            terrain_projection[:, 3] / float(image_height)
        )
        terrain_reference = rasterize_triangle_reference(
            terrain_triangles,
            viewpoint_camera.world_view_transform,
            terrain_projection,
            image_height,
            image_width,
        )
        terrain_proxy_depth = terrain_reference[
            "sand_triangle_reference_depth_raw"
        ]
        terrain_proxy_coverage = terrain_reference[
            "sand_triangle_reference_coverage"
        ]
        (
            occlusion_depth,
            terrain_proxy_used,
            terrain_proxy_rejected_foreground,
            terrain_proxy_source_delta,
        ) = resolve_terrain_depth_proxy(
            opaque_surface_depth=opaque_surface_depth,
            opaque_surface_sand_owner=opaque_surface_sand_owner,
            terrain_depth=terrain_proxy_depth,
            terrain_coverage=terrain_proxy_coverage,
            water_valid=water_valid,
            source_tolerance=float(getattr(
                args, "water_terrain_proxy_source_tolerance", 0.10
            )),
        )
        terrain_proxy_depth_bias = float(getattr(
            args, "water_terrain_proxy_depth_bias", 0.0
        ))
        if (
            not math.isfinite(terrain_proxy_depth_bias)
            or terrain_proxy_depth_bias < 0
        ):
            raise ValueError(
                "water_terrain_proxy_depth_bias must be finite and non-negative"
            )
        occlusion_bias = torch.where(
            terrain_proxy_used,
            torch.full_like(occlusion_bias, terrain_proxy_depth_bias),
            occlusion_bias,
        )
    # Keep the raw depth ordering separate for diagnostics.  A zero plane
    # depth compares as nearer than every real opaque surface, so it must
    # not itself be interpreted as a visible water sample.
    water_depth_in_front_without_proxy = torch.where(
        opaque_valid,
        water_depth < (opaque_surface_depth - water_depth_bias),
        torch.ones_like(water_valid),
    )
    occlusion_valid = torch.isfinite(occlusion_depth) & (occlusion_depth > 0.0)
    water_depth_in_front = torch.where(
        occlusion_valid,
        water_depth < (occlusion_depth - occlusion_bias),
        torch.ones_like(water_valid),
    )
    water_depth_in_front_without_physical_reference = water_depth_in_front
    physical_reference_depth = torch.zeros_like(water_depth)
    physical_reference_coverage = torch.zeros_like(
        water_depth, dtype=torch.bool
    )
    physical_reference_trusted = torch.zeros_like(
        water_depth, dtype=torch.bool
    )
    physical_reference_recovered = torch.zeros_like(
        water_depth, dtype=torch.bool
    )
    physical_reference_removed = torch.zeros_like(
        water_depth, dtype=torch.bool
    )
    physical_reference_rejected_opaque = torch.zeros_like(
        water_depth, dtype=torch.bool
    )
    if bool(getattr(
        args, "water_physical_reference_depth_recovery", False
    )):
        water_triangles = getattr(
            pc,
            "water_surface_reference_triangles_gs_t",
            torch.empty(
                (0, 3, 3), dtype=water_depth.dtype, device=water_depth.device
            ),
        )
        # An isolated wet cell has a Gaussian but cannot form an all-wet
        # triangle. In that case the opt-in reference simply has no trusted
        # pixels and the original Gaussian visibility remains unchanged.
        if water_triangles.shape[0] > 0:
            water_projection = full_proj_transform.clone()
            water_projection[:, 0] += (
                water_projection[:, 3] / float(image_width)
            )
            water_projection[:, 1] += (
                water_projection[:, 3] / float(image_height)
            )
            water_reference = rasterize_triangle_reference(
                water_triangles,
                viewpoint_camera.world_view_transform,
                water_projection,
                image_height,
                image_width,
            )
            physical_reference_depth = water_reference[
                "sand_triangle_reference_depth_raw"
            ]
            physical_reference_coverage = water_reference[
                "sand_triangle_reference_coverage"
            ]
            (
                _,
                water_depth_in_front,
                physical_reference_trusted,
                physical_reference_recovered,
                physical_reference_removed,
                physical_reference_rejected_opaque,
            ) = resolve_water_physical_reference_visibility(
                water_depth=water_depth,
                water_valid=water_valid,
                water_coverage=water_coverage,
                physical_depth=physical_reference_depth,
                physical_coverage=physical_reference_coverage,
                occlusion_depth=occlusion_depth,
                occlusion_bias=occlusion_bias,
                erosion_pixels=int(getattr(
                    args, "water_physical_reference_erosion_pixels", 2
                )),
                gaussian_coverage_min=float(getattr(
                    args,
                    "water_physical_reference_gaussian_coverage_min",
                    0.5,
                )),
            )
    # A recovered center-depth candidate is not allowed to bypass normal
    # source+sand occlusion.  Export this rejection separately so an
    # experiment can detect whether fallback is reviving hidden water.
    water_depth_fallback_rejected_by_opaque = (
        water_depth_fallback_used & (~water_depth_in_front)
    )
    water_surface_valid = (
        water_valid & water_depth_in_front & (water_coverage >= 0.5)
    )
    # RGB coverage stays continuous for antialiasing, but invalid water
    # geometry falls back to the immutable source/source+sand substrate.
    # Previously this omitted water_valid, so depth=0 pixels could pass the
    # front test, sample a zero reflection buffer and become black dots.
    water_visible_coverage = valid_visible_coverage(
        water_coverage, water_valid, water_depth_in_front
    )
    water_invalid_visible_candidate = (
        water_coverage
        * water_depth_in_front.float()
        * (~water_valid).float()
    )
    # Diagnostic-only partition of the exact gates above.  Keeping this
    # beside the production masks ensures that a saved class map explains
    # the actual final-composition decision without changing it.
    water_stage_diagnostics = classify_water_gbuffer_stages(
        water_alpha,
        water_coverage,
        water_plane_valid,
        water_depth_fallback_used,
        water_depth_in_front,
        alpha_min=water_alpha_min,
        owner_coverage_min=0.5,
    )
    water_gbuffer_validation = None
    if bool(getattr(args, "water_validate_gbuffer", True)):
        water_gbuffer_validation = validate_surface_gbuffer(
            water_coverage,
            water_depth,
            water_normal_world,
            water_surface_valid,
        )
        print("P4 water G-buffer validated:", water_gbuffer_validation)

    return WaterGBuffer(
        depth=water_depth,
        plane_depth_valid=water_plane_valid,
        depth_fallback_used=water_depth_fallback_used,
        depth_fallback_rejected_by_opaque=(
            water_depth_fallback_rejected_by_opaque
        ),
        surface_valid=water_surface_valid,
        visible_coverage=water_visible_coverage,
        invalid_visible_candidate=water_invalid_visible_candidate,
        stage_diagnostics=water_stage_diagnostics,
        validation=water_gbuffer_validation,
        normal_camera=water_normal_camera,
        normal_world=water_normal_world,
        thickness=water_thickness,
        raw_alpha=water_alpha,
        raw_coverage=water_coverage,
        depth_in_front=water_depth_in_front,
        depth_in_front_without_proxy=water_depth_in_front_without_proxy,
        depth_bias=water_depth_bias,
        occlusion_depth=occlusion_depth,
        terrain_proxy_depth=terrain_proxy_depth,
        terrain_proxy_coverage=terrain_proxy_coverage,
        terrain_proxy_used=terrain_proxy_used,
        terrain_proxy_rejected_foreground=(
            terrain_proxy_rejected_foreground
        ),
        terrain_proxy_source_delta=terrain_proxy_source_delta,
        occlusion_bias=occlusion_bias,
        depth_in_front_without_physical_reference=(
            water_depth_in_front_without_physical_reference
        ),
        physical_reference_depth=physical_reference_depth,
        physical_reference_coverage=physical_reference_coverage,
        physical_reference_trusted=physical_reference_trusted,
        physical_reference_recovered=physical_reference_recovered,
        physical_reference_removed=physical_reference_removed,
        physical_reference_rejected_opaque=(
            physical_reference_rejected_opaque
        ),
    )
