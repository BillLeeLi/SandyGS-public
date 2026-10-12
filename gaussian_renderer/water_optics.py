"""SandyGS P4 reflection, refraction and final water composition."""

import torch

from bvh import ScreenRayTracer
from sandygs.rendering import (
    beer_lambert_transmittance,
    linear_to_srgb,
    occluded_reflection_proxy,
    perspective_view_directions_camera,
    schlick_fresnel,
    screen_space_refraction_grid,
    srgb_to_linear,
)
from utils.my_utils import apply_gaussian_smoothing


def compose_water_optics(
    *,
    args,
    image_height,
    image_width,
    opaque_surface_depth,
    opaque_surface_normal_camera_facing,
    pc,
    rendered_rgb_composite,
    result,
    viewpoint_camera,
    water_gbuffer,
):
    """Apply SandyGS water optics to an immutable opaque substrate."""

    water_depth = water_gbuffer.depth
    water_plane_valid = water_gbuffer.plane_depth_valid
    water_depth_fallback_used = water_gbuffer.depth_fallback_used
    water_depth_fallback_rejected_by_opaque = (
        water_gbuffer.depth_fallback_rejected_by_opaque
    )
    water_surface_valid = water_gbuffer.surface_valid
    water_visible_coverage = water_gbuffer.visible_coverage
    water_invalid_visible_candidate = (
        water_gbuffer.invalid_visible_candidate
    )
    water_stage_diagnostics = water_gbuffer.stage_diagnostics
    water_gbuffer_validation = water_gbuffer.validation
    water_normal_camera = water_gbuffer.normal_camera
    water_normal_world = water_gbuffer.normal_world
    water_thickness = water_gbuffer.thickness
    water_alpha = water_gbuffer.raw_alpha
    water_coverage = water_gbuffer.raw_coverage
    water_depth_in_front = water_gbuffer.depth_in_front
    water_depth_in_front_without_proxy = (
        water_gbuffer.depth_in_front_without_proxy
    )
    water_depth_bias = water_gbuffer.depth_bias

    # Orient the physical normal toward the incident/view half-space.
    # This changes only the per-pixel optical evaluation; the stored
    # height-field normal remains camera-independent in the G-buffer.
    view_camera = perspective_view_directions_camera(
        image_height, image_width,
        viewpoint_camera.fx, viewpoint_camera.fy,
        viewpoint_camera.cx, viewpoint_camera.cy,
        device=water_depth.device, dtype=water_depth.dtype,
    )
    facing = (water_normal_camera * view_camera).sum(0, keepdim=True)
    optical_normal_camera = torch.where(
        (facing < 0.0).repeat(3, 1, 1),
        -water_normal_camera,
        water_normal_camera,
    )
    cosine = torch.clamp(
        (optical_normal_camera * view_camera).sum(0, keepdim=True),
        0.0, 1.0,
    )
    fresnel = schlick_fresnel(
        cosine, float(getattr(args, "water_f0", 0.02))
    )

    # Refraction uses an image-space approximation whose offset is
    # proportional to the local free-water thickness. Samples crossing the
    # water silhouette or landing in front of the water use the substrate.
    opaque_linear = srgb_to_linear(rendered_rgb_composite)
    refraction_grid = screen_space_refraction_grid(
        optical_normal_camera,
        water_thickness,
        float(getattr(args, "water_refraction_pixels_per_unit", 8.0)),
    )
    refracted_linear = torch.nn.functional.grid_sample(
        opaque_linear[None], refraction_grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    )[0]
    sampled_water = torch.nn.functional.grid_sample(
        water_surface_valid.float()[None], refraction_grid,
        mode="nearest", padding_mode="zeros", align_corners=True,
    )[0] > 0.5
    sampled_opaque_depth = torch.nn.functional.grid_sample(
        opaque_surface_depth[None], refraction_grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    )[0]
    valid_refraction = sampled_water & (
        sampled_opaque_depth > water_depth + water_depth_bias
    )
    refracted_linear = torch.where(
        valid_refraction.repeat(3, 1, 1),
        refracted_linear,
        opaque_linear,
    )

    # Beer-Lambert absorption uses the view-angle-corrected path length.
    # Garden is not metrically calibrated, so the explicit scale below
    # converts solver-height units to an appearance distance.
    path_length = (
        water_thickness
        * float(getattr(args, "water_optical_distance_scale", 1.0))
        / torch.clamp(cosine, min=0.15)
    )
    absorption = torch.tensor(
        getattr(args, "water_absorption_rgb", [0.15, 0.055, 0.025]),
        device="cuda", dtype=torch.float32,
    )[:, None, None]
    transmittance = beer_lambert_transmittance(path_length, absorption)
    water_tint = srgb_to_linear(
        torch.tensor(
            getattr(args, "water_scatter_color", [34, 86, 92]),
            device="cuda", dtype=torch.float32,
        ) / 255.0
    )[:, None, None]
    transmitted_linear = (
        transmittance * refracted_linear
        + (1.0 - transmittance) * water_tint
    )

    #################################
    # P4 SSR: march against the composed opaque source+sand depth.
    #################################
    reflection_linear = torch.zeros_like(opaque_linear)
    ssr_hit_map = torch.zeros_like(water_surface_valid, dtype=torch.float32)
    ssr_backface_map = torch.zeros_like(ssr_hit_map)
    ssr_ambiguous_map = torch.zeros_like(ssr_hit_map)
    ssr_normal_dot_map = torch.zeros_like(ssr_hit_map)
    ssr_proxy_linear = torch.zeros_like(opaque_linear)
    reflection_direction_camera = torch.nn.functional.normalize(
        2.0 * cosine * optical_normal_camera - view_camera,
        dim=0, eps=1.0e-6,
    )
    active_flat = water_surface_valid[0].reshape(-1)
    active_indices = torch.nonzero(active_flat, as_tuple=False)[:, 0]
    if active_indices.numel() > 0:
        height_index = torch.div(
            active_indices, image_width, rounding_mode="floor"
        )
        width_index = active_indices % image_width
        start_coords = torch.stack(
            (width_index, height_index), dim=-1
        ).float()
        fx = torch.as_tensor(viewpoint_camera.fx, device="cuda").float()
        fy = torch.as_tensor(viewpoint_camera.fy, device="cuda").float()
        cx = torch.as_tensor(viewpoint_camera.cx, device="cuda").float()
        cy = torch.as_tensor(viewpoint_camera.cy, device="cuda").float()
        depth_active = water_depth[0].reshape(-1)[active_indices]
        start_camera = torch.stack(
            (
                (width_index.float() - cx) / fx * depth_active,
                (height_index.float() - cy) / fy * depth_active,
                depth_active,
            ), dim=-1,
        )
        direction_active = (
            reflection_direction_camera.permute(1, 2, 0)
            .reshape(-1, 3)[active_indices]
        )
        ray_length = float(getattr(args, "water_ssr_ray_length", 500.0))
        end_camera = start_camera + ray_length * direction_active
        end_z = torch.clamp(end_camera[:, 2], min=1.0e-4)
        end_coords = torch.stack(
            (fx * end_camera[:, 0] / end_z + cx,
             fy * end_camera[:, 1] / end_z + cy), dim=-1,
        )
        ssr = ScreenRayTracer(
            refra_k=0.0,
            thickness=float(getattr(args, "water_ssr_thickness", 0.03)),
        ).trace(
            opaque_surface_depth[0].contiguous(),
            opaque_surface_normal_camera_facing.permute(
                1, 2, 0
            ).contiguous(),
            start_coords.contiguous(), end_coords.contiguous(),
            start_camera.contiguous(), end_camera.contiguous(),
            start_step=int(getattr(args, "water_ssr_start_step", 2)),
            frontface_cos_threshold=float(getattr(
                args, "water_ssr_frontface_cos_threshold", 0.08
            )),
            backface_cos_threshold=float(getattr(
                args, "water_ssr_backface_cos_threshold", 0.08
            )),
            max_depth_jump_absolute=float(getattr(
                args, "water_ssr_max_depth_jump_absolute", 0.25
            )),
            max_depth_jump_relative=float(getattr(
                args, "water_ssr_max_depth_jump_relative", 0.05
            )),
        )
        hit_coords = ssr["hit_coords"]
        hit_status = ssr["hit_status"]
        hit_valid = hit_status == 1
        backface_only = hit_status == 2
        ambiguous = hit_status == 3
        ssr_hit_flat = ssr_hit_map.reshape(-1)
        ssr_hit_flat[active_indices[hit_valid]] = 1.0
        ssr_backface_map.reshape(-1)[
            active_indices[backface_only]
        ] = 1.0
        ssr_ambiguous_map.reshape(-1)[
            active_indices[ambiguous]
        ] = 1.0
        # Store signed R.N as [0,1] for PNG inspection. Gray 0.5 means
        # no classified crossing; front hits are dark, back faces bright.
        classified = hit_status != 0
        ssr_normal_dot_map.reshape(-1)[
            active_indices[classified]
        ] = 0.5 * (
            ssr["hit_normal_dot"][classified] + 1.0
        )
        opaque_linear_hwc = opaque_linear.permute(1, 2, 0)
        reflected_active = torch.empty(
            (active_indices.numel(), 3), device="cuda", dtype=torch.float32
        )
        if torch.any(hit_valid):
            hit = hit_coords[hit_valid]
            reflected_active[hit_valid] = opaque_linear_hwc[
                hit[:, 1].long(), hit[:, 0].long()
            ]
        # No hit and ambiguous/tangent crossings use the ordinary
        # environment fallback. A back-face-only ray is handled below
        # by a separate occluded-surface proxy, never by clear top color.
        environment_fallback = (hit_status == 0) | ambiguous
        inverse_rotation = torch.linalg.inv(
            viewpoint_camera.world_view_transform[:3, :3]
        )

        def sample_reflection_environment(selection, roughness_value):
            """Sample the scene environment for selected active rays."""
            if not torch.any(selection):
                return torch.empty(
                    (0, 3), device="cuda", dtype=torch.float32
                )
            direction_world = torch.nn.functional.normalize(
                direction_active[selection] @ inverse_rotation,
                dim=-1, eps=1.0e-6,
            )
            roughness = torch.full(
                (1, 1, direction_world.shape[0], 1),
                roughness_value,
                device="cuda", dtype=torch.float32,
            )
            environment_srgb = pc.brdf_mlp_fluid.shade_ssr_nohit(
                direction_world[None, None], roughness
            )[0, 0]
            return srgb_to_linear(environment_srgb)

        if torch.any(environment_fallback):
            reflected_active[environment_fallback] = (
                sample_reflection_environment(
                    environment_fallback,
                    float(getattr(args, "water_roughness", 0.05)),
                )
            )

        if torch.any(backface_only):
            # The G-buffer contains the camera-visible side only. For a
            # rejected back face (e.g. tabletop viewed from below), use
            # its location merely as low-frequency context and synthesize
            # a dark rough proxy for the genuinely unobserved underside.
            blur_radius = int(getattr(
                args, "water_occluded_proxy_blur_radius", 12
            ))
            if blur_radius < 0:
                raise ValueError(
                    "water_occluded_proxy_blur_radius must be non-negative"
                )
            if blur_radius > 0:
                kernel_size = 2 * blur_radius + 1
                opaque_low_frequency = torch.nn.functional.avg_pool2d(
                    opaque_linear[None], kernel_size=kernel_size,
                    stride=1, padding=blur_radius,
                    count_include_pad=False,
                )[0]
            else:
                opaque_low_frequency = opaque_linear
            proxy_coords = ssr["proxy_coords"][backface_only]
            low_frequency_color = opaque_low_frequency.permute(1, 2, 0)[
                proxy_coords[:, 1].long(), proxy_coords[:, 0].long()
            ]
            proxy_environment = sample_reflection_environment(
                backface_only,
                float(getattr(
                    args, "water_occluded_proxy_roughness", 0.80
                )),
            )
            proxy_color = occluded_reflection_proxy(
                low_frequency_color,
                proxy_environment,
                darkness=float(getattr(
                    args, "water_occluded_proxy_darkness", 0.25
                )),
                saturation=float(getattr(
                    args, "water_occluded_proxy_saturation", 0.45
                )),
                environment_weight=float(getattr(
                    args, "water_occluded_proxy_environment_weight", 0.10
                )),
            )
            reflected_active[backface_only] = proxy_color
            proxy_flat = ssr_proxy_linear.permute(1, 2, 0).reshape(-1, 3)
            proxy_flat[active_indices[backface_only]] = proxy_color
            ssr_proxy_linear = proxy_flat.view(
                image_height, image_width, 3
            ).permute(2, 0, 1)
        reflection_flat = reflection_linear.permute(1, 2, 0).reshape(-1, 3)
        reflection_flat[active_indices] = reflected_active
        reflection_linear = reflection_flat.view(
            image_height, image_width, 3
        ).permute(2, 0, 1)

    # Filter discrete SSR samples with a 5x5, sigma=2 Gaussian
    # before Fresnel composition. Keep the unfiltered radiance only for
    # diagnostics, then apply the same operation here so nearest-pixel DDA
    # hits and hard hit-status transitions are not exposed directly by a
    # strong Fresnel term. Depth, normals, masks and refraction stay intact.
    reflection_linear_raw = reflection_linear
    reflection_linear = apply_gaussian_smoothing(
        reflection_linear, kernel_size=5, sigma=2.0
    )

    # A restrained direct highlight uses profile-controlled parameters.
    light_camera = torch.nn.functional.normalize(
        pc.water_light_direction_gs_t
        @ viewpoint_camera.world_view_transform[:3, :3],
        dim=0, eps=1.0e-6,
    )
    half_vector_water = torch.nn.functional.normalize(
        light_camera[:, None, None] + view_camera,
        dim=0, eps=1.0e-6,
    )
    highlight = float(getattr(args, "water_highlight_strength", 0.10)) * torch.pow(
        torch.clamp(
            (optical_normal_camera * half_vector_water).sum(0, keepdim=True),
            min=0.0,
        ),
        float(getattr(args, "water_highlight_exponent", 96.0)),
    )
    water_linear = (
        (1.0 - fresnel) * transmitted_linear
        + fresnel * torch.clamp(reflection_linear + highlight, 0.0, 1.0)
    )
    final_linear = (
        opaque_linear * (1.0 - water_visible_coverage)
        + water_linear * water_visible_coverage
    )
    final_srgb = torch.clamp(linear_to_srgb(final_linear), 0.0, 1.0)
    result.update(
        {
            "render": final_srgb,
            "rendered_image_src_refl_save": linear_to_srgb(
                opaque_linear * (1.0 - water_visible_coverage)
                + reflection_linear * water_visible_coverage
            ).clamp(0.0, 1.0),
            "rendered_image_src_refl_refra_save": final_srgb,
            "water_surface_depth": water_depth,
            "water_surface_plane_depth_valid": water_plane_valid,
            "water_depth_fallback_used": (
                water_depth_fallback_used.repeat(3, 1, 1)
            ),
            "water_depth_fallback_rejected_by_opaque": (
                water_depth_fallback_rejected_by_opaque.repeat(3, 1, 1)
            ),
            "water_surface_normal_world": water_normal_world,
            "water_surface_valid": water_surface_valid,
            "water_surface_thickness": water_thickness,
            "water_surface_coverage": water_visible_coverage,
            "water_raw_alpha": water_alpha.repeat(3, 1, 1),
            "water_raw_coverage": water_coverage.repeat(3, 1, 1),
            "water_plane_depth_valid": (
                water_plane_valid.repeat(3, 1, 1)
            ),
            "water_depth_in_front": (
                water_depth_in_front.repeat(3, 1, 1)
            ),
            "water_depth_in_front_without_proxy": (
                water_depth_in_front_without_proxy.repeat(3, 1, 1)
            ),
            "water_depth_in_front_without_physical_reference": (
                water_gbuffer.depth_in_front_without_physical_reference.repeat(
                    3, 1, 1
                )
            ),
            "water_occlusion_depth_raw": water_gbuffer.occlusion_depth,
            "water_occlusion_bias_raw": water_gbuffer.occlusion_bias,
            "water_terrain_proxy_depth_raw": (
                water_gbuffer.terrain_proxy_depth
            ),
            "water_terrain_proxy_coverage": (
                water_gbuffer.terrain_proxy_coverage.repeat(3, 1, 1)
            ),
            "water_terrain_proxy_used": (
                water_gbuffer.terrain_proxy_used.repeat(3, 1, 1)
            ),
            "water_terrain_proxy_rejected_foreground": (
                water_gbuffer.terrain_proxy_rejected_foreground.repeat(3, 1, 1)
            ),
            "water_terrain_proxy_source_delta_raw": (
                water_gbuffer.terrain_proxy_source_delta
            ),
            "water_physical_reference_depth_raw": (
                water_gbuffer.physical_reference_depth
            ),
            "water_physical_reference_coverage": (
                water_gbuffer.physical_reference_coverage.repeat(3, 1, 1)
            ),
            "water_physical_reference_trusted": (
                water_gbuffer.physical_reference_trusted.repeat(3, 1, 1)
            ),
            "water_physical_reference_recovered": (
                water_gbuffer.physical_reference_recovered.repeat(3, 1, 1)
            ),
            "water_physical_reference_removed": (
                water_gbuffer.physical_reference_removed.repeat(3, 1, 1)
            ),
            "water_physical_reference_rejected_opaque": (
                water_gbuffer.physical_reference_rejected_opaque.repeat(
                    3, 1, 1
                )
            ),
            "water_surface_valid_debug": (
                water_surface_valid.repeat(3, 1, 1)
            ),
            "water_gbuffer_validation": water_gbuffer_validation,
            "water_reflection": linear_to_srgb(reflection_linear).clamp(0.0, 1.0),
            "water_refraction": linear_to_srgb(refracted_linear).clamp(0.0, 1.0),
            "water_fresnel": fresnel.repeat(3, 1, 1) * water_surface_valid,
            "water_transmittance": transmittance * water_surface_valid,
            "water_coverage": water_visible_coverage.repeat(3, 1, 1),
            "water_invalid_coverage": (
                water_invalid_visible_candidate.repeat(3, 1, 1)
            ),
            "water_normal": (
                water_normal_world * 0.5 + 0.5
            ) * water_surface_valid,
            "water_ssr_hit": ssr_hit_map.repeat(3, 1, 1),
            "water_ssr_frontface_hit": ssr_hit_map.repeat(3, 1, 1),
            "water_ssr_backface_rejected": ssr_backface_map.repeat(3, 1, 1),
            "water_ssr_ambiguous": ssr_ambiguous_map.repeat(3, 1, 1),
            "water_ssr_hit_normal_dot": ssr_normal_dot_map.repeat(3, 1, 1),
            "water_ssr_occluded_proxy": linear_to_srgb(
                ssr_proxy_linear
            ).clamp(0.0, 1.0),
            "water_ssr_raw_reflection": linear_to_srgb(
                reflection_linear_raw
            ).clamp(0.0, 1.0),
            "water_ssr_final_reflection": linear_to_srgb(
                reflection_linear
            ).clamp(0.0, 1.0),
        }
    )
    # Boolean failure-stage masks remain in the render package for both
    # PNG export and exact per-frame counts; the RGB legend map is already
    # float-valued and can be saved directly.
    result.update(
        {
            key: value if value.ndim == 3 and value.shape[0] == 3
            else value.repeat(3, 1, 1)
            for key, value in water_stage_diagnostics.items()
        }
    )
    return result
