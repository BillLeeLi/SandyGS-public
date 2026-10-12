#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

"""Opaque source/sand G-buffer construction for SandyGS rendering."""

import math

import torch
from diff_plane_rasterization import (
    GaussianRasterizationSettings as PlaneGaussianRasterizationSettings,
)
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer

from sandygs.materials import (
    SandPBRMaterial,
    ambient_occlusion_from_residual,
    apply_sand_pbr_microtexture,
    apply_sand_texture_primary,
    reconstruct_world_positions,
    resolve_material_root,
    sample_sand_pbr_material,
    sample_sand_pbr_material_triplanar,
    sand_texture_uv_checker,
    simulation_xy_uv,
    world_positions_to_simulation,
)
from sandygs.rendering import (
    align_reference_normals,
    camera_normals_to_world,
    classify_sand_center_trusted_invalid_depth,
    classify_sand_depth_competition,
    classify_sand_gbuffer_stages,
    classify_sand_invalid_plane_depth,
    classify_sand_physical_reference_depth_loss,
    classify_sand_plane_depth_reference,
    classify_sand_primitive_overcoverage,
    classify_sand_t3_prerequisites,
    classify_sand_triangle_footprint,
    compare_sand_soft_boundary,
    guide_sand_coverage_with_soft_boundary,
    linear_to_srgb,
    normalize_environment_irradiance,
    opaque_surface_coverage,
    opaque_surface_owner,
    orient_plane_normals_toward_camera,
    perspective_view_directions_camera,
    rasterize_triangle_reference,
    resolve_sand_center_trusted_invalid_depth,
    resolve_sand_center_trusted_invalid_normal,
    resolve_sand_oriented_plane_depth,
    resolve_sand_physical_reference_depth,
    resolve_sand_physical_reference_normal,
    resolve_sand_plane_depth_stability,
    sand_height_to_soft_coverage,
    select_sand_recovery_attribute_region,
    split_sum_specular_reflectance,
    srgb_to_linear,
    validate_surface_gbuffer,
)


def compose_opaque_gbuffer(
    *,
    args,
    bg_color,
    colors_precomp,
    full_proj_transform,
    image_height,
    image_width,
    means2D,
    means2D_abs,
    means3D,
    num_gausssian_all,
    num_sand_gaussian,
    opacity,
    orthographic,
    ortho_x_min,
    ortho_x_max,
    ortho_y_min,
    ortho_y_max,
    out_all_map_src,
    pc,
    pipe,
    rendered_depth_src,
    rendered_rgb_src,
    rotations,
    sand_span,
    scales,
    scaling_modifier,
    shs,
    tanfovx,
    tanfovy,
    viewpoint_camera,
):
    """Build the immutable source-or-source-plus-sand opaque G-buffer."""

    # A scene without water/rain is complete after opaque composition.
    # Source and sand are deliberately rasterized separately: 3DGS alpha
    # blending is volumetric, so placing both in one pass lets paving and
    # furniture colors/normals leak through an allegedly opaque sand pile.
    if num_sand_gaussian > 0:
        sand_slice = slice(sand_span.start, sand_span.end)
        means3D_sand = means3D[sand_slice]
        means2D_sand = means2D[sand_slice]
        means2D_abs_sand = means2D_abs[sand_slice]
        shs_sand = None if shs is None else shs[sand_slice]
        colors_precomp_sand = (
            None
            if colors_precomp is None
            else colors_precomp[sand_slice]
        )
        opacity_sand = opacity[sand_slice]
        scales_sand = scales[sand_slice]
        rotations_sand = rotations[sand_slice]

        # Sand is a height-field surface with a stable upward orientation.
        # Keep that established normal in the production plane pass; a
        # separately rasterized camera-facing candidate may later recover
        # only well-supported invalid interior pixels.
        stable_normal_sand = pc.get_stored_layer_normals("sand")
        local_normal_sand = (
            stable_normal_sand
            @ viewpoint_camera.world_view_transform[:3, :3]
        )
        pts_in_cam_sand = (
            means3D_sand
            @ viewpoint_camera.world_view_transform[:3, :3]
            + viewpoint_camera.world_view_transform[3, :3]
        )
        local_distance_sand = (
            local_normal_sand * pts_in_cam_sand
        ).sum(-1).abs()
        input_all_map_sand = torch.zeros(
            (num_sand_gaussian, 5),
            dtype=torch.float32,
            device="cuda",
        )
        input_all_map_sand[:, :3] = local_normal_sand
        input_all_map_sand[:, 3] = 1.0
        input_all_map_sand[:, 4] = local_distance_sand

        # Black background makes the returned RGB premultiplied by sand
        # alpha.  all_map channel 3 is the matching accumulated alpha and
        # plane_depth is computed from sand normals/distances only.
        sand_surface_settings = PlaneGaussianRasterizationSettings(
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
        sand_surface_rasterizer = PlaneGaussianRasterizer(
            raster_settings=sand_surface_settings
        )
        (
            sand_rgb_premultiplied,
            _,
            _,
            out_all_map_sand,
            sand_depth,
        ) = sand_surface_rasterizer(
            means3D=means3D_sand,
            means2D=means2D_sand,
            means2D_abs=means2D_abs_sand,
            shs=shs_sand,
            colors_precomp=colors_precomp_sand,
            opacities=opacity_sand,
            scales=scales_sand,
            rotations=rotations_sand,
            all_map=input_all_map_sand,
            cov3D_precomp=None,
        )
        sand_alpha = torch.clamp(out_all_map_sand[3:4], 0.0, 1.0)
        sand_rgb = sand_rgb_premultiplied / torch.clamp(
            sand_alpha, min=1.0e-4
        )

        orientation_recovery_enabled = bool(getattr(
            args, "sand_plane_orientation_recovery", False
        ))
        orientation_recovery_diagnostics_enabled = bool(getattr(
            args,
            "sand_plane_orientation_recovery_diagnostics",
            False,
        ))
        evaluate_orientation_recovery = (
            orientation_recovery_enabled
            or orientation_recovery_diagnostics_enabled
        )
        oriented_sand_depth = None
        oriented_normal_flip_sand = torch.zeros(
            num_sand_gaussian,
            dtype=torch.bool,
            device=local_normal_sand.device,
        )
        if evaluate_orientation_recovery:
            oriented_normal_sand, oriented_normal_flip_sand = (
                orient_plane_normals_toward_camera(
                    local_normal_sand,
                    pts_in_cam_sand,
                    orthographic=orthographic,
                )
            )
            oriented_distance_sand = (
                oriented_normal_sand * pts_in_cam_sand
            ).sum(-1).abs()
            oriented_all_map_sand = torch.zeros_like(input_all_map_sand)
            oriented_all_map_sand[:, :3] = oriented_normal_sand
            oriented_all_map_sand[:, 3] = 1.0
            oriented_all_map_sand[:, 4] = oriented_distance_sand
            # This pass is a candidate generator only. Its RGB and alpha
            # never replace the established sand coverage/material pass.
            _, _, _, _, oriented_sand_depth = sand_surface_rasterizer(
                means3D=means3D_sand,
                means2D=means2D_sand,
                means2D_abs=means2D_abs_sand,
                shs=None,
                colors_precomp=torch.zeros(
                    (num_sand_gaussian, 3),
                    dtype=torch.float32,
                    device=means3D_sand.device,
                ),
                opacities=opacity_sand,
                scales=scales_sand,
                rotations=rotations_sand,
                all_map=oriented_all_map_sand,
                cov3D_precomp=None,
            )

        def rasterize_sand_channels(values):
            rendered, _, _, _, _ = sand_surface_rasterizer(
                means3D=means3D_sand,
                means2D=means2D_sand,
                means2D_abs=means2D_abs_sand,
                shs=None,
                colors_precomp=values,
                opacities=opacity_sand,
                scales=scales_sand,
                rotations=rotations_sand,
                all_map=input_all_map_sand,
                cov3D_precomp=None,
            )
            return rendered / torch.clamp(sand_alpha, min=1.0e-4)

        def rasterize_sand_scalar(values):
            colors = values[:, None].repeat(1, 3)
            return rasterize_sand_channels(colors).mean(
                dim=0, keepdim=True
            )

        def rasterize_sand_subset_alpha(primitive_mask):
            """Rasterize one T2.3 primitive subset with production splats.

            Subsets reuse the exact means, opacity, scales and rotations
            of the production sand pass. Their alphas are diagnostic
            supports; independently rendered subsets are not assumed to
            add up under front-to-back compositing.
            """

            if primitive_mask.shape != (num_sand_gaussian,):
                raise ValueError(
                    "sand primitive mask must match the sand layer"
                )
            if not torch.any(primitive_mask):
                return torch.zeros_like(sand_alpha)
            _, _, _, subset_all_map, _ = sand_surface_rasterizer(
                means3D=means3D_sand[primitive_mask],
                means2D=means2D_sand[primitive_mask],
                means2D_abs=means2D_abs_sand[primitive_mask],
                shs=None,
                colors_precomp=torch.zeros(
                    (int(primitive_mask.sum().item()), 3),
                    dtype=means3D_sand.dtype,
                    device=means3D_sand.device,
                ),
                opacities=opacity_sand[primitive_mask],
                scales=scales_sand[primitive_mask],
                rotations=rotations_sand[primitive_mask],
                all_map=input_all_map_sand[primitive_mask],
                cov3D_precomp=None,
            )
            return torch.clamp(subset_all_map[3:4], 0.0, 1.0)

        # De-premultiplication can overshoot the convex scalar range by a
        # few ulps at faint Gaussian tails.  These are bounded material
        # parameters, so clamp at the rasterizer boundary while the later
        # validation still rejects NaN/Inf and genuine contract failures.
        roughness = torch.clamp(
            rasterize_sand_scalar(pc.sand_roughness_t), 0.0, 1.0
        )
        specular_strength = torch.clamp(
            rasterize_sand_scalar(pc.sand_specular_t), 0.0, 1.0
        )
        wetness = torch.clamp(
            rasterize_sand_scalar(pc.sand_wetness_t), 0.0, 1.0
        )
        normal_map_camera = torch.nn.functional.normalize(
            out_all_map_sand[:3], dim=0, eps=1.0e-6
        )
        normal_map_world = camera_normals_to_world(
            normal_map_camera,
            viewpoint_camera.world_view_transform[:3, :3],
        )
        light_local = (
            pc.sand_light_direction_gs_t
            @ viewpoint_camera.world_view_transform[:3, :3]
        )
        light_local = torch.nn.functional.normalize(
            light_local, dim=0, eps=1.0e-6
        )
        ndotl = torch.clamp(
            (normal_map_camera * light_local[:, None, None]).sum(dim=0),
            min=0.0,
        )
        ambient = float(getattr(args, "sand_ambient", 0.30))
        if not 0.0 <= ambient <= 1.0:
            raise ValueError("sand_ambient must lie in [0,1]")
        directional_diffuse = (
            ambient + (1.0 - ambient) * ndotl
        )[None, ...].repeat(3, 1, 1)

        environment_weight = float(
            getattr(args, "sand_environment_weight", 0.45)
        )
        environment_min = float(
            getattr(args, "sand_environment_min", 0.35)
        )
        environment_max = float(
            getattr(args, "sand_environment_max", 1.75)
        )
        if not 0.0 <= environment_weight <= 1.0:
            raise ValueError("sand_environment_weight must lie in [0,1]")
        environment_diffuse = directional_diffuse
        if environment_weight > 0.0:
            if not hasattr(pc.brdf_mlp_fluid, "sample_diffuse"):
                raise RuntimeError(
                    "P3 environment lighting requires an EnvironmentLight "
                    "with diffuse cubemap sampling"
                )
            # The scene's LDR environment map is an uncalibrated reflection
            # fallback.  Normalize its mean luminance, retaining only its
            # directional/color variation so it cannot arbitrarily change
            # the configured sand exposure.
            environment_srgb = pc.brdf_mlp_fluid.sample_diffuse(
                normal_map_world.movedim(0, -1)[None, ...]
            )[0]
            environment_reference_srgb = pc.brdf_mlp_fluid.diffuse
            environment_linear = normalize_environment_irradiance(
                srgb_to_linear(environment_srgb),
                srgb_to_linear(environment_reference_srgb),
                environment_min,
                environment_max,
            )
            environment_diffuse = environment_linear.movedim(-1, 0)
        diffuse_irradiance = (
            (1.0 - environment_weight) * directional_diffuse
            + environment_weight * environment_diffuse
        )

        # Use the actual perspective ray instead of the old image-wide
        # [0,0,-1] approximation.  P4 will reuse the same direction for
        # Fresnel and the layered water/sand BRDF.
        view_local = perspective_view_directions_camera(
            image_height,
            image_width,
            viewpoint_camera.fx,
            viewpoint_camera.fy,
            viewpoint_camera.cx,
            viewpoint_camera.cy,
            device=normal_map_camera.device,
            dtype=normal_map_camera.dtype,
        )
        view_world = camera_normals_to_world(
            view_local,
            viewpoint_camera.world_view_transform[:3, :3],
        )
        specular_environment_enabled = bool(getattr(
            args, "sand_specular_environment_enabled", False
        ))
        specular_environment_intensity = float(getattr(
            args, "sand_specular_environment_intensity", 1.0
        ))
        specular_environment_min = float(getattr(
            args, "sand_specular_environment_min", 0.0
        ))
        specular_environment_max = float(getattr(
            args, "sand_specular_environment_max", 4.0
        ))
        if (
            not math.isfinite(specular_environment_intensity)
            or specular_environment_intensity < 0.0
        ):
            raise ValueError(
                "sand_specular_environment_intensity must be finite and "
                "non-negative"
            )
        if not (
            math.isfinite(specular_environment_min)
            and math.isfinite(specular_environment_max)
            and 0.0 <= specular_environment_min
            < specular_environment_max
        ):
            raise ValueError(
                "sand specular environment clamp must satisfy finite "
                "0 <= min < max"
            )
        specular_environment_active = (
            specular_environment_enabled
            and specular_environment_intensity > 0.0
        )

        def evaluate_sand_specular_environment(
            shading_normal_world,
            shading_roughness,
            shading_f0,
        ):
            """Evaluate roughness-aware split-sum IBL for opaque sand.

            The scene-owned environment cubemap is sampled at sand pixels after
            sand coverage/depth have been established.  The normal-map
            detail therefore affects appearance without changing owner,
            plane depth, water SSR, or the simulated height field.
            """

            zero = torch.zeros(
                (3, image_height, image_width),
                dtype=shading_normal_world.dtype,
                device=shading_normal_world.device,
            )
            if (
                not specular_environment_active
            ):
                return zero
            environment_light = pc.brdf_mlp_fluid
            if not (
                hasattr(environment_light, "sample_specular")
                and hasattr(environment_light, "sample_fg_lut")
            ):
                raise RuntimeError(
                    "sand specular environment lighting requires an "
                    "EnvironmentLight with split-sum sampling"
                )

            normal_world = torch.nn.functional.normalize(
                shading_normal_world, dim=0, eps=1.0e-6
            )
            raw_ndotv = torch.sum(normal_world * view_world, dim=0)
            front_facing = raw_ndotv > 0.0
            ndotv = torch.clamp(raw_ndotv, min=1.0e-4, max=1.0)
            # ``view_world`` points from the surface toward the camera.
            # Reflecting its opposite incoming direction is equivalently
            # 2(N.V)N-V, the outgoing cubemap lookup direction below.
            reflection_world = torch.nn.functional.normalize(
                2.0 * ndotv[None, ...] * normal_world - view_world,
                dim=0,
                eps=1.0e-6,
            )
            reflection_query = reflection_world.movedim(0, -1)[None, ...]
            roughness_query = torch.clamp(
                shading_roughness, 0.0, 1.0
            ).movedim(0, -1)[None, ...]
            ndotv_query = ndotv[None, ..., None]
            f0_query = torch.clamp(
                shading_f0, 0.0, 1.0
            ).movedim(0, -1)[None, ...]

            environment_srgb = environment_light.sample_specular(
                reflection_query, roughness_query
            )
            environment_linear = normalize_environment_irradiance(
                srgb_to_linear(environment_srgb),
                srgb_to_linear(environment_light.base),
                specular_environment_min,
                specular_environment_max,
            )
            fg_lookup = environment_light.sample_fg_lut(
                ndotv_query, roughness_query
            )
            reflectance = split_sum_specular_reflectance(
                f0_query, fg_lookup
            )
            specular_environment = (
                specular_environment_intensity
                * environment_linear
                * reflectance
            )[0].movedim(-1, 0)
            return (
                specular_environment
                * front_facing[None, ...].to(specular_environment.dtype)
            )

        half_vector = torch.nn.functional.normalize(
            light_local[:, None, None] + view_local,
            dim=0,
            eps=1.0e-6,
        )
        ndoth = torch.clamp(
            (normal_map_camera * half_vector).sum(dim=0),
            min=0.0,
        )
        exponent = torch.clamp(
            2.0 / torch.clamp(roughness, min=0.08) ** 2 - 2.0,
            min=1.0,
            max=256.0,
        )
        specular = specular_strength * torch.pow(
            ndoth[None, ...], exponent
        )
        # A texture-enabled render immediately re-evaluates shading with
        # its sampled roughness and micro-normal below. Avoid a redundant
        # full-resolution cubemap/LUT lookup for the discarded base material.
        if bool(getattr(args, "sand_texture_enabled", False)):
            sand_specular_environment = torch.zeros(
                (3, image_height, image_width),
                dtype=normal_map_world.dtype,
                device=normal_map_world.device,
            )
        else:
            sand_specular_environment = evaluate_sand_specular_environment(
                normal_map_world,
                roughness,
                specular_strength,
            )
        # Preserve the exact legacy expression when the opt-in is off;
        # adding an explicit zero tensor would needlessly perturb the
        # frozen baseline's floating-point operation graph.
        specular_total = specular
        if specular_environment_active:
            specular_total = specular + sand_specular_environment
        # The checkpoint and configured colors are in display/sRGB space.
        # Decode only the new material, evaluate its illumination in linear
        # RGB, then encode it before compositing with PGSR's display-space
        # output.  This avoids the gamma-space multiply that flattened the
        # first opaque-sand correction.
        sand_shaded = torch.clamp(
            linear_to_srgb(
                srgb_to_linear(sand_rgb) * diffuse_irradiance
                + specular_total
            ),
            0.0,
            1.0,
        )
        # Keep explicit production material aliases so the texture-off
        # branch remains byte-for-byte identical.  T4.1 may replace only
        # these BRDF inputs after coverage/depth visibility is resolved;
        # the macroscopic normal above remains the geometry/SSR normal.
        sand_render_albedo = sand_rgb
        sand_render_normal_world = normal_map_world
        sand_render_irradiance = diffuse_irradiance
        sand_texture_ao_raw = torch.ones_like(roughness)
        sand_texture_ao_dark_residual = torch.zeros_like(roughness)
        sand_texture_ao_micro = torch.ones_like(roughness)
        sand_texture_diagnostics = {}
        sand_recovery_attribute_diagnostics = {}

        def evaluate_diagnostic_sand_shading(
            diagnostic_normal_camera,
            diagnostic_normal_world,
            diagnostic_albedo,
            diagnostic_roughness,
            diagnostic_specular_strength,
            diagnostic_ambient_occlusion=None,
        ):
            """Re-evaluate appearance for a diagnostic normal/material.

            This deliberately mirrors the production equations above but
            is called only when the T4 shading-band audit is enabled.  Its
            outputs never feed depth, ownership, water optics, or the final
            image; they are counterfactual views used to identify whether
            a stripe first appears in material, mesh normals, Gaussian
            normal blending, or lighting.
            """

            diagnostic_ndotl = torch.clamp(
                (
                    diagnostic_normal_camera
                    * light_local[:, None, None]
                ).sum(dim=0),
                min=0.0,
            )
            diagnostic_directional = (
                ambient + (1.0 - ambient) * diagnostic_ndotl
            )[None, ...].repeat(3, 1, 1)
            diagnostic_environment = diagnostic_directional
            if environment_weight > 0.0:
                diagnostic_environment_srgb = (
                    pc.brdf_mlp_fluid.sample_diffuse(
                        diagnostic_normal_world.movedim(0, -1)[None, ...]
                    )[0]
                )
                diagnostic_environment = normalize_environment_irradiance(
                    srgb_to_linear(diagnostic_environment_srgb),
                    srgb_to_linear(environment_reference_srgb),
                    environment_min,
                    environment_max,
                ).movedim(-1, 0)
            if diagnostic_ambient_occlusion is None:
                # Preserve the original operation graph when AO is off.
                diagnostic_irradiance = (
                    (1.0 - environment_weight) * diagnostic_directional
                    + environment_weight * diagnostic_environment
                )
            else:
                if (
                    diagnostic_ambient_occlusion.shape
                    != diagnostic_roughness.shape
                ):
                    raise ValueError(
                        "sand ambient occlusion must have shape [1,H,W]"
                    )
                # Split the former directional term into true direct light
                # and its ambient floor. AO attenuates only indirect light:
                # the ambient floor plus the diffuse environment cubemap.
                diagnostic_direct = (
                    (1.0 - environment_weight)
                    * (1.0 - ambient)
                    * diagnostic_ndotl
                )[None, ...].repeat(3, 1, 1)
                diagnostic_indirect = (
                    (1.0 - environment_weight) * ambient
                    + environment_weight * diagnostic_environment
                )
                diagnostic_irradiance = (
                    diagnostic_direct
                    + diagnostic_ambient_occlusion * diagnostic_indirect
                )
            diagnostic_ndoth = torch.clamp(
                (
                    diagnostic_normal_camera * half_vector
                ).sum(dim=0),
                min=0.0,
            )
            diagnostic_exponent = torch.clamp(
                2.0
                / torch.clamp(diagnostic_roughness, min=0.08) ** 2
                - 2.0,
                min=1.0,
                max=256.0,
            )
            diagnostic_specular = diagnostic_specular_strength * torch.pow(
                diagnostic_ndoth[None, ...], diagnostic_exponent
            )
            diagnostic_specular_environment = (
                evaluate_sand_specular_environment(
                    diagnostic_normal_world,
                    diagnostic_roughness,
                    diagnostic_specular_strength,
                )
            )
            diagnostic_specular_total = diagnostic_specular
            if specular_environment_active:
                diagnostic_specular_total = (
                    diagnostic_specular
                    + diagnostic_specular_environment
                )
            diagnostic_diffuse_linear = (
                srgb_to_linear(diagnostic_albedo) * diagnostic_irradiance
            )
            diagnostic_shaded = torch.clamp(
                linear_to_srgb(
                    diagnostic_diffuse_linear + diagnostic_specular_total
                ),
                0.0,
                1.0,
            )
            return {
                "shaded": diagnostic_shaded,
                "directional": diagnostic_directional,
                "environment": diagnostic_environment,
                "irradiance": diagnostic_irradiance,
                "specular": diagnostic_specular_total,
                "specular_direct": diagnostic_specular,
                "specular_environment": diagnostic_specular_environment,
                "diffuse_linear": diagnostic_diffuse_linear,
            }

        alpha_low = float(
            getattr(
                args,
                "sand_coverage_alpha_low",
                getattr(args, "sand_opaque_alpha_low", 0.02),
            )
        )
        alpha_high = float(
            getattr(
                args,
                "sand_coverage_alpha_high",
                getattr(args, "sand_opaque_alpha_high", 0.80),
            )
        )
        # This is geometric pixel coverage, not sand transparency.  The
        # wider upper threshold preserves a sub-pixel Gaussian silhouette
        # while the densely covered interior still reaches exactly one.
        unguided_surface_coverage = opaque_surface_coverage(
            sand_alpha, alpha_low, alpha_high
        )
        surface_coverage = unguided_surface_coverage

        # T2.1--T2.4 share one independent triangle reference. Build it
        # before RGB/owner composition so the opt-in T2.4 branch can use
        # the physical soft target as a production contour constraint.
        # The Gaussian material/depth passes above remain unchanged.
        triangle_footprint_diagnostics = {}
        triangle_reference_enabled = bool(getattr(
            args, "sand_triangle_reference_diagnostics", False
        ))
        soft_boundary_enabled = bool(getattr(
            args, "sand_soft_boundary_diagnostics", False
        ))
        primitive_attribution_enabled = bool(getattr(
            args, "sand_primitive_attribution_diagnostics", False
        ))
        soft_boundary_guidance_enabled = bool(getattr(
            args, "sand_soft_boundary_guidance", False
        ))
        t3_prerequisite_diagnostics_enabled = bool(getattr(
            args, "sand_t3_prerequisite_diagnostics", False
        ))
        physical_reference_depth_diagnostics_enabled = bool(getattr(
            args,
            "sand_physical_reference_depth_diagnostics",
            False,
        ))
        physical_reference_depth_recovery_enabled = bool(getattr(
            args,
            "sand_physical_reference_depth_recovery",
            False,
        ))
        physical_reference_normal_recovery_enabled = bool(getattr(
            args,
            "sand_physical_reference_normal_recovery",
            False,
        ))
        physical_reference_attribute_diagnostics_enabled = bool(getattr(
            args,
            "sand_physical_reference_attribute_diagnostics",
            False,
        ))
        center_trusted_invalid_depth_diagnostics_enabled = bool(getattr(
            args,
            "sand_center_trusted_invalid_depth_diagnostics",
            False,
        ))
        center_trusted_invalid_depth_recovery_enabled = bool(getattr(
            args,
            "sand_center_trusted_invalid_depth_recovery",
            False,
        ))
        center_trusted_invalid_normal_recovery_enabled = bool(getattr(
            args,
            "sand_center_trusted_invalid_normal_recovery",
            False,
        ))
        center_trusted_invalid_attribute_diagnostics_enabled = bool(
            getattr(
                args,
                "sand_center_trusted_invalid_attribute_diagnostics",
                False,
            )
        )
        evaluate_center_trusted_invalid_depth = (
            center_trusted_invalid_depth_diagnostics_enabled
            or center_trusted_invalid_depth_recovery_enabled
            or center_trusted_invalid_normal_recovery_enabled
            or center_trusted_invalid_attribute_diagnostics_enabled
        )
        evaluate_physical_reference_depth = (
            physical_reference_depth_diagnostics_enabled
            or physical_reference_depth_recovery_enabled
            or physical_reference_normal_recovery_enabled
            or physical_reference_attribute_diagnostics_enabled
        )
        evaluate_independent_depth_reference = (
            evaluate_physical_reference_depth
            or evaluate_center_trusted_invalid_depth
        )
        triangle_reference = None
        target_soft_coverage = None
        if (
            t3_prerequisite_diagnostics_enabled
            and not soft_boundary_guidance_enabled
        ):
            raise ValueError(
                "sand_t3_prerequisite_diagnostics requires the frozen "
                "T2.4 sand_soft_boundary_guidance baseline"
            )
        if soft_boundary_guidance_enabled and (
            triangle_reference_enabled
            or soft_boundary_enabled
            or primitive_attribution_enabled
        ):
            raise ValueError(
                "sand_soft_boundary_guidance already emits the shared "
                "triangle/soft-target products; do not combine it with "
                "T2.1--T2.3 diagnostic switches in one branch"
            )
        if (
            evaluate_independent_depth_reference
            and not soft_boundary_guidance_enabled
        ):
            raise ValueError(
                "sand independent depth-reference diagnostics/recovery "
                "require sand_soft_boundary_guidance so they can reuse "
                "the frozen T2.4 physical triangle reference"
            )
        if soft_boundary_guidance_enabled:
            reference_triangles = getattr(
                pc,
                "sand_reference_triangles_gs_t",
                torch.empty(
                    (0, 3, 3), dtype=means3D_sand.dtype,
                    device=means3D_sand.device,
                ),
            )
            reference_vertex_heights = getattr(
                pc,
                "sand_reference_vertex_heights_t",
                torch.empty(
                    (0, 3), dtype=means3D_sand.dtype,
                    device=means3D_sand.device,
                ),
            )
            reference_face_wetness = getattr(
                pc,
                "sand_reference_face_wetness_t",
                torch.empty(
                    0,
                    dtype=means3D_sand.dtype,
                    device=means3D_sand.device,
                ),
            )
            if reference_triangles.shape[0] == 0:
                raise ValueError(
                    "sand triangle boundary features require triangle_games"
                )
            triangle_reference = rasterize_triangle_reference(
                reference_triangles,
                viewpoint_camera.world_view_transform,
                full_proj_transform,
                image_height,
                image_width,
                reference_vertex_heights,
                reference_face_wetness,
            )
            triangle_footprint_diagnostics.update(triangle_reference)

            if soft_boundary_guidance_enabled:
                sand_epsilon = float(getattr(
                    pc, "sand_reference_sand_epsilon", 0.0
                ))
                fade_height = float(getattr(
                    args,
                    "sand_boundary_fade_height",
                    getattr(
                        pc,
                        "sand_reference_full_coverage_height",
                        1.0,
                    ),
                ))
                target_soft_coverage = sand_height_to_soft_coverage(
                    triangle_reference[
                        "sand_triangle_reference_sand_height_raw"
                    ],
                    triangle_reference[
                        "sand_triangle_reference_coverage"
                    ],
                    sand_epsilon=sand_epsilon,
                    fade_height=fade_height,
                )
                # Always compare against the original Gaussian silhouette;
                # otherwise a guided branch would trivially report that it
                # has no overcoverage after clipping itself to the target.
                triangle_footprint_diagnostics.update(
                    compare_sand_soft_boundary(
                        target_soft_coverage,
                        unguided_surface_coverage,
                    )
                )
                triangle_footprint_diagnostics.update({
                    "sand_boundary_sand_epsilon": sand_epsilon,
                    "sand_boundary_fade_height": fade_height,
                })

        if soft_boundary_guidance_enabled:
            if target_soft_coverage is None:
                raise RuntimeError(
                    "sand soft-boundary guidance has no physical target"
                )
            guidance_diagnostics = (
                guide_sand_coverage_with_soft_boundary(
                    unguided_surface_coverage,
                    target_soft_coverage,
                    enabled=True,
                )
            )
            surface_coverage = guidance_diagnostics[
                "sand_guided_coverage"
            ]
            triangle_footprint_diagnostics.update(guidance_diagnostics)
            triangle_footprint_diagnostics[
                "sand_soft_boundary_guidance_enabled"
            ] = True

        source_alpha_min = float(
            getattr(args, "sand_source_alpha_min", 0.01)
        )
        depth_bias = float(getattr(args, "sand_depth_bias", 0.002))
        depth_softness = float(
            getattr(args, "sand_depth_softness", 0.005)
        )
        if (
            not 0.0 <= source_alpha_min <= 1.0
            or depth_bias < 0.0
            or depth_softness < 0.0
        ):
            raise ValueError(
                "sand depth-composition parameters must be non-negative"
            )
        source_alpha = torch.clamp(out_all_map_src[3:4], 0.0, 1.0)
        source_depth = rendered_depth_src[:1]
        valid_source = (
            (source_alpha > source_alpha_min)
            & torch.isfinite(source_depth)
            & (source_depth > 0.0)
        )
        fallback_enabled = bool(
            getattr(args, "sand_plane_depth_stability_fallback", False)
        )
        save_sand_debug_maps = bool(
            getattr(args, "sand_save_debug_maps", False)
        )
        need_center_depth_reference = (
            fallback_enabled
            or evaluate_orientation_recovery
            or save_sand_debug_maps
            or t3_prerequisite_diagnostics_enabled
            or evaluate_independent_depth_reference
        )
        sand_plane_depth_valid = (
            torch.isfinite(sand_depth) & (sand_depth > 0.0)
        )

        sand_center_depth = None
        sand_center_depth_std = None
        sand_fallback_diagnostics = None
        sand_orientation_recovery_diagnostics = {}
        resolved_sand_depth = sand_depth
        if need_center_depth_reference:
            # One RGB pass carries E[z] and E[z^2]. Their variance detects
            # pixels whose center-depth reference itself mixes multiple
            # sand layers, in which case it must not override plane depth.
            center_z = pts_in_cam_sand[:, 2]
            center_moments = rasterize_sand_channels(
                torch.stack(
                    (
                        center_z,
                        center_z * center_z,
                        torch.zeros_like(center_z),
                    ),
                    dim=1,
                )
            )
            sand_center_depth = center_moments[0:1]
            center_second_moment = center_moments[1:2]
            center_variance = torch.clamp(
                center_second_moment
                - sand_center_depth * sand_center_depth,
                min=0.0,
            )
            sand_center_depth_std = torch.sqrt(center_variance)
            sand_fallback_diagnostics = resolve_sand_plane_depth_stability(
                sand_depth,
                sand_center_depth,
                sand_center_depth_std,
                source_depth,
                valid_source,
                unguided_surface_coverage,
                enabled=fallback_enabled,
                depth_bias=depth_bias,
                depth_softness=depth_softness,
                coverage_min=float(getattr(
                    args, "sand_plane_fallback_coverage_min", 0.95
                )),
                erode_kernel=int(getattr(
                    args, "sand_plane_fallback_erode_kernel", 5
                )),
                sigma_scale=float(getattr(
                    args, "sand_plane_fallback_sigma_scale", 3.0
                )),
                relative_margin=float(getattr(
                    args, "sand_plane_fallback_relative_margin", 0.002
                )),
                center_std_relative_max=float(getattr(
                    args,
                    "sand_plane_fallback_center_std_relative_max",
                    0.01,
                )),
            )
            resolved_sand_depth = sand_fallback_diagnostics[
                "sand_plane_fallback_resolved_depth"
            ]
            if evaluate_orientation_recovery:
                original_invalid_depth = (
                    (sand_alpha > alpha_low)
                    & (~sand_plane_depth_valid)
                )
                sand_orientation_recovery_diagnostics = (
                    resolve_sand_oriented_plane_depth(
                        sand_depth,
                        original_invalid_depth,
                        oriented_sand_depth,
                        sand_center_depth,
                        sand_center_depth_std,
                        source_depth,
                        valid_source,
                        unguided_surface_coverage,
                        enabled=orientation_recovery_enabled,
                        depth_bias=depth_bias,
                        depth_softness=depth_softness,
                        coverage_min=float(getattr(
                            args,
                            "sand_orientation_recovery_coverage_min",
                            0.95,
                        )),
                        erode_kernel=int(getattr(
                            args,
                            "sand_orientation_recovery_erode_kernel",
                            5,
                        )),
                        sigma_scale=float(getattr(
                            args,
                            "sand_orientation_recovery_sigma_scale",
                            3.0,
                        )),
                        relative_margin=float(getattr(
                            args,
                            "sand_orientation_recovery_relative_margin",
                            0.002,
                        )),
                        center_std_relative_max=float(getattr(
                            args,
                            "sand_orientation_recovery_center_std_relative_max",
                            0.01,
                        )),
                    )
                )
                if orientation_recovery_enabled:
                    orientation_used = (
                        sand_orientation_recovery_diagnostics[
                            "sand_orientation_recovery_used"
                        ]
                    )
                    orientation_resolved_depth = (
                        sand_orientation_recovery_diagnostics[
                            "sand_orientation_recovery_resolved_depth"
                        ]
                    )
                    # Both selectors are evaluated independently against
                    # the same original plane/source evidence, so enabling
                    # both cannot make one rule qualify pixels for the
                    # other.  Their accepted sets are combined only here:
                    # center fallback handles its remaining candidates,
                    # while the more geometric oriented-plane estimate wins
                    # where both conservative rules accepted the pixel.
                    resolved_sand_depth = torch.where(
                        orientation_used,
                        orientation_resolved_depth,
                        resolved_sand_depth,
                    )

        resolved_sand_depth_valid = (
            torch.isfinite(resolved_sand_depth)
            & (resolved_sand_depth > 0.0)
        )
        valid_sand = (
            (sand_alpha > alpha_low) & resolved_sand_depth_valid
        )
        # Keep the original plane-only decision for diagnostics, then use
        # the resolved depth consistently for RGB, owner and the P4 opaque
        # G-buffer. Positive delta means sand lies in front.
        plane_depth_delta = source_depth - sand_depth + depth_bias
        depth_delta = source_depth - resolved_sand_depth + depth_bias

        def depth_front_weight(delta, depth_valid):
            if depth_softness > 0.0:
                weight = torch.clamp(
                    0.5 + 0.5 * delta / depth_softness,
                    0.0,
                    1.0,
                )
                weight = weight * weight * (3.0 - 2.0 * weight)
            else:
                weight = (delta >= 0.0).to(torch.float32)
            weight = torch.where(
                valid_source, weight, torch.ones_like(weight)
            )
            return torch.where(
                depth_valid, weight, torch.zeros_like(weight)
            )

        plane_valid_sand = (
            (sand_alpha > alpha_low) & sand_plane_depth_valid
        )
        plane_sand_in_front = depth_front_weight(
            plane_depth_delta, plane_valid_sand
        )
        sand_in_front = depth_front_weight(depth_delta, valid_sand)
        sand_composite_mask = surface_coverage * sand_in_front

        center_trusted_invalid_depth_diagnostics = {}
        if evaluate_center_trusted_invalid_depth:
            if (
                triangle_reference is None
                or target_soft_coverage is None
                or sand_center_depth is None
                or sand_fallback_diagnostics is None
            ):
                raise RuntimeError(
                    "center-trusted invalid-depth diagnosis is missing "
                    "its triangle, soft-coverage, center, or fallback evidence"
                )
            center_trusted_invalid_depth_diagnostics = (
                classify_sand_center_trusted_invalid_depth(
                    target_soft_coverage,
                    surface_coverage,
                    resolved_sand_depth_valid,
                    sand_center_depth,
                    sand_fallback_diagnostics[
                        "sand_plane_fallback_margin_raw"
                    ],
                    sand_fallback_diagnostics[
                        "sand_plane_fallback_reference_reliable"
                    ],
                    triangle_reference[
                        "sand_triangle_reference_depth_raw"
                    ],
                    triangle_reference[
                        "sand_triangle_reference_coverage"
                    ],
                    source_depth,
                    valid_source,
                    depth_bias=depth_bias,
                    depth_softness=depth_softness,
                    physical_core_min=float(getattr(
                        args,
                        "sand_center_trusted_invalid_physical_core_min",
                        0.5,
                    )),
                    owner_coverage_min=float(getattr(
                        args,
                        "sand_center_trusted_invalid_owner_coverage_min",
                        0.5,
                    )),
                    relative_margin=float(getattr(
                        args,
                        "sand_center_trusted_invalid_relative_margin",
                        0.002,
                    )),
                )
            )
            center_trusted_candidate = (
                center_trusted_invalid_depth_diagnostics[
                    "sand_center_trusted_invalid_candidate"
                ]
            )
            # Explain why the already-enabled center fallback did not
            # resolve each otherwise trusted core hole. Intersections keep
            # these reason maps disjoint from unrelated fallback pixels.
            fallback_reason_keys = {
                "boundary": "sand_plane_fallback_rejected_by_boundary",
                "uncertain_center": (
                    "sand_plane_fallback_rejected_by_uncertain_center"
                ),
                "adaptive_margin": (
                    "sand_plane_fallback_rejected_by_adaptive_margin"
                ),
                "used": "sand_plane_fallback_used",
            }
            accounted = torch.zeros_like(center_trusted_candidate)
            for reason, fallback_key in fallback_reason_keys.items():
                overlap = (
                    center_trusted_candidate
                    & sand_fallback_diagnostics[fallback_key]
                )
                center_trusted_invalid_depth_diagnostics[
                    f"sand_center_trusted_invalid_fallback_{reason}"
                ] = overlap
                accounted = accounted | overlap
            center_trusted_invalid_depth_diagnostics[
                "sand_center_trusted_invalid_fallback_unaccounted"
            ] = center_trusted_candidate & (~accounted)
        physical_reference_depth_diagnostics = {}
        if evaluate_physical_reference_depth:
            if (
                triangle_reference is None
                or target_soft_coverage is None
                or sand_center_depth is None
                or sand_fallback_diagnostics is None
            ):
                raise RuntimeError(
                    "physical-reference depth diagnosis is missing its "
                    "triangle, soft-coverage, or center-depth evidence"
                )
            # Reclassify only the previously observed blue population.
            # The direct mesh z-buffer is independent of the Gaussian
            # plane/normal mixture, so it can distinguish real rear thin
            # sand from a source surface that genuinely occludes it.
            physical_reference_depth_diagnostics = (
                classify_sand_physical_reference_depth_loss(
                    target_soft_coverage,
                    surface_coverage,
                    sand_in_front,
                    sand_center_depth,
                    sand_fallback_diagnostics[
                        "sand_plane_fallback_margin_raw"
                    ],
                    triangle_reference[
                        "sand_triangle_reference_depth_raw"
                    ],
                    triangle_reference[
                        "sand_triangle_reference_coverage"
                    ],
                    source_depth,
                    valid_source,
                    depth_bias=depth_bias,
                    depth_softness=depth_softness,
                    support_min=float(getattr(
                        args,
                        "sand_physical_reference_support_min",
                        0.02,
                    )),
                    rejected_in_front_max=float(getattr(
                        args,
                        "sand_physical_reference_rejected_front_max",
                        0.02,
                    )),
                    relative_margin=float(getattr(
                        args,
                        "sand_physical_reference_relative_margin",
                        0.002,
                    )),
                )
            )
            physical_reference_depth_diagnostics.update(
                resolve_sand_physical_reference_depth(
                    resolved_sand_depth,
                    triangle_reference[
                        "sand_triangle_reference_depth_raw"
                    ],
                    physical_reference_depth_diagnostics[
                        "sand_physical_reference_sand_front"
                    ],
                    enabled=physical_reference_depth_recovery_enabled,
                )
            )

            baseline_sand_in_front = sand_in_front
            if physical_reference_depth_recovery_enabled:
                # Re-enter the shared depth/coverage/owner pipeline with
                # the trusted mesh depth. This updates RGB, discrete owner,
                # and the P4 opaque G-buffer together; it is not a late
                # color patch. Red source occlusions and yellow near-plane
                # pixels never enter ``recovery_used`` above.
                resolved_sand_depth = physical_reference_depth_diagnostics[
                    "sand_physical_reference_recovery_resolved_depth"
                ]
                resolved_sand_depth_valid = (
                    torch.isfinite(resolved_sand_depth)
                    & (resolved_sand_depth > 0.0)
                )
                valid_sand = (
                    (sand_alpha > alpha_low)
                    & resolved_sand_depth_valid
                )
                depth_delta = (
                    source_depth - resolved_sand_depth + depth_bias
                )
                sand_in_front = depth_front_weight(
                    depth_delta, valid_sand
                )
                sand_composite_mask = surface_coverage * sand_in_front

            added_in_front = torch.clamp(
                sand_in_front - baseline_sand_in_front,
                min=0.0,
                max=1.0,
            )
            physical_reference_depth_diagnostics.update({
                "sand_physical_reference_recovery_added_in_front": (
                    added_in_front
                ),
                "sand_physical_reference_recovery_added_coverage": (
                    surface_coverage * added_in_front
                ),
                "sand_physical_reference_recovery_enabled": (
                    physical_reference_depth_recovery_enabled
                ),
            })

        if evaluate_center_trusted_invalid_depth:
            # The first physical recovery and this core-hole recovery are
            # diagnosed from the same pre-recovery buffers and are
            # disjoint by center-order confidence. Apply the second mask
            # after the first so both depth substitutions enter one shared
            # RGB/owner/P4 G-buffer decision without using center z.
            center_recovery_baseline_in_front = sand_in_front
            # Resolve against the depth *after* the first physical
            # recovery.  This preserves its disjoint thin-sand repairs
            # when both production switches are enabled.
            center_trusted_invalid_depth_diagnostics.update(
                resolve_sand_center_trusted_invalid_depth(
                    resolved_sand_depth,
                    triangle_reference[
                        "sand_triangle_reference_depth_raw"
                    ],
                    center_trusted_invalid_depth_diagnostics[
                        "sand_center_trusted_invalid_triangle_sand_front"
                    ],
                    enabled=(
                        center_trusted_invalid_depth_recovery_enabled
                    ),
                )
            )
            if center_trusted_invalid_depth_recovery_enabled:
                resolved_sand_depth = (
                    center_trusted_invalid_depth_diagnostics[
                        "sand_center_trusted_invalid_recovery_resolved_depth"
                    ]
                )
                resolved_sand_depth_valid = (
                    torch.isfinite(resolved_sand_depth)
                    & (resolved_sand_depth > 0.0)
                )
                valid_sand = (
                    (sand_alpha > alpha_low)
                    & resolved_sand_depth_valid
                )
                depth_delta = (
                    source_depth - resolved_sand_depth + depth_bias
                )
                sand_in_front = depth_front_weight(
                    depth_delta, valid_sand
                )
                sand_composite_mask = surface_coverage * sand_in_front

            center_recovery_added_in_front = torch.clamp(
                sand_in_front - center_recovery_baseline_in_front,
                min=0.0,
                max=1.0,
            )
            center_trusted_invalid_depth_diagnostics.update({
                "sand_center_trusted_invalid_recovery_added_in_front": (
                    center_recovery_added_in_front
                ),
                "sand_center_trusted_invalid_recovery_added_coverage": (
                    surface_coverage * center_recovery_added_in_front
                ),
                "sand_center_trusted_invalid_recovery_enabled": (
                    center_trusted_invalid_depth_recovery_enabled
                ),
            })

        # Both conservative depth recoveries write the independently
        # rasterized triangle depth. Optionally take the macro normal from
        # that same primitive on each recovery's exact mask; otherwise a
        # foreground/background Gaussian mixture can retain a contaminated
        # shading normal at the newly restored pixel. Their switches and
        # diagnostics remain separate because the first recovery includes
        # much fainter soft-boundary coverage than the second.
        resolved_sand_macro_normal_world = normal_map_world
        resolved_sand_macro_normal_camera = normal_map_camera
        combined_normal_recovery_used = torch.zeros_like(
            surface_coverage, dtype=torch.bool
        )
        if evaluate_physical_reference_depth:
            physical_normal_recovery = (
                resolve_sand_physical_reference_normal(
                    resolved_sand_macro_normal_world,
                    triangle_reference[
                        "sand_triangle_reference_normal_world_raw"
                    ],
                    physical_reference_depth_diagnostics[
                        "sand_physical_reference_recovery_used"
                    ],
                    enabled=physical_reference_normal_recovery_enabled,
                )
            )
            physical_reference_depth_diagnostics.update(
                physical_normal_recovery
            )
            physical_normal_candidate = physical_normal_recovery[
                "sand_physical_reference_normal_recovery_candidate"
            ]
            physical_normal_candidate_rgb = (
                physical_normal_candidate.repeat(3, 1, 1)
            )
            physical_reference_depth_diagnostics.update({
                "sand_physical_reference_normal_recovery_original": (
                    (normal_map_world * 0.5 + 0.5)
                    * physical_normal_candidate_rgb
                ),
                "sand_physical_reference_normal_recovery_resolved_visualization": (
                    (
                        physical_normal_recovery[
                            "sand_physical_reference_normal_recovery_resolved"
                        ]
                        * 0.5
                        + 0.5
                    )
                    * physical_normal_candidate_rgb
                ),
                "sand_physical_reference_normal_recovery_angle_visualization": (
                    torch.clamp(
                        physical_normal_recovery[
                            "sand_physical_reference_normal_recovery_angle_degrees_raw"
                        ]
                        / 90.0,
                        0.0,
                        1.0,
                    ).repeat(3, 1, 1)
                ),
            })
            resolved_sand_macro_normal_world = physical_normal_recovery[
                "sand_physical_reference_normal_recovery_resolved"
            ]
            combined_normal_recovery_used = (
                combined_normal_recovery_used
                | physical_normal_recovery[
                    "sand_physical_reference_normal_recovery_used"
                ]
            )

        if evaluate_center_trusted_invalid_depth:
            normal_recovery = resolve_sand_center_trusted_invalid_normal(
                resolved_sand_macro_normal_world,
                triangle_reference[
                    "sand_triangle_reference_normal_world_raw"
                ],
                center_trusted_invalid_depth_diagnostics[
                    "sand_center_trusted_invalid_recovery_used"
                ],
                enabled=(
                    center_trusted_invalid_normal_recovery_enabled
                ),
            )
            center_trusted_invalid_depth_diagnostics.update(
                normal_recovery
            )
            normal_recovery_candidate = normal_recovery[
                "sand_center_trusted_invalid_normal_recovery_candidate"
            ]
            normal_recovery_candidate_rgb = (
                normal_recovery_candidate.repeat(3, 1, 1)
            )
            center_trusted_invalid_depth_diagnostics.update({
                "sand_center_trusted_invalid_normal_recovery_original": (
                    (normal_map_world * 0.5 + 0.5)
                    * normal_recovery_candidate_rgb
                ),
                "sand_center_trusted_invalid_normal_recovery_resolved_visualization": (
                    (
                        normal_recovery[
                            "sand_center_trusted_invalid_normal_recovery_resolved"
                        ]
                        * 0.5
                        + 0.5
                    )
                    * normal_recovery_candidate_rgb
                ),
                "sand_center_trusted_invalid_normal_recovery_angle_visualization": (
                    torch.clamp(
                        normal_recovery[
                            "sand_center_trusted_invalid_normal_recovery_angle_degrees_raw"
                        ]
                        / 90.0,
                        0.0,
                        1.0,
                    ).repeat(3, 1, 1)
                ),
            })
            resolved_sand_macro_normal_world = normal_recovery[
                "sand_center_trusted_invalid_normal_recovery_resolved"
            ]
            combined_normal_recovery_used = (
                combined_normal_recovery_used
                | normal_recovery[
                    "sand_center_trusted_invalid_normal_recovery_used"
                ]
            )

        any_normal_recovery_enabled = (
            physical_reference_normal_recovery_enabled
            or center_trusted_invalid_normal_recovery_enabled
        )
        if any_normal_recovery_enabled:
            resolved_sand_macro_normal_camera = (
                torch.nn.functional.normalize(
                    torch.matmul(
                        resolved_sand_macro_normal_world.movedim(0, -1),
                        viewpoint_camera.world_view_transform[:3, :3],
                    ).movedim(-1, 0),
                    dim=0,
                    eps=1.0e-6,
                )
            )

        # Texture-enabled branches re-evaluate all material shading below.
        # Preserve complete behavior for texture-off sand too, but splice
        # the corrected result only into the union of exact recovery masks
        # so all other pixels retain the frozen operation graph.
        if (
            any_normal_recovery_enabled
            and not bool(getattr(args, "sand_texture_enabled", False))
        ):
            recovered_base_shading = evaluate_diagnostic_sand_shading(
                resolved_sand_macro_normal_camera,
                resolved_sand_macro_normal_world,
                sand_rgb,
                roughness,
                specular_strength,
            )
            recovered_normal_rgb = combined_normal_recovery_used.repeat(
                3, 1, 1
            )
            sand_shaded = torch.where(
                recovered_normal_rgb,
                recovered_base_shading["shaded"],
                sand_shaded,
            )
            sand_render_normal_world = resolved_sand_macro_normal_world
            sand_render_irradiance = torch.where(
                recovered_normal_rgb,
                recovered_base_shading["irradiance"],
                sand_render_irradiance,
            )
            specular_total = torch.where(
                recovered_normal_rgb,
                recovered_base_shading["specular"],
                specular_total,
            )
            sand_specular_environment = torch.where(
                recovered_normal_rgb,
                recovered_base_shading["specular_environment"],
                sand_specular_environment,
            )

        if bool(getattr(args, "sand_texture_enabled", False)):
            if orthographic:
                raise ValueError(
                    "T4.1 sand textures currently require a perspective camera"
                )
            texture_root = resolve_material_root(
                getattr(args, "sand_texture_root")
            )
            texture_cache_key = (
                str(texture_root),
                str(means3D_sand.device),
                str(means3D_sand.dtype),
                bool(getattr(
                    args, "sand_texture_normal_highpass_enabled", False
                )),
                float(getattr(
                    args,
                    "sand_texture_normal_highpass_sigma_texels",
                    8.0,
                )),
                float(getattr(
                    args,
                    "sand_texture_normal_highpass_strength",
                    1.0,
                )),
                bool(getattr(args, "sand_texture_ao_enabled", False)),
                float(getattr(
                    args, "sand_texture_ao_highpass_sigma_texels", 8.0
                )),
            )
            cached_texture = getattr(pc, "_sand_pbr_texture_cache", None)
            if (
                cached_texture is None
                or cached_texture[0] != texture_cache_key
            ):
                # Cache one GPU mip pyramid on the Gaussian model. Views
                # and frames reuse it; switching profiles replaces it
                # instead of accumulating candidate assets in VRAM.
                cached_texture = (
                    texture_cache_key,
                    SandPBRMaterial.load(
                        texture_root,
                        device=means3D_sand.device,
                        dtype=means3D_sand.dtype,
                        normal_highpass_enabled=bool(getattr(
                            args,
                            "sand_texture_normal_highpass_enabled",
                            False,
                        )),
                        normal_highpass_sigma_texels=float(getattr(
                            args,
                            "sand_texture_normal_highpass_sigma_texels",
                            8.0,
                        )),
                        normal_highpass_strength=float(getattr(
                            args,
                            "sand_texture_normal_highpass_strength",
                            1.0,
                        )),
                        ambient_occlusion_enabled=bool(getattr(
                            args, "sand_texture_ao_enabled", False
                        )),
                        ambient_occlusion_highpass_sigma_texels=float(getattr(
                            args,
                            "sand_texture_ao_highpass_sigma_texels",
                            8.0,
                        )),
                    ),
                )
                pc._sand_pbr_texture_cache = cached_texture
            sand_material = cached_texture[1]

            # This mask follows the continuous RGB contribution, not the
            # discrete G-buffer owner. Consequently antialiased sand edges
            # receive the same material without allowing texture detail to
            # manufacture depth, ownership or opacity outside the sand.
            texture_valid = (
                valid_sand
                & (surface_coverage > 1.0e-6)
                & (sand_in_front > 0.0)
            )
            sim_to_gs = torch.as_tensor(
                pc.sand_water_adapter.sim_to_gs,
                dtype=means3D_sand.dtype,
                device=means3D_sand.device,
            )
            sand_world_position = reconstruct_world_positions(
                resolved_sand_depth,
                viewpoint_camera.fx,
                viewpoint_camera.fy,
                viewpoint_camera.cx,
                viewpoint_camera.cy,
                viewpoint_camera.world_view_transform,
            )
            sand_sim_position = world_positions_to_simulation(
                sand_world_position, sim_to_gs
            )
            texture_period = float(getattr(
                args, "sand_texture_period_sim", 0.10
            ))
            texture_rotation = float(getattr(
                args, "sand_texture_rotation_degrees", 0.0
            ))
            texture_offset = getattr(
                args, "sand_texture_offset", [0.0, 0.0]
            )
            texture_lod_bias = float(getattr(
                args, "sand_texture_lod_bias", 0.0
            ))
            texture_normal_strength = float(getattr(
                args,
                "sand_texture_normal_strength",
                (
                    1.0
                    if getattr(
                        args,
                        "sand_texture_material_mode",
                        "perturbation",
                    )
                    == "texture_primary"
                    else 0.50
                ),
            ))
            texture_projection = str(getattr(
                args, "sand_texture_projection", "simulation_xy"
            ))
            if texture_projection == "simulation_xy":
                texture_uv = simulation_xy_uv(
                    sand_sim_position,
                    period=texture_period,
                    rotation_degrees=texture_rotation,
                    offset=texture_offset,
                )
                # Invalid plane-depth pixels may contain sentinel positions.
                # Sanitize them before finite differences; texture_valid
                # still provides the semantic mask used by sampling.
                texture_uv = torch.nan_to_num(
                    texture_uv, nan=0.0, posinf=0.0, neginf=0.0
                )
                sampled_texture = sample_sand_pbr_material(
                    sand_material,
                    texture_uv,
                    texture_valid,
                    lod_bias=texture_lod_bias,
                )
            elif texture_projection == "simulation_triplanar":
                texture_uv = None
                sampled_texture = sample_sand_pbr_material_triplanar(
                    sand_material,
                    sand_sim_position,
                    resolved_sand_macro_normal_world,
                    sim_to_gs,
                    texture_valid,
                    period=texture_period,
                    rotation_degrees=texture_rotation,
                    offset=texture_offset,
                    sharpness=float(getattr(
                        args,
                        "sand_texture_triplanar_sharpness",
                        4.0,
                    )),
                    lod_bias=texture_lod_bias,
                    normal_strength=texture_normal_strength,
                )
            else:
                raise ValueError(
                    "sand texture projection must be simulation_xy or "
                    "simulation_triplanar"
                )
            sand_texture_ao_raw = sampled_texture[
                "ambient_occlusion_raw"
            ]
            sand_texture_ao_dark_residual = sampled_texture[
                "ambient_occlusion_dark_residual"
            ]
            ambient_occlusion_enabled = bool(getattr(
                args, "sand_texture_ao_enabled", False
            ))
            if ambient_occlusion_enabled:
                sand_texture_ao_micro = ambient_occlusion_from_residual(
                    sand_texture_ao_dark_residual,
                    strength=float(getattr(
                        args, "sand_texture_ao_strength", 1.0
                    )),
                    minimum=float(getattr(
                        args, "sand_texture_ao_min", 0.4
                    )),
                    response=str(getattr(
                        args, "sand_texture_ao_response", "linear"
                    )),
                    sparse_low=float(getattr(
                        args, "sand_texture_ao_sparse_low", 0.04
                    )),
                    sparse_high=float(getattr(
                        args, "sand_texture_ao_sparse_high", 0.08
                    )),
                    sparse_contrast=float(getattr(
                        args, "sand_texture_ao_sparse_contrast", 0.40
                    )),
                )
            texture_material_mode = getattr(
                args, "sand_texture_material_mode", "perturbation"
            )
            if texture_material_mode == "perturbation":
                textured_material = apply_sand_pbr_microtexture(
                    sand_rgb,
                    roughness,
                    resolved_sand_macro_normal_world,
                    sampled_texture,
                    sand_material,
                    sim_to_gs,
                    texture_valid,
                    albedo_strength=float(getattr(
                        args, "sand_texture_albedo_strength", 0.35
                    )),
                    roughness_strength=float(getattr(
                        args, "sand_texture_roughness_strength", 0.20
                    )),
                    normal_strength=texture_normal_strength,
                    albedo_detail_min=float(getattr(
                        args, "sand_texture_albedo_detail_min", 0.5
                    )),
                    albedo_detail_max=float(getattr(
                        args, "sand_texture_albedo_detail_max", 1.5
                    )),
                    roughness_min=float(getattr(
                        args, "sand_texture_roughness_min", 0.08
                    )),
                    roughness_max=float(getattr(
                        args, "sand_texture_roughness_max", 1.0
                    )),
                )
            elif texture_material_mode == "texture_primary":
                # T4.1e/T4.1m endpoint: sampled maps replace the
                # artist-authored color/roughness instead of acting as a
                # weak zero-mean perturbation. Optional pore-water
                # darkening acts only on linear-RGB Albedo; free-water
                # reflection/refraction remains in the separate P4 layer.
                textured_material = apply_sand_texture_primary(
                    sand_rgb,
                    roughness,
                    resolved_sand_macro_normal_world,
                    sampled_texture,
                    sim_to_gs,
                    texture_valid,
                    albedo_gain=getattr(
                        args,
                        "sand_texture_primary_albedo_gain",
                        [1.0, 1.0, 1.0],
                    ),
                    roughness_scale=float(getattr(
                        args,
                        "sand_texture_primary_roughness_scale",
                        1.0,
                    )),
                    normal_strength=texture_normal_strength,
                    roughness_min=float(getattr(
                        args, "sand_texture_roughness_min", 0.08
                    )),
                    roughness_max=float(getattr(
                        args, "sand_texture_roughness_max", 1.0
                    )),
                    wetness=wetness,
                    wet_adjustment_enabled=bool(getattr(
                        args, "sand_texture_wet_adjustment_enabled", False
                    )),
                    wet_albedo_scale=float(getattr(
                        args, "sand_texture_wet_albedo_scale", 0.65
                    )),
                    wet_response_rate=float(getattr(
                        args, "sand_texture_wet_response_rate", 4.0
                    )),
                )
                # The downloaded material has no specular map. Use the
                # existing dry dielectric endpoint (normally 0.04).
                # Pore-water wetness deliberately does not alter it in the
                # first wet appearance model; surface-water Fresnel is
                # evaluated later by the explicit P4 water layer.
                dry_specular = float(getattr(
                    args, "sand_dry_specular", 0.04
                ))
                specular_strength = torch.where(
                    texture_valid,
                    torch.full_like(specular_strength, dry_specular),
                    specular_strength,
                )
            else:
                raise ValueError(
                    "sand_texture_material_mode must be perturbation or "
                    "texture_primary"
                )
            sand_render_albedo = textured_material["albedo_srgb"]
            roughness = textured_material["roughness"]
            sand_render_normal_world = textured_material["normal_world"]
            sand_render_normal_camera = torch.nn.functional.normalize(
                torch.matmul(
                    sand_render_normal_world.movedim(0, -1),
                    viewpoint_camera.world_view_transform[:3, :3],
                ).movedim(-1, 0),
                dim=0,
                eps=1.0e-6,
            )
            textured_shading = evaluate_diagnostic_sand_shading(
                sand_render_normal_camera,
                sand_render_normal_world,
                sand_render_albedo,
                roughness,
                specular_strength,
                (
                    sand_texture_ao_micro
                    if ambient_occlusion_enabled
                    else None
                ),
            )
            sand_shaded = textured_shading["shaded"]
            sand_render_irradiance = textured_shading["irradiance"]
            specular_total = textured_shading["specular"]
            sand_specular_environment = textured_shading[
                "specular_environment"
            ]

            if bool(getattr(args, "sand_texture_diagnostics", False)):
                texture_rgb_mask = texture_valid.to(
                    sand_rgb.dtype
                ).repeat(3, 1, 1)
                max_lod = max(1, sand_material.max_lod)
                albedo_detail = torch.clamp(
                    sampled_texture["albedo_linear"]
                    / sand_material.albedo_mean_linear,
                    min=float(getattr(
                        args, "sand_texture_albedo_detail_min", 0.5
                    )),
                    max=float(getattr(
                        args, "sand_texture_albedo_detail_max", 1.5
                    )),
                )
                texture_uv_checker = sampled_texture.get("uv_checker")
                if texture_uv_checker is None:
                    texture_uv_checker = sand_texture_uv_checker(
                        texture_uv, texture_valid
                    )
                triplanar_weights = sampled_texture.get(
                    "triplanar_weights",
                    torch.zeros_like(sand_render_albedo),
                )
                triplanar_lod = sampled_texture.get(
                    "triplanar_lod",
                    torch.zeros_like(sand_render_albedo),
                )
                sand_texture_diagnostics = {
                    "sand_texture_valid": texture_rgb_mask,
                    "sand_texture_uv_checker": texture_uv_checker,
                    # RGB encodes XY/XZ/YZ blend weights. Side faces should
                    # transition away from red (XY) toward green/blue.
                    "sand_texture_triplanar_weights": triplanar_weights,
                    # RGB stores the independently estimated XY/XZ/YZ LODs.
                    "sand_texture_triplanar_lod": (
                        triplanar_lod / float(max_lod)
                    ),
                    "sand_texture_lod": (
                        sampled_texture["lod"].repeat(3, 1, 1)
                        / float(max_lod)
                    ),
                    "sand_texture_albedo_raw": (
                        linear_to_srgb(sampled_texture["albedo_linear"])
                        * texture_rgb_mask
                    ),
                    # Neutral detail is 0.5 gray; brighter/darker pixels
                    # show multiplicative variation around the wetness base.
                    "sand_texture_albedo_neutralized": (
                        torch.clamp(albedo_detail * 0.5, 0.0, 1.0)
                        * texture_rgb_mask
                    ),
                    "sand_texture_albedo_final": (
                        sand_render_albedo * texture_rgb_mask
                    ),
                    # White means fully saturated response; brightness is
                    # the actual neutral multiplier applied in linear RGB.
                    "sand_texture_wet_response": (
                        textured_material["wet_response"].repeat(3, 1, 1)
                        * texture_rgb_mask
                    ),
                    "sand_texture_wet_brightness": (
                        textured_material["wet_brightness"].repeat(3, 1, 1)
                        * texture_rgb_mask
                    ),
                    "sand_texture_roughness_raw": (
                        sampled_texture["roughness"].repeat(3, 1, 1)
                        * texture_rgb_mask
                    ),
                    "sand_texture_roughness_final": (
                        roughness.repeat(3, 1, 1) * texture_rgb_mask
                    ),
                    "sand_texture_normal_tangent": (
                        (textured_material["normal_tangent"] * 0.5 + 0.5)
                        * texture_rgb_mask
                    ),
                    "sand_texture_normal_world": (
                        (sand_render_normal_world * 0.5 + 0.5)
                        * texture_rgb_mask
                    ),
                    "sand_texture_ao_raw": (
                        sand_texture_ao_raw.repeat(3, 1, 1)
                        * texture_rgb_mask
                    ),
                    # Residual is deliberately stored at physical scale;
                    # the full-resolution exporter also saves a normalized
                    # visualization for easier filter inspection.
                    "sand_texture_ao_dark_residual": (
                        sand_texture_ao_dark_residual.repeat(3, 1, 1)
                        * texture_rgb_mask
                    ),
                    "sand_texture_ao_micro": (
                        sand_texture_ao_micro.repeat(3, 1, 1)
                        * texture_rgb_mask
                    ),
                    "sand_texture_final": (
                        sand_shaded * surface_coverage * texture_rgb_mask
                    ),
                }
            recovery_attribute_diagnostics_enabled = (
                physical_reference_attribute_diagnostics_enabled
                or center_trusted_invalid_attribute_diagnostics_enabled
            )
            if recovery_attribute_diagnostics_enabled:
                # Counterfactual audit only: production depth recovery
                # intentionally retains the Gaussian alpha-blended
                # material. Replace one attribute at a time below, but do
                # not feed any counterfactual back into RGB or the P4
                # opaque G-buffer.
                if (
                    physical_reference_attribute_diagnostics_enabled
                    and not physical_reference_depth_recovery_enabled
                ):
                    raise ValueError(
                        "sand_physical_reference_attribute_diagnostics "
                        "requires sand_physical_reference_depth_recovery"
                    )
                if (
                    center_trusted_invalid_attribute_diagnostics_enabled
                    and not center_trusted_invalid_depth_recovery_enabled
                ):
                    raise ValueError(
                        "sand_center_trusted_invalid_attribute_diagnostics "
                        "requires sand_center_trusted_invalid_depth_recovery"
                    )
                if texture_material_mode != "texture_primary":
                    raise ValueError(
                        "sand recovery attribute diagnostics require "
                        "sand_texture_material_mode=texture_primary"
                    )
                if texture_projection != "simulation_triplanar":
                    raise ValueError(
                        "sand recovery attribute diagnostics require "
                        "sand_texture_projection=simulation_triplanar"
                    )

                recovery_attribute_selection = (
                    select_sand_recovery_attribute_region(
                        physical_reference_depth_diagnostics.get(
                            "sand_physical_reference_recovery_used",
                            torch.zeros_like(texture_valid),
                        ),
                        center_trusted_invalid_depth_diagnostics.get(
                            "sand_center_trusted_invalid_recovery_used",
                            torch.zeros_like(texture_valid),
                        ),
                        texture_valid,
                        triangle_reference[
                            "sand_triangle_reference_coverage"
                        ],
                        physical_enabled=(
                            physical_reference_attribute_diagnostics_enabled
                        ),
                        center_enabled=(
                            center_trusted_invalid_attribute_diagnostics_enabled
                        ),
                    )
                )
                recovery_region = recovery_attribute_selection[
                    "sand_recovery_attribute_region_mask"
                ]
                reference_macro_normal, reference_normal_angle = (
                    align_reference_normals(
                        triangle_reference[
                            "sand_triangle_reference_normal_world_raw"
                        ],
                        normal_map_world,
                        recovery_region,
                    )
                )
                reference_macro_normal = torch.where(
                    recovery_region.repeat(3, 1, 1),
                    reference_macro_normal,
                    normal_map_world,
                )
                reference_wetness = torch.where(
                    recovery_region,
                    triangle_reference[
                        "sand_triangle_reference_face_wetness_raw"
                    ],
                    wetness,
                )

                # Macro normals also select the triplanar projections and
                # their tangent frames. Re-sample with the reference face
                # normal so the normal counterfactual remains internally
                # consistent rather than retaining contaminated weights.
                reference_sampled_texture = (
                    sample_sand_pbr_material_triplanar(
                        sand_material,
                        sand_sim_position,
                        reference_macro_normal,
                        sim_to_gs,
                        texture_valid,
                        period=texture_period,
                        rotation_degrees=texture_rotation,
                        offset=texture_offset,
                        sharpness=float(getattr(
                            args,
                            "sand_texture_triplanar_sharpness",
                            4.0,
                        )),
                        lod_bias=texture_lod_bias,
                        normal_strength=texture_normal_strength,
                    )
                )

                def counterfactual_ao(sampled):
                    if not ambient_occlusion_enabled:
                        return None
                    return ambient_occlusion_from_residual(
                        sampled["ambient_occlusion_dark_residual"],
                        strength=float(getattr(
                            args, "sand_texture_ao_strength", 1.0
                        )),
                        minimum=float(getattr(
                            args, "sand_texture_ao_min", 0.4
                        )),
                        response=str(getattr(
                            args, "sand_texture_ao_response", "linear"
                        )),
                        sparse_low=float(getattr(
                            args, "sand_texture_ao_sparse_low", 0.04
                        )),
                        sparse_high=float(getattr(
                            args, "sand_texture_ao_sparse_high", 0.08
                        )),
                        sparse_contrast=float(getattr(
                            args,
                            "sand_texture_ao_sparse_contrast",
                            0.40,
                        )),
                    )

                def evaluate_primary_counterfactual(
                    counterfactual_sampled,
                    counterfactual_macro_normal,
                    counterfactual_wetness,
                ):
                    material = apply_sand_texture_primary(
                        sand_rgb,
                        roughness,
                        counterfactual_macro_normal,
                        counterfactual_sampled,
                        sim_to_gs,
                        texture_valid,
                        albedo_gain=getattr(
                            args,
                            "sand_texture_primary_albedo_gain",
                            [1.0, 1.0, 1.0],
                        ),
                        roughness_scale=float(getattr(
                            args,
                            "sand_texture_primary_roughness_scale",
                            1.0,
                        )),
                        normal_strength=texture_normal_strength,
                        roughness_min=float(getattr(
                            args, "sand_texture_roughness_min", 0.08
                        )),
                        roughness_max=float(getattr(
                            args, "sand_texture_roughness_max", 1.0
                        )),
                        wetness=counterfactual_wetness,
                        wet_adjustment_enabled=bool(getattr(
                            args,
                            "sand_texture_wet_adjustment_enabled",
                            False,
                        )),
                        wet_albedo_scale=float(getattr(
                            args,
                            "sand_texture_wet_albedo_scale",
                            0.65,
                        )),
                        wet_response_rate=float(getattr(
                            args,
                            "sand_texture_wet_response_rate",
                            4.0,
                        )),
                    )
                    normal_world = material["normal_world"]
                    normal_camera = torch.nn.functional.normalize(
                        torch.matmul(
                            normal_world.movedim(0, -1),
                            viewpoint_camera.world_view_transform[:3, :3],
                        ).movedim(-1, 0),
                        dim=0,
                        eps=1.0e-6,
                    )
                    counterfactual_specular = torch.where(
                        texture_valid,
                        torch.full_like(
                            roughness,
                            float(getattr(args, "sand_dry_specular", 0.04)),
                        ),
                        specular_strength,
                    )
                    shading = evaluate_diagnostic_sand_shading(
                        normal_camera,
                        normal_world,
                        material["albedo_srgb"],
                        material["roughness"],
                        counterfactual_specular,
                        counterfactual_ao(counterfactual_sampled),
                    )
                    return material, shading

                wet_material, wet_shading = evaluate_primary_counterfactual(
                    sampled_texture,
                    normal_map_world,
                    reference_wetness,
                )
                normal_material, normal_shading = (
                    evaluate_primary_counterfactual(
                        reference_sampled_texture,
                        reference_macro_normal,
                        wetness,
                    )
                )
                both_material, both_shading = (
                    evaluate_primary_counterfactual(
                        reference_sampled_texture,
                        reference_macro_normal,
                        reference_wetness,
                    )
                )
                region = recovery_region.to(sand_rgb.dtype)
                region_rgb = region.repeat(3, 1, 1)
                rgb_weight = region_rgb * surface_coverage

                def masked_rgb(value):
                    return value * rgb_weight

                def absolute_difference(value):
                    return torch.abs(value - sand_shaded) * rgb_weight

                sand_recovery_attribute_diagnostics = {
                    "sand_recovery_attribute_scope": (
                        recovery_attribute_selection[
                            "sand_recovery_attribute_scope"
                        ]
                    ),
                    "sand_recovery_attribute_region": region_rgb,
                    "sand_recovery_attribute_current_shading": masked_rgb(
                        sand_shaded
                    ),
                    "sand_recovery_attribute_triangle_normal_shading": (
                        masked_rgb(normal_shading["shaded"])
                    ),
                    "sand_recovery_attribute_triangle_wetness_shading": (
                        masked_rgb(wet_shading["shaded"])
                    ),
                    "sand_recovery_attribute_triangle_normal_wetness_shading": (
                        masked_rgb(both_shading["shaded"])
                    ),
                    "sand_recovery_attribute_triangle_normal_absolute_difference": (
                        absolute_difference(normal_shading["shaded"])
                    ),
                    "sand_recovery_attribute_triangle_wetness_absolute_difference": (
                        absolute_difference(wet_shading["shaded"])
                    ),
                    "sand_recovery_attribute_triangle_normal_wetness_absolute_difference": (
                        absolute_difference(both_shading["shaded"])
                    ),
                    "sand_recovery_attribute_current_albedo": (
                        sand_render_albedo * region_rgb
                    ),
                    "sand_recovery_attribute_triangle_normal_albedo": (
                        normal_material["albedo_srgb"] * region_rgb
                    ),
                    "sand_recovery_attribute_triangle_wetness_albedo": (
                        wet_material["albedo_srgb"] * region_rgb
                    ),
                    "sand_recovery_attribute_triangle_normal_wetness_albedo": (
                        both_material["albedo_srgb"] * region_rgb
                    ),
                    "sand_recovery_attribute_current_irradiance": (
                        textured_shading["irradiance"] * region_rgb
                    ),
                    "sand_recovery_attribute_triangle_normal_irradiance": (
                        normal_shading["irradiance"] * region_rgb
                    ),
                    "sand_recovery_attribute_current_macro_normal": (
                        (normal_map_world * 0.5 + 0.5) * region_rgb
                    ),
                    "sand_recovery_attribute_triangle_macro_normal": (
                        (reference_macro_normal * 0.5 + 0.5) * region_rgb
                    ),
                    "sand_recovery_attribute_current_wetness": (
                        wetness.repeat(3, 1, 1) * region_rgb
                    ),
                    "sand_recovery_attribute_triangle_face_wetness": (
                        reference_wetness.repeat(3, 1, 1) * region_rgb
                    ),
                    "sand_recovery_attribute_normal_angle_degrees_raw": (
                        reference_normal_angle
                    ),
                    "sand_recovery_attribute_wetness_difference_raw": (
                        torch.where(
                            recovery_region,
                            reference_wetness - wetness,
                            torch.zeros_like(wetness),
                        )
                    ),
                }
        rendered_rgb_composite = (
            rendered_rgb_src * (1.0 - sand_composite_mask)
            + sand_shaded * sand_composite_mask
        )

        # P4 must never interpolate camera depth or physical normals across
        # an antialiased edge: those values describe one surface owner, not
        # optical transparency.  Use a discrete half-coverage owner buffer
        # while retaining the continuous mask only for final RGB coverage.
        # Plane depth becomes ill-conditioned in extremely faint Gaussian
        # tails, so those pixels are also excluded from the ray-traceable
        # sand G-buffer even though they remain useful for RGB antialiasing.
        sand_depth_candidate = valid_sand & (surface_coverage >= 0.5)
        sand_owns_pixel = opaque_surface_owner(
            surface_coverage,
            sand_in_front,
            sand_depth_candidate,
        )
        sand_shading_diagnostics = {}
        if bool(getattr(args, "sand_shading_diagnostics", False)):
            if triangle_reference is None:
                raise ValueError(
                    "sand_shading_diagnostics requires triangle_games with "
                    "the T2.4 sand_soft_boundary_guidance reference"
                )
            reference_coverage = triangle_reference[
                "sand_triangle_reference_coverage"
            ].to(torch.bool)
            diagnostic_region = (
                sand_owns_pixel
                & reference_coverage
                & (surface_coverage >= 0.99)
            )
            aligned_face_normal_world, face_blended_angle = (
                align_reference_normals(
                    triangle_reference[
                        "sand_triangle_reference_normal_world_raw"
                    ],
                    normal_map_world,
                    diagnostic_region,
                )
            )
            aligned_face_normal_camera = torch.nn.functional.normalize(
                (
                    aligned_face_normal_world.movedim(0, -1)
                    @ viewpoint_camera.world_view_transform[:3, :3]
                ).movedim(-1, 0),
                dim=0,
                eps=1.0e-6,
            )

            # A hard independent triangle reference distinguishes the
            # height-field's piecewise-planar normal from any extra banding
            # introduced by alpha blending neighboring Gaussian normals.
            face_shading = evaluate_diagnostic_sand_shading(
                aligned_face_normal_camera,
                aligned_face_normal_world,
                sand_rgb,
                roughness,
                specular_strength,
            )

            # Use the simulation's +Z direction as an intentionally simple
            # counterfactual. If stripes survive here, they cannot be
            # attributed to spatially varying surface normals.
            sim_to_gs_rotation = torch.as_tensor(
                pc.sand_water_adapter.sim_to_gs[:3, :3],
                device=normal_map_world.device,
                dtype=normal_map_world.dtype,
            )
            up_world = torch.tensor(
                [0.0, 0.0, 1.0],
                device=normal_map_world.device,
                dtype=normal_map_world.dtype,
            ) @ sim_to_gs_rotation.T
            up_world = torch.nn.functional.normalize(
                up_world, dim=0, eps=1.0e-6
            )
            up_camera = torch.nn.functional.normalize(
                up_world @ viewpoint_camera.world_view_transform[:3, :3],
                dim=0,
                eps=1.0e-6,
            )
            flat_normal_world = up_world[:, None, None].expand_as(
                normal_map_world
            )
            flat_normal_camera = up_camera[:, None, None].expand_as(
                normal_map_camera
            )
            flat_shading = evaluate_diagnostic_sand_shading(
                flat_normal_camera,
                flat_normal_world,
                sand_rgb,
                roughness,
                specular_strength,
            )

            # Hold the material at the configured dry-sand constants while
            # preserving the production blended normal. Bands remaining in
            # this image are caused by geometry/lighting, not wetness or
            # material interpolation.
            dry_albedo = torch.tensor(
                getattr(args, "sand_dry_color", [194, 154, 96]),
                device=sand_rgb.device,
                dtype=sand_rgb.dtype,
            )[:, None, None] / 255.0
            dry_albedo = linear_to_srgb(
                srgb_to_linear(dry_albedo)
                * float(getattr(args, "sand_exposure", 1.0))
            ).expand_as(sand_rgb)
            dry_roughness = torch.full_like(
                roughness,
                float(getattr(args, "sand_dry_roughness", 0.90)),
            )
            dry_specular_strength = torch.full_like(
                specular_strength,
                float(getattr(args, "sand_dry_specular", 0.04)),
            )
            constant_material_shading = evaluate_diagnostic_sand_shading(
                normal_map_camera,
                normal_map_world,
                dry_albedo,
                dry_roughness,
                dry_specular_strength,
            )

            visible = valid_sand.to(torch.float32)
            rgb_coverage = surface_coverage
            specular_total_rgb = (
                specular_total.repeat(3, 1, 1)
                if specular_total.shape[0] == 1
                else specular_total
            )
            sand_shading_diagnostics = {
                "sand_shading_diagnostic_region": (
                    diagnostic_region.repeat(3, 1, 1)
                ),
                "sand_shading_albedo": sand_rgb * visible,
                "sand_shading_wetness": wetness.repeat(3, 1, 1) * visible,
                "sand_shading_roughness": (
                    roughness.repeat(3, 1, 1) * visible
                ),
                "sand_shading_specular_strength": (
                    specular_strength.repeat(3, 1, 1) * visible
                ),
                "sand_shading_directional_diffuse": (
                    torch.clamp(directional_diffuse, 0.0, 1.0) * visible
                ),
                "sand_shading_environment_diffuse": (
                    torch.clamp(environment_diffuse, 0.0, 1.0) * visible
                ),
                "sand_shading_diffuse_irradiance": (
                    torch.clamp(diffuse_irradiance, 0.0, 1.0) * visible
                ),
                "sand_shading_diffuse_only": (
                    torch.clamp(
                        linear_to_srgb(
                            srgb_to_linear(sand_rgb) * diffuse_irradiance
                        ),
                        0.0,
                        1.0,
                    )
                    * rgb_coverage
                ),
                "sand_shading_specular_term": (
                    torch.clamp(
                        linear_to_srgb(specular_total_rgb),
                        0.0,
                        1.0,
                    )
                    * visible
                ),
                "sand_shading_specular_environment": (
                    torch.clamp(
                        linear_to_srgb(sand_specular_environment),
                        0.0,
                        1.0,
                    )
                    * visible
                ),
                "sand_shading_blended_normal": (
                    (normal_map_world * 0.5 + 0.5) * visible
                ),
                "sand_shading_mesh_face_normal": (
                    (aligned_face_normal_world * 0.5 + 0.5)
                    * diagnostic_region.to(torch.float32)
                ),
                # White denotes >=45 degrees; the exact angle is also
                # exported as float32 NPY for quantitative inspection.
                "sand_shading_normal_disagreement": (
                    torch.clamp(face_blended_angle / 45.0, 0.0, 1.0)
                    .repeat(3, 1, 1)
                ),
                "sand_shading_counterfactual_constant_material": (
                    constant_material_shading["shaded"] * rgb_coverage
                ),
                "sand_shading_counterfactual_mesh_face_normal": (
                    face_shading["shaded"]
                    * diagnostic_region.to(torch.float32)
                ),
                "sand_shading_counterfactual_flat_normal": (
                    flat_shading["shaded"] * rgb_coverage
                ),
                "sand_shading_mesh_face_irradiance": (
                    torch.clamp(face_shading["irradiance"], 0.0, 1.0)
                    * diagnostic_region.to(torch.float32)
                ),
                "sand_shading_flat_normal_irradiance": (
                    torch.clamp(flat_shading["irradiance"], 0.0, 1.0)
                    * visible
                ),
                "sand_shading_face_blended_angle_degrees_raw": (
                    face_blended_angle
                ),
            }
        if t3_prerequisite_diagnostics_enabled:
            if (
                triangle_reference is None
                or target_soft_coverage is None
                or sand_center_depth is None
                or sand_center_depth_std is None
            ):
                raise RuntimeError(
                    "T3 prerequisite audit is missing its T2.4 reference "
                    "or center-depth buffers"
                )
            # This gate is diagnostic-only. It audits the dense, visible
            # interior after T2.4 ownership has been decided and never
            # changes coverage, depth, normals, shading or composition.
            t3_thresholds = {
                "core_coverage_min": float(getattr(
                    args, "sand_t3_core_coverage_min", 0.95
                )),
                "erode_kernel": int(getattr(
                    args, "sand_t3_erode_kernel", 5
                )),
                "depth_absolute_tolerance": float(getattr(
                    args, "sand_t3_depth_absolute_tolerance", 0.005
                )),
                "depth_relative_tolerance": float(getattr(
                    args, "sand_t3_depth_relative_tolerance", 0.002
                )),
                "normal_angle_tolerance_degrees": float(getattr(
                    args,
                    "sand_t3_normal_angle_tolerance_degrees",
                    15.0,
                )),
                "center_std_relative_tolerance": float(getattr(
                    args,
                    "sand_t3_center_std_relative_tolerance",
                    0.01,
                )),
                "large_projected_edge_pixels": float(getattr(
                    args,
                    "sand_t3_large_projected_edge_pixels",
                    4.0,
                )),
            }
            triangle_footprint_diagnostics.update(
                classify_sand_t3_prerequisites(
                    target_soft_coverage,
                    surface_coverage,
                    sand_owns_pixel,
                    resolved_sand_depth,
                    sand_center_depth,
                    sand_center_depth_std,
                    normal_map_world,
                    triangle_reference[
                        "sand_triangle_reference_coverage"
                    ],
                    triangle_reference[
                        "sand_triangle_reference_depth_raw"
                    ],
                    triangle_reference[
                        "sand_triangle_reference_normal_world_raw"
                    ],
                    triangle_reference[
                        "sand_triangle_reference_projected_max_edge_raw"
                    ],
                    triangle_reference[
                        "sand_triangle_reference_projected_area_raw"
                    ],
                    **t3_thresholds,
                )
            )
            triangle_footprint_diagnostics.update({
                "sand_t3_thresholds": t3_thresholds,
                "sand_t3_decision_limits": {
                    "depth_inconsistent_fraction": float(getattr(
                        args,
                        "sand_t3_gate_max_depth_fraction",
                        0.01,
                    )),
                    "normal_mismatch_fraction": float(getattr(
                        args,
                        "sand_t3_gate_max_normal_fraction",
                        0.02,
                    )),
                    "center_depth_mixed_fraction": float(getattr(
                        args,
                        "sand_t3_gate_max_center_mixed_fraction",
                        0.01,
                    )),
                    "any_failure_fraction": float(getattr(
                        args,
                        "sand_t3_gate_max_any_failure_fraction",
                        0.02,
                    )),
                    "minimum_failure_on_large_triangle_fraction": float(
                        getattr(
                            args,
                            "sand_t3_gate_min_large_triangle_correlation",
                            0.50,
                        )
                    ),
                },
            })
        if soft_boundary_guidance_enabled:
            unguided_composite_mask = (
                unguided_surface_coverage * sand_in_front
            )
            unguided_depth_candidate = valid_sand & (
                unguided_surface_coverage >= 0.5
            )
            unguided_owner = opaque_surface_owner(
                unguided_surface_coverage,
                sand_in_front,
                unguided_depth_candidate,
            )
            unguided_rgb_composite = (
                rendered_rgb_src * (1.0 - unguided_composite_mask)
                + sand_shaded * unguided_composite_mask
            )
            triangle_footprint_diagnostics.update({
                "sand_guidance_owner_unguided": unguided_owner,
                "sand_guidance_owner_removed": (
                    unguided_owner & ~sand_owns_pixel
                ),
                "sand_guidance_owner_added": (
                    sand_owns_pixel & ~unguided_owner
                ),
                "sand_guidance_rgb_absolute_difference": torch.abs(
                    unguided_rgb_composite - rendered_rgb_composite
                ),
                "sand_only_unguided": (
                    sand_shaded * unguided_surface_coverage
                ),
            })
        if not soft_boundary_guidance_enabled:
            triangle_footprint_diagnostics = {}
        triangle_reference_enabled = bool(getattr(
            args, "sand_triangle_reference_diagnostics", False
        ))
        soft_boundary_enabled = bool(getattr(
            args, "sand_soft_boundary_diagnostics", False
        ))
        primitive_attribution_enabled = bool(getattr(
            args, "sand_primitive_attribution_diagnostics", False
        ))
        if (
            triangle_reference_enabled
            or soft_boundary_enabled
            or primitive_attribution_enabled
        ):
            reference_triangles = getattr(
                pc,
                "sand_reference_triangles_gs_t",
                torch.empty(
                    (0, 3, 3), dtype=means3D_sand.dtype,
                    device=means3D_sand.device,
                ),
            )
            reference_vertex_heights = getattr(
                pc,
                "sand_reference_vertex_heights_t",
                torch.empty(
                    (0, 3), dtype=means3D_sand.dtype,
                    device=means3D_sand.device,
                ),
            )
            reference_face_wetness = getattr(
                pc,
                "sand_reference_face_wetness_t",
                torch.empty(
                    0,
                    dtype=means3D_sand.dtype,
                    device=means3D_sand.device,
                ),
            )
            if (
                (soft_boundary_enabled or primitive_attribution_enabled)
                and reference_triangles.shape[0] == 0
            ):
                raise ValueError(
                    "sand boundary diagnostics require triangle_games"
                )
            # This hard triangle pass is diagnostic-only.  It bypasses the
            # Gaussian ellipse, opacity and plane-depth reconstruction so
            # it can serve as an independent expected-coverage reference.
            triangle_reference = rasterize_triangle_reference(
                reference_triangles,
                viewpoint_camera.world_view_transform,
                full_proj_transform,
                image_height,
                image_width,
                reference_vertex_heights,
                reference_face_wetness,
            )
            triangle_footprint_diagnostics = dict(triangle_reference)
            if triangle_reference_enabled:
                triangle_footprint_diagnostics.update(
                    classify_sand_triangle_footprint(
                        triangle_reference[
                            "sand_triangle_reference_coverage"
                        ],
                        triangle_reference[
                            "sand_triangle_reference_depth_raw"
                        ],
                        sand_alpha,
                        surface_coverage,
                        resolved_sand_depth,
                        resolved_sand_depth_valid,
                        sand_in_front,
                        sand_owns_pixel,
                        alpha_min=alpha_low,
                        owner_coverage_min=0.5,
                        footprint_epsilon=float(getattr(
                            args,
                            "sand_triangle_reference_footprint_epsilon",
                            1.0e-8,
                        )),
                        boundary_kernel=int(getattr(
                            args,
                            "sand_triangle_reference_boundary_kernel",
                            5,
                        )),
                    )
                )
            if soft_boundary_enabled or primitive_attribution_enabled:
                sand_epsilon = float(getattr(
                    pc, "sand_reference_sand_epsilon", 0.0
                ))
                fade_height = float(getattr(
                    args,
                    "sand_boundary_fade_height",
                    getattr(
                        pc,
                        "sand_reference_full_coverage_height",
                        1.0,
                    ),
                ))
                target_soft_coverage = sand_height_to_soft_coverage(
                    triangle_reference[
                        "sand_triangle_reference_sand_height_raw"
                    ],
                    triangle_reference[
                        "sand_triangle_reference_coverage"
                    ],
                    sand_epsilon=sand_epsilon,
                    fade_height=fade_height,
                )
                # T2.2 is deliberately comparison-only: these products
                # are saved below, but production RGB/owner still use the
                # unmodified Gaussian ``surface_coverage``.
                triangle_footprint_diagnostics.update(
                    compare_sand_soft_boundary(
                        target_soft_coverage, surface_coverage
                    )
                )
                triangle_footprint_diagnostics.update({
                    "sand_boundary_sand_epsilon": sand_epsilon,
                    "sand_boundary_fade_height": fade_height,
                })
                if primitive_attribution_enabled:
                    primitive_boundary = getattr(
                        pc,
                        "sand_primitive_boundary_t",
                        torch.empty(
                            0,
                            dtype=torch.bool,
                            device=means3D_sand.device,
                        ),
                    )
                    primitive_slope = getattr(
                        pc,
                        "sand_primitive_slope_t",
                        torch.empty(
                            0,
                            dtype=means3D_sand.dtype,
                            device=means3D_sand.device,
                        ),
                    )
                    if (
                        primitive_boundary.shape
                        != (num_sand_gaussian,)
                        or primitive_slope.shape
                        != (num_sand_gaussian,)
                    ):
                        raise ValueError(
                            "T2.3 primitive labels must match sand Gaussians"
                        )
                    steep_threshold = float(getattr(
                        args,
                        "sand_primitive_steep_slope_threshold",
                        1.0,
                    ))
                    facing_epsilon = float(getattr(
                        args,
                        "sand_primitive_facing_epsilon",
                        0.0,
                    ))
                    if steep_threshold < 0.0 or facing_epsilon < 0.0:
                        raise ValueError(
                            "T2.3 slope/facing thresholds must be non-negative"
                        )
                    view_to_camera = torch.nn.functional.normalize(
                        -pts_in_cam_sand, dim=1, eps=1.0e-8
                    )
                    facing_cosine = torch.sum(
                        local_normal_sand * view_to_camera, dim=1
                    )
                    steep = primitive_slope >= steep_threshold
                    steep_front = steep & (
                        facing_cosine > facing_epsilon
                    )
                    steep_rear = steep & (
                        facing_cosine < -facing_epsilon
                    )
                    # Grazing primitives are kept in the residual class so
                    # a numerical sign around zero cannot masquerade as
                    # evidence for either the front- or rear-facing theory.
                    other_primitive = ~(steep_front | steep_rear)
                    steep_front_alpha = rasterize_sand_subset_alpha(
                        steep_front
                    )
                    steep_rear_alpha = rasterize_sand_subset_alpha(
                        steep_rear
                    )
                    other_alpha = rasterize_sand_subset_alpha(
                        other_primitive
                    )
                    boundary_steep_front = (
                        primitive_boundary & steep_front
                    )
                    boundary_steep_rear = (
                        primitive_boundary & steep_rear
                    )
                    boundary_steep_front_alpha = (
                        rasterize_sand_subset_alpha(boundary_steep_front)
                    )
                    boundary_steep_rear_alpha = (
                        rasterize_sand_subset_alpha(boundary_steep_rear)
                    )
                    depth_absolute_tolerance = float(getattr(
                        args,
                        "sand_primitive_depth_absolute_tolerance",
                        0.005,
                    ))
                    depth_relative_tolerance = float(getattr(
                        args,
                        "sand_primitive_depth_relative_tolerance",
                        0.002,
                    ))
                    primitive_diagnostics = (
                        classify_sand_primitive_overcoverage(
                            target_soft_coverage,
                            surface_coverage,
                            triangle_reference[
                                "sand_triangle_reference_coverage"
                            ],
                            steep_front_alpha,
                            steep_rear_alpha,
                            other_alpha,
                            resolved_sand_depth,
                            triangle_reference[
                                "sand_triangle_reference_depth_raw"
                            ],
                            overcoverage_epsilon=float(getattr(
                                args,
                                "sand_primitive_overcoverage_epsilon",
                                1.0e-6,
                            )),
                            support_epsilon=float(getattr(
                                args,
                                "sand_primitive_support_epsilon",
                                1.0e-8,
                            )),
                            depth_absolute_tolerance=(
                                depth_absolute_tolerance
                            ),
                            depth_relative_tolerance=(
                                depth_relative_tolerance
                            ),
                        )
                    )
                    soft_overcoverage = torch.clamp(
                        surface_coverage - target_soft_coverage, min=0.0
                    )
                    triangle_footprint_diagnostics.update({
                        "sand_steep_front_alpha": steep_front_alpha,
                        "sand_steep_rear_alpha": steep_rear_alpha,
                        "sand_boundary_steep_front_alpha": (
                            boundary_steep_front_alpha
                        ),
                        "sand_boundary_steep_rear_alpha": (
                            boundary_steep_rear_alpha
                        ),
                        "sand_other_primitive_alpha": other_alpha,
                        # These two nested-support maps are deliberately
                        # not part of the mutually-exclusive dominant
                        # attribution. They answer whether the stricter
                        # clipped-boundary hypothesis overlaps the error.
                        "sand_boundary_steep_front_overcoverage": (
                            torch.minimum(
                                soft_overcoverage,
                                boundary_steep_front_alpha,
                            )
                        ),
                        "sand_boundary_steep_rear_overcoverage": (
                            torch.minimum(
                                soft_overcoverage,
                                boundary_steep_rear_alpha,
                            )
                        ),
                        "sand_primitive_count": num_sand_gaussian,
                        "sand_primitive_boundary_count": int(
                            primitive_boundary.sum().item()
                        ),
                        "sand_primitive_steep_count": int(
                            steep.sum().item()
                        ),
                        "sand_primitive_steep_front_count": int(
                            steep_front.sum().item()
                        ),
                        "sand_primitive_steep_rear_count": int(
                            steep_rear.sum().item()
                        ),
                        "sand_primitive_other_count": int(
                            other_primitive.sum().item()
                        ),
                        "sand_primitive_boundary_steep_count": int(
                            (primitive_boundary & steep).sum().item()
                        ),
                        "sand_primitive_boundary_steep_front_count": int(
                            boundary_steep_front.sum().item()
                        ),
                        "sand_primitive_boundary_steep_rear_count": int(
                            boundary_steep_rear.sum().item()
                        ),
                        "sand_primitive_steep_slope_threshold": (
                            steep_threshold
                        ),
                        "sand_primitive_facing_epsilon": facing_epsilon,
                        "sand_primitive_depth_absolute_tolerance": (
                            depth_absolute_tolerance
                        ),
                        "sand_primitive_depth_relative_tolerance": (
                            depth_relative_tolerance
                        ),
                        **primitive_diagnostics,
                    })
        # ``sand_surface_valid`` means visible ray-traceable substrate, not
        # merely "a sand splat touched this pixel". Water is composed later
        # in P4, so testing only against the static source here still keeps
        # sand below water while rejecting plane-depth outliers behind it.
        sand_surface_valid = sand_owns_pixel
        source_normal_camera = torch.nn.functional.normalize(
            out_all_map_src[:3], dim=0, eps=1.0e-6
        )
        source_normal_world = camera_normals_to_world(
            source_normal_camera,
            viewpoint_camera.world_view_transform[:3, :3],
        )
        opaque_surface_depth = torch.where(
            sand_owns_pixel, resolved_sand_depth, source_depth
        )
        opaque_surface_normal_world = torch.where(
            sand_owns_pixel.repeat(3, 1, 1),
            resolved_sand_macro_normal_world,
            source_normal_world,
        )
        opaque_surface_normal_camera = torch.where(
            sand_owns_pixel.repeat(3, 1, 1),
            resolved_sand_macro_normal_camera,
            source_normal_camera,
        )
        # SSR classifies the side from which its secondary ray reaches an
        # opaque surface. Orient every target normal toward the primary
        # camera first, so R.N < 0 consistently means a visible/front face.
        opaque_normal_facing = (
            opaque_surface_normal_camera * view_local
        ).sum(0, keepdim=True)
        opaque_surface_normal_camera_facing = torch.where(
            (opaque_normal_facing < 0.0).repeat(3, 1, 1),
            -opaque_surface_normal_camera,
            opaque_surface_normal_camera,
        )
        gbuffer_validation = None
        if bool(getattr(args, "sand_validate_gbuffer", True)):
            gbuffer_validation = validate_surface_gbuffer(
                surface_coverage,
                resolved_sand_depth,
                resolved_sand_macro_normal_world,
                sand_surface_valid,
            )
            for material_name, material_value in (
                ("roughness", roughness),
                ("specular", specular_strength),
                ("wetness", wetness),
            ):
                if not torch.isfinite(material_value[valid_sand]).all():
                    raise ValueError(
                        f"sand {material_name} contains NaN or infinity"
                    )
                if torch.any(
                    (material_value[valid_sand] < 0.0)
                    | (material_value[valid_sand] > 1.0)
                ):
                    raise ValueError(
                        f"sand {material_name} lies outside [0,1]"
                    )
            print("P4 sand G-buffer validated:", gbuffer_validation)

        result = {
            "render": rendered_rgb_composite,
            # Unlike the old dry branch, src now means the true source-only
            # pass. The other established folders keep the final composite
            # until reflection/refraction layers are added in P4.
            "rendered_src_save": rendered_rgb_src,
            "rendered_image_src_refl_save": rendered_rgb_composite,
            "rendered_image_src_refl_refra_save": rendered_rgb_composite,
            "num_gausssian_all": num_gausssian_all,
            # P3/P4 opaque-layer contract.  Coverage remains continuous for
            # antialiasing; depth and normal use the discrete owner above.
            "source_surface_rgb": rendered_rgb_src,
            "source_surface_depth": source_depth,
            "source_surface_normal_world": source_normal_world,
            "sand_surface_albedo": sand_render_albedo,
            "sand_surface_rgb": sand_shaded,
            "sand_surface_raw_alpha": sand_alpha,
            "sand_surface_coverage": surface_coverage,
            "sand_surface_visible_coverage": sand_composite_mask,
            "sand_surface_depth": resolved_sand_depth,
            "sand_surface_normal_world": resolved_sand_macro_normal_world,
            # Normal-map detail is deliberately separate from the macro
            # surface normal consumed by plane depth, owner and water SSR.
            "sand_surface_shading_normal_world": (
                sand_render_normal_world
            ),
            "sand_surface_valid": sand_surface_valid,
            "sand_surface_view_direction_camera": view_local,
            "sand_surface_roughness": roughness,
            "sand_surface_specular": specular_strength,
            # Linear-RGB additive IBL term, kept separate so appearance
            # experiments can audit it independently of direct highlights.
            "sand_surface_specular_environment": (
                sand_specular_environment
            ),
            "sand_surface_wetness": wetness,
            "opaque_surface_rgb": rendered_rgb_composite,
            "opaque_surface_depth": opaque_surface_depth,
            "opaque_surface_normal_world": opaque_surface_normal_world,
            "opaque_surface_normal_camera_facing": (
                opaque_surface_normal_camera_facing
            ),
            "opaque_surface_sand_owner": sand_owns_pixel,
            "sand_gbuffer_validation": gbuffer_validation,
        }
        if bool(getattr(args, "sand_save_debug_maps", False)):
            # These masks are diagnostic only.  In particular, do not use
            # the classification map to modify the established soft RGB
            # composition or the discrete P4 owner decision above.
            sand_stage_diagnostics = classify_sand_gbuffer_stages(
                sand_alpha,
                surface_coverage,
                resolved_sand_depth_valid,
                sand_in_front,
                alpha_min=alpha_low,
                owner_coverage_min=0.5,
                owner_in_front_min=0.5,
            )
            # Preserve the original plane-only failure population even
            # when the production fallback is enabled. This lets the A/B
            # show which red cracks were repaired instead of erasing the
            # evidence used to select them.
            plane_stage_diagnostics = classify_sand_gbuffer_stages(
                sand_alpha,
                surface_coverage,
                sand_plane_depth_valid,
                plane_sand_in_front,
                alpha_min=alpha_low,
                owner_coverage_min=0.5,
                owner_in_front_min=0.5,
            )
            # Refine only the orange depth-rejection class.  These maps do
            # not feed composition: they distinguish near-coplanar source
            # depth noise from genuine foreground occluders before we
            # consider changing the production owner rule.
            sand_depth_diagnostics = classify_sand_depth_competition(
                plane_depth_delta,
                valid_source & plane_valid_sand,
                plane_stage_diagnostics[
                    "sand_failure_behind_source"
                ],
                surface_coverage,
                depth_softness=depth_softness,
            )
            sand_plane_reference = classify_sand_plane_depth_reference(
                sand_depth,
                sand_center_depth,
                source_depth,
                valid_source,
                plane_valid_sand,
                plane_stage_diagnostics[
                    "sand_failure_behind_source"
                ],
                sand_depth_diagnostics[
                    "sand_depth_high_coverage_interior"
                ],
                depth_bias=depth_bias,
                depth_softness=depth_softness,
            )
            # Diagnose invalid plane depths without changing production
            # ownership. Two normal-moment passes distinguish a genuinely
            # grazing shared plane from alpha cancellation between several
            # individually valid tangent planes. A third pass projects
            # per-primitive construction/orientation flags into the same
            # screen-space contribution weights.
            invalid_depth_diagnostics_enabled = bool(
                getattr(args, "sand_invalid_depth_diagnostics", False)
            )
            # These three extra rasterization passes are deliberately
            # opt-in: ordinary long-video debug renders should not pay the
            # cost of the detailed invalid-depth investigation.
            normal_second_diagonal = (
                rasterize_sand_channels(
                    local_normal_sand * local_normal_sand
                )
                if invalid_depth_diagnostics_enabled
                else torch.zeros_like(out_all_map_sand[:3])
            )
            normal_second_cross = (
                rasterize_sand_channels(
                    torch.stack(
                        (
                            local_normal_sand[:, 0]
                            * local_normal_sand[:, 1],
                            local_normal_sand[:, 0]
                            * local_normal_sand[:, 2],
                            local_normal_sand[:, 1]
                            * local_normal_sand[:, 2],
                        ),
                        dim=1,
                    )
                )
                if invalid_depth_diagnostics_enabled
                else torch.zeros_like(out_all_map_sand[:3])
            )
            normal_second_moments = torch.cat(
                (normal_second_diagonal, normal_second_cross), dim=0
            )

            primitive_normal_norm = torch.linalg.vector_norm(
                local_normal_sand, dim=1
            )
            primitive_numeric_good = (
                torch.isfinite(pts_in_cam_sand).all(dim=1)
                & (pts_in_cam_sand[:, 2] > 0.0)
                & torch.isfinite(local_normal_sand).all(dim=1)
                & (primitive_normal_norm > 1.0e-6)
                & torch.isfinite(local_distance_sand)
                & (local_distance_sand >= 0.0)
                & torch.isfinite(opacity_sand).all(dim=1)
                & (opacity_sand >= 0.0).all(dim=1)
                & (opacity_sand <= 1.0).all(dim=1)
            )
            if scales_sand is not None:
                primitive_numeric_good = (
                    primitive_numeric_good
                    & torch.isfinite(scales_sand).all(dim=1)
                    & (scales_sand > 0.0).all(dim=1)
                )
            safe_center_z = torch.where(
                primitive_numeric_good,
                pts_in_cam_sand[:, 2],
                torch.ones_like(pts_in_cam_sand[:, 2]),
            )
            center_ray = pts_in_cam_sand / safe_center_z[:, None]
            center_ray_norm = torch.linalg.vector_norm(center_ray, dim=1)
            center_denominator = -torch.sum(
                local_normal_sand * center_ray, dim=1
            )
            center_incidence = center_denominator / torch.clamp(
                primitive_normal_norm * center_ray_norm, min=1.0e-6
            )
            primitive_orientation_bad = (
                primitive_numeric_good
                & torch.isfinite(center_incidence)
                & (
                    center_incidence
                    < -float(getattr(
                        args,
                        "sand_invalid_depth_incidence_epsilon",
                        0.02,
                    ))
                )
            )
            primitive_numeric_bad = ~primitive_numeric_good
            primitive_flags = (
                rasterize_sand_channels(
                    torch.stack(
                        (
                            primitive_numeric_bad.to(torch.float32),
                            primitive_orientation_bad.to(torch.float32),
                            oriented_normal_flip_sand.to(torch.float32),
                        ),
                        dim=1,
                    )
                )
                if invalid_depth_diagnostics_enabled
                else torch.zeros_like(out_all_map_sand[:3])
            )
            primitive_numeric_bad_fraction = torch.clamp(
                primitive_flags[0:1], 0.0, 1.0
            )
            primitive_orientation_bad_fraction = torch.clamp(
                primitive_flags[1:2], 0.0, 1.0
            )
            sand_invalid_depth_diagnostics = (
                classify_sand_invalid_plane_depth(
                    plane_stage_diagnostics[
                        "sand_failure_invalid_plane_depth"
                    ],
                    surface_coverage,
                    sand_center_depth,
                    sand_center_depth_std,
                    source_depth,
                    valid_source,
                    out_all_map_sand[:3]
                    / torch.clamp(sand_alpha, min=1.0e-4),
                    normal_second_moments,
                    -view_local,
                    primitive_numeric_bad_fraction,
                    primitive_orientation_bad_fraction,
                    depth_bias=depth_bias,
                    depth_softness=depth_softness,
                    coverage_min=float(getattr(
                        args,
                        "sand_invalid_depth_coverage_min",
                        0.95,
                    )),
                    erode_kernel=int(getattr(
                        args,
                        "sand_invalid_depth_erode_kernel",
                        5,
                    )),
                    primitive_bad_fraction_max=float(getattr(
                        args,
                        "sand_invalid_depth_primitive_bad_fraction_max",
                        0.01,
                    )),
                    incidence_epsilon=float(getattr(
                        args,
                        "sand_invalid_depth_incidence_epsilon",
                        0.02,
                    )),
                    incidence_std_max=float(getattr(
                        args,
                        "sand_invalid_depth_incidence_std_max",
                        0.02,
                    )),
                    normal_variance_max=float(getattr(
                        args,
                        "sand_invalid_depth_normal_variance_max",
                        0.02,
                    )),
                    center_std_relative_max=float(getattr(
                        args,
                        "sand_invalid_depth_center_std_relative_max",
                        0.01,
                    )),
                    recovery_sigma_scale=float(getattr(
                        args,
                        "sand_plane_fallback_sigma_scale",
                        3.0,
                    )),
                    recovery_relative_margin=float(getattr(
                        args,
                        "sand_plane_fallback_relative_margin",
                        0.002,
                    )),
                )
                if invalid_depth_diagnostics_enabled
                else {}
            )
            valid_debug_depth = sand_surface_valid
            sand_depth_vis = torch.zeros_like(resolved_sand_depth)
            if torch.any(valid_debug_depth):
                depth_values = resolved_sand_depth[valid_debug_depth]
                depth_min = torch.min(depth_values)
                depth_range = torch.clamp(
                    torch.max(depth_values) - depth_min, min=1.0e-6
                )
                sand_depth_vis[valid_debug_depth] = (
                    resolved_sand_depth[valid_debug_depth] - depth_min
                ) / depth_range
            result.update(
                {
                    "sand_only": sand_shaded * surface_coverage,
                    "sand_alpha": sand_alpha.repeat(3, 1, 1),
                    "sand_coverage": surface_coverage.repeat(3, 1, 1),
                    "sand_normal": (
                        (resolved_sand_macro_normal_world * 0.5 + 0.5)
                        * valid_sand.to(torch.float32)
                    ),
                    "sand_depth": sand_depth_vis.repeat(3, 1, 1),
                    "sand_occlusion": sand_composite_mask.repeat(3, 1, 1),
                    # ``sand_occlusion`` is the continuous RGB blend mask;
                    # this is the actual discrete G-buffer owner used by P4.
                    "sand_owner": sand_owns_pixel.repeat(3, 1, 1),
                    "sand_plane_depth_valid": (
                        sand_plane_depth_valid.repeat(3, 1, 1)
                    ),
                    "sand_resolved_depth_valid": (
                        resolved_sand_depth_valid.repeat(3, 1, 1)
                    ),
                    "sand_plane_depth_in_front": (
                        plane_sand_in_front.repeat(3, 1, 1)
                    ),
                    "sand_depth_in_front": sand_in_front.repeat(3, 1, 1),
                    # Preserve the signed scalar for exact statistics/NPY;
                    # the RGB visualization is separately bounded.
                    "sand_depth_delta_raw": plane_depth_delta,
                    "sand_resolved_depth_delta_raw": depth_delta,
                    "sand_plane_depth_raw": sand_depth,
                    "sand_center_depth_raw": sand_center_depth,
                    "sand_irradiance": torch.clamp(
                        sand_render_irradiance, 0.0, 1.0
                    ) * valid_sand.to(torch.float32),
                    "sand_specular_environment": (
                        torch.clamp(
                            linear_to_srgb(sand_specular_environment),
                            0.0,
                            1.0,
                        )
                        * valid_sand.to(torch.float32)
                    ),
                    **sand_texture_diagnostics,
                    **sand_recovery_attribute_diagnostics,
                    **sand_shading_diagnostics,
                    **sand_stage_diagnostics,
                    **sand_depth_diagnostics,
                    **sand_plane_reference,
                    **sand_invalid_depth_diagnostics,
                    **sand_fallback_diagnostics,
                    **sand_orientation_recovery_diagnostics,
                    **center_trusted_invalid_depth_diagnostics,
                    **physical_reference_depth_diagnostics,
                    **triangle_footprint_diagnostics,
                }
            )
    else:
        # Pure-water P4 has no dynamic opaque material. Reuse the exact
        # source-only PGSR pass as its immutable substrate instead of
        # manufacturing an empty/dummy sand layer. This makes the optical
        # path identical after the opaque G-buffer boundary.
        source_depth = rendered_depth_src[:1]
        source_normal_camera = torch.nn.functional.normalize(
            out_all_map_src[:3], dim=0, eps=1.0e-6
        )
        source_normal_world = camera_normals_to_world(
            source_normal_camera,
            viewpoint_camera.world_view_transform[:3, :3],
        )
        view_local = perspective_view_directions_camera(
            image_height,
            image_width,
            viewpoint_camera.fx,
            viewpoint_camera.fy,
            viewpoint_camera.cx,
            viewpoint_camera.cy,
            device=source_depth.device,
            dtype=source_depth.dtype,
        )
        source_normal_facing = (
            source_normal_camera * view_local
        ).sum(0, keepdim=True)
        opaque_surface_normal_camera_facing = torch.where(
            (source_normal_facing < 0.0).repeat(3, 1, 1),
            -source_normal_camera,
            source_normal_camera,
        )
        rendered_rgb_composite = rendered_rgb_src
        opaque_surface_depth = source_depth
        opaque_surface_normal_world = source_normal_world
        result = {
            "render": rendered_rgb_src,
            "rendered_src_save": rendered_rgb_src,
            "rendered_image_src_refl_save": rendered_rgb_src,
            "rendered_image_src_refl_refra_save": rendered_rgb_src,
            "num_gausssian_all": num_gausssian_all,
            "source_surface_rgb": rendered_rgb_src,
            "source_surface_depth": source_depth,
            "source_surface_normal_world": source_normal_world,
            "opaque_surface_rgb": rendered_rgb_src,
            "opaque_surface_depth": opaque_surface_depth,
            "opaque_surface_normal_world": opaque_surface_normal_world,
            "opaque_surface_normal_camera_facing": (
                opaque_surface_normal_camera_facing
            ),
            "opaque_surface_sand_owner": torch.zeros_like(
                source_depth, dtype=torch.bool
            ),
            "sand_gbuffer_validation": None,
        }

    return result

