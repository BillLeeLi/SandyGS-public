"""Persist SandyGS renderer diagnostics without changing render tensors."""

import os
from os import makedirs

import numpy as np
import torch
import torchvision
import yaml


def save_sand_debug_maps(
    render_pkg,
    model_path,
    name,
    image_index,
    output_keys=None,
):
    """Persist optional P3/P4 material views without changing legacy outputs.

    ``output_keys`` is an optional allow-list used by compact comparison runs.
    It limits PNG/NPY files written to disk, while the YAML statistics remain
    available for auditing. Existing runs omit it and retain the full layout.
    """

    selected_keys = None
    if output_keys is not None:
        if not isinstance(output_keys, (list, tuple)) or not all(
            isinstance(key, str) and key for key in output_keys
        ):
            raise ValueError("sand_debug_output_keys must be a list of strings")
        selected_keys = set(output_keys)

    image_stem = (
        f"{image_index:05d}" if isinstance(image_index, int) else str(image_index)
    )

    for key in (
        "sand_only",
        "sand_alpha",
        "sand_coverage",
        "sand_normal",
        "sand_depth",
        "sand_occlusion",
        "sand_owner",
        "sand_plane_depth_valid",
        "sand_resolved_depth_valid",
        "sand_plane_depth_in_front",
        "sand_depth_in_front",
        "sand_depth_comparison_valid",
        "sand_depth_high_coverage_interior",
        "sand_depth_behind_boundary",
        "sand_depth_behind_near_coplanar",
        "sand_depth_behind_strong_occluder",
        "sand_depth_interior_near_coplanar",
        "sand_depth_interior_strong_occluder",
        "sand_depth_competition_classification",
        "sand_depth_delta_visualization",
        "sand_plane_reference_valid",
        "sand_plane_reference_invalid",
        "sand_plane_reference_sand_in_front",
        "sand_plane_reference_near_behind",
        "sand_plane_reference_strong_behind",
        "sand_plane_reference_interior_disagreement",
        "sand_plane_reference_interior_near_behind",
        "sand_plane_reference_interior_strong_behind",
        "sand_plane_reference_classification",
        "sand_plane_center_residual_visualization",
        "sand_plane_fallback_candidate",
        "sand_plane_fallback_used",
        "sand_plane_fallback_rejected_by_boundary",
        "sand_plane_fallback_rejected_by_uncertain_center",
        "sand_plane_fallback_rejected_by_adaptive_margin",
        "sand_plane_fallback_reference_reliable",
        "sand_plane_fallback_interior",
        "sand_physical_reference_depth_loss_candidate",
        "sand_physical_reference_sand_front",
        "sand_physical_reference_source_front",
        "sand_physical_reference_near_coplanar",
        "sand_physical_reference_invalid",
        "sand_physical_reference_center_confident_front",
        "sand_physical_reference_depth_loss_classification",
        "sand_physical_reference_recovery_candidate",
        "sand_physical_reference_recovery_used",
        "sand_physical_reference_recovery_added_in_front",
        "sand_physical_reference_recovery_added_coverage",
        "sand_physical_reference_normal_recovery_candidate",
        "sand_physical_reference_normal_recovery_used",
        "sand_physical_reference_normal_recovery_original",
        "sand_physical_reference_normal_recovery_resolved_visualization",
        "sand_physical_reference_normal_recovery_absolute_difference",
        "sand_physical_reference_normal_recovery_angle_visualization",
        "sand_center_trusted_invalid_candidate",
        "sand_center_trusted_invalid_triangle_sand_front",
        "sand_center_trusted_invalid_source_front",
        "sand_center_trusted_invalid_near_coplanar",
        "sand_center_trusted_invalid_reference_invalid",
        "sand_center_trusted_invalid_classification",
        "sand_center_trusted_invalid_physical_core",
        "sand_center_trusted_invalid_owner_support",
        "sand_center_trusted_invalid_center_confident_front",
        "sand_center_trusted_invalid_center_reliable",
        "sand_center_trusted_invalid_center_unreliable",
        "sand_center_trusted_invalid_fallback_boundary",
        "sand_center_trusted_invalid_fallback_uncertain_center",
        "sand_center_trusted_invalid_fallback_adaptive_margin",
        "sand_center_trusted_invalid_fallback_used",
        "sand_center_trusted_invalid_fallback_unaccounted",
        "sand_center_trusted_invalid_recovery_candidate",
        "sand_center_trusted_invalid_recovery_used",
        "sand_center_trusted_invalid_recovery_added_in_front",
        "sand_center_trusted_invalid_recovery_added_coverage",
        "sand_center_trusted_invalid_normal_recovery_candidate",
        "sand_center_trusted_invalid_normal_recovery_used",
        "sand_center_trusted_invalid_normal_recovery_original",
        "sand_center_trusted_invalid_normal_recovery_resolved_visualization",
        "sand_center_trusted_invalid_normal_recovery_absolute_difference",
        "sand_center_trusted_invalid_normal_recovery_angle_visualization",
        "sand_invalid_depth_boundary",
        "sand_invalid_depth_primitive_numeric",
        "sand_invalid_depth_primitive_orientation",
        "sand_invalid_depth_stable_singular",
        "sand_invalid_depth_mixed_planes",
        "sand_invalid_depth_unclassified",
        "sand_invalid_depth_cause_classification",
        "sand_invalid_depth_recovery_uncertain",
        "sand_invalid_depth_recovery_center_behind",
        "sand_invalid_depth_recovery_weak_front",
        "sand_invalid_depth_recovery_confident_front",
        "sand_invalid_depth_recovery_classification",
        "sand_invalid_depth_interior",
        "sand_orientation_recovery_candidate",
        "sand_orientation_recovery_boundary",
        "sand_orientation_recovery_candidate_invalid",
        "sand_orientation_recovery_uncertain_reference",
        "sand_orientation_recovery_depth_disagreement",
        "sand_orientation_recovery_insufficient_front_margin",
        "sand_orientation_recovery_eligible",
        "sand_orientation_recovery_used",
        "sand_orientation_recovery_classification",
        "sand_orientation_recovery_interior",
        "sand_failure_no_footprint",
        "sand_failure_low_alpha",
        "sand_failure_invalid_plane_depth",
        "sand_failure_behind_source",
        "sand_failure_valid_partial",
        "sand_failure_valid_owner",
        "sand_failure_classification",
        "sand_triangle_reference_coverage",
        "sand_triangle_reference_interior",
        "sand_triangle_reference_boundary",
        "sand_triangle_reference_strict_outside",
        "sand_triangle_internal_no_footprint",
        "sand_triangle_internal_low_alpha",
        "sand_triangle_internal_low_coverage",
        "sand_triangle_internal_invalid_depth",
        "sand_triangle_internal_behind_source",
        "sand_triangle_internal_owner",
        "sand_triangle_internal_unclassified",
        "sand_triangle_internal_failure_classification",
        "sand_triangle_overcoverage_no_footprint",
        "sand_triangle_overcoverage_low_alpha",
        "sand_triangle_overcoverage_low_coverage",
        "sand_triangle_overcoverage_invalid_depth",
        "sand_triangle_overcoverage_behind_source",
        "sand_triangle_overcoverage_owner",
        "sand_triangle_overcoverage_unclassified",
        "sand_triangle_overcoverage_classification",
        "sand_target_soft_coverage",
        "sand_boundary_under_coverage",
        "sand_boundary_over_coverage",
        "sand_boundary_coverage_match",
        "sand_boundary_error_visualization",
        "sand_boundary_target_current_overlay",
        "sand_coverage_unguided",
        "sand_guided_coverage",
        "sand_guidance_removed_coverage",
        "sand_guidance_residual_undercoverage",
        "sand_guidance_owner_unguided",
        "sand_guidance_owner_removed",
        "sand_guidance_owner_added",
        "sand_guidance_rgb_absolute_difference",
        "sand_only_unguided",
        "sand_t3_physical_interior",
        "sand_t3_audit_region",
        "sand_t3_depth_inconsistent",
        "sand_t3_normal_mismatch",
        "sand_t3_center_depth_mixed",
        "sand_t3_any_failure",
        "sand_t3_multiple_failure",
        "sand_t3_large_triangle_support",
        "sand_t3_failure_on_large_triangle",
        "sand_t3_failure_classification",
        "sand_steep_front_alpha",
        "sand_steep_rear_alpha",
        "sand_boundary_steep_front_alpha",
        "sand_boundary_steep_rear_alpha",
        "sand_other_primitive_alpha",
        "sand_boundary_steep_front_overcoverage",
        "sand_boundary_steep_rear_overcoverage",
        "sand_steep_front_overcoverage",
        "sand_steep_rear_overcoverage",
        "sand_other_primitive_overcoverage",
        "sand_unattributed_overcoverage",
        "sand_overcoverage_strict_outside",
        "sand_overcoverage_inside_hard_mesh",
        "sand_overcoverage_primitive_classification",
        "sand_mesh_depth_inconsistent_support",
        "sand_mesh_depth_consistency_classification",
        "sand_irradiance",
        "sand_specular_environment",
        "sand_shading_diagnostic_region",
        "sand_shading_albedo",
        "sand_shading_wetness",
        "sand_shading_roughness",
        "sand_shading_specular_strength",
        "sand_shading_directional_diffuse",
        "sand_shading_environment_diffuse",
        "sand_shading_diffuse_irradiance",
        "sand_shading_diffuse_only",
        "sand_shading_specular_term",
        "sand_shading_specular_environment",
        "sand_shading_blended_normal",
        "sand_shading_mesh_face_normal",
        "sand_shading_normal_disagreement",
        "sand_shading_counterfactual_constant_material",
        "sand_shading_counterfactual_mesh_face_normal",
        "sand_shading_counterfactual_flat_normal",
        "sand_shading_mesh_face_irradiance",
        "sand_shading_flat_normal_irradiance",
        "sand_texture_valid",
        "sand_texture_uv_checker",
        "sand_texture_triplanar_weights",
        "sand_texture_triplanar_lod",
        "sand_texture_lod",
        "sand_texture_albedo_raw",
        "sand_texture_albedo_neutralized",
        "sand_texture_albedo_final",
        "sand_texture_wet_response",
        "sand_texture_wet_brightness",
        "sand_texture_roughness_raw",
        "sand_texture_roughness_final",
        "sand_texture_normal_tangent",
        "sand_texture_normal_world",
        "sand_texture_ao_raw",
        "sand_texture_ao_dark_residual",
        "sand_texture_ao_micro",
        "sand_texture_final",
        "sand_recovery_attribute_region",
        "sand_recovery_attribute_current_shading",
        "sand_recovery_attribute_triangle_normal_shading",
        "sand_recovery_attribute_triangle_wetness_shading",
        "sand_recovery_attribute_triangle_normal_wetness_shading",
        "sand_recovery_attribute_triangle_normal_absolute_difference",
        "sand_recovery_attribute_triangle_wetness_absolute_difference",
        "sand_recovery_attribute_triangle_normal_wetness_absolute_difference",
        "sand_recovery_attribute_current_albedo",
        "sand_recovery_attribute_triangle_normal_albedo",
        "sand_recovery_attribute_triangle_wetness_albedo",
        "sand_recovery_attribute_triangle_normal_wetness_albedo",
        "sand_recovery_attribute_current_irradiance",
        "sand_recovery_attribute_triangle_normal_irradiance",
        "sand_recovery_attribute_current_macro_normal",
        "sand_recovery_attribute_triangle_macro_normal",
        "sand_recovery_attribute_current_wetness",
        "sand_recovery_attribute_triangle_face_wetness",
        "water_reflection",
        "water_refraction",
        "water_fresnel",
        "water_transmittance",
        "water_coverage",
        "water_invalid_coverage",
        "water_raw_alpha",
        "water_raw_coverage",
        "water_plane_depth_valid",
        "water_depth_in_front",
        "water_depth_in_front_without_proxy",
        "water_depth_in_front_without_physical_reference",
        "water_terrain_proxy_coverage",
        "water_terrain_proxy_used",
        "water_terrain_proxy_rejected_foreground",
        "water_physical_reference_coverage",
        "water_physical_reference_trusted",
        "water_physical_reference_recovered",
        "water_physical_reference_removed",
        "water_physical_reference_rejected_opaque",
        "water_surface_valid_debug",
        "water_failure_no_footprint",
        "water_failure_low_alpha",
        "water_failure_invalid_plane_depth",
        "water_failure_behind_opaque",
        "water_failure_valid_partial",
        "water_failure_valid_owner",
        "water_failure_classification",
        "water_depth_fallback_used",
        "water_depth_fallback_rejected_by_opaque",
        "water_normal",
        "water_ssr_hit",
        "water_ssr_frontface_hit",
        "water_ssr_backface_rejected",
        "water_ssr_ambiguous",
        "water_ssr_hit_normal_dot",
        "water_ssr_occluded_proxy",
        "water_ssr_raw_reflection",
        "water_ssr_final_reflection",
    ):
        if selected_keys is not None and key not in selected_keys:
            continue
        image = render_pkg.get(key)
        if image is None:
            continue
        debug_group = "water_debug" if key.startswith("water_") else "sand_debug"
        output = os.path.join(model_path, name, debug_group, key)
        makedirs(output, exist_ok=True)
        torchvision.utils.save_image(
            image.clamp(0.0, 1.0),
            os.path.join(output, f"{image_stem}.png"),
        )

    # PNG cannot preserve a signed camera-depth residual. Keep both the raw
    # value and its softness-normalized form as float32 arrays for quantitative
    # inspection; the bounded diverging PNG above is only a visual aid.
    for key in (
        "sand_depth_delta_raw",
        "sand_depth_delta_normalized",
        "sand_plane_depth_raw",
        "sand_center_depth_raw",
        "sand_plane_center_residual_raw",
        "sand_plane_center_residual_normalized",
        "sand_source_center_delta_raw",
        "sand_resolved_depth_delta_raw",
        "sand_plane_fallback_margin_raw",
        "sand_plane_fallback_center_std_raw",
        "sand_plane_fallback_resolved_depth",
        "sand_physical_reference_source_delta_raw",
        "sand_physical_reference_margin_raw",
        "sand_physical_reference_recovery_resolved_depth",
        "sand_physical_reference_normal_recovery_resolved",
        "sand_physical_reference_normal_recovery_angle_degrees_raw",
        "sand_center_trusted_invalid_center_source_delta_raw",
        "sand_center_trusted_invalid_triangle_source_delta_raw",
        "sand_center_trusted_invalid_reference_margin_raw",
        "sand_center_trusted_invalid_recovery_resolved_depth",
        "sand_center_trusted_invalid_normal_recovery_resolved",
        "sand_center_trusted_invalid_normal_recovery_angle_degrees_raw",
        "sand_invalid_depth_incidence_signed_raw",
        "sand_invalid_depth_incidence_abs_raw",
        "sand_invalid_depth_incidence_std_raw",
        "sand_invalid_depth_normal_variance_raw",
        "sand_invalid_depth_primitive_numeric_fraction_raw",
        "sand_invalid_depth_primitive_orientation_fraction_raw",
        "sand_invalid_depth_recovery_margin_raw",
        "sand_orientation_recovery_candidate_depth_raw",
        "sand_orientation_recovery_center_residual_raw",
        "sand_orientation_recovery_margin_raw",
        "sand_orientation_recovery_resolved_depth",
        "sand_triangle_reference_depth_raw",
        "sand_triangle_reference_depth_residual_raw",
        "sand_triangle_reference_sand_height_raw",
        "sand_triangle_reference_face_wetness_raw",
        "sand_triangle_reference_normal_world_raw",
        "sand_triangle_reference_projected_max_edge_raw",
        "sand_triangle_reference_projected_area_raw",
        "sand_boundary_current_coverage_raw",
        "sand_boundary_signed_error_raw",
        "sand_mesh_depth_residual_raw",
        "sand_mesh_depth_tolerance_raw",
        "sand_t3_depth_residual_raw",
        "sand_t3_depth_tolerance_raw",
        "sand_t3_normal_angle_degrees_raw",
        "sand_t3_center_depth_relative_std_raw",
        "sand_t3_projected_max_edge_raw",
        "sand_t3_projected_area_raw",
        "sand_shading_face_blended_angle_degrees_raw",
        "sand_recovery_attribute_normal_angle_degrees_raw",
        "sand_recovery_attribute_wetness_difference_raw",
    ):
        if selected_keys is not None and key not in selected_keys:
            continue
        value = render_pkg.get(key)
        if value is None:
            continue
        output = os.path.join(model_path, name, "sand_debug", key)
        makedirs(output, exist_ok=True)
        np.save(
            os.path.join(output, f"{image_stem}.npy"),
            value.detach().to(torch.float32).squeeze(0).cpu().numpy(),
            allow_pickle=False,
        )

    for key in (
        "water_occlusion_depth_raw",
        "water_occlusion_bias_raw",
        "water_terrain_proxy_depth_raw",
        "water_terrain_proxy_source_delta_raw",
        "water_physical_reference_depth_raw",
    ):
        if selected_keys is not None and key not in selected_keys:
            continue
        value = render_pkg.get(key)
        if value is None:
            continue
        output = os.path.join(model_path, name, "water_debug", key)
        makedirs(output, exist_ok=True)
        np.save(
            os.path.join(output, f"{image_stem}.npy"),
            value.detach().to(torch.float32).squeeze(0).cpu().numpy(),
            allow_pickle=False,
        )

    # T4 shading diagnostics form a causal rather than merely visual audit:
    # fixed material tests material variation, hard mesh normals test Gaussian
    # normal blending, and a constant +Z normal removes all spatial normal
    # variation. Keep compact numerical comparisons beside the PNGs so a later
    # appearance change can be reviewed without estimating differences by eye.
    shading_region = render_pkg.get("sand_shading_diagnostic_region")
    if (
        shading_region is not None
        and (selected_keys is None or "sand_shading_diagnostic_stats" in selected_keys)
    ):
        region = shading_region[:1].to(torch.bool)
        region_pixels = int(region.sum().item())

        def rgb_mae(left_key, right_key):
            left = render_pkg[left_key]
            right = render_pkg[right_key]
            selected = region.repeat(3, 1, 1)
            if not torch.any(selected):
                return None
            return float(torch.mean(torch.abs(left[selected] - right[selected])).item())

        angle = render_pkg[
            "sand_shading_face_blended_angle_degrees_raw"
        ][region]
        if angle.numel():
            levels = torch.tensor(
                [0.0, 0.5, 0.9, 0.95, 0.99, 1.0],
                dtype=angle.dtype,
                device=angle.device,
            )
            names = ("min", "median", "p90", "p95", "p99", "max")
            angle_quantiles = dict(zip(
                names,
                map(float, torch.quantile(angle, levels).tolist()),
            ))
        else:
            angle_quantiles = None
        output = os.path.join(
            model_path, name, "sand_debug", "sand_shading_diagnostic_stats"
        )
        makedirs(output, exist_ok=True)
        with open(
            os.path.join(output, f"{image_stem}.yaml"),
            "w",
            encoding="utf-8",
        ) as stream:
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "diagnostic_region_pixels": region_pixels,
                    "rgb_mae_in_region": {
                        "constant_material_vs_original": rgb_mae(
                            "sand_shading_counterfactual_constant_material",
                            "sand_only",
                        ),
                        "mesh_face_normal_vs_original": rgb_mae(
                            "sand_shading_counterfactual_mesh_face_normal",
                            "sand_only",
                        ),
                        "flat_normal_vs_original": rgb_mae(
                            "sand_shading_counterfactual_flat_normal",
                            "sand_only",
                        ),
                    },
                    "face_vs_blended_normal_angle_degrees": angle_quantiles,
                    "interpretation": {
                        "constant_material": (
                            "bands remaining here are not caused by wetness or "
                            "material interpolation"
                        ),
                        "mesh_face_normal": (
                            "bands here originate in the piecewise-planar mesh; "
                            "differences from original measure Gaussian normal blending"
                        ),
                        "flat_normal": (
                            "bands remaining here originate outside spatially varying normals"
                        ),
                    },
                },
                stream,
                sort_keys=False,
            )

    # The second conservative depth recovery can expose a latent material
    # mixture that was already present in the Gaussian pass. Compare that
    # current appearance against the recovered triangle's own face normal and
    # pre-rasterization wetness, independently and jointly. These statistics
    # diagnose color provenance; they never alter the production render.
    recovery_attribute_region = render_pkg.get(
        "sand_recovery_attribute_region"
    )
    if (
        recovery_attribute_region is not None
        and (
            selected_keys is None
            or "sand_recovery_attribute_stats" in selected_keys
        )
    ):
        region = recovery_attribute_region[:1].to(torch.bool)
        region_rgb = region.repeat(3, 1, 1)
        region_pixels = int(region.sum().item())

        def selected_rgb(key):
            return render_pkg[key][region_rgb].reshape(3, -1)

        def luma_values(key):
            rgb = selected_rgb(key)
            if rgb.shape[1] == 0:
                return rgb.new_empty(0)
            weights = torch.tensor(
                [0.2126, 0.7152, 0.0722],
                device=rgb.device,
                dtype=rgb.dtype,
            )[:, None]
            return torch.sum(rgb * weights, dim=0)

        def quantiles(values):
            values = values.detach().to(torch.float64).flatten().cpu()
            if not values.numel():
                return None
            levels = torch.tensor(
                [0.0, 0.1, 0.5, 0.9, 1.0], dtype=values.dtype
            )
            names = ("min", "p10", "median", "p90", "max")
            return dict(zip(
                names,
                map(float, torch.quantile(values, levels).tolist()),
            ))

        current_shading_key = "sand_recovery_attribute_current_shading"
        variants = {
            "triangle_normal": (
                "sand_recovery_attribute_triangle_normal_shading"
            ),
            "triangle_wetness": (
                "sand_recovery_attribute_triangle_wetness_shading"
            ),
            "triangle_normal_and_wetness": (
                "sand_recovery_attribute_triangle_normal_wetness_shading"
            ),
        }
        current_rgb = selected_rgb(current_shading_key)
        current_luma = luma_values(current_shading_key)
        surface_coverage = render_pkg["sand_coverage"][:1][region]
        comparisons = {}
        for label, key in variants.items():
            variant_rgb = selected_rgb(key)
            variant_luma = luma_values(key)
            delta = variant_luma - current_luma
            comparisons[label] = {
                "mean_luminance": (
                    float(variant_luma.mean().item())
                    if variant_luma.numel()
                    else None
                ),
                "signed_mean_luminance_change_vs_current": (
                    float(delta.mean().item()) if delta.numel() else None
                ),
                "mean_absolute_rgb_change_vs_current": (
                    float(torch.mean(torch.abs(
                        variant_rgb - current_rgb
                    )).item())
                    if variant_rgb.numel()
                    else None
                ),
                "fraction_brighter_than_current": (
                    float((delta > 0.0).to(torch.float32).mean().item())
                    if delta.numel()
                    else None
                ),
                "luminance_change_quantiles": quantiles(delta),
            }

        # The first recovery spans both faint antialiased sand and dense
        # support. Split the normal counterfactual by coverage so a large
        # intrinsic correction at a nearly invisible edge is not confused
        # with a visible correction in the core.
        triangle_normal_luma = luma_values(
            "sand_recovery_attribute_triangle_normal_shading"
        )

        def coverage_bin_summary(lower, upper, include_upper=False):
            if include_upper:
                selected = (
                    (surface_coverage >= lower)
                    & (surface_coverage <= upper)
                )
            else:
                selected = (
                    (surface_coverage >= lower)
                    & (surface_coverage < upper)
                )
            count = int(selected.sum().item())
            if count == 0:
                return {"pixel_count": 0}
            coverage = surface_coverage[selected]
            current = current_luma[selected]
            triangle = triangle_normal_luma[selected]
            safe_coverage = torch.clamp(coverage, min=1.0e-6)
            return {
                "pixel_count": count,
                "mean_surface_coverage": float(coverage.mean().item()),
                "current_coverage_weighted_mean_luminance": float(
                    current.mean().item()
                ),
                "triangle_normal_coverage_weighted_mean_luminance": float(
                    triangle.mean().item()
                ),
                "signed_mean_luminance_change": float(
                    (triangle - current).mean().item()
                ),
                "current_intrinsic_mean_luminance": float(
                    (current / safe_coverage).mean().item()
                ),
                "triangle_normal_intrinsic_mean_luminance": float(
                    (triangle / safe_coverage).mean().item()
                ),
            }

        coverage_strata = {
            "coverage_0.02_to_0.10": coverage_bin_summary(0.02, 0.10),
            "coverage_0.10_to_0.50": coverage_bin_summary(0.10, 0.50),
            "coverage_0.50_to_1.00": coverage_bin_summary(
                0.50, 1.0, include_upper=True
            ),
        }

        current_wetness = render_pkg[
            "sand_recovery_attribute_current_wetness"
        ][:1][region]
        face_wetness = render_pkg[
            "sand_recovery_attribute_triangle_face_wetness"
        ][:1][region]
        wetness_delta = render_pkg[
            "sand_recovery_attribute_wetness_difference_raw"
        ][region]
        normal_angle = render_pkg[
            "sand_recovery_attribute_normal_angle_degrees_raw"
        ][region]
        current_albedo_luma = luma_values(
            "sand_recovery_attribute_current_albedo"
        )
        normal_albedo_luma = luma_values(
            "sand_recovery_attribute_triangle_normal_albedo"
        )
        wet_albedo_luma = luma_values(
            "sand_recovery_attribute_triangle_wetness_albedo"
        )
        current_irradiance_luma = luma_values(
            "sand_recovery_attribute_current_irradiance"
        )
        normal_irradiance_luma = luma_values(
            "sand_recovery_attribute_triangle_normal_irradiance"
        )

        output = os.path.join(
            model_path,
            name,
            "sand_debug",
            "sand_recovery_attribute_stats",
        )
        makedirs(output, exist_ok=True)
        with open(
            os.path.join(output, f"{image_stem}.yaml"),
            "w",
            encoding="utf-8",
        ) as stream:
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "diagnostic_only": True,
                    "production_rgb_unchanged": True,
                    "diagnostic_scope": render_pkg.get(
                        "sand_recovery_attribute_scope", "unknown"
                    ),
                    "recovered_region_pixels": region_pixels,
                    "current_mean_luminance": (
                        float(current_luma.mean().item())
                        if current_luma.numel()
                        else None
                    ),
                    "counterfactual_comparisons": comparisons,
                    "triangle_normal_by_surface_coverage": coverage_strata,
                    "attribute_statistics": {
                        "current_alpha_blended_wetness": {
                            "mean": (
                                float(current_wetness.mean().item())
                                if current_wetness.numel()
                                else None
                            ),
                            "quantiles": quantiles(current_wetness),
                        },
                        "triangle_face_wetness": {
                            "mean": (
                                float(face_wetness.mean().item())
                                if face_wetness.numel()
                                else None
                            ),
                            "quantiles": quantiles(face_wetness),
                        },
                        "triangle_minus_blended_wetness": {
                            "mean": (
                                float(wetness_delta.mean().item())
                                if wetness_delta.numel()
                                else None
                            ),
                            "quantiles": quantiles(wetness_delta),
                        },
                        "triangle_vs_blended_macro_normal_angle_degrees": (
                            quantiles(normal_angle)
                        ),
                        "normal_counterfactual_albedo_luminance_change": (
                            quantiles(
                                normal_albedo_luma - current_albedo_luma
                            )
                        ),
                        "wetness_counterfactual_albedo_luminance_change": (
                            quantiles(wet_albedo_luma - current_albedo_luma)
                        ),
                        "normal_counterfactual_irradiance_luminance_change": (
                            quantiles(
                                normal_irradiance_luma
                                - current_irradiance_luma
                            )
                        ),
                    },
                    "interpretation": {
                        "triangle_normal": (
                            "replaces the blended macro normal and consistently "
                            "re-samples the triplanar material"
                        ),
                        "triangle_wetness": (
                            "replaces only screen-space blended wetness with the "
                            "owning triangle Gaussian's primitive wetness"
                        ),
                        "triangle_normal_and_wetness": (
                            "applies both independent primitive attributes"
                        ),
                    },
                },
                stream,
                sort_keys=False,
            )

    # T3 is intentionally gated by residual errors in the dense, visible
    # physical interior. Boundary antialiasing and source-scene occlusion were
    # already resolved by T2.4 and cannot justify increasing mesh density.
    t3_audit = render_pkg.get("sand_t3_audit_region")
    if t3_audit is not None:
        audit = t3_audit.to(torch.bool)
        audit_pixels = int(audit.sum().item())
        masks = {
            "depth_inconsistent": render_pkg[
                "sand_t3_depth_inconsistent"
            ].to(torch.bool),
            "normal_mismatch": render_pkg[
                "sand_t3_normal_mismatch"
            ].to(torch.bool),
            "center_depth_mixed": render_pkg[
                "sand_t3_center_depth_mixed"
            ].to(torch.bool),
            "any_failure": render_pkg[
                "sand_t3_any_failure"
            ].to(torch.bool),
            "multiple_failure": render_pkg[
                "sand_t3_multiple_failure"
            ].to(torch.bool),
        }
        counts = {
            key: int(value.sum().item()) for key, value in masks.items()
        }
        fractions = {
            key: value / max(audit_pixels, 1)
            for key, value in counts.items()
        }
        large = render_pkg["sand_t3_large_triangle_support"].to(torch.bool)
        failure_on_large = render_pkg[
            "sand_t3_failure_on_large_triangle"
        ].to(torch.bool)
        failure_on_large_count = int(failure_on_large.sum().item())
        failure_on_large_fraction = (
            failure_on_large_count / max(counts["any_failure"], 1)
        )

        def audited_quantiles(key):
            values = render_pkg[key][audit].detach().to(torch.float64).cpu()
            if not values.numel():
                return None
            levels = torch.tensor(
                [0.0, 0.5, 0.9, 0.95, 0.99, 1.0], dtype=values.dtype
            )
            names = ("min", "median", "p90", "p95", "p99", "max")
            return dict(zip(
                names,
                map(float, torch.quantile(values, levels).tolist()),
            ))

        thresholds = dict(render_pkg["sand_t3_thresholds"])
        limits = dict(render_pkg["sand_t3_decision_limits"])
        exceeds_quality_limit = any(
            fractions[key] > float(limits[f"{key}_fraction"])
            for key in (
                "depth_inconsistent",
                "normal_mismatch",
                "center_depth_mixed",
                "any_failure",
            )
        )
        correlated_with_large_triangles = (
            failure_on_large_fraction
            >= float(limits["minimum_failure_on_large_triangle_fraction"])
        )
        if not exceeds_quality_limit:
            recommendation = "skip_t3"
            reason = "all dense-interior failure fractions are within limits"
        elif correlated_with_large_triangles:
            recommendation = "consider_t3"
            reason = "residual failures exceed limits and correlate with large triangles"
        else:
            recommendation = "investigate_non_subdivision_cause"
            reason = "residual failures exceed limits but do not correlate with large triangles"

        stats_path = os.path.join(
            model_path, name, "sand_debug", "sand_t3_prerequisite_stats"
        )
        makedirs(stats_path, exist_ok=True)
        with open(
            os.path.join(stats_path, f"{image_stem}.yaml"),
            "w",
            encoding="utf-8",
        ) as stream:
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "audit_pixels": audit_pixels,
                    "counts": counts,
                    "fractions_of_audit": fractions,
                    "large_triangle_support_pixels": int(large.sum().item()),
                    "failure_on_large_triangle_pixels": failure_on_large_count,
                    "failure_on_large_triangle_fraction_of_failures": float(
                        failure_on_large_fraction
                    ),
                    "quantiles": {
                        "absolute_depth_residual": audited_quantiles(
                            "sand_t3_depth_residual_raw"
                        ),
                        "normal_angle_degrees": audited_quantiles(
                            "sand_t3_normal_angle_degrees_raw"
                        ),
                        "center_depth_relative_std": audited_quantiles(
                            "sand_t3_center_depth_relative_std_raw"
                        ),
                        "projected_max_edge_pixels": audited_quantiles(
                            "sand_t3_projected_max_edge_raw"
                        ),
                        "projected_area_pixels": audited_quantiles(
                            "sand_t3_projected_area_raw"
                        ),
                    },
                    "thresholds": thresholds,
                    "decision_limits": limits,
                    "decision": {
                        "recommendation": recommendation,
                        "reason": reason,
                        "heuristic_only": True,
                    },
                    "classification_legend_rgb": {
                        "black": "outside the dense visible audit region",
                        "gray": "audited and within all tolerances",
                        "red": "depth inconsistency only",
                        "green": "normal mismatch only",
                        "blue": "center-depth mixing only",
                        "magenta": "two or more failure conditions",
                    },
                },
                stream,
                sort_keys=False,
            )

    # T2.1 compares Gaussian coverage against an independently rasterized hard
    # triangle mesh.  Internal and strict-outside partitions are exact, while
    # a neutral boundary band prevents pixel-center/antialias edge rules from
    # dominating the diagnosis.
    triangle_internal_names = (
        "no_footprint",
        "low_alpha",
        "low_coverage",
        "invalid_depth",
        "behind_source",
        "owner",
        "unclassified",
    )
    triangle_overcoverage_names = triangle_internal_names
    triangle_internal_keys = tuple(
        f"sand_triangle_internal_{name}"
        for name in triangle_internal_names
    )
    triangle_overcoverage_keys = tuple(
        f"sand_triangle_overcoverage_{name}"
        for name in triangle_overcoverage_names
    )
    if all(
        render_pkg.get(key) is not None
        for key in triangle_internal_keys + triangle_overcoverage_keys
    ):
        internal_counts = {
            name: int(render_pkg[key].to(torch.int64).sum().item())
            for name, key in zip(
                triangle_internal_names, triangle_internal_keys
            )
        }
        overcoverage_counts = {
            name: int(render_pkg[key].to(torch.int64).sum().item())
            for name, key in zip(
                triangle_overcoverage_names, triangle_overcoverage_keys
            )
        }
        internal_total = sum(internal_counts.values())
        overcoverage_total = sum(overcoverage_counts.values())
        reference_pixels = int(
            render_pkg["sand_triangle_reference_coverage"]
            .to(torch.int64).sum().item()
        )
        boundary_pixels = int(
            render_pkg["sand_triangle_reference_boundary"]
            .to(torch.int64).sum().item()
        )
        residual = render_pkg["sand_triangle_reference_depth_residual_raw"]
        overlap = render_pkg["sand_triangle_reference_depth_overlap"].to(
            torch.bool
        )
        residual_values = residual[overlap].detach().to(torch.float64).cpu()
        residual_quantiles = None
        if residual_values.numel():
            levels = torch.tensor(
                [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0],
                dtype=residual_values.dtype,
            )
            values = torch.quantile(residual_values, levels).tolist()
            residual_quantiles = dict(zip(
                ("min", "p05", "p25", "median", "p75", "p95", "max"),
                map(float, values),
            ))
        stats_path = os.path.join(
            model_path, name, "sand_debug", "sand_triangle_footprint_stats"
        )
        makedirs(stats_path, exist_ok=True)
        with open(
            os.path.join(stats_path, f"{image_stem}.yaml"),
            "w",
            encoding="utf-8",
        ) as stream:
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "reference_pixels": reference_pixels,
                    "reference_interior_pixels": internal_total,
                    "neutral_boundary_band_pixels": boundary_pixels,
                    "internal_counts": internal_counts,
                    "internal_fractions": {
                        key: value / max(internal_total, 1)
                        for key, value in internal_counts.items()
                    },
                    "internal_missing_fraction": (
                        (internal_total - internal_counts["owner"])
                        / max(internal_total, 1)
                    ),
                    "strict_outside_gaussian_support_pixels": (
                        overcoverage_total
                    ),
                    "overcoverage_counts": overcoverage_counts,
                    "overcoverage_fractions": {
                        key: value / max(overcoverage_total, 1)
                        for key, value in overcoverage_counts.items()
                    },
                    "resolved_minus_triangle_depth_quantiles": (
                        residual_quantiles
                    ),
                    "classification_legend_rgb": {
                        "no_footprint": [0, 89, 255],
                        "low_alpha": [0, 255, 255],
                        "low_coverage": [255, 255, 0],
                        "invalid_depth": [255, 0, 255],
                        "behind_source": [255, 128, 0],
                        "owner": [0, 255, 0],
                        "unclassified": [255, 0, 0],
                    },
                },
                stream,
                sort_keys=False,
            )

    target_soft_coverage = render_pkg.get("sand_target_soft_coverage")
    current_soft_coverage = render_pkg.get(
        "sand_boundary_current_coverage_raw"
    )
    if target_soft_coverage is not None and current_soft_coverage is not None:
        target = torch.clamp(target_soft_coverage, 0.0, 1.0)
        current = torch.clamp(current_soft_coverage, 0.0, 1.0)
        absolute_error = torch.abs(current - target)
        under = torch.clamp(target - current, min=0.0)
        over = torch.clamp(current - target, min=0.0)
        soft_intersection = torch.minimum(target, current).sum().item()
        soft_union = torch.maximum(target, current).sum().item()
        target_area = target.sum().item()
        current_area = current.sum().item()
        union_support = torch.maximum(target, current) > 1.0e-6

        def mask_boundary(mask):
            """Return a one-pixel inner contour, including image borders."""

            inverse = (~mask).to(torch.float32).unsqueeze(0)
            inverse = torch.nn.functional.pad(
                inverse, (1, 1, 1, 1), value=1.0
            )
            has_outside_neighbor = torch.nn.functional.max_pool2d(
                inverse, kernel_size=3, stride=1
            ).squeeze(0).to(torch.bool)
            return mask & has_outside_neighbor

        def dilate(mask, radius):
            if radius == 0:
                return mask
            return torch.nn.functional.max_pool2d(
                mask.to(torch.float32).unsqueeze(0),
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            ).squeeze(0).to(torch.bool)

        target_binary = target >= 0.5
        current_binary = current >= 0.5
        target_boundary = mask_boundary(target_binary)
        current_boundary = mask_boundary(current_binary)
        target_boundary_count = int(target_boundary.sum().item())
        current_boundary_count = int(current_boundary.sum().item())
        boundary_metrics = {}
        for tolerance in (1, 2, 4):
            matched_current = current_boundary & dilate(
                target_boundary, tolerance
            )
            matched_target = target_boundary & dilate(
                current_boundary, tolerance
            )
            precision = (
                float(matched_current.sum().item())
                / max(current_boundary_count, 1)
            )
            recall = (
                float(matched_target.sum().item())
                / max(target_boundary_count, 1)
            )
            boundary_metrics[f"tolerance_{tolerance}px"] = {
                "precision": precision,
                "recall": recall,
                "f_score": (
                    2.0 * precision * recall / max(
                        precision + recall, 1.0e-12
                    )
                ),
            }

        stats_path = os.path.join(
            model_path, name, "sand_debug", "sand_soft_boundary_stats"
        )
        makedirs(stats_path, exist_ok=True)
        with open(
            os.path.join(stats_path, f"{image_stem}.yaml"),
            "w",
            encoding="utf-8",
        ) as stream:
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "sand_epsilon": float(render_pkg.get(
                        "sand_boundary_sand_epsilon", 0.0
                    )),
                    "fade_height": float(render_pkg.get(
                        "sand_boundary_fade_height", 1.0
                    )),
                    "target_soft_area": float(target_area),
                    "gaussian_soft_area": float(current_area),
                    "signed_area_bias_gaussian_minus_target": float(
                        current_area - target_area
                    ),
                    "relative_area_bias": float(
                        (current_area - target_area) / max(target_area, 1.0e-12)
                    ),
                    "integrated_under_coverage": float(under.sum().item()),
                    "integrated_over_coverage": float(over.sum().item()),
                    "under_fraction_of_target": float(
                        under.sum().item() / max(target_area, 1.0e-12)
                    ),
                    "over_fraction_of_target": float(
                        over.sum().item() / max(target_area, 1.0e-12)
                    ),
                    "soft_iou": float(
                        soft_intersection / max(soft_union, 1.0e-12)
                    ),
                    "mean_absolute_error_on_union_support": float(
                        absolute_error[union_support].mean().item()
                        if torch.any(union_support)
                        else 0.0
                    ),
                    "target_boundary_pixels_at_0_5": target_boundary_count,
                    "gaussian_boundary_pixels_at_0_5": (
                        current_boundary_count
                    ),
                    "boundary_metrics": boundary_metrics,
                    "visualization_legend": {
                        "sand_boundary_error_visualization": {
                            "red": "Gaussian over target",
                            "green": "matched coverage",
                            "blue": "target under-covered",
                        },
                        "sand_boundary_target_current_overlay": {
                            "red": "current Gaussian coverage",
                            "green": "physical target coverage",
                            "yellow": "agreement",
                        },
                    },
                },
                stream,
                sort_keys=False,
            )

    guided_coverage = render_pkg.get("sand_guided_coverage")
    if guided_coverage is not None:
        target = torch.clamp(
            render_pkg["sand_target_soft_coverage"], 0.0, 1.0
        )
        unguided = torch.clamp(
            render_pkg["sand_coverage_unguided"], 0.0, 1.0
        )
        guided = torch.clamp(guided_coverage, 0.0, 1.0)
        removed = torch.clamp(unguided - guided, min=0.0)
        residual_under = torch.clamp(target - guided, min=0.0)
        residual_over = torch.clamp(guided - target, min=0.0)
        soft_intersection = torch.minimum(target, guided).sum().item()
        soft_union = torch.maximum(target, guided).sum().item()
        owner_before = render_pkg[
            "sand_guidance_owner_unguided"
        ].to(torch.bool)
        owner_after = render_pkg["sand_surface_valid"].to(torch.bool)
        owner_removed = render_pkg[
            "sand_guidance_owner_removed"
        ].to(torch.bool)
        owner_added = render_pkg["sand_guidance_owner_added"].to(torch.bool)
        rgb_difference = render_pkg[
            "sand_guidance_rgb_absolute_difference"
        ]
        changed_rgb_pixels = torch.any(rgb_difference > 1.0e-6, dim=0)
        stats_path = os.path.join(
            model_path, name, "sand_debug", "sand_soft_guidance_stats"
        )
        makedirs(stats_path, exist_ok=True)
        with open(
            os.path.join(stats_path, f"{image_stem}.yaml"),
            "w",
            encoding="utf-8",
        ) as stream:
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "rule": "guided=min(unguided,target_soft_coverage)",
                    "areas": {
                        "target_soft": float(target.sum().item()),
                        "unguided": float(unguided.sum().item()),
                        "guided": float(guided.sum().item()),
                        "removed": float(removed.sum().item()),
                        "residual_undercoverage": float(
                            residual_under.sum().item()
                        ),
                        "residual_overcoverage": float(
                            residual_over.sum().item()
                        ),
                    },
                    "fractions": {
                        "removed_of_unguided": float(
                            removed.sum().item()
                            / max(unguided.sum().item(), 1.0e-12)
                        ),
                        "residual_under_of_target": float(
                            residual_under.sum().item()
                            / max(target.sum().item(), 1.0e-12)
                        ),
                    },
                    "guided_soft_iou": float(
                        soft_intersection / max(soft_union, 1.0e-12)
                    ),
                    "owner_pixels": {
                        "unguided": int(owner_before.sum().item()),
                        "guided": int(owner_after.sum().item()),
                        "removed": int(owner_removed.sum().item()),
                        "added": int(owner_added.sum().item()),
                    },
                    "rgb": {
                        "changed_pixels": int(changed_rgb_pixels.sum().item()),
                        "mean_absolute_difference": float(
                            rgb_difference.mean().item()
                        ),
                        "maximum_absolute_difference": float(
                            rgb_difference.max().item()
                        ),
                    },
                },
                stream,
                sort_keys=False,
            )

    primitive_overcoverage = render_pkg.get(
        "sand_primitive_overcoverage_mask"
    )
    if primitive_overcoverage is not None:
        over = torch.clamp(
            render_pkg["sand_boundary_over_coverage"], 0.0, 1.0
        )
        total_pixels = int(primitive_overcoverage.sum().item())
        total_error = float(over.sum().item())
        attribution_keys = {
            "steep_rear": "sand_primitive_attribution_rear",
            "steep_front": "sand_primitive_attribution_front",
            "other": "sand_primitive_attribution_other",
            "unattributed": "sand_primitive_attribution_unattributed",
        }
        attribution_counts = {
            name: int(render_pkg[key].sum().item())
            for name, key in attribution_keys.items()
        }
        attribution_error = {
            name: float(
                (over * render_pkg[key].to(over.dtype)).sum().item()
            )
            for name, key in attribution_keys.items()
        }
        depth_keys = {
            "consistent": "sand_mesh_depth_consistent",
            "inconsistent": "sand_mesh_depth_inconsistent",
            "no_reference": "sand_mesh_depth_no_reference",
            "invalid_resolved_depth": "sand_mesh_depth_invalid",
        }
        depth_counts = {
            name: int(render_pkg[key].sum().item())
            for name, key in depth_keys.items()
        }
        strict_outside_error = float(
            render_pkg["sand_overcoverage_strict_outside"].sum().item()
        )
        inside_hard_error = float(
            render_pkg["sand_overcoverage_inside_hard_mesh"].sum().item()
        )
        evaluated = render_pkg["sand_mesh_depth_evaluated"].to(torch.bool)
        residual = render_pkg["sand_mesh_depth_residual_raw"]
        residual_quantiles = None
        absolute_residual_quantiles = None
        if torch.any(evaluated):
            residual_values = residual[evaluated].detach().to(torch.float64).cpu()
            levels = torch.tensor(
                [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0],
                dtype=residual_values.dtype,
            )
            names = ("min", "p05", "p25", "median", "p75", "p95", "max")
            residual_quantiles = dict(zip(
                names,
                map(float, torch.quantile(residual_values, levels).tolist()),
            ))
            absolute_residual_quantiles = dict(zip(
                names,
                map(
                    float,
                    torch.quantile(torch.abs(residual_values), levels).tolist(),
                ),
            ))

        stats_path = os.path.join(
            model_path, name, "sand_debug", "sand_primitive_attribution_stats"
        )
        makedirs(stats_path, exist_ok=True)
        with open(
            os.path.join(stats_path, f"{image_stem}.yaml"),
            "w",
            encoding="utf-8",
        ) as stream:
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "thresholds": {
                        "steep_slope": float(render_pkg[
                            "sand_primitive_steep_slope_threshold"
                        ]),
                        "facing_epsilon": float(render_pkg[
                            "sand_primitive_facing_epsilon"
                        ]),
                        "depth_absolute_tolerance": float(render_pkg[
                            "sand_primitive_depth_absolute_tolerance"
                        ]),
                        "depth_relative_tolerance": float(render_pkg[
                            "sand_primitive_depth_relative_tolerance"
                        ]),
                    },
                    "primitive_counts": {
                        "total": int(render_pkg["sand_primitive_count"]),
                        "boundary": int(render_pkg[
                            "sand_primitive_boundary_count"
                        ]),
                        "steep": int(render_pkg[
                            "sand_primitive_steep_count"
                        ]),
                        "steep_front": int(render_pkg[
                            "sand_primitive_steep_front_count"
                        ]),
                        "steep_rear": int(render_pkg[
                            "sand_primitive_steep_rear_count"
                        ]),
                        "boundary_steep": int(render_pkg[
                            "sand_primitive_boundary_steep_count"
                        ]),
                        "boundary_steep_front": int(render_pkg[
                            "sand_primitive_boundary_steep_front_count"
                        ]),
                        "boundary_steep_rear": int(render_pkg[
                            "sand_primitive_boundary_steep_rear_count"
                        ]),
                        "other": int(render_pkg[
                            "sand_primitive_other_count"
                        ]),
                    },
                    "overcoverage": {
                        "pixels": total_pixels,
                        "integrated_error": total_error,
                        "strict_outside_integrated_error": strict_outside_error,
                        "inside_hard_mesh_integrated_error": inside_hard_error,
                        "strict_outside_fraction": (
                            strict_outside_error / max(total_error, 1.0e-12)
                        ),
                        "inside_hard_mesh_fraction": (
                            inside_hard_error / max(total_error, 1.0e-12)
                        ),
                    },
                    "dominant_primitive_attribution": {
                        "counts": attribution_counts,
                        "pixel_fractions": {
                            key: value / max(total_pixels, 1)
                            for key, value in attribution_counts.items()
                        },
                        "integrated_error": attribution_error,
                        "error_fractions": {
                            key: value / max(total_error, 1.0e-12)
                            for key, value in attribution_error.items()
                        },
                    },
                    "clipped_boundary_steep_overlap": {
                        "front_integrated_support": float(render_pkg[
                            "sand_boundary_steep_front_overcoverage"
                        ].sum().item()),
                        "rear_integrated_support": float(render_pkg[
                            "sand_boundary_steep_rear_overcoverage"
                        ].sum().item()),
                    },
                    "mesh_depth_consistency": {
                        "counts": depth_counts,
                        "fractions_of_overcoverage_pixels": {
                            key: value / max(total_pixels, 1)
                            for key, value in depth_counts.items()
                        },
                        "residual_quantiles": residual_quantiles,
                        "absolute_residual_quantiles": (
                            absolute_residual_quantiles
                        ),
                    },
                    "visualization_legends": {
                        "sand_overcoverage_primitive_classification": {
                            "red": "steep + rear dominant",
                            "green": "steep + front dominant",
                            "blue": "other primitive dominant",
                            "yellow": "no independently rasterized subset support",
                        },
                        "sand_mesh_depth_consistency_classification": {
                            "red": "depth inconsistent",
                            "green": "depth consistent",
                            "blue": "outside hard mesh; no reference depth",
                            "yellow": "invalid resolved depth with mesh reference",
                        },
                    },
                    "interpretation_note": (
                        "Subset alphas are rasterized independently; dominant "
                        "attribution is diagnostic rather than an additive "
                        "alpha decomposition."
                    ),
                },
                stream,
                sort_keys=False,
            )

    # Store exact class counts alongside the visual maps. PNG inspection is
    # convenient, but thresholding an 8-bit image should not be required to
    # decide which G-buffer gate rejected a pixel.
    stage_keys = (
        "water_failure_no_footprint",
        "water_failure_low_alpha",
        "water_failure_invalid_plane_depth",
        "water_failure_behind_opaque",
        "water_failure_valid_partial",
        "water_failure_valid_owner",
    )
    if all(render_pkg.get(key) is not None for key in stage_keys):
        counts = {
            key.removeprefix("water_failure_"): int(
                render_pkg[key][0].to(torch.int64).sum().item()
            )
            for key in stage_keys
        }
        total = sum(counts.values())
        stats_path = os.path.join(
            model_path, name, "water_debug", "water_failure_stats"
        )
        makedirs(stats_path, exist_ok=True)
        proxy_used = render_pkg.get("water_terrain_proxy_used")
        proxy_rejected = render_pkg.get(
            "water_terrain_proxy_rejected_foreground"
        )
        proxy_recovered = None
        without_proxy = render_pkg.get(
            "water_depth_in_front_without_proxy"
        )
        with_proxy = render_pkg.get(
            "water_depth_in_front_without_physical_reference"
        )
        if with_proxy is None:
            with_proxy = render_pkg.get("water_depth_in_front")
        if without_proxy is not None and with_proxy is not None:
            proxy_recovered = (
                (~without_proxy[0].to(torch.bool))
                & with_proxy[0].to(torch.bool)
            )
        with open(
            os.path.join(stats_path, f"{image_stem}.yaml"),
            "w",
            encoding="utf-8",
        ) as stream:
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "total_pixels": total,
                    "counts": counts,
                    "fractions": {
                        key: value / total for key, value in counts.items()
                    },
                    "classification_legend_rgb": {
                        "no_footprint": [0, 0, 0],
                        "low_alpha": [0, 89, 255],
                        "invalid_plane_depth": [255, 0, 255],
                        "behind_opaque": [255, 128, 0],
                        "valid_partial": [0, 255, 255],
                        "valid_owner": [0, 255, 0],
                    },
                    "terrain_proxy": {
                        "used_pixels": int(proxy_used[0].sum().item())
                        if proxy_used is not None else 0,
                        "rejected_foreground_pixels": int(
                            proxy_rejected[0].sum().item()
                        ) if proxy_rejected is not None else 0,
                        "recovered_visibility_pixels": int(
                            proxy_recovered.sum().item()
                        ) if proxy_recovered is not None else 0,
                    },
                    "water_physical_reference": {
                        "trusted_pixels": int(
                            render_pkg[
                                "water_physical_reference_trusted"
                            ][0].sum().item()
                        ) if render_pkg.get(
                            "water_physical_reference_trusted"
                        ) is not None else 0,
                        "recovered_visibility_pixels": int(
                            render_pkg[
                                "water_physical_reference_recovered"
                            ][0].sum().item()
                        ) if render_pkg.get(
                            "water_physical_reference_recovered"
                        ) is not None else 0,
                        "removed_visibility_pixels": int(
                            render_pkg[
                                "water_physical_reference_removed"
                            ][0].sum().item()
                        ) if render_pkg.get(
                            "water_physical_reference_removed"
                        ) is not None else 0,
                        "rejected_opaque_pixels": int(
                            render_pkg[
                                "water_physical_reference_rejected_opaque"
                            ][0].sum().item()
                        ) if render_pkg.get(
                            "water_physical_reference_rejected_opaque"
                        ) is not None else 0,
                    },
                },
                stream,
                sort_keys=False,
            )

    sand_stage_keys = (
        "sand_failure_no_footprint",
        "sand_failure_low_alpha",
        "sand_failure_invalid_plane_depth",
        "sand_failure_behind_source",
        "sand_failure_valid_partial",
        "sand_failure_valid_owner",
    )
    if all(render_pkg.get(key) is not None for key in sand_stage_keys):
        counts = {
            key.removeprefix("sand_failure_"): int(
                render_pkg[key][0].to(torch.int64).sum().item()
            )
            for key in sand_stage_keys
        }
        total = sum(counts.values())
        stats_path = os.path.join(
            model_path, name, "sand_debug", "sand_failure_stats"
        )
        makedirs(stats_path, exist_ok=True)
        report = {
            "schema_version": 4,
            "total_pixels": total,
            "counts": counts,
            "fractions": {
                key: value / total for key, value in counts.items()
            },
            "classification_legend_rgb": {
                "no_footprint": [0, 0, 0],
                "low_alpha": [0, 89, 255],
                "invalid_plane_depth": [255, 0, 255],
                "behind_source": [255, 128, 0],
                "valid_partial": [0, 255, 255],
                "valid_owner": [0, 255, 0],
            },
        }

        depth_stage_keys = (
            "sand_depth_comparison_valid",
            "sand_depth_high_coverage_interior",
            "sand_depth_behind_boundary",
            "sand_depth_behind_near_coplanar",
            "sand_depth_behind_strong_occluder",
            "sand_depth_interior_near_coplanar",
            "sand_depth_interior_strong_occluder",
        )
        raw_delta = render_pkg.get("sand_depth_delta_raw")
        normalized_delta = render_pkg.get("sand_depth_delta_normalized")
        if (
            all(render_pkg.get(key) is not None for key in depth_stage_keys)
            and raw_delta is not None
            and normalized_delta is not None
        ):
            depth_counts = {
                key.removeprefix("sand_depth_"): int(
                    render_pkg[key][0].to(torch.int64).sum().item()
                )
                for key in depth_stage_keys
            }
            behind_mask = render_pkg["sand_failure_behind_source"].to(torch.bool)
            behind_raw = raw_delta[behind_mask].detach().to(torch.float64).cpu()
            behind_normalized = (
                normalized_delta[behind_mask]
                .detach()
                .to(torch.float64)
                .cpu()
            )

            def quantiles(values):
                if values.numel() == 0:
                    return None
                levels = torch.tensor(
                    [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0],
                    dtype=values.dtype,
                )
                result = torch.quantile(values, levels).tolist()
                return {
                    name: float(value)
                    for name, value in zip(
                        ("min", "p05", "p25", "median", "p75", "p95", "max"),
                        result,
                    )
                }

            normalized_bins = (
                ("less_than_-8", None, -8.0),
                ("-8_to_-4", -8.0, -4.0),
                ("-4_to_-2", -4.0, -2.0),
                ("-2_to_-1", -2.0, -1.0),
                ("-1_to_-0.5", -1.0, -0.5),
                ("-0.5_to_-0.25", -0.5, -0.25),
                ("-0.25_to_0", -0.25, 0.0),
            )
            histogram = {}
            for label, lower, upper in normalized_bins:
                selected = behind_normalized < upper
                if lower is not None:
                    selected = selected & (behind_normalized >= lower)
                histogram[label] = int(selected.sum().item())

            report["depth_competition"] = {
                "definition": "source_depth - sand_depth + sand_depth_bias",
                "negative_means": "source_wins_depth_test",
                "interior_definition": (
                    "coverage >= 0.95 after 5x5 binary erosion"
                ),
                "counts": depth_counts,
                "behind_source_raw_delta_quantiles": quantiles(behind_raw),
                "behind_source_delta_over_softness_quantiles": quantiles(
                    behind_normalized
                ),
                "behind_source_delta_over_softness_histogram": histogram,
                "classification_legend_rgb": {
                    "background_or_not_behind": [0, 0, 0],
                    "boundary_behind": [255, 128, 0],
                    "interior_near_coplanar": [255, 255, 0],
                    "interior_strong_occluder": [255, 0, 0],
                },
            }

            reference_stage_keys = (
                "sand_plane_reference_valid",
                "sand_plane_reference_invalid",
                "sand_plane_reference_sand_in_front",
                "sand_plane_reference_near_behind",
                "sand_plane_reference_strong_behind",
                "sand_plane_reference_interior_disagreement",
                "sand_plane_reference_interior_near_behind",
                "sand_plane_reference_interior_strong_behind",
            )
            plane_center_residual = render_pkg.get(
                "sand_plane_center_residual_raw"
            )
            source_center_delta = render_pkg.get("sand_source_center_delta_raw")
            if (
                all(
                    render_pkg.get(key) is not None
                    for key in reference_stage_keys
                )
                and plane_center_residual is not None
                and source_center_delta is not None
            ):
                reference_counts = {
                    key.removeprefix("sand_plane_reference_"): int(
                        render_pkg[key][0].to(torch.int64).sum().item()
                    )
                    for key in reference_stage_keys
                }
                reference_valid = render_pkg[
                    "sand_plane_reference_valid"
                ].to(torch.bool)
                rejected_reference_valid = behind_mask & reference_valid
                rejected_plane_center = (
                    plane_center_residual[rejected_reference_valid]
                    .detach()
                    .to(torch.float64)
                    .cpu()
                )
                rejected_source_center = (
                    source_center_delta[rejected_reference_valid]
                    .detach()
                    .to(torch.float64)
                    .cpu()
                )
                interior_strong = render_pkg[
                    "sand_depth_interior_strong_occluder"
                ].to(torch.bool)
                interior_disagreement = reference_counts[
                    "interior_disagreement"
                ]
                interior_strong_count = int(interior_strong.sum().item())
                report["plane_depth_reference"] = {
                    "reference": (
                        "alpha-composited camera-space Gaussian-center z"
                    ),
                    "diagnostic_only": True,
                    "counts": reference_counts,
                    "interior_strong_plane_failures": interior_strong_count,
                    "interior_strong_decision_disagreement_fraction": (
                        interior_disagreement / interior_strong_count
                        if interior_strong_count > 0
                        else None
                    ),
                    "behind_source_plane_minus_center_quantiles": quantiles(
                        rejected_plane_center
                    ),
                    "behind_source_source_minus_center_plus_bias_quantiles": (
                        quantiles(rejected_source_center)
                    ),
                    "classification_legend_rgb": {
                        "not_behind_source": [0, 0, 0],
                        "reference_invalid": [255, 0, 255],
                        "center_reference_says_sand_in_front": [0, 255, 0],
                        "center_reference_near_behind": [255, 255, 0],
                        "center_reference_strong_behind": [255, 0, 0],
                    },
                }

                fallback_stage_keys = (
                    "sand_plane_fallback_candidate",
                    "sand_plane_fallback_used",
                    "sand_plane_fallback_rejected_by_boundary",
                    "sand_plane_fallback_rejected_by_uncertain_center",
                    "sand_plane_fallback_rejected_by_adaptive_margin",
                    "sand_plane_fallback_reference_reliable",
                    "sand_plane_fallback_interior",
                )
                fallback_margin = render_pkg.get(
                    "sand_plane_fallback_margin_raw"
                )
                fallback_center_std = render_pkg.get(
                    "sand_plane_fallback_center_std_raw"
                )
                if (
                    all(
                        render_pkg.get(key) is not None
                        for key in fallback_stage_keys
                    )
                    and fallback_margin is not None
                    and fallback_center_std is not None
                ):
                    fallback_counts = {
                        key.removeprefix("sand_plane_fallback_"): int(
                            render_pkg[key][0].to(torch.int64).sum().item()
                        )
                        for key in fallback_stage_keys
                    }
                    candidate_mask = render_pkg[
                        "sand_plane_fallback_candidate"
                    ].to(torch.bool)
                    candidate_margin = (
                        fallback_margin[candidate_mask]
                        .detach()
                        .to(torch.float64)
                        .cpu()
                    )
                    candidate_std = (
                        fallback_center_std[candidate_mask]
                        .detach()
                        .to(torch.float64)
                        .cpu()
                    )
                    candidate_count = fallback_counts["candidate"]
                    report["plane_depth_stability_fallback"] = {
                        "diagnostic_when_disabled": True,
                        "counts": fallback_counts,
                        "used_fraction_of_candidate": (
                            fallback_counts["used"] / candidate_count
                            if candidate_count > 0
                            else None
                        ),
                        "candidate_margin_quantiles": quantiles(
                            candidate_margin
                        ),
                        "candidate_center_std_quantiles": quantiles(
                            candidate_std
                        ),
                    }

                orientation_stage_keys = (
                    "sand_orientation_recovery_candidate",
                    "sand_orientation_recovery_boundary",
                    "sand_orientation_recovery_candidate_invalid",
                    "sand_orientation_recovery_uncertain_reference",
                    "sand_orientation_recovery_depth_disagreement",
                    "sand_orientation_recovery_insufficient_front_margin",
                    "sand_orientation_recovery_eligible",
                    "sand_orientation_recovery_used",
                )
                if all(
                    render_pkg.get(key) is not None
                    for key in orientation_stage_keys
                ):
                    orientation_counts = {
                        key.removeprefix("sand_orientation_recovery_"): int(
                            render_pkg[key][0].to(torch.int64).sum().item()
                        )
                        for key in orientation_stage_keys
                    }
                    rejected_partition = sum(
                        orientation_counts[key]
                        for key in (
                            "boundary",
                            "candidate_invalid",
                            "uncertain_reference",
                            "depth_disagreement",
                            "insufficient_front_margin",
                            "eligible",
                        )
                    )
                    report["oriented_plane_depth_recovery"] = {
                        "changes_only_original_invalid_depth": True,
                        "diagnostic_when_disabled": True,
                        "counts": orientation_counts,
                        "partition_matches_invalid": (
                            rejected_partition
                            == orientation_counts["candidate"]
                        ),
                        "classification_legend_rgb": {
                            "not_originally_invalid": [0, 0, 0],
                            "boundary": [255, 128, 0],
                            "candidate_invalid": [255, 0, 255],
                            "uncertain_reference": [0, 255, 255],
                            "candidate_center_disagreement": [255, 0, 0],
                            "insufficient_front_margin": [255, 255, 0],
                            "eligible": [0, 255, 0],
                        },
                    }

            physical_reference_keys = (
                "sand_physical_reference_depth_loss_candidate",
                "sand_physical_reference_sand_front",
                "sand_physical_reference_source_front",
                "sand_physical_reference_near_coplanar",
                "sand_physical_reference_invalid",
            )
            physical_source_delta = render_pkg.get(
                "sand_physical_reference_source_delta_raw"
            )
            physical_margin = render_pkg.get(
                "sand_physical_reference_margin_raw"
            )
            physical_sand_height = render_pkg.get(
                "sand_triangle_reference_sand_height_raw"
            )
            physical_center_std = render_pkg.get(
                "sand_plane_fallback_center_std_raw"
            )
            if (
                all(
                    render_pkg.get(key) is not None
                    for key in physical_reference_keys
                )
                and physical_source_delta is not None
                and physical_margin is not None
                and physical_sand_height is not None
                and physical_center_std is not None
            ):
                physical_counts = {
                    key.removeprefix("sand_physical_reference_"): int(
                        render_pkg[key][0].to(torch.int64).sum().item()
                    )
                    for key in physical_reference_keys
                }
                candidate = render_pkg[
                    "sand_physical_reference_depth_loss_candidate"
                ].to(torch.bool)
                sand_front = render_pkg[
                    "sand_physical_reference_sand_front"
                ].to(torch.bool)
                source_front = render_pkg[
                    "sand_physical_reference_source_front"
                ].to(torch.bool)
                near_coplanar = render_pkg[
                    "sand_physical_reference_near_coplanar"
                ].to(torch.bool)
                reference_invalid = render_pkg[
                    "sand_physical_reference_invalid"
                ].to(torch.bool)
                candidate_count = physical_counts["depth_loss_candidate"]
                partition_count = sum(
                    physical_counts[key]
                    for key in (
                        "sand_front",
                        "source_front",
                        "near_coplanar",
                        "invalid",
                    )
                )

                def selected_quantiles(value, mask):
                    return quantiles(
                        value[mask].detach().to(torch.float64).cpu()
                    )

                report["physical_reference_depth_loss"] = {
                    "diagnostic_only": True,
                    "candidate_definition": (
                        "target_soft_coverage>0.02 and gaussian_coverage>0.02 "
                        "and depth_in_front<0.02 and center_depth_not_"
                        "confidently_in_front"
                    ),
                    "reference": (
                        "direct camera-space z-buffer of the simulated sand "
                        "triangle mesh"
                    ),
                    "counts": physical_counts,
                    "partition_matches_candidate": (
                        partition_count == candidate_count
                    ),
                    "reference_sand_front_fraction_of_candidate": (
                        physical_counts["sand_front"] / candidate_count
                        if candidate_count > 0
                        else None
                    ),
                    "source_front_fraction_of_candidate": (
                        physical_counts["source_front"] / candidate_count
                        if candidate_count > 0
                        else None
                    ),
                    "source_minus_triangle_depth_plus_bias_quantiles": {
                        "reference_sand_front": selected_quantiles(
                            physical_source_delta, sand_front
                        ),
                        "source_front": selected_quantiles(
                            physical_source_delta, source_front
                        ),
                        "near_coplanar": selected_quantiles(
                            physical_source_delta, near_coplanar
                        ),
                    },
                    "triangle_sand_height_quantiles": {
                        "candidate": selected_quantiles(
                            physical_sand_height, candidate
                        ),
                        "reference_sand_front": selected_quantiles(
                            physical_sand_height, sand_front
                        ),
                        "source_front": selected_quantiles(
                            physical_sand_height, source_front
                        ),
                        "near_coplanar": selected_quantiles(
                            physical_sand_height, near_coplanar
                        ),
                    },
                    "gaussian_center_depth_std_quantiles": {
                        "candidate": selected_quantiles(
                            physical_center_std, candidate
                        ),
                        "reference_sand_front": selected_quantiles(
                            physical_center_std, sand_front
                        ),
                        "source_front": selected_quantiles(
                            physical_center_std, source_front
                        ),
                    },
                    "reference_margin_quantiles": selected_quantiles(
                        physical_margin, candidate & (~reference_invalid)
                    ),
                    "classification_legend_rgb": {
                        "outside_ambiguous_depth_loss": [0, 0, 0],
                        "triangle_reference_says_sand_in_front": [0, 255, 0],
                        "source_confidently_in_front": [255, 0, 0],
                        "near_coplanar": [255, 255, 0],
                        "triangle_reference_invalid": [255, 0, 255],
                    },
                }
                recovery_candidate = render_pkg.get(
                    "sand_physical_reference_recovery_candidate"
                )
                recovery_used = render_pkg.get(
                    "sand_physical_reference_recovery_used"
                )
                recovery_added_in_front = render_pkg.get(
                    "sand_physical_reference_recovery_added_in_front"
                )
                recovery_added_coverage = render_pkg.get(
                    "sand_physical_reference_recovery_added_coverage"
                )
                if all(
                    value is not None
                    for value in (
                        recovery_candidate,
                        recovery_used,
                        recovery_added_in_front,
                        recovery_added_coverage,
                    )
                ):
                    recovery_candidate_mask = recovery_candidate.to(torch.bool)
                    recovery_used_mask = recovery_used.to(torch.bool)
                    added_mask = recovery_added_in_front > 1.0e-6
                    report["physical_reference_depth_loss"][
                        "production_recovery"
                    ] = {
                        "enabled": bool(render_pkg.get(
                            "sand_physical_reference_recovery_enabled",
                            False,
                        )),
                        "candidate_count": int(
                            recovery_candidate_mask.sum().item()
                        ),
                        "used_count": int(recovery_used_mask.sum().item()),
                        "added_in_front_pixel_count": int(
                            added_mask.sum().item()
                        ),
                        "added_in_front_weight_sum": float(
                            recovery_added_in_front.sum().item()
                        ),
                        "added_visible_coverage_area": float(
                            recovery_added_coverage.sum().item()
                        ),
                        "used_is_subset_of_green_class": bool(
                            torch.all(
                                (~recovery_used_mask) | sand_front
                            ).item()
                        ),
                        "red_source_front_pixels_recovered": int(
                            (recovery_used_mask & source_front).sum().item()
                        ),
                        "yellow_near_coplanar_pixels_recovered": int(
                            (recovery_used_mask & near_coplanar).sum().item()
                        ),
                    }

                physical_normal_candidate = render_pkg.get(
                    "sand_physical_reference_normal_recovery_candidate"
                )
                physical_normal_used = render_pkg.get(
                    "sand_physical_reference_normal_recovery_used"
                )
                physical_normal_angle = render_pkg.get(
                    "sand_physical_reference_normal_recovery_angle_degrees_raw"
                )
                if all(
                    value is not None
                    for value in (
                        physical_normal_candidate,
                        physical_normal_used,
                        physical_normal_angle,
                    )
                ):
                    physical_normal_candidate_mask = (
                        physical_normal_candidate.to(torch.bool)
                    )
                    physical_normal_used_mask = (
                        physical_normal_used.to(torch.bool)
                    )
                    normal_enabled = bool(render_pkg.get(
                        "sand_physical_reference_normal_recovery_enabled",
                        False,
                    ))
                    report["physical_reference_depth_loss"][
                        "production_normal_recovery"
                    ] = {
                        "enabled": normal_enabled,
                        "candidate_count": int(
                            physical_normal_candidate_mask.sum().item()
                        ),
                        "used_count": int(
                            physical_normal_used_mask.sum().item()
                        ),
                        "used_equals_depth_recovery_used": bool(
                            torch.equal(
                                physical_normal_used_mask,
                                render_pkg[
                                    "sand_physical_reference_recovery_used"
                                ].to(torch.bool),
                            )
                            if normal_enabled
                            else not torch.any(
                                physical_normal_used_mask
                            ).item()
                        ),
                        "triangle_vs_blended_normal_angle_degrees": quantiles(
                            physical_normal_angle[
                                physical_normal_candidate_mask
                            ].detach().to(torch.float64).cpu()
                        ),
                        "scope": (
                            "exact first depth-recovery mask; no dilation or "
                            "threshold relaxation"
                        ),
                    }

            center_trusted_keys = (
                "sand_center_trusted_invalid_candidate",
                "sand_center_trusted_invalid_triangle_sand_front",
                "sand_center_trusted_invalid_source_front",
                "sand_center_trusted_invalid_near_coplanar",
                "sand_center_trusted_invalid_reference_invalid",
            )
            center_trusted_reason_keys = (
                "sand_center_trusted_invalid_fallback_boundary",
                "sand_center_trusted_invalid_fallback_uncertain_center",
                "sand_center_trusted_invalid_fallback_adaptive_margin",
                "sand_center_trusted_invalid_fallback_used",
                "sand_center_trusted_invalid_fallback_unaccounted",
            )
            center_trusted_reliability_keys = (
                "sand_center_trusted_invalid_center_reliable",
                "sand_center_trusted_invalid_center_unreliable",
            )
            center_trusted_center_delta = render_pkg.get(
                "sand_center_trusted_invalid_center_source_delta_raw"
            )
            center_trusted_triangle_delta = render_pkg.get(
                "sand_center_trusted_invalid_triangle_source_delta_raw"
            )
            center_trusted_margin = render_pkg.get(
                "sand_center_trusted_invalid_reference_margin_raw"
            )
            if (
                all(render_pkg.get(key) is not None for key in center_trusted_keys)
                and all(
                    render_pkg.get(key) is not None
                    for key in center_trusted_reliability_keys
                )
                and all(
                    render_pkg.get(key) is not None
                    for key in center_trusted_reason_keys
                )
                and center_trusted_center_delta is not None
                and center_trusted_triangle_delta is not None
                and center_trusted_margin is not None
            ):
                center_counts = {
                    key.removeprefix("sand_center_trusted_invalid_"): int(
                        render_pkg[key][0].to(torch.int64).sum().item()
                    )
                    for key in center_trusted_keys
                }
                reason_counts = {
                    key.removeprefix(
                        "sand_center_trusted_invalid_fallback_"
                    ): int(render_pkg[key][0].to(torch.int64).sum().item())
                    for key in center_trusted_reason_keys
                }
                reliability_counts = {
                    key.removeprefix("sand_center_trusted_invalid_center_"): int(
                        render_pkg[key][0].to(torch.int64).sum().item()
                    )
                    for key in center_trusted_reliability_keys
                }
                center_candidate = render_pkg[
                    "sand_center_trusted_invalid_candidate"
                ].to(torch.bool)
                triangle_sand_front = render_pkg[
                    "sand_center_trusted_invalid_triangle_sand_front"
                ].to(torch.bool)
                triangle_reference_valid = (
                    triangle_sand_front
                    | render_pkg[
                        "sand_center_trusted_invalid_source_front"
                    ].to(torch.bool)
                    | render_pkg[
                        "sand_center_trusted_invalid_near_coplanar"
                    ].to(torch.bool)
                )
                center_candidate_count = center_counts["candidate"]
                center_partition_count = sum(
                    center_counts[key]
                    for key in (
                        "triangle_sand_front",
                        "source_front",
                        "near_coplanar",
                        "reference_invalid",
                    )
                )
                report["center_trusted_invalid_depth"] = {
                    "diagnostic_only": not bool(render_pkg.get(
                        "sand_center_trusted_invalid_recovery_enabled",
                        False,
                    )),
                    "candidate_definition": (
                        "target_soft_coverage>=0.5 and gaussian_coverage>=0.5 "
                        "and resolved_plane_depth_invalid and center_depth_"
                        "ordering_confidently_in_front"
                    ),
                    "counts": center_counts,
                    "center_reliability_counts": reliability_counts,
                    "center_reliability_partitions_candidate": (
                        sum(reliability_counts.values())
                        == center_candidate_count
                    ),
                    "partition_matches_candidate": (
                        center_partition_count == center_candidate_count
                    ),
                    "fallback_reason_counts": reason_counts,
                    "fallback_reasons_partition_candidate": (
                        sum(reason_counts.values()) == center_candidate_count
                    ),
                    "triangle_sand_front_fraction_of_candidate": (
                        center_counts["triangle_sand_front"]
                        / center_candidate_count
                        if center_candidate_count > 0
                        else None
                    ),
                    "center_source_delta_quantiles": quantiles(
                        center_trusted_center_delta[center_candidate]
                        .detach().to(torch.float64).cpu()
                    ),
                    "triangle_source_delta_quantiles": quantiles(
                        center_trusted_triangle_delta[triangle_reference_valid]
                        .detach().to(torch.float64).cpu()
                    ),
                    "triangle_reference_margin_quantiles": quantiles(
                        center_trusted_margin[triangle_reference_valid]
                        .detach().to(torch.float64).cpu()
                    ),
                    "triangle_sand_height_quantiles": (
                        quantiles(
                            physical_sand_height[triangle_sand_front]
                            .detach().to(torch.float64).cpu()
                        )
                        if physical_sand_height is not None
                        else None
                    ),
                    "classification_legend_rgb": {
                        "outside_candidate": [0, 0, 0],
                        "triangle_reference_says_sand_in_front": [0, 255, 0],
                        "source_confidently_in_front": [255, 0, 0],
                        "near_coplanar": [255, 255, 0],
                        "triangle_reference_invalid": [255, 0, 255],
                    },
                }
                center_recovery_candidate = render_pkg.get(
                    "sand_center_trusted_invalid_recovery_candidate"
                )
                center_recovery_used = render_pkg.get(
                    "sand_center_trusted_invalid_recovery_used"
                )
                center_recovery_added_in_front = render_pkg.get(
                    "sand_center_trusted_invalid_recovery_added_in_front"
                )
                center_recovery_added_coverage = render_pkg.get(
                    "sand_center_trusted_invalid_recovery_added_coverage"
                )
                if all(
                    value is not None
                    for value in (
                        center_recovery_candidate,
                        center_recovery_used,
                        center_recovery_added_in_front,
                        center_recovery_added_coverage,
                    )
                ):
                    center_recovery_candidate_mask = (
                        center_recovery_candidate.to(torch.bool)
                    )
                    center_recovery_used_mask = (
                        center_recovery_used.to(torch.bool)
                    )
                    center_added_mask = (
                        center_recovery_added_in_front > 1.0e-6
                    )
                    center_source_front = render_pkg[
                        "sand_center_trusted_invalid_source_front"
                    ].to(torch.bool)
                    center_near_coplanar = render_pkg[
                        "sand_center_trusted_invalid_near_coplanar"
                    ].to(torch.bool)
                    report["center_trusted_invalid_depth"][
                        "production_recovery"
                    ] = {
                        "enabled": bool(render_pkg.get(
                            "sand_center_trusted_invalid_recovery_enabled",
                            False,
                        )),
                        "candidate_count": int(
                            center_recovery_candidate_mask.sum().item()
                        ),
                        "used_count": int(
                            center_recovery_used_mask.sum().item()
                        ),
                        "added_in_front_pixel_count": int(
                            center_added_mask.sum().item()
                        ),
                        "added_in_front_weight_sum": float(
                            center_recovery_added_in_front.sum().item()
                        ),
                        "added_visible_coverage_area": float(
                            center_recovery_added_coverage.sum().item()
                        ),
                        "used_is_subset_of_green_class": bool(
                            torch.all(
                                (~center_recovery_used_mask)
                                | triangle_sand_front
                            ).item()
                        ),
                        "red_source_front_pixels_recovered": int(
                            (
                                center_recovery_used_mask
                                & center_source_front
                            ).sum().item()
                        ),
                        "yellow_near_coplanar_pixels_recovered": int(
                            (
                                center_recovery_used_mask
                                & center_near_coplanar
                            ).sum().item()
                        ),
                    }

                normal_recovery_candidate = render_pkg.get(
                    "sand_center_trusted_invalid_normal_recovery_candidate"
                )
                normal_recovery_used = render_pkg.get(
                    "sand_center_trusted_invalid_normal_recovery_used"
                )
                normal_recovery_angle = render_pkg.get(
                    "sand_center_trusted_invalid_normal_recovery_angle_degrees_raw"
                )
                if all(
                    value is not None
                    for value in (
                        normal_recovery_candidate,
                        normal_recovery_used,
                        normal_recovery_angle,
                    )
                ):
                    normal_candidate_mask = (
                        normal_recovery_candidate.to(torch.bool)
                    )
                    normal_used_mask = normal_recovery_used.to(torch.bool)
                    report["center_trusted_invalid_depth"][
                        "production_normal_recovery"
                    ] = {
                        "enabled": bool(render_pkg.get(
                            "sand_center_trusted_invalid_normal_recovery_enabled",
                            False,
                        )),
                        "candidate_count": int(
                            normal_candidate_mask.sum().item()
                        ),
                        "used_count": int(normal_used_mask.sum().item()),
                        "used_equals_depth_recovery_used": bool(
                            torch.equal(
                                normal_used_mask,
                                center_recovery_used_mask,
                            )
                            if bool(render_pkg.get(
                                "sand_center_trusted_invalid_normal_recovery_enabled",
                                False,
                            ))
                            else not torch.any(normal_used_mask).item()
                        ),
                        "triangle_vs_blended_normal_angle_degrees": quantiles(
                            normal_recovery_angle[normal_candidate_mask]
                            .detach().to(torch.float64).cpu()
                        ),
                        "scope": (
                            "exact second depth-recovery mask; no dilation or "
                            "threshold relaxation"
                        ),
                    }

            invalid_cause_keys = (
                "sand_invalid_depth_boundary",
                "sand_invalid_depth_primitive_numeric",
                "sand_invalid_depth_primitive_orientation",
                "sand_invalid_depth_stable_singular",
                "sand_invalid_depth_mixed_planes",
                "sand_invalid_depth_unclassified",
            )
            invalid_recovery_keys = (
                "sand_invalid_depth_boundary",
                "sand_invalid_depth_recovery_uncertain",
                "sand_invalid_depth_recovery_center_behind",
                "sand_invalid_depth_recovery_weak_front",
                "sand_invalid_depth_recovery_confident_front",
            )
            invalid_raw_keys = (
                "sand_invalid_depth_incidence_abs_raw",
                "sand_invalid_depth_incidence_std_raw",
                "sand_invalid_depth_normal_variance_raw",
                "sand_invalid_depth_primitive_numeric_fraction_raw",
                "sand_invalid_depth_primitive_orientation_fraction_raw",
            )
            if (
                all(render_pkg.get(key) is not None for key in invalid_cause_keys)
                and all(
                    render_pkg.get(key) is not None
                    for key in invalid_recovery_keys
                )
                and all(render_pkg.get(key) is not None for key in invalid_raw_keys)
            ):
                # The detailed cause/recovery maps intentionally classify the
                # *original* plane-depth failures.  Once conservative recovery
                # is enabled, ``sand_failure_invalid_plane_depth`` is the
                # smaller production remainder and is no longer the matching
                # population.  Prefer the explicit candidate mask so the YAML
                # partition remains auditable before and after recovery.
                original_invalid = render_pkg.get(
                    "sand_orientation_recovery_candidate"
                )
                invalid_mask = (
                    original_invalid
                    if original_invalid is not None
                    else render_pkg["sand_failure_invalid_plane_depth"]
                ).to(torch.bool)
                invalid_count = int(invalid_mask.sum().item())
                cause_counts = {
                    key.removeprefix("sand_invalid_depth_"): int(
                        render_pkg[key][0].to(torch.int64).sum().item()
                    )
                    for key in invalid_cause_keys
                }
                recovery_counts = {
                    key.removeprefix("sand_invalid_depth_"): int(
                        render_pkg[key][0].to(torch.int64).sum().item()
                    )
                    for key in invalid_recovery_keys
                }
                invalid_values = {
                    key.removeprefix("sand_invalid_depth_").removesuffix(
                        "_raw"
                    ): quantiles(
                        render_pkg[key][invalid_mask]
                        .detach()
                        .to(torch.float64)
                        .cpu()
                    )
                    for key in invalid_raw_keys
                }
                report["invalid_plane_depth_diagnostic"] = {
                    "diagnostic_only": True,
                    "invalid_pixel_count": invalid_count,
                    "cause_counts": cause_counts,
                    "cause_partition_matches_invalid": (
                        sum(cause_counts.values()) == invalid_count
                    ),
                    "recovery_counts": recovery_counts,
                    "recovery_partition_matches_invalid": (
                        sum(recovery_counts.values()) == invalid_count
                    ),
                    "invalid_pixel_quantiles": invalid_values,
                    "cause_legend_rgb": {
                        "not_invalid": [0, 0, 0],
                        "boundary_or_low_coverage": [255, 128, 0],
                        "primitive_numeric_invalid": [255, 0, 255],
                        "primitive_orientation_invalid": [0, 89, 255],
                        "stable_grazing_singularity": [255, 255, 0],
                        "alpha_mixed_planes": [255, 0, 0],
                        "unclassified": [0, 255, 255],
                    },
                    "recovery_legend_rgb": {
                        "not_invalid": [0, 0, 0],
                        "boundary_or_low_coverage": [255, 128, 0],
                        "center_reference_uncertain": [255, 0, 255],
                        "center_reference_behind_source": [255, 0, 0],
                        "center_reference_weakly_in_front": [255, 255, 0],
                        "center_reference_confidently_in_front": [0, 255, 0],
                    },
                }

        with open(
            os.path.join(stats_path, f"{image_stem}.yaml"),
            "w",
            encoding="utf-8",
        ) as stream:
            yaml.safe_dump(report, stream, sort_keys=False)


def save_render_intermediates(render_pkg, model_path, name, image_stem):
    """Save source/reflection diagnostics only when a schedule asks for them."""

    for key, directory in (
        ("rendered_src_save", "src"),
        ("rendered_image_src_refl_save", "src_refl"),
        ("rendered_image_src_refl_refra_save", "src_refl_refra"),
    ):
        image = render_pkg.get(key)
        if image is None:
            continue
        output = os.path.join(model_path, name, directory)
        makedirs(output, exist_ok=True)
        torchvision.utils.save_image(
            image.clamp(0.0, 1.0),
            os.path.join(output, f"{image_stem}.png"),
        )
