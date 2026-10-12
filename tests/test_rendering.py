from __future__ import annotations

import unittest

import torch

from sandygs.rendering import (
    align_reference_normals,
    beer_lambert_transmittance,
    camera_normals_to_world,
    classify_sand_gbuffer_stages,
    classify_sand_depth_competition,
    classify_sand_invalid_plane_depth,
    classify_sand_center_trusted_invalid_depth,
    classify_sand_physical_reference_depth_loss,
    classify_sand_plane_depth_reference,
    classify_sand_triangle_footprint,
    classify_sand_primitive_overcoverage,
    classify_sand_t3_prerequisites,
    classify_water_gbuffer_stages,
    compare_sand_soft_boundary,
    guide_sand_coverage_with_soft_boundary,
    fallback_invalid_plane_depth,
    linear_to_srgb,
    normalize_environment_irradiance,
    occluded_reflection_proxy,
    opaque_surface_coverage,
    opaque_surface_owner,
    orient_plane_normals_toward_camera,
    perspective_view_directions_camera,
    rasterize_triangle_reference,
    resolve_sand_oriented_plane_depth,
    resolve_sand_center_trusted_invalid_depth,
    resolve_sand_center_trusted_invalid_normal,
    resolve_sand_physical_reference_depth,
    resolve_sand_physical_reference_normal,
    resolve_sand_plane_depth_stability,
    select_sand_recovery_attribute_region,
    sand_height_to_soft_coverage,
    schlick_fresnel,
    screen_space_refraction_grid,
    srgb_to_linear,
    split_sum_specular_reflectance,
    valid_visible_coverage,
    validate_surface_gbuffer,
)


class SandRenderingMathTests(unittest.TestCase):
    def test_physical_normal_recovery_has_independent_exact_mask(self) -> None:
        blended = torch.tensor(
            [
                [[0.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 0.0]],
            ]
        )
        triangle = torch.tensor(
            [
                [[0.0, 0.0]],
                [[0.0, 1.0]],
                [[-1.0, 0.0]],
            ]
        )
        mask = torch.tensor([[[True, False]]])
        resolved = resolve_sand_physical_reference_normal(
            blended, triangle, mask, enabled=True
        )
        self.assertTrue(torch.equal(
            resolved["sand_physical_reference_normal_recovery_used"], mask
        ))
        torch.testing.assert_close(
            resolved["sand_physical_reference_normal_recovery_resolved"][:, 0, 0],
            torch.tensor([0.0, 0.0, 1.0]),
        )

    def test_recovery_attribute_region_selects_one_exact_scope(self) -> None:
        physical = torch.tensor([[[True, True, False]]])
        center = torch.tensor([[[False, False, True]]])
        texture_valid = torch.tensor([[[True, False, True]]])
        triangle_valid = torch.tensor([[[True, True, True]]])

        selected = select_sand_recovery_attribute_region(
            physical,
            center,
            texture_valid,
            triangle_valid,
            physical_enabled=True,
            center_enabled=False,
        )
        self.assertEqual(selected["sand_recovery_attribute_scope"], "physical_reference")
        torch.testing.assert_close(
            selected["sand_recovery_attribute_region_mask"],
            torch.tensor([[[True, False, False]]]),
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            select_sand_recovery_attribute_region(
                physical,
                center,
                texture_valid,
                triangle_valid,
                physical_enabled=True,
                center_enabled=True,
            )

    def test_center_trusted_normal_recovery_is_exactly_masked(self) -> None:
        blended = torch.tensor(
            [
                [[0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0]],
            ]
        )
        triangle = torch.tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, -1.0]],
                [[-1.0, 1.0, 0.0]],
            ]
        )
        mask = torch.tensor([[[True, False, False]]])

        disabled = resolve_sand_center_trusted_invalid_normal(
            blended, triangle, mask, enabled=False
        )
        torch.testing.assert_close(
            disabled[
                "sand_center_trusted_invalid_normal_recovery_resolved"
            ],
            blended,
        )
        self.assertFalse(
            disabled[
                "sand_center_trusted_invalid_normal_recovery_used"
            ].any()
        )

        enabled = resolve_sand_center_trusted_invalid_normal(
            blended, triangle, mask, enabled=True
        )
        expected = blended.clone()
        # Triangle winding is opposite at the recovered pixel, so the helper
        # aligns it to the original normal hemisphere before replacement.
        expected[:, 0, 0] = torch.tensor([0.0, 0.0, 1.0])
        torch.testing.assert_close(
            enabled[
                "sand_center_trusted_invalid_normal_recovery_resolved"
            ],
            expected,
        )
        torch.testing.assert_close(
            enabled[
                "sand_center_trusted_invalid_normal_recovery_used"
            ],
            mask,
        )

    def test_triangle_reference_rejects_misaligned_face_wetness(self) -> None:
        triangles = torch.zeros((2, 3, 3))
        with self.assertRaisesRegex(ValueError, "face_wetness"):
            rasterize_triangle_reference(
                triangles,
                torch.eye(4),
                torch.eye(4),
                8,
                8,
                triangle_face_wetness=torch.zeros(3),
            )

    def test_reference_normal_alignment_ignores_triangle_winding(self) -> None:
        shading = torch.tensor(
            [[[0.0, 1.0]], [[0.0, 0.0]], [[1.0, 0.0]]]
        )
        reference = torch.tensor(
            [[[0.0, 0.0]], [[0.0, 1.0]], [[-1.0, 0.0]]]
        )
        valid = torch.tensor([[[True, False]]])

        aligned, angle = align_reference_normals(reference, shading, valid)

        torch.testing.assert_close(
            aligned[:, 0, 0], torch.tensor([0.0, 0.0, 1.0])
        )
        torch.testing.assert_close(angle[:, 0, 0], torch.tensor([0.0]))
        torch.testing.assert_close(aligned[:, 0, 1], torch.zeros(3))
        torch.testing.assert_close(angle[:, 0, 1], torch.zeros(1))

    def test_t3_prerequisite_gate_classifies_only_dense_visible_interior(
        self,
    ) -> None:
        scalar = torch.ones((1, 1, 5))
        resolved_depth = torch.full_like(scalar, 4.0)
        resolved_depth[0, 0, 1] = 4.1
        resolved_depth[0, 0, 4] = 4.1
        center_depth = torch.full_like(scalar, 4.0)
        center_std = torch.zeros_like(scalar)
        center_std[0, 0, 3] = 0.08
        gaussian_normal = torch.zeros((3, 1, 5))
        gaussian_normal[2] = 1.0
        reference_normal = gaussian_normal.clone()
        reference_normal[:, 0, 2] = torch.tensor([1.0, 0.0, 0.0])
        reference_normal[:, 0, 4] = torch.tensor([1.0, 0.0, 0.0])
        projected_edge = torch.tensor([[[2.0, 2.0, 2.0, 2.0, 6.0]]])

        result = classify_sand_t3_prerequisites(
            scalar,
            scalar,
            scalar.to(torch.bool),
            resolved_depth,
            center_depth,
            center_std,
            gaussian_normal,
            scalar.to(torch.bool),
            torch.full_like(scalar, 4.0),
            reference_normal,
            projected_edge,
            torch.ones_like(scalar),
            erode_kernel=1,
        )

        self.assertTrue(bool(result["sand_t3_audit_region"].all()))
        self.assertTrue(result["sand_t3_depth_inconsistent"][0, 0, 1])
        self.assertTrue(result["sand_t3_normal_mismatch"][0, 0, 2])
        self.assertTrue(result["sand_t3_center_depth_mixed"][0, 0, 3])
        self.assertTrue(result["sand_t3_multiple_failure"][0, 0, 4])
        self.assertTrue(result["sand_t3_failure_on_large_triangle"][0, 0, 4])
        torch.testing.assert_close(
            result["sand_t3_failure_classification"][:, 0, 0],
            torch.tensor([0.22, 0.22, 0.22]),
        )
        torch.testing.assert_close(
            result["sand_t3_failure_classification"][:, 0, 4],
            torch.tensor([1.0, 0.0, 1.0]),
        )

    def test_soft_boundary_guidance_only_removes_gaussian_coverage(self) -> None:
        current = torch.tensor([[[0.8, 0.4, 0.2, 0.0]]])
        target = torch.tensor([[[0.5, 0.7, 0.0, 0.4]]])
        result = guide_sand_coverage_with_soft_boundary(
            current, target, enabled=True
        )
        torch.testing.assert_close(
            result["sand_guided_coverage"],
            torch.tensor([[[0.5, 0.4, 0.0, 0.0]]]),
        )
        torch.testing.assert_close(
            result["sand_guidance_removed_coverage"],
            torch.tensor([[[0.3, 0.0, 0.2, 0.0]]]),
        )
        torch.testing.assert_close(
            result["sand_guidance_residual_undercoverage"],
            torch.tensor([[[0.0, 0.3, 0.0, 0.4]]]),
        )
        disabled = guide_sand_coverage_with_soft_boundary(
            current, target, enabled=False
        )
        torch.testing.assert_close(
            disabled["sand_guided_coverage"], current
        )

    def test_primitive_overcoverage_attribution_and_depth_classes(self) -> None:
        target = torch.tensor([[[0.0, 0.2, 0.5, 0.0]]])
        current = torch.tensor([[[0.8, 0.8, 0.8, 0.4]]])
        hard_reference = torch.tensor([[[False, True, True, True]]])
        front = torch.tensor([[[0.1, 0.7, 0.2, 0.0]]])
        rear = torch.tensor([[[0.9, 0.2, 0.1, 0.0]]])
        other = torch.tensor([[[0.2, 0.3, 0.8, 0.0]]])
        resolved_depth = torch.tensor([[[4.0, 4.0, 4.2, -1.0]]])
        reference_depth = torch.tensor([[[0.0, 4.0, 4.0, 4.0]]])

        result = classify_sand_primitive_overcoverage(
            target,
            current,
            hard_reference,
            front,
            rear,
            other,
            resolved_depth,
            reference_depth,
            depth_absolute_tolerance=0.05,
            depth_relative_tolerance=0.0,
        )
        torch.testing.assert_close(
            result["sand_steep_rear_overcoverage"],
            torch.tensor([[[0.8, 0.0, 0.0, 0.0]]]),
        )
        torch.testing.assert_close(
            result["sand_steep_front_overcoverage"],
            torch.tensor([[[0.0, 0.6, 0.0, 0.0]]]),
        )
        torch.testing.assert_close(
            result["sand_other_primitive_overcoverage"],
            torch.tensor([[[0.0, 0.0, 0.3, 0.0]]]),
        )
        self.assertTrue(
            result["sand_primitive_attribution_unattributed"][0, 0, 3]
        )
        self.assertTrue(result["sand_mesh_depth_no_reference"][0, 0, 0])
        self.assertTrue(result["sand_mesh_depth_consistent"][0, 0, 1])
        self.assertTrue(result["sand_mesh_depth_inconsistent"][0, 0, 2])
        self.assertTrue(result["sand_mesh_depth_invalid"][0, 0, 3])

    def test_physical_sand_height_defines_soft_boundary_independent_of_alpha(
        self,
    ) -> None:
        height = torch.tensor([[[0.1, 0.2, 0.3, 0.5, 0.3]]])
        reference = torch.tensor([[[True, True, True, True, False]]])
        target = sand_height_to_soft_coverage(
            height,
            reference,
            sand_epsilon=0.1,
            fade_height=0.3,
        )
        torch.testing.assert_close(
            target, torch.tensor([[[0.0, 0.5, 1.0, 1.0, 0.0]]])
        )

        diagnostics = compare_sand_soft_boundary(
            target, torch.tensor([[[0.25, 0.25, 1.0, 0.75, 0.0]]])
        )
        torch.testing.assert_close(
            diagnostics["sand_boundary_signed_error_raw"],
            torch.tensor([[[0.25, -0.25, 0.0, -0.25, 0.0]]]),
        )
        torch.testing.assert_close(
            diagnostics["sand_boundary_over_coverage"],
            torch.tensor([[[0.25, 0.0, 0.0, 0.0, 0.0]]]),
        )
        torch.testing.assert_close(
            diagnostics["sand_boundary_under_coverage"],
            torch.tensor([[[0.0, 0.25, 0.0, 0.25, 0.0]]]),
        )

    def test_oriented_plane_recovery_is_restricted_to_safe_invalid_interior(
        self,
    ) -> None:
        shape = (1, 7, 7)
        legacy = torch.full(shape, 4.0)
        invalid = torch.zeros(shape, dtype=torch.bool)
        oriented = torch.full(shape, 4.0)
        center = torch.full(shape, 4.0)
        center_std = torch.full(shape, 0.001)
        source = torch.full(shape, 5.0)
        source_valid = torch.ones(shape, dtype=torch.bool)
        coverage = torch.ones(shape)

        boundary_at = (0, 0, 0)
        candidate_invalid_at = (0, 2, 2)
        uncertain_at = (0, 2, 3)
        disagreement_at = (0, 3, 2)
        insufficient_front_at = (0, 3, 3)
        eligible_at = (0, 4, 4)
        for index in (
            boundary_at,
            candidate_invalid_at,
            uncertain_at,
            disagreement_at,
            insufficient_front_at,
            eligible_at,
        ):
            invalid[index] = True
            legacy[index] = -1.0

        oriented[candidate_invalid_at] = -1.0
        center_std[uncertain_at] = 1.0
        oriented[disagreement_at] = 4.5
        source[insufficient_front_at] = 4.002

        result = resolve_sand_oriented_plane_depth(
            legacy,
            invalid,
            oriented,
            center,
            center_std,
            source,
            source_valid,
            coverage,
            enabled=True,
            depth_bias=0.0,
            depth_softness=0.005,
            erode_kernel=3,
        )

        self.assertTrue(result["sand_orientation_recovery_boundary"][boundary_at])
        self.assertTrue(
            result["sand_orientation_recovery_candidate_invalid"][
                candidate_invalid_at
            ]
        )
        self.assertTrue(
            result["sand_orientation_recovery_uncertain_reference"][uncertain_at]
        )
        self.assertTrue(
            result["sand_orientation_recovery_depth_disagreement"][disagreement_at]
        )
        self.assertTrue(
            result["sand_orientation_recovery_insufficient_front_margin"][
                insufficient_front_at
            ]
        )
        self.assertTrue(result["sand_orientation_recovery_eligible"][eligible_at])
        self.assertTrue(result["sand_orientation_recovery_used"][eligible_at])
        self.assertEqual(
            float(result["sand_orientation_recovery_resolved_depth"][eligible_at]),
            4.0,
        )
        # A valid legacy plane is never changed, even when the candidate pass
        # exists. This preserves every prior behind-source decision.
        self.assertFalse(result["sand_orientation_recovery_used"][0, 5, 5])
        self.assertEqual(
            float(result["sand_orientation_recovery_resolved_depth"][0, 5, 5]),
            4.0,
        )
        partition_keys = (
            "sand_orientation_recovery_boundary",
            "sand_orientation_recovery_candidate_invalid",
            "sand_orientation_recovery_uncertain_reference",
            "sand_orientation_recovery_depth_disagreement",
            "sand_orientation_recovery_insufficient_front_margin",
            "sand_orientation_recovery_eligible",
        )
        self.assertEqual(
            sum(int(result[key].sum()) for key in partition_keys),
            int(invalid.sum()),
        )

    def test_plane_normals_are_oriented_before_depth_blending(self) -> None:
        normals = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        centers = torch.tensor(
            [
                [1.0, 0.0, 4.0],
                [-1.0, 0.0, 4.0],
                [0.0, 0.0, 4.0],
            ]
        )
        oriented, flipped = orient_plane_normals_toward_camera(
            normals, centers
        )
        center_rays = centers / centers[:, 2:3]
        denominators = -torch.sum(oriented * center_rays, dim=1)

        self.assertTrue(flipped[0])
        self.assertFalse(flipped[1])
        self.assertFalse(flipped[2])
        self.assertGreater(float(denominators[0]), 0.0)
        self.assertGreater(float(denominators[1]), 0.0)
        # A true grazing plane stays at zero for the existing invalid-depth
        # guard instead of being assigned an arbitrary orientation.
        self.assertEqual(float(denominators[2]), 0.0)
        torch.testing.assert_close(
            torch.abs(torch.sum(oriented * centers, dim=1)),
            torch.abs(torch.sum(normals * centers, dim=1)),
        )

        ortho_normals, ortho_flipped = orient_plane_normals_toward_camera(
            torch.tensor([[1.0, 0.0, 0.1]]),
            torch.tensor([[-100.0, 0.0, 4.0]]),
            orthographic=True,
        )
        self.assertTrue(ortho_flipped[0])
        self.assertGreater(float(-ortho_normals[0, 2]), 0.0)

    def test_invalid_plane_depth_cause_and_recovery_partitions(self) -> None:
        shape = (1, 7, 7)
        invalid = torch.zeros(shape, dtype=torch.bool)
        boundary_at = (0, 0, 0)
        singular_at = (0, 3, 1)
        mixed_at = (0, 3, 2)
        numeric_at = (0, 3, 3)
        orientation_at = (0, 3, 4)
        unclassified_at = (0, 3, 5)
        for index in (
            boundary_at,
            singular_at,
            mixed_at,
            numeric_at,
            orientation_at,
            unclassified_at,
        ):
            invalid[index] = True

        coverage = torch.ones(shape)
        center = torch.full(shape, 4.0)
        center_std = torch.full(shape, 0.001)
        source = torch.full(shape, 5.0)
        source_valid = torch.ones(shape, dtype=torch.bool)
        # Default pixels are a coherent front-facing plane. With a +Z camera
        # ray, n=[0,0,-1] has a positive rasterizer denominator.
        mean_normal = torch.zeros((3, 7, 7))
        mean_normal[2] = -1.0
        moments = torch.zeros((6, 7, 7))
        moments[2] = 1.0
        ray = torch.zeros((3, 7, 7))
        ray[2] = 1.0
        numeric_fraction = torch.zeros(shape)
        orientation_fraction = torch.zeros(shape)

        # A coherent x-facing plane is genuinely grazing for the +Z ray.
        mean_normal[:, singular_at[1], singular_at[2]] = torch.tensor(
            [1.0, 0.0, 0.0]
        )
        moments[:, singular_at[1], singular_at[2]] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        # Zero mean with non-zero E[n_z^2] represents cancellation between
        # opposite normals rather than a single stable grazing plane.
        mean_normal[:, mixed_at[1], mixed_at[2]] = 0.0
        moments[:, mixed_at[1], mixed_at[2]] = torch.tensor(
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        )
        numeric_fraction[numeric_at] = 0.5
        orientation_fraction[orientation_at] = 0.5

        # Exercise all recovery-reference branches independently of cause.
        center_std[mixed_at] = 1.0
        center[numeric_at] = 6.0
        center[orientation_at] = 4.998
        result = classify_sand_invalid_plane_depth(
            invalid,
            coverage,
            center,
            center_std,
            source,
            source_valid,
            mean_normal,
            moments,
            ray,
            numeric_fraction,
            orientation_fraction,
            depth_bias=0.0,
            depth_softness=0.005,
            erode_kernel=3,
        )

        self.assertTrue(result["sand_invalid_depth_boundary"][boundary_at])
        self.assertTrue(
            result["sand_invalid_depth_stable_singular"][singular_at]
        )
        self.assertTrue(result["sand_invalid_depth_mixed_planes"][mixed_at])
        self.assertTrue(
            result["sand_invalid_depth_primitive_numeric"][numeric_at]
        )
        self.assertTrue(
            result["sand_invalid_depth_primitive_orientation"][orientation_at]
        )
        self.assertTrue(
            result["sand_invalid_depth_unclassified"][unclassified_at]
        )
        cause_keys = (
            "sand_invalid_depth_boundary",
            "sand_invalid_depth_primitive_numeric",
            "sand_invalid_depth_primitive_orientation",
            "sand_invalid_depth_stable_singular",
            "sand_invalid_depth_mixed_planes",
            "sand_invalid_depth_unclassified",
        )
        self.assertEqual(
            sum(int(result[key].sum()) for key in cause_keys),
            int(invalid.sum()),
        )
        self.assertTrue(
            result["sand_invalid_depth_recovery_uncertain"][mixed_at]
        )
        self.assertTrue(
            result["sand_invalid_depth_recovery_center_behind"][numeric_at]
        )
        self.assertTrue(
            result["sand_invalid_depth_recovery_weak_front"][orientation_at]
        )
        self.assertTrue(
            result["sand_invalid_depth_recovery_confident_front"][singular_at]
        )

    def test_sand_plane_depth_fallback_requires_reliable_interior_flip(self) -> None:
        # Pixel 0: stable contradiction -> fallback. Pixel 1: real source
        # occluder according to both depths. Pixel 2: uncertain reference.
        # Pixel 3: contradiction at an eroded image boundary.
        plane = torch.tensor([[[6.0, 6.0, 6.0, 6.0]]]).repeat(1, 7, 2)
        center = torch.tensor([[[4.0, 7.0, 4.0, 4.0]]]).repeat(1, 7, 2)
        std = torch.tensor([[[0.001, 0.001, 1.0, 0.001]]]).repeat(1, 7, 2)
        source = torch.full_like(plane, 5.0)
        valid = torch.ones_like(plane, dtype=torch.bool)
        coverage = torch.ones_like(plane)
        result = resolve_sand_plane_depth_stability(
            plane,
            center,
            std,
            source,
            valid,
            coverage,
            enabled=True,
            depth_bias=0.0,
            depth_softness=0.005,
            erode_kernel=3,
        )
        # Select an interior row. Columns repeat every four entries.
        self.assertTrue(result["sand_plane_fallback_used"][0, 3, 4])
        self.assertFalse(result["sand_plane_fallback_candidate"][0, 3, 5])
        self.assertTrue(
            result["sand_plane_fallback_rejected_by_uncertain_center"][0, 3, 6]
        )
        self.assertTrue(
            result["sand_plane_fallback_rejected_by_boundary"][0, 0, 0]
        )
        self.assertEqual(
            float(result["sand_plane_fallback_resolved_depth"][0, 3, 4]),
            4.0,
        )
        disabled = resolve_sand_plane_depth_stability(
            plane,
            center,
            std,
            source,
            valid,
            coverage,
            enabled=False,
            depth_bias=0.0,
            depth_softness=0.005,
            erode_kernel=3,
        )
        self.assertFalse(torch.any(disabled["sand_plane_fallback_used"]))
        torch.testing.assert_close(
            disabled["sand_plane_fallback_resolved_depth"], plane
        )

    def test_sand_plane_depth_reference_detects_owner_flip(self) -> None:
        plane = torch.tensor([[[6.0, 6.0, 6.0, 6.0]]])
        center = torch.tensor([[[4.0, 5.002, 5.5, 0.0]]])
        source = torch.tensor([[[5.0, 5.0, 5.0, 5.0]]])
        valid = torch.ones_like(plane, dtype=torch.bool)
        behind = torch.ones_like(valid)
        interior = torch.tensor([[[True, True, True, False]]])
        result = classify_sand_plane_depth_reference(
            plane,
            center,
            source,
            valid,
            valid,
            behind,
            interior,
            depth_bias=0.0,
            depth_softness=0.005,
        )
        self.assertTrue(result["sand_plane_reference_sand_in_front"][0, 0, 0])
        self.assertTrue(result["sand_plane_reference_near_behind"][0, 0, 1])
        self.assertTrue(result["sand_plane_reference_strong_behind"][0, 0, 2])
        self.assertTrue(result["sand_plane_reference_invalid"][0, 0, 3])
        self.assertTrue(
            result["sand_plane_reference_interior_disagreement"][0, 0, 0]
        )
        partition = (
            result["sand_plane_reference_sand_in_front"]
            | result["sand_plane_reference_near_behind"]
            | result["sand_plane_reference_strong_behind"]
            | result["sand_plane_reference_invalid"]
        )
        torch.testing.assert_close(partition, behind)

    def test_physical_reference_splits_ambiguous_thin_sand_loss(self) -> None:
        target = torch.ones((1, 1, 5))
        coverage = torch.ones_like(target)
        depth_in_front = torch.zeros_like(target)
        source = torch.full_like(target, 5.0)
        center = torch.tensor([[[6.0, 6.0, 6.0, 6.0, 4.0]]])
        center_margin = torch.full_like(target, 0.01)
        reference_depth = torch.tensor([[[4.0, 6.0, 5.0, 0.0, 4.0]]])
        reference_coverage = torch.tensor(
            [[[True, True, True, False, True]]]
        )
        valid_source = torch.ones_like(target, dtype=torch.bool)

        result = classify_sand_physical_reference_depth_loss(
            target,
            coverage,
            depth_in_front,
            center,
            center_margin,
            reference_depth,
            reference_coverage,
            source,
            valid_source,
            depth_bias=0.0,
            depth_softness=0.01,
        )

        candidate = result["sand_physical_reference_depth_loss_candidate"]
        torch.testing.assert_close(
            candidate,
            torch.tensor([[[True, True, True, True, False]]]),
        )
        self.assertTrue(result["sand_physical_reference_sand_front"][0, 0, 0])
        self.assertTrue(result["sand_physical_reference_source_front"][0, 0, 1])
        self.assertTrue(result["sand_physical_reference_near_coplanar"][0, 0, 2])
        self.assertTrue(result["sand_physical_reference_invalid"][0, 0, 3])
        partition = (
            result["sand_physical_reference_sand_front"]
            | result["sand_physical_reference_source_front"]
            | result["sand_physical_reference_near_coplanar"]
            | result["sand_physical_reference_invalid"]
        )
        torch.testing.assert_close(partition, candidate)
        torch.testing.assert_close(
            result["sand_physical_reference_depth_loss_classification"][:, 0, 0],
            torch.tensor([0.0, 1.0, 0.0]),
        )

    def test_physical_reference_recovery_changes_only_green_depths(self) -> None:
        current = torch.tensor([[[0.0, 6.0, 7.0]]])
        reference = torch.tensor([[[4.0, 5.0, 0.0]]])
        green = torch.tensor([[[True, False, True]]])

        disabled = resolve_sand_physical_reference_depth(
            current, reference, green, enabled=False
        )
        torch.testing.assert_close(
            disabled["sand_physical_reference_recovery_resolved_depth"],
            current,
        )
        self.assertFalse(
            torch.any(disabled["sand_physical_reference_recovery_used"])
        )

        enabled = resolve_sand_physical_reference_depth(
            current, reference, green, enabled=True
        )
        torch.testing.assert_close(
            enabled["sand_physical_reference_recovery_resolved_depth"],
            torch.tensor([[[4.0, 6.0, 7.0]]]),
        )
        torch.testing.assert_close(
            enabled["sand_physical_reference_recovery_used"],
            torch.tensor([[[True, False, False]]]),
        )

    def test_center_trusted_invalid_depth_uses_independent_triangle_split(self) -> None:
        target = torch.ones((1, 1, 6))
        coverage = torch.ones_like(target)
        resolved_valid = torch.zeros_like(target, dtype=torch.bool)
        center = torch.full_like(target, 4.0)
        center_margin = torch.full_like(target, 0.01)
        center_reliable = torch.ones_like(target, dtype=torch.bool)
        reference_depth = torch.tensor([[[4.0, 6.0, 5.0, 0.0, 4.0, 4.0]]])
        reference_coverage = torch.tensor(
            [[[True, True, True, False, True, True]]]
        )
        source = torch.full_like(target, 5.0)
        source_valid = torch.ones_like(target, dtype=torch.bool)
        # The last pixel is not a core owner candidate; the penultimate center
        # reference is deliberately marked unreliable but remains an ordering-
        # confident candidate so the diagnostic can expose that distinction.
        coverage[0, 0, 5] = 0.49
        center_reliable[0, 0, 4] = False

        result = classify_sand_center_trusted_invalid_depth(
            target,
            coverage,
            resolved_valid,
            center,
            center_margin,
            center_reliable,
            reference_depth,
            reference_coverage,
            source,
            source_valid,
            depth_bias=0.0,
            depth_softness=0.01,
        )

        candidate = result["sand_center_trusted_invalid_candidate"]
        torch.testing.assert_close(
            candidate,
            torch.tensor([[[True, True, True, True, True, False]]]),
        )
        self.assertTrue(
            result["sand_center_trusted_invalid_triangle_sand_front"][0, 0, 0]
        )
        self.assertTrue(
            result["sand_center_trusted_invalid_source_front"][0, 0, 1]
        )
        self.assertTrue(
            result["sand_center_trusted_invalid_near_coplanar"][0, 0, 2]
        )
        self.assertTrue(
            result["sand_center_trusted_invalid_reference_invalid"][0, 0, 3]
        )
        self.assertTrue(
            result["sand_center_trusted_invalid_center_reliable"][0, 0, 0]
        )
        self.assertTrue(
            result["sand_center_trusted_invalid_center_unreliable"][0, 0, 4]
        )
        partition = (
            result["sand_center_trusted_invalid_triangle_sand_front"]
            | result["sand_center_trusted_invalid_source_front"]
            | result["sand_center_trusted_invalid_near_coplanar"]
            | result["sand_center_trusted_invalid_reference_invalid"]
        )
        torch.testing.assert_close(partition, candidate)

    def test_center_trusted_invalid_recovery_uses_only_green_triangle_depth(self) -> None:
        current = torch.tensor([[[0.0, 6.0, 7.0]]])
        triangle = torch.tensor([[[4.0, 5.0, 0.0]]])
        green = torch.tensor([[[True, False, True]]])

        disabled = resolve_sand_center_trusted_invalid_depth(
            current, triangle, green, enabled=False
        )
        torch.testing.assert_close(
            disabled["sand_center_trusted_invalid_recovery_resolved_depth"],
            current,
        )
        self.assertFalse(torch.any(
            disabled["sand_center_trusted_invalid_recovery_used"]
        ))

        enabled = resolve_sand_center_trusted_invalid_depth(
            current, triangle, green, enabled=True
        )
        torch.testing.assert_close(
            enabled["sand_center_trusted_invalid_recovery_resolved_depth"],
            torch.tensor([[[4.0, 6.0, 7.0]]]),
        )
        torch.testing.assert_close(
            enabled["sand_center_trusted_invalid_recovery_used"],
            torch.tensor([[[True, False, False]]]),
        )

    def test_sand_depth_competition_splits_interior_rejections(self) -> None:
        depth_delta = torch.ones((1, 7, 7))
        comparison_valid = torch.ones_like(depth_delta, dtype=torch.bool)
        behind = torch.zeros_like(comparison_valid)
        behind[0, 3, 2] = True
        behind[0, 3, 3] = True
        behind[0, 0, 0] = True
        depth_delta[0, 3, 2] = -0.25
        depth_delta[0, 3, 3] = -2.0
        depth_delta[0, 0, 0] = -0.25
        result = classify_sand_depth_competition(
            depth_delta,
            comparison_valid,
            behind,
            torch.ones_like(depth_delta),
            depth_softness=1.0,
            interior_erode_kernel=3,
        )
        self.assertTrue(result["sand_depth_interior_near_coplanar"][0, 3, 2])
        self.assertTrue(result["sand_depth_interior_strong_occluder"][0, 3, 3])
        self.assertTrue(result["sand_depth_behind_boundary"][0, 0, 0])
        torch.testing.assert_close(
            (
                result["sand_depth_behind_boundary"]
                | result["sand_depth_interior_near_coplanar"]
                | result["sand_depth_interior_strong_occluder"]
            ),
            behind,
        )

    def test_sand_gbuffer_stage_classification_matches_owner_gates(self) -> None:
        alpha = torch.tensor([[[0.0, 0.01, 1.0, 1.0, 0.20, 1.0]]])
        coverage = torch.tensor([[[0.0, 0.0, 1.0, 1.0, 0.20, 1.0]]])
        plane_valid = torch.tensor(
            [[[False, False, False, True, True, True]]]
        )
        depth_weight = torch.tensor([[[1.0, 1.0, 1.0, 0.0, 1.0, 1.0]]])
        result = classify_sand_gbuffer_stages(
            alpha,
            coverage,
            plane_valid,
            depth_weight,
            alpha_min=0.02,
        )
        keys = (
            "sand_failure_no_footprint",
            "sand_failure_low_alpha",
            "sand_failure_invalid_plane_depth",
            "sand_failure_behind_source",
            "sand_failure_valid_partial",
            "sand_failure_valid_owner",
        )
        masks = [result[key] for key in keys]
        torch.testing.assert_close(
            torch.stack(masks).to(torch.int64).sum(0),
            torch.ones_like(alpha, dtype=torch.int64),
        )
        self.assertTrue(
            all(bool(mask.flatten()[index]) for index, mask in enumerate(masks))
        )
        expected_owner = opaque_surface_owner(
            coverage,
            depth_weight,
            (alpha > 0.02) & plane_valid & (coverage >= 0.5),
        )
        torch.testing.assert_close(
            result["sand_failure_valid_owner"], expected_owner
        )

    def test_water_gbuffer_stage_classification_is_exhaustive(self) -> None:
        alpha = torch.tensor([[[0.0, 0.01, 1.0, 1.0, 0.20, 1.0]]])
        coverage = torch.tensor([[[0.0, 0.0, 1.0, 1.0, 0.20, 1.0]]])
        plane_valid = torch.tensor(
            [[[False, False, False, True, True, True]]]
        )
        fallback_used = torch.zeros_like(plane_valid)
        in_front = torch.tensor([[[True, True, True, False, True, True]]])
        result = classify_water_gbuffer_stages(
            alpha,
            coverage,
            plane_valid,
            fallback_used,
            in_front,
            alpha_min=0.02,
        )
        masks = [
            result[key]
            for key in (
                "water_failure_no_footprint",
                "water_failure_low_alpha",
                "water_failure_invalid_plane_depth",
                "water_failure_behind_opaque",
                "water_failure_valid_partial",
                "water_failure_valid_owner",
            )
        ]
        torch.testing.assert_close(
            torch.stack(masks).to(torch.int64).sum(0),
            torch.ones_like(alpha, dtype=torch.int64),
        )
        self.assertTrue(
            all(
                bool(mask.flatten()[index])
                for index, mask in enumerate(masks)
            )
        )

    def test_opaque_coverage_is_soft_only_at_the_silhouette(self) -> None:
        alpha = torch.tensor([0.0, 0.02, 0.20, 0.45, 0.80, 1.0])
        coverage = opaque_surface_coverage(alpha, 0.02, 0.80)
        self.assertEqual(float(coverage[0]), 0.0)
        self.assertEqual(float(coverage[1]), 0.0)
        self.assertEqual(float(coverage[-2]), 1.0)
        self.assertEqual(float(coverage[-1]), 1.0)
        self.assertTrue(torch.all(coverage[1:] >= coverage[:-1]))

        # The former 0.45 upper threshold made this alpha fully opaque.  The
        # wider P3 correction retains it as antialiased geometric coverage.
        self.assertLess(float(coverage[3]), 1.0)

    def test_opaque_owner_does_not_average_depth_or_normal(self) -> None:
        coverage = torch.tensor([[[0.49, 0.50, 1.00]]])
        in_front = torch.tensor([[[1.00, 1.00, 0.49]]])
        valid = torch.ones_like(coverage, dtype=torch.bool)
        owner = opaque_surface_owner(coverage, in_front, valid)
        torch.testing.assert_close(
            owner,
            torch.tensor([[[False, True, False]]]),
        )

    def test_invalid_surface_never_replaces_opaque_substrate(self) -> None:
        coverage = torch.tensor([[[1.0, 0.75, 0.40, 0.25]]])
        valid = torch.tensor([[[True, False, True, False]]])
        in_front = torch.tensor([[[True, True, False, False]]])
        visible = valid_visible_coverage(coverage, valid, in_front)
        torch.testing.assert_close(
            visible,
            torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        )

        with self.assertRaisesRegex(ValueError, "equal shapes"):
            valid_visible_coverage(
                coverage, valid[..., :2], in_front
            )

    def test_water_depth_fallback_is_gated_and_keeps_occlusion_separate(self) -> None:
        plane = torch.tensor([[[2.0, 0.0, float("nan"), -1.0]]])
        center = torch.tensor([[[2.1, 3.0, 4.0, 5.0]]])
        alpha = torch.tensor([[[1.0, 1.0, 0.05, 1.0]]])
        coverage = torch.tensor([[[1.0, 1.0, 1.0, 0.40]]])
        resolved, plane_valid, fallback_used = fallback_invalid_plane_depth(
            plane,
            center,
            alpha,
            coverage,
            enabled=True,
            alpha_min=0.10,
            coverage_min=0.50,
        )
        torch.testing.assert_close(
            plane_valid,
            torch.tensor([[[True, False, False, False]]]),
        )
        torch.testing.assert_close(
            fallback_used,
            torch.tensor([[[False, True, False, False]]]),
        )
        self.assertEqual(float(resolved[0, 0, 0]), 2.0)
        self.assertEqual(float(resolved[0, 0, 1]), 3.0)

        # The helper recovers geometry only.  A center depth of 3 remains
        # behind opaque depth 2 and is rejected by the unchanged depth test.
        opaque_depth = torch.tensor([[[4.0, 2.0, 6.0, 6.0]]])
        in_front = resolved < opaque_depth
        self.assertTrue(bool(in_front[0, 0, 0]))
        self.assertFalse(bool(in_front[0, 0, 1]))

        disabled, _, disabled_used = fallback_invalid_plane_depth(
            plane,
            center,
            alpha,
            coverage,
            enabled=False,
        )
        torch.testing.assert_close(disabled, plane, equal_nan=True)
        self.assertFalse(bool(disabled_used.any()))

    def test_camera_normal_transform_inverts_row_vector_convention(self) -> None:
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        normal_world = torch.tensor([1.0, 0.0, 0.0])
        normal_camera = normal_world @ rotation
        recovered = camera_normals_to_world(
            normal_camera[:, None, None], rotation
        )
        torch.testing.assert_close(
            recovered[:, 0, 0], normal_world, atol=1.0e-6, rtol=1.0e-6
        )

    def test_perspective_view_direction_varies_across_image(self) -> None:
        directions = perspective_view_directions_camera(
            3,
            3,
            1.0,
            1.0,
            1.0,
            1.0,
            device="cpu",
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            directions[:, 1, 1], torch.tensor([0.0, 0.0, -1.0])
        )
        self.assertGreater(float(directions[0, 0, 0]), 0.0)
        self.assertGreater(float(directions[1, 0, 0]), 0.0)
        self.assertLess(float(directions[2, 0, 0]), 0.0)
        torch.testing.assert_close(
            torch.linalg.vector_norm(directions, dim=0),
            torch.ones((3, 3)),
        )

    def test_srgb_round_trip_and_environment_normalization(self) -> None:
        srgb = torch.tensor([0.0, 0.02, 0.18, 0.5, 1.0])
        torch.testing.assert_close(
            linear_to_srgb(srgb_to_linear(srgb)),
            srgb,
            atol=1.0e-6,
            rtol=1.0e-6,
        )

        irradiance = torch.full((1, 2, 2, 3), 0.25)
        reference = torch.full((6, 2, 2, 3), 0.5)
        normalized = normalize_environment_irradiance(
            irradiance, reference, 0.1, 2.0
        )
        torch.testing.assert_close(normalized, torch.full_like(normalized, 0.5))

    def test_split_sum_specular_reflectance_uses_f0_scale_and_bias(self) -> None:
        f0 = torch.tensor([[[[0.04], [0.20]]]])
        fg = torch.tensor([[[[0.75, 0.02], [0.50, 0.10]]]])
        reflectance = split_sum_specular_reflectance(f0, fg)
        torch.testing.assert_close(
            reflectance,
            torch.tensor([[[[0.05], [0.20]]]]),
        )
        with self.assertRaisesRegex(ValueError, "spatial shapes"):
            split_sum_specular_reflectance(f0, fg[:, :, :1])

    def test_occluded_reflection_proxy_removes_clear_bright_surface_color(self) -> None:
        local = torch.tensor([[0.8, 0.4, 0.1]])
        environment = torch.tensor([[0.2, 0.3, 0.4]])
        proxy = occluded_reflection_proxy(
            local,
            environment,
            darkness=0.25,
            saturation=0.0,
            environment_weight=0.10,
        )
        # Zero saturation removes the tabletop hue and the darkness factor
        # keeps an unobserved underside substantially dimmer than its top.
        self.assertLess(float(proxy.max()), float(local.max()))
        self.assertLess(float(proxy.max() - proxy.min()), 0.03)

        with self.assertRaisesRegex(ValueError, "darkness"):
            occluded_reflection_proxy(local, environment, darkness=1.1)

    def test_triangle_footprint_diagnostic_partitions_internal_and_outside(
        self,
    ) -> None:
        shape = (1, 1, 14)
        reference = torch.zeros(shape, dtype=torch.bool)
        reference[..., :7] = True
        reference_depth = torch.full(shape, 3.0)
        alpha = torch.full(shape, 0.4)
        alpha[..., 0] = 0.0
        alpha[..., 1] = 0.01
        alpha[..., 7] = 0.0  # strict outside without Gaussian support
        alpha[..., 8] = 0.01
        coverage = torch.full(shape, 0.8)
        coverage[..., 2] = 0.4
        coverage[..., 9] = 0.4
        resolved_depth = torch.full(shape, 3.0)
        depth_valid = torch.ones(shape, dtype=torch.bool)
        depth_valid[..., 3] = False
        depth_valid[..., 10] = False
        in_front = torch.ones(shape)
        in_front[..., 4] = 0.0
        in_front[..., 11] = 0.0
        owner = torch.zeros(shape, dtype=torch.bool)
        owner[..., 5] = True
        owner[..., 12] = True

        result = classify_sand_triangle_footprint(
            reference,
            reference_depth,
            alpha,
            coverage,
            resolved_depth,
            depth_valid,
            in_front,
            owner,
            alpha_min=0.02,
            boundary_kernel=1,
        )

        internal_keys = (
            "no_footprint", "low_alpha", "low_coverage", "invalid_depth",
            "behind_source", "owner", "unclassified",
        )
        for index, name in enumerate(internal_keys):
            self.assertTrue(
                bool(result[f"sand_triangle_internal_{name}"][0, 0, index])
            )
        internal_sum = sum(
            result[f"sand_triangle_internal_{name}"].to(torch.int32)
            for name in internal_keys
        )
        torch.testing.assert_close(internal_sum, reference.to(torch.int32))
        # Pixel 7 has no splat and is intentionally absent from the
        # overcoverage population; pixels 8..13 cover the six later stages.
        self.assertFalse(bool(result["sand_triangle_overcoverage_no_footprint"].any()))
        for index, name in zip(range(8, 14), internal_keys[1:]):
            self.assertTrue(bool(
                result[f"sand_triangle_overcoverage_{name}"][0, 0, index]
            ))

    def test_p4_surface_gbuffer_contract(self) -> None:
        coverage = torch.tensor([[[0.25, 1.0]]])
        depth = torch.tensor([[[2.0, 3.0]]])
        normals = torch.tensor(
            [[[0.0, 0.0]], [[0.0, 0.0]], [[1.0, 1.0]]]
        )
        valid = torch.ones_like(coverage, dtype=torch.bool)
        metrics = validate_surface_gbuffer(
            coverage, depth, normals, valid
        )
        self.assertEqual(metrics["valid_pixels"], 2)
        self.assertEqual(metrics["coverage_max"], 1.0)
        self.assertEqual(metrics["normal_max_length_error"], 0.0)

        with self.assertRaisesRegex(ValueError, "finite and positive"):
            validate_surface_gbuffer(
                coverage, torch.tensor([[[0.0, 3.0]]]), normals, valid
            )

    def test_p4_water_optics_are_thickness_and_angle_dependent(self) -> None:
        cosine = torch.tensor([1.0, 0.0])
        fresnel = schlick_fresnel(cosine, 0.02)
        torch.testing.assert_close(fresnel, torch.tensor([0.02, 1.0]))

        path = torch.tensor([[[0.0, 2.0]]])
        sigma = torch.tensor([1.0, 0.5, 0.25])[:, None, None]
        transmission = beer_lambert_transmittance(path, sigma)
        torch.testing.assert_close(transmission[:, 0, 0], torch.ones(3))
        self.assertTrue(torch.all(transmission[:, 0, 1] < 1.0))

        normal = torch.zeros((3, 2, 3))
        normal[0] = 1.0
        thickness = torch.zeros((1, 2, 3))
        thickness[:, :, 1] = 2.0
        grid = screen_space_refraction_grid(normal, thickness, 1.0)
        self.assertEqual(tuple(grid.shape), (1, 2, 3, 2))
        self.assertGreater(float(grid[0, 0, 1, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
