from __future__ import annotations

from pathlib import Path
import unittest

import torch
import yaml

from sandygs.materials import (
    SandPBRMaterial,
    ambient_occlusion_from_residual,
    apply_sand_pbr_microtexture,
    apply_sand_texture_primary,
    build_simulation_tangent_basis,
    decode_tangent_normal,
    encode_tangent_normal,
    estimate_texture_lod,
    extract_ambient_occlusion_microdetail,
    highpass_tangent_normal,
    reconstruct_world_positions,
    sample_sand_pbr_material,
    sample_sand_pbr_material_triplanar,
    simulation_triplanar_uvs,
    simulation_triplanar_weights,
    simulation_xy_uv,
    world_positions_to_simulation,
)
from sandygs.render_schedule import expand_render_schedule


def synthetic_material() -> SandPBRMaterial:
    albedo = torch.tensor(
        [
            [[0.20, 0.40], [0.60, 0.80]],
            [[0.30, 0.50], [0.70, 0.90]],
            [[0.40, 0.60], [0.80, 1.00]],
        ]
    )
    roughness = torch.tensor([[[0.20, 0.40], [0.60, 0.80]]])
    normal = torch.zeros((3, 2, 2))
    normal[2] = 1.0
    ambient_occlusion = torch.ones((1, 2, 2))
    ambient_occlusion_dark_residual = torch.zeros((1, 2, 2))
    packed = torch.cat(
        (
            albedo,
            roughness,
            normal,
            ambient_occlusion,
            ambient_occlusion_dark_residual,
        ),
        dim=0,
    )[None]
    return SandPBRMaterial(
        root=Path("/synthetic"),
        levels=(packed,),
        albedo_mean_linear=albedo.mean(dim=(1, 2), keepdim=True),
        roughness_mean=roughness.mean().reshape(1, 1, 1),
        base_height=2,
        base_width=2,
    )


class SandMaterialTests(unittest.TestCase):
    def test_normal_highpass_removes_broad_slope_and_preserves_micro_detail(self) -> None:
        height = width = 16
        columns = torch.arange(width).reshape(1, width)
        checker = ((columns % 2) * 2 - 1).to(torch.float32).expand(height, width)
        slopes = torch.stack(
            (
                0.30 + 0.08 * checker,
                torch.full((height, width), -0.20),
                torch.ones((height, width)),
            )
        )
        normal = torch.nn.functional.normalize(slopes, dim=0)
        filtered = highpass_tangent_normal(
            normal, sigma_texels=1.0, strength=1.0
        )

        # The broad X/Y tilt belongs to the simulated macro surface and is
        # removed, while the alternating grain-scale X slope remains.
        self.assertLess(abs(float(filtered[0].mean())), 1.0e-5)
        self.assertLess(abs(float(filtered[1].mean())), 1.0e-5)
        self.assertGreater(float(filtered[0].std()), 0.05)
        torch.testing.assert_close(
            torch.linalg.vector_norm(filtered, dim=0),
            torch.ones((height, width)),
            atol=1.0e-6,
            rtol=0.0,
        )

        # Encoding is an auditable image export only; decoding it recovers the
        # same unit-vector field up to floating-point precision.
        torch.testing.assert_close(
            decode_tangent_normal(encode_tangent_normal(filtered)),
            filtered,
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_ao_highpass_keeps_only_dark_microdetail(self) -> None:
        ambient_occlusion = torch.full((1, 16, 16), 0.8)
        ambient_occlusion[0, 8, 8] = 0.2
        ambient_occlusion[0, 4, 4] = 1.0
        detail = extract_ambient_occlusion_microdetail(
            ambient_occlusion, sigma_texels=1.0
        )
        self.assertGreater(float(detail["dark_residual"][0, 8, 8]), 0.3)
        self.assertEqual(float(detail["dark_residual"][0, 4, 4]), 0.0)

        micro = ambient_occlusion_from_residual(
            detail["dark_residual"], strength=1.0, minimum=0.4
        )
        self.assertLess(float(micro[0, 8, 8]), 0.7)
        self.assertEqual(float(micro[0, 4, 4]), 1.0)
        self.assertGreaterEqual(float(micro.min()), 0.4)

    def test_sparse_ao_separates_density_thresholds_from_contrast(self) -> None:
        residual = torch.tensor([[[0.00, 0.04, 0.06, 0.08, 0.12]]])
        micro = ambient_occlusion_from_residual(
            residual,
            strength=99.0,  # Sparse mode must not reuse the linear gain.
            minimum=0.4,
            response="sparse_smoothstep",
            sparse_low=0.04,
            sparse_high=0.08,
            sparse_contrast=0.50,
        )
        torch.testing.assert_close(
            micro,
            torch.tensor([[[1.00, 1.00, 0.75, 0.50, 0.50]]]),
            atol=1.0e-6,
            rtol=0.0,
        )

        # Contrast changes darkness but never admits sub-threshold residuals.
        lighter = ambient_occlusion_from_residual(
            residual,
            strength=0.0,
            minimum=0.4,
            response="sparse_smoothstep",
            sparse_low=0.04,
            sparse_high=0.08,
            sparse_contrast=0.25,
        )
        self.assertEqual(float(lighter[0, 0, 1]), 1.0)
        self.assertEqual(float(lighter[0, 0, 4]), 0.75)

    def test_texture_primary_uses_sampled_maps_as_dry_endpoint(self) -> None:
        base_albedo = torch.full((3, 1, 2), 0.25)
        base_roughness = torch.full((1, 1, 2), 0.90)
        macro_normal = torch.zeros((3, 1, 2))
        macro_normal[2] = 1.0
        sampled_linear = torch.tensor(
            [
                [[0.10, 0.20]],
                [[0.30, 0.40]],
                [[0.50, 0.60]],
            ]
        )
        sampled = {
            "albedo_linear": sampled_linear,
            "roughness": torch.tensor([[[0.35, 0.45]]]),
            "normal_tangent": torch.tensor(
                [[[0.0, 0.0]], [[0.0, 0.0]], [[1.0, 1.0]]]
            ),
        }
        valid = torch.tensor([[[True, False]]])

        result = apply_sand_texture_primary(
            base_albedo,
            base_roughness,
            macro_normal,
            sampled,
            torch.eye(4),
            valid,
            albedo_gain=(1.0, 1.0, 1.0),
            roughness_scale=1.0,
            normal_strength=1.0,
        )
        expected_srgb = torch.where(
            sampled_linear <= 0.0031308,
            sampled_linear * 12.92,
            1.055 * torch.pow(sampled_linear, 1.0 / 2.4) - 0.055,
        )

        # Unit gains make the valid dry pixel exactly the sampled material;
        # invalid pixels keep the legacy material and cannot leak texture.
        torch.testing.assert_close(
            result["albedo_srgb"][:, 0, 0], expected_srgb[:, 0, 0]
        )
        torch.testing.assert_close(
            result["roughness"][:, 0, 0], sampled["roughness"][:, 0, 0]
        )
        torch.testing.assert_close(
            result["normal_world"][:, 0, 0], macro_normal[:, 0, 0]
        )
        torch.testing.assert_close(
            result["albedo_srgb"][:, 0, 1], base_albedo[:, 0, 1]
        )
        torch.testing.assert_close(
            result["roughness"][:, 0, 1], base_roughness[:, 0, 1]
        )

    def test_texture_primary_wetness_only_scales_linear_albedo(self) -> None:
        base_albedo = torch.full((3, 1, 4), 0.25)
        base_roughness = torch.full((1, 1, 4), 0.90)
        macro_normal = torch.zeros((3, 1, 4))
        macro_normal[2] = 1.0
        sampled_linear = torch.tensor(
            [
                [[0.20, 0.20, 0.20, 0.20]],
                [[0.40, 0.40, 0.40, 0.40]],
                [[0.60, 0.60, 0.60, 0.60]],
            ]
        )
        sampled = {
            "albedo_linear": sampled_linear,
            "roughness": torch.full((1, 1, 4), 0.45),
            "normal_tangent": torch.tensor(
                [
                    [[0.10, 0.10, 0.10, 0.10]],
                    [[0.00, 0.00, 0.00, 0.00]],
                    [[0.995, 0.995, 0.995, 0.995]],
                ]
            ),
        }
        wetness = torch.tensor([[[0.0, 0.5, 1.0, 1.0]]])
        valid = torch.tensor([[[True, True, True, False]]])
        rate = 4.0
        wet_scale = 0.65

        result = apply_sand_texture_primary(
            base_albedo,
            base_roughness,
            macro_normal,
            sampled,
            torch.eye(4),
            valid,
            wetness=wetness,
            wet_adjustment_enabled=True,
            wet_albedo_scale=wet_scale,
            wet_response_rate=rate,
        )
        response = -torch.expm1(-rate * wetness) / (-torch.expm1(
            torch.tensor(-rate)
        ))
        brightness = 1.0 - (1.0 - wet_scale) * response
        expected_linear = sampled_linear * brightness
        expected_srgb = torch.where(
            expected_linear <= 0.0031308,
            expected_linear * 12.92,
            1.055 * torch.pow(expected_linear, 1.0 / 2.4) - 0.055,
        )

        # Dry is exactly the accepted texture endpoint, full wetness reaches
        # the scalar brightness endpoint, and the middle sample follows the
        # normalized exponential rather than a linear interpolation.
        torch.testing.assert_close(
            result["albedo_srgb"][:, 0, :3], expected_srgb[:, 0, :3]
        )
        self.assertGreater(float(result["wet_response"][0, 0, 1]), 0.5)
        self.assertEqual(float(result["wet_response"][0, 0, 0]), 0.0)
        self.assertEqual(float(result["wet_response"][0, 0, 2]), 1.0)
        # Roughness and normals are independent of pore-water wetness in this
        # first model, while an invalid pixel retains the original material.
        torch.testing.assert_close(
            result["roughness"][:, 0, :3], sampled["roughness"][:, 0, :3]
        )
        torch.testing.assert_close(
            result["normal_world"][:, 0, 0],
            result["normal_world"][:, 0, 2],
        )
        torch.testing.assert_close(
            result["albedo_srgb"][:, 0, 3], base_albedo[:, 0, 3]
        )
        self.assertEqual(float(result["wet_response"][0, 0, 3]), 0.0)
        self.assertEqual(float(result["wet_brightness"][0, 0, 3]), 1.0)

    def test_triplanar_uvs_and_axis_weights(self) -> None:
        positions = torch.tensor(
            [
                [[0.0, 2.0, 4.0]],
                [[1.0, 3.0, 5.0]],
                [[4.0, 5.0, 6.0]],
            ]
        )
        uvs = simulation_triplanar_uvs(positions, period=2.0)
        torch.testing.assert_close(
            uvs["xy"], simulation_xy_uv(positions, period=2.0)
        )
        torch.testing.assert_close(
            uvs["xz"],
            torch.tensor([[[0.0, 1.0, 2.0]], [[2.0, 2.5, 3.0]]]),
        )
        torch.testing.assert_close(
            uvs["yz"],
            torch.tensor([[[0.5, 1.5, 2.5]], [[2.0, 2.5, 3.0]]]),
        )

        # +Z chooses XY, +Y chooses XZ and +X chooses YZ.
        axis_normals = torch.tensor(
            [
                [[0.0, 0.0, 1.0]],
                [[0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0]],
            ]
        )
        weights = simulation_triplanar_weights(
            axis_normals, torch.eye(4), sharpness=4.0
        )
        torch.testing.assert_close(weights, torch.eye(3).reshape(3, 1, 3))

    def test_triplanar_neutral_normal_preserves_macro_surface(self) -> None:
        material = synthetic_material()
        positions = torch.tensor(
            [
                [[0.10, 0.35, 0.70]],
                [[0.20, 0.45, 0.80]],
                [[0.30, 0.55, 0.90]],
            ]
        )
        macro_normal = torch.tensor(
            [
                [[0.0, 0.0, 1.0]],
                [[0.0, 1.0, 1.0]],
                [[1.0, 0.0, 1.0]],
            ]
        )
        macro_normal = torch.nn.functional.normalize(macro_normal, dim=0)
        valid = torch.ones((1, 1, 3), dtype=torch.bool)
        sampled = sample_sand_pbr_material_triplanar(
            material,
            positions,
            macro_normal,
            torch.eye(4),
            valid,
            period=1.0,
            sharpness=4.0,
            normal_strength=3.0,
        )
        torch.testing.assert_close(
            sampled["normal_world_override"],
            macro_normal,
            atol=1.0e-6,
            rtol=0.0,
        )

        # A horizontal surface must sample the exact legacy XY material.
        horizontal = torch.zeros_like(macro_normal)
        horizontal[2] = 1.0
        triplanar_horizontal = sample_sand_pbr_material_triplanar(
            material,
            positions,
            horizontal,
            torch.eye(4),
            valid,
            period=1.0,
            sharpness=4.0,
            normal_strength=1.0,
        )
        xy = sample_sand_pbr_material(
            material,
            simulation_xy_uv(positions, period=1.0),
            valid,
        )
        torch.testing.assert_close(
            triplanar_horizontal["albedo_linear"], xy["albedo_linear"]
        )
        torch.testing.assert_close(
            triplanar_horizontal["roughness"], xy["roughness"]
        )

    def test_depth_backprojection_and_exchange_inverse(self) -> None:
        depth = torch.tensor([[[2.0, 2.0]]])
        world = reconstruct_world_positions(
            depth,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            world_view_transform=torch.eye(4),
        )
        torch.testing.assert_close(
            world,
            torch.tensor([[[0.0, 2.0]], [[0.0, 0.0]], [[2.0, 2.0]]]),
        )

        sim_to_gs = torch.eye(4)
        sim_to_gs[:3, 3] = torch.tensor([10.0, 20.0, 30.0])
        translated_world = world + torch.tensor([10.0, 20.0, 30.0])[:, None, None]
        simulation = world_positions_to_simulation(
            translated_world, sim_to_gs
        )
        torch.testing.assert_close(simulation, world)

    def test_simulation_uv_rotation_and_period(self) -> None:
        positions = torch.tensor([[[2.0]], [[0.0]], [[5.0]]])
        uv = simulation_xy_uv(
            positions,
            period=2.0,
            rotation_degrees=90.0,
            offset=(0.25, 0.50),
        )
        torch.testing.assert_close(
            uv[:, 0, 0], torch.tensor([0.25, 1.50]), atol=1.0e-6, rtol=0.0
        )

    def test_repeat_sampling_is_invariant_to_integer_uv_offsets(self) -> None:
        material = synthetic_material()
        uv = torch.tensor(
            [
                [[0.25, 1.25, -0.75]],
                [[0.25, 1.25, -0.75]],
            ]
        )
        valid = torch.ones((1, 1, 3), dtype=torch.bool)
        sampled = sample_sand_pbr_material(material, uv, valid)
        torch.testing.assert_close(
            sampled["albedo_linear"][:, :, 0],
            sampled["albedo_linear"][:, :, 1],
        )
        torch.testing.assert_close(
            sampled["albedo_linear"][:, :, 0],
            sampled["albedo_linear"][:, :, 2],
        )

    def test_lod_uses_texture_space_pixel_footprint(self) -> None:
        uv = torch.tensor(
            [
                [[0.0, 0.25, 0.50]],
                [[0.0, 0.0, 0.0]],
            ]
        )
        lod = estimate_texture_lod(
            uv,
            texture_height=8,
            texture_width=8,
            max_lod=3,
        )
        # A one-pixel screen step spans two base texture pixels: log2(2)=1.
        torch.testing.assert_close(lod, torch.ones_like(lod))

    def test_zero_strengths_preserve_existing_sand_material(self) -> None:
        material = synthetic_material()
        base_albedo = torch.tensor(
            [
                [[0.7, 0.6]],
                [[0.5, 0.4]],
                [[0.3, 0.2]],
            ]
        )
        base_roughness = torch.tensor([[[0.9, 0.4]]])
        macro_normal = torch.tensor(
            [
                [[0.0, 0.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
            ]
        )
        sampled = {
            "albedo_linear": torch.ones((3, 1, 2)),
            "roughness": torch.zeros((1, 1, 2)),
            "normal_tangent": torch.tensor(
                [[[0.8, 0.8]], [[0.2, 0.2]], [[0.6, 0.6]]]
            ),
        }
        valid = torch.tensor([[[True, False]]])
        result = apply_sand_pbr_microtexture(
            base_albedo,
            base_roughness,
            macro_normal,
            sampled,
            material,
            torch.eye(4),
            valid,
            albedo_strength=0.0,
            roughness_strength=0.0,
            normal_strength=0.0,
        )
        torch.testing.assert_close(result["albedo_srgb"], base_albedo)
        torch.testing.assert_close(result["roughness"], base_roughness)
        torch.testing.assert_close(result["normal_world"], macro_normal)

    def test_microtexture_changes_only_valid_sand_pixels(self) -> None:
        material = synthetic_material()
        base_albedo = torch.full((3, 1, 2), 0.5)
        base_roughness = torch.full((1, 1, 2), 0.5)
        macro_normal = torch.zeros((3, 1, 2))
        macro_normal[2] = 1.0
        sampled = {
            "albedo_linear": torch.ones((3, 1, 2)),
            "roughness": torch.ones((1, 1, 2)),
            "normal_tangent": torch.tensor(
                [[[0.6, 0.6]], [[0.0, 0.0]], [[0.8, 0.8]]]
            ),
        }
        valid = torch.tensor([[[True, False]]])
        result = apply_sand_pbr_microtexture(
            base_albedo,
            base_roughness,
            macro_normal,
            sampled,
            material,
            torch.eye(4),
            valid,
            albedo_strength=0.5,
            roughness_strength=0.5,
            normal_strength=1.0,
        )
        self.assertFalse(torch.equal(
            result["albedo_srgb"][:, 0, 0], base_albedo[:, 0, 0]
        ))
        self.assertFalse(torch.equal(
            result["roughness"][:, 0, 0], base_roughness[:, 0, 0]
        ))
        self.assertGreater(float(result["normal_world"][0, 0, 0]), 0.0)
        torch.testing.assert_close(
            result["albedo_srgb"][:, 0, 1], base_albedo[:, 0, 1]
        )
        torch.testing.assert_close(
            result["roughness"][:, 0, 1], base_roughness[:, 0, 1]
        )
        torch.testing.assert_close(
            result["normal_world"][:, 0, 1], macro_normal[:, 0, 1]
        )

    def test_identity_transform_builds_simulation_xy_tangent_basis(self) -> None:
        normal = torch.tensor([[[0.0]], [[0.0]], [[1.0]]])
        tangent, bitangent = build_simulation_tangent_basis(
            normal, torch.eye(4)
        )
        torch.testing.assert_close(
            tangent[:, 0, 0], torch.tensor([1.0, 0.0, 0.0])
        )
        torch.testing.assert_close(
            bitangent[:, 0, 0], torch.tensor([0.0, 1.0, 0.0])
        )


if __name__ == "__main__":
    unittest.main()
