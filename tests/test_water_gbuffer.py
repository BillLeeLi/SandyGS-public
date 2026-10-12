from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gaussian_renderer import water_gbuffer as water_gbuffer_module


class _FakeWaterRasterizer:
    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.call_count = 0

    def __call__(self, **kwargs):
        self.call_count += 1
        rendered = torch.zeros(
            (3, self.height, self.width), dtype=torch.float32
        )
        all_map = torch.zeros(
            (5, self.height, self.width), dtype=torch.float32
        )
        all_map[2] = 1.0
        all_map[3] = 1.0
        depth = torch.ones(
            (1, self.height, self.width), dtype=torch.float32
        )
        if self.call_count == 2:
            rendered.fill_(0.25)
        return rendered, None, None, all_map, depth


class WaterGBufferTests(unittest.TestCase):
    def test_builds_visible_water_contract_without_running_optics(self) -> None:
        height, width, count = 2, 3, 2
        rasterizer = _FakeWaterRasterizer(height, width)
        real_zeros = torch.zeros
        real_ones = torch.ones

        def cpu_zeros(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs = dict(kwargs)
                kwargs["device"] = "cpu"
            return real_zeros(*args, **kwargs)

        def cpu_ones(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs = dict(kwargs)
                kwargs["device"] = "cpu"
            return real_ones(*args, **kwargs)

        camera = SimpleNamespace(
            world_view_transform=torch.eye(4, dtype=torch.float32),
            camera_center=torch.zeros(3, dtype=torch.float32),
        )
        pc = SimpleNamespace(
            active_sh_degree=3,
            water_thickness_t=torch.full((count,), 0.25),
            get_stored_layer_normals=lambda name: torch.tensor(
                [[0.0, 0.0, 1.0]] * count, dtype=torch.float32
            ),
        )
        means3d = torch.tensor(
            [[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]], dtype=torch.float32
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(
                water_gbuffer_module,
                "PlaneGaussianRasterizationSettings",
                side_effect=lambda **kwargs: kwargs,
            ))
            stack.enter_context(patch.object(
                water_gbuffer_module,
                "PlaneGaussianRasterizer",
                return_value=rasterizer,
            ))
            stack.enter_context(patch.object(
                water_gbuffer_module.torch, "zeros", cpu_zeros
            ))
            stack.enter_context(patch.object(
                water_gbuffer_module.torch, "ones", cpu_ones
            ))
            result = water_gbuffer_module.build_water_gbuffer(
                args=SimpleNamespace(water_validate_gbuffer=False),
                bg_color=torch.zeros(3),
                fluid_span=SimpleNamespace(start=0, end=count),
                full_proj_transform=torch.eye(4),
                image_height=height,
                image_width=width,
                means2D=torch.zeros_like(means3d),
                means2D_abs=torch.zeros_like(means3d),
                means3D=means3d,
                num_fluid_gaussian=count,
                opacity=torch.ones((count, 1)),
                opaque_surface_depth=torch.full((1, height, width), 2.0),
                opaque_surface_sand_owner=torch.zeros(
                    (1, height, width), dtype=torch.bool
                ),
                orthographic=False,
                ortho_x_min=-1.0,
                ortho_x_max=1.0,
                ortho_y_min=-1.0,
                ortho_y_max=1.0,
                pc=pc,
                pipe=SimpleNamespace(debug=False),
                rotations=torch.zeros((count, 4)),
                scales=torch.ones((count, 3)),
                scaling_modifier=1.0,
                tanfovx=1.0,
                tanfovy=1.0,
                viewpoint_camera=camera,
            )

        self.assertEqual(rasterizer.call_count, 2)
        torch.testing.assert_close(result.depth, torch.ones_like(result.depth))
        torch.testing.assert_close(
            result.thickness, torch.full_like(result.thickness, 0.25)
        )
        self.assertTrue(torch.all(result.plane_depth_valid))
        self.assertTrue(torch.all(result.surface_valid))
        self.assertTrue(torch.all(result.visible_coverage == 1.0))
        self.assertFalse(torch.any(result.depth_fallback_used))
        self.assertFalse(torch.any(result.depth_fallback_rejected_by_opaque))
        self.assertFalse(torch.any(result.invalid_visible_candidate))
        self.assertIsNone(result.validation)
        self.assertEqual(result.depth_bias, 0.002)
        torch.testing.assert_close(
            result.occlusion_bias,
            torch.full_like(result.occlusion_bias, 0.002),
        )
        self.assertFalse(torch.any(result.terrain_proxy_used))
        torch.testing.assert_close(
            result.occlusion_depth,
            torch.full_like(result.occlusion_depth, 2.0),
        )
        self.assertIn("water_failure_valid_owner", result.stage_diagnostics)
        self.assertTrue(
            torch.all(result.stage_diagnostics["water_failure_valid_owner"])
        )

    def test_terrain_proxy_preserves_sand_and_clear_foreground(self) -> None:
        opaque = torch.tensor([[[1.95, 1.70, 1.98, 2.20]]])
        terrain = torch.full_like(opaque, 2.0)
        sand = torch.tensor([[[False, False, True, False]]])
        resolved, used, rejected, delta = (
            water_gbuffer_module.resolve_terrain_depth_proxy(
                opaque_surface_depth=opaque,
                opaque_surface_sand_owner=sand,
                terrain_depth=terrain,
                terrain_coverage=torch.ones_like(opaque, dtype=torch.bool),
                water_valid=torch.ones_like(opaque, dtype=torch.bool),
                source_tolerance=0.10,
            )
        )
        torch.testing.assert_close(
            resolved, torch.tensor([[[2.00, 1.70, 1.98, 2.20]]])
        )
        self.assertEqual(used.tolist(), [[[True, False, False, False]]])
        self.assertEqual(rejected.tolist(), [[[False, True, False, False]]])
        torch.testing.assert_close(
            delta, torch.tensor([[[-0.05, -0.30, -0.02, 0.20]]])
        )

    def test_physical_water_reference_changes_only_trusted_visibility(self) -> None:
        water_depth = torch.tensor([[[2.10, 1.90, 1.90, 2.10]]])
        physical_depth = torch.tensor([[[1.80, 2.20, 1.80, 1.80]]])
        coverage = torch.ones_like(water_depth)
        physical_coverage = torch.tensor([[[True, True, False, True]]])
        water_valid = torch.tensor([[[True, True, True, False]]])
        occlusion_depth = torch.full_like(water_depth, 2.0)
        occlusion_bias = torch.zeros_like(water_depth)
        (
            visibility_depth,
            in_front,
            trusted,
            recovered,
            removed,
            rejected,
        ) = water_gbuffer_module.resolve_water_physical_reference_visibility(
            water_depth=water_depth,
            water_valid=water_valid,
            water_coverage=coverage,
            physical_depth=physical_depth,
            physical_coverage=physical_coverage,
            occlusion_depth=occlusion_depth,
            occlusion_bias=occlusion_bias,
            erosion_pixels=0,
            gaussian_coverage_min=0.5,
        )
        torch.testing.assert_close(
            visibility_depth, torch.tensor([[[1.80, 2.20, 1.90, 2.10]]])
        )
        self.assertEqual(in_front.tolist(), [[[True, False, True, False]]])
        self.assertEqual(trusted.tolist(), [[[True, True, False, False]]])
        self.assertEqual(recovered.tolist(), [[[True, False, False, False]]])
        self.assertEqual(removed.tolist(), [[[False, True, False, False]]])
        self.assertEqual(rejected.tolist(), [[[False, True, False, False]]])

    def test_physical_water_reference_erodes_its_silhouette(self) -> None:
        shape = (1, 5, 5)
        result = water_gbuffer_module.resolve_water_physical_reference_visibility(
            water_depth=torch.full(shape, 2.1),
            water_valid=torch.ones(shape, dtype=torch.bool),
            water_coverage=torch.ones(shape),
            physical_depth=torch.full(shape, 1.8),
            physical_coverage=torch.ones(shape, dtype=torch.bool),
            occlusion_depth=torch.full(shape, 2.0),
            occlusion_bias=torch.zeros(shape),
            erosion_pixels=2,
            gaussian_coverage_min=0.5,
        )
        trusted = result[2]
        self.assertEqual(int(trusted.sum()), 1)
        self.assertTrue(bool(trusted[0, 2, 2]))


if __name__ == "__main__":
    unittest.main()
