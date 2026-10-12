from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gaussian_renderer import water_optics as water_optics_module
from gaussian_renderer.water_gbuffer import WaterGBuffer


class WaterOpticsTests(unittest.TestCase):
    def test_empty_visible_water_preserves_opaque_substrate_and_diagnostics(
        self,
    ) -> None:
        height, width = 3, 4
        opaque_rgb = torch.full((3, height, width), 0.4)
        opaque_depth = torch.full((1, height, width), 2.0)
        normal = torch.zeros((3, height, width))
        normal[2] = 1.0
        scalar = torch.ones((1, height, width))
        false_mask = torch.zeros((1, height, width), dtype=torch.bool)
        true_mask = torch.ones((1, height, width), dtype=torch.bool)
        stage_mask = torch.ones((1, height, width), dtype=torch.bool)
        gbuffer = WaterGBuffer(
            depth=torch.ones((1, height, width)),
            plane_depth_valid=true_mask,
            depth_fallback_used=false_mask,
            depth_fallback_rejected_by_opaque=false_mask,
            surface_valid=false_mask,
            visible_coverage=torch.zeros_like(scalar),
            invalid_visible_candidate=torch.zeros_like(scalar),
            stage_diagnostics={"water_failure_no_footprint": stage_mask},
            validation=None,
            normal_camera=normal,
            normal_world=normal,
            thickness=torch.full_like(scalar, 0.25),
            raw_alpha=scalar,
            raw_coverage=scalar,
            depth_in_front=true_mask,
            depth_in_front_without_proxy=true_mask,
            depth_bias=0.002,
            occlusion_depth=opaque_depth,
            terrain_proxy_depth=torch.zeros_like(scalar),
            terrain_proxy_coverage=false_mask,
            terrain_proxy_used=false_mask,
            terrain_proxy_rejected_foreground=false_mask,
            terrain_proxy_source_delta=torch.zeros_like(scalar),
            occlusion_bias=torch.full_like(scalar, 0.002),
            depth_in_front_without_physical_reference=true_mask,
            physical_reference_depth=torch.zeros_like(scalar),
            physical_reference_coverage=false_mask,
            physical_reference_trusted=false_mask,
            physical_reference_recovered=false_mask,
            physical_reference_removed=false_mask,
            physical_reference_rejected_opaque=false_mask,
        )
        camera = SimpleNamespace(
            world_view_transform=torch.eye(4),
            fx=2.0,
            fy=2.0,
            cx=1.5,
            cy=1.0,
        )
        pc = SimpleNamespace(
            water_light_direction_gs_t=torch.tensor([0.0, 0.0, 1.0])
        )
        result = {
            "opaque_surface_rgb": opaque_rgb,
            "opaque_surface_depth": opaque_depth,
        }
        real_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs = dict(kwargs)
                kwargs["device"] = "cpu"
            return real_tensor(*args, **kwargs)

        with ExitStack() as stack:
            tracer = stack.enter_context(patch.object(
                water_optics_module, "ScreenRayTracer"
            ))
            stack.enter_context(patch.object(
                water_optics_module.torch, "tensor", cpu_tensor
            ))
            stack.enter_context(patch.object(
                water_optics_module,
                "apply_gaussian_smoothing",
                side_effect=lambda image, **kwargs: image,
            ))
            returned = water_optics_module.compose_water_optics(
                args=SimpleNamespace(),
                image_height=height,
                image_width=width,
                opaque_surface_depth=opaque_depth,
                opaque_surface_normal_camera_facing=normal,
                pc=pc,
                rendered_rgb_composite=opaque_rgb,
                result=result,
                viewpoint_camera=camera,
                water_gbuffer=gbuffer,
            )

        self.assertIs(returned, result)
        tracer.assert_not_called()
        torch.testing.assert_close(returned["render"], opaque_rgb)
        torch.testing.assert_close(
            returned["water_surface_depth"], gbuffer.depth
        )
        torch.testing.assert_close(
            returned["water_surface_normal_world"], gbuffer.normal_world
        )
        self.assertFalse(torch.any(returned["water_surface_valid"]))
        self.assertEqual(
            returned["water_failure_no_footprint"].shape,
            (3, height, width),
        )
        self.assertTrue(torch.all(returned["water_failure_no_footprint"]))


if __name__ == "__main__":
    unittest.main()
