from types import SimpleNamespace
import unittest

import torch

from gaussian_renderer.opaque_gbuffer import compose_opaque_gbuffer


class OpaqueGBufferTests(unittest.TestCase):
    def test_source_only_contract_preserves_rgb_depth_normals_and_owner(self) -> None:
        height, width = 2, 3
        rendered_rgb = torch.arange(
            3 * height * width, dtype=torch.float32
        ).reshape(3, height, width)
        rendered_depth = torch.stack(
            (
                torch.full((height, width), 2.0),
                torch.full((height, width), 3.0),
                torch.full((height, width), 4.0),
            )
        )
        source_all_map = torch.zeros((5, height, width), dtype=torch.float32)
        source_all_map[2] = 1.0
        camera = SimpleNamespace(
            world_view_transform=torch.eye(4, dtype=torch.float32),
            fx=2.0,
            fy=2.0,
            cx=1.0,
            cy=0.5,
        )

        result = compose_opaque_gbuffer(
            args=SimpleNamespace(),
            bg_color=None,
            colors_precomp=None,
            full_proj_transform=None,
            image_height=height,
            image_width=width,
            means2D=None,
            means2D_abs=None,
            means3D=None,
            num_gausssian_all=17,
            num_sand_gaussian=0,
            opacity=None,
            orthographic=False,
            ortho_x_min=None,
            ortho_x_max=None,
            ortho_y_min=None,
            ortho_y_max=None,
            out_all_map_src=source_all_map,
            pc=None,
            pipe=None,
            rendered_depth_src=rendered_depth,
            rendered_rgb_src=rendered_rgb,
            rotations=None,
            sand_span=None,
            scales=None,
            scaling_modifier=1.0,
            shs=None,
            tanfovx=None,
            tanfovy=None,
            viewpoint_camera=camera,
        )

        self.assertIs(result["render"], rendered_rgb)
        self.assertIs(result["opaque_surface_rgb"], rendered_rgb)
        torch.testing.assert_close(
            result["opaque_surface_depth"], rendered_depth[:1]
        )
        torch.testing.assert_close(
            result["opaque_surface_normal_world"], source_all_map[:3]
        )
        torch.testing.assert_close(
            result["opaque_surface_normal_camera_facing"],
            -source_all_map[:3],
        )
        self.assertEqual(result["num_gausssian_all"], 17)
        self.assertEqual(
            result["opaque_surface_sand_owner"].dtype, torch.bool
        )
        self.assertFalse(torch.any(result["opaque_surface_sand_owner"]))
        self.assertIsNone(result["sand_gbuffer_validation"])


if __name__ == "__main__":
    unittest.main()
