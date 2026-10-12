from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from sandygs.exchange import (
    ExchangeValidationError,
    GridSpec,
    apply_transform,
    build_legacy_sim_to_gs,
    cell_center_coordinates,
    rainygs_endpoint_image_to_cell_field,
    validate_bundle,
)
from sandygs.prepare import base_metadata, write_field, write_solver_terrain


class CoordinateTests(unittest.TestCase):
    def test_legacy_transform_round_trip(self) -> None:
        sim_to_gs = build_legacy_sim_to_gs(
            (241.209, 0.298943, 70.0),
            (0.102286, -0.4026, 4.4),
            (1.0, 1.0, 1.0),
        )
        points = np.array(
            [[-5.25, -2.25, 1.1], [0.25, 0.75, 2.0], [4.75, 1.25, 3.5]]
        )
        gs_points = apply_transform(points, sim_to_gs)
        np.testing.assert_allclose(
            apply_transform(np.zeros(3), sim_to_gs),
            [0.36628877367797885, 3.741850994326079, 2.3231299697465086],
            rtol=0.0,
            atol=1e-12,
        )
        restored = apply_transform(gs_points, np.linalg.inv(sim_to_gs))
        np.testing.assert_allclose(restored, points, rtol=0.0, atol=1e-12)

    def test_endpoint_yx_conversion_to_cell_center_xy(self) -> None:
        grid = GridSpec((8, 6), (-2.0, -1.5), (2.0, 1.5))
        source_x = np.linspace(-2.0, 2.0, 8)
        source_y = np.linspace(-1.5, 1.5, 6)
        image_yx = source_x[None, :] + 10.0 * source_y[:, None]
        converted = rainygs_endpoint_image_to_cell_field(image_yx, grid)
        target_x, target_y = cell_center_coordinates(grid)
        expected = target_x[:, None] + 10.0 * target_y[None, :]
        self.assertEqual(converted.shape, (8, 6))
        self.assertEqual(converted.dtype, np.float32)
        np.testing.assert_allclose(converted, expected, rtol=0.0, atol=2e-6)

        legacy_dx = 4.0 / (grid.resolution[0] - 1)
        self.assertNotEqual(legacy_dx, grid.spacing[0])

    def test_pulse_slope_and_rectangle_axes(self) -> None:
        grid = GridSpec((8, 6), (-2.0, -1.5), (2.0, 1.5))
        pulse = np.zeros(grid.resolution)
        pulse[5, 1] = 1.0
        self.assertEqual(np.unravel_index(np.argmax(pulse), pulse.shape), (5, 1))

        x, _ = cell_center_coordinates(grid)
        slope = np.broadcast_to(x[:, None], grid.resolution)
        self.assertTrue(np.all(np.diff(slope, axis=0) > 0.0))
        self.assertTrue(np.all(np.diff(slope, axis=1) == 0.0))

        rectangle = np.zeros(grid.resolution)
        rectangle[1:4, 3:5] = 1.0
        indices = np.argwhere(rectangle > 0.0)
        np.testing.assert_array_equal(indices.min(axis=0), [1, 3])
        np.testing.assert_array_equal(indices.max(axis=0), [3, 4])


class BundleTests(unittest.TestCase):
    def make_bundle(self, directory: Path) -> dict:
        grid = GridSpec((4, 3), (-2.0, -1.5), (2.0, 1.5))
        metadata = base_metadata(
            scene_id="unit-test",
            grid=grid,
            sim_to_gs=np.eye(4),
            scale_status="unverified",
            meters_per_simulation_unit=None,
            scale_source="unit test",
        )
        descriptor = write_field(
            directory, "frames/000000/terrain.npy", np.zeros(grid.resolution)
        )
        descriptor["semantic"] = "static_terrain_elevation"
        metadata["frames"] = [
            {
                "frame_index": 0,
                "step_index": 0,
                "time_seconds": 0.0,
                "fields": {"terrain": descriptor},
            }
        ]
        with (directory / "metadata.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(metadata, stream, sort_keys=False)
        return metadata

    def test_valid_bundle_and_solver_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            metadata = self.make_bundle(directory)
            validate_bundle(directory)
            terrain = np.load(directory / "frames/000000/terrain.npy")
            grid = GridSpec((4, 3), (-2.0, -1.5), (2.0, 1.5))
            write_solver_terrain(directory, terrain, grid, boundary_width=1)
            solver_terrain = np.load(directory / "solver/terrain.npy")
            self.assertEqual(solver_terrain.dtype, np.float64)
            with (directory / "solver/scene.yaml").open(
                "r", encoding="utf-8"
            ) as stream:
                scene = yaml.safe_load(stream)
            self.assertEqual(scene["grid"]["resolution"], [6, 5])
            self.assertEqual(
                scene["initial"]["terrain"]["origin"],
                metadata["grid"]["cell_center_origin"],
            )

    def test_duplicate_frame_and_non_monotonic_time_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            metadata = self.make_bundle(directory)
            duplicate = deepcopy(metadata["frames"][0])
            duplicate["step_index"] = 1
            duplicate["time_seconds"] = -1.0
            metadata["frames"].append(duplicate)
            with (directory / "metadata.yaml").open(
                "w", encoding="utf-8"
            ) as stream:
                yaml.safe_dump(metadata, stream, sort_keys=False)
            with self.assertRaises(ExchangeValidationError) as context:
                validate_bundle(directory)
            message = str(context.exception)
            self.assertIn("duplicates 0", message)
            self.assertIn("time_seconds must be finite and non-negative", message)

if __name__ == "__main__":
    unittest.main()
