from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from sandygs.exchange import GridSpec
from sandygs.prepare import (
    _interior_feather_weight,
    base_metadata,
    prepare_pure_water_case,
    prepare_sand_water_case,
    sample_height_spec,
    write_field,
)
from sandygs.preview import write_solver_evolution_preview
from sandygs.simulation import SandWaterRunAdapter, SimulationRunError


def write_yaml(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)


class SandWaterAdapterTests(unittest.TestCase):
    def make_exchange_and_run(
        self,
        root: Path,
        *,
        sand_height: float = 0.0,
        mixed_water: bool = False,
    ) -> tuple[Path, Path, np.ndarray]:
        exchange = root / "exchange"
        exchange.mkdir()
        grid = GridSpec((4, 3), (-2.0, -1.5), (2.0, 1.5))
        x = np.arange(4, dtype=np.float32)[:, None]
        y = np.arange(3, dtype=np.float32)[None, :]
        terrain = 0.1 * x + 0.01 * y
        metadata = base_metadata(
            scene_id="adapter-test",
            grid=grid,
            sim_to_gs=np.eye(4),
            scale_status="assumed",
            meters_per_simulation_unit=None,
            scale_source="unit test",
        )
        descriptor = write_field(
            exchange, "frames/000000/terrain.npy", terrain
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
        write_yaml(exchange / "metadata.yaml", metadata)

        run = root / "run"
        (run / "frames/000000").mkdir(parents=True)
        (run / "frames/000002").mkdir(parents=True)
        write_yaml(
            run / "metadata.yaml",
            {
                "format_version": 7,
                "grid": {
                    "resolution": [6, 5],
                    "spacing": 1.0,
                    "boundary_width": 1,
                    "center": [0.0, 0.0],
                },
                # yaml-cpp emits this valid numeric spelling, while PyYAML
                # commonly loads it as a string under YAML 1.1 rules.
                "time": {"height_epsilon": "1e-08"},
            },
        )
        with (run / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "time"])
            writer.writeheader()
            writer.writerows(
                [
                    {"step": 0, "time": 0.0},
                    {"step": 1, "time": 0.1},
                    {"step": 2, "time": 0.2},
                ]
            )

        def with_ghost(values: np.ndarray) -> np.ndarray:
            return np.pad(values, 1, mode="edge").astype(np.float64)

        for step, water in ((0, 0.0), (2, 0.2)):
            frame = run / "frames" / f"{step:06d}"
            fields = {
                "terrain": terrain,
                "sand_height": np.full((4, 3), sand_height),
                "pure_water_height": np.full(
                    (4, 3), 0.0 if mixed_water else water
                ),
                "mixed_water_height": np.full(
                    (4, 3), water if mixed_water else 0.0
                ),
                "total_water": np.full((4, 3), water),
            }
            for name, values in fields.items():
                np.save(frame / f"{name}.npy", with_ghost(values))
        return exchange, run, terrain

    @staticmethod
    def replace_sand_scene(
        frame,
        adapter: SandWaterRunAdapter,
        terrain: np.ndarray,
        sand: np.ndarray,
    ):
        """Build a renderer-only geometry fixture from explicit height fields."""

        terrain = np.asarray(terrain, dtype=np.float32)
        sand = np.asarray(sand, dtype=np.float32)
        return replace(
            frame,
            terrain=terrain,
            sand_height=sand,
            sand_surface=terrain + sand,
            sand_mask=sand > adapter.sand_epsilon,
            saturation=np.zeros_like(sand),
            mixed_water_wetness=np.zeros_like(sand),
        )

    def test_solver_evolution_preview_uses_render_snapshot_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, run, _ = self.make_exchange_and_run(
                root, sand_height=0.2, mixed_water=True
            )
            report = write_solver_evolution_preview(run, root / "preview")

            self.assertEqual(report["first_step"], 0)
            self.assertEqual(report["final_step"], 2)
            self.assertEqual(report["shape_after_ghost_crop"], [4, 3])
            self.assertAlmostEqual(report["water_volume_final"], 2.4)
            self.assertAlmostEqual(report["mixed_water_volume_final"], 2.4)
            self.assertTrue((root / "preview/evolution_overview.png").is_file())
            self.assertTrue((root / "preview/preview.yaml").is_file())

    def test_time_interpolation_mask_and_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exchange, run, terrain = self.make_exchange_and_run(
                Path(temporary)
            )
            adapter = SandWaterRunAdapter(
                exchange,
                run,
                free_water_epsilon=0.05,
            )
            frame = adapter.sample(0.1)
            self.assertEqual((frame.lower_step, frame.upper_step), (0, 2))
            self.assertAlmostEqual(frame.interpolation_weight, 0.5)
            np.testing.assert_allclose(frame.pure_water_height, 0.1)
            np.testing.assert_allclose(
                frame.absolute_surface, terrain + 0.1, atol=1e-7
            )
            self.assertTrue(frame.free_water_mask.all())

            geometry = adapter.geometry(frame)
            self.assertEqual(geometry.xyz_gs.shape, (12, 3))
            self.assertEqual(geometry.normals_gs.shape, (12, 3))
            np.testing.assert_allclose(geometry.pure_water_thickness, 0.1)
            np.testing.assert_array_equal(
                geometry.active_cell_indices[0], [0, 0]
            )
            np.testing.assert_array_equal(
                geometry.active_cell_indices[-1], [3, 2]
            )
            # Identity T_sim_to_gs means the first point is exactly the first
            # cell center plus the sampled absolute surface elevation.
            np.testing.assert_allclose(
                geometry.xyz_gs[0], [-1.5, -1.0, 0.1], atol=1e-7
            )
            np.testing.assert_allclose(
                np.linalg.norm(geometry.normals_gs, axis=1),
                1.0,
                atol=1e-7,
            )

    def test_water_normals_do_not_read_dry_obstacle_heights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exchange, run, _ = self.make_exchange_and_run(Path(temporary))
            adapter = SandWaterRunAdapter(
                exchange,
                run,
                free_water_epsilon=0.05,
                water_normal_gradient_mode="free_water_masked",
            )
            frame = adapter.sample(0.1)

            # The 2x2 wet patch has the plane z=x+2y.  Surrounding dry cells
            # deliberately contain extreme obstacle heights; they must not
            # tilt shoreline normals.  The disconnected cell also verifies
            # that a water sample with no axial wet neighbor stays horizontal.
            active = np.zeros((4, 3), dtype=np.bool_)
            active[:2, :2] = True
            active[3, 2] = True
            surface = np.full((4, 3), 1000.0, dtype=np.float32)
            surface[0, 0] = 0.0
            surface[0, 1] = 2.0
            surface[1, 0] = 1.0
            surface[1, 1] = 3.0
            surface[3, 2] = -500.0
            frame = replace(
                frame,
                absolute_surface=surface,
                free_water_mask=active,
            )

            geometry = adapter.geometry(frame)
            expected_sloped = np.array([-1.0, -2.0, 1.0])
            expected_sloped /= np.linalg.norm(expected_sloped)
            np.testing.assert_allclose(
                geometry.normals_gs[:4],
                np.broadcast_to(expected_sloped, (4, 3)),
                atol=1e-7,
            )
            np.testing.assert_allclose(
                geometry.normals_gs[4], [0.0, 0.0, 1.0], atol=1e-7
            )

            legacy = SandWaterRunAdapter(
                exchange,
                run,
                free_water_epsilon=0.05,
                water_normal_gradient_mode="legacy_full",
            ).geometry(frame)
            self.assertGreater(
                float(np.max(np.abs(legacy.normals_gs - geometry.normals_gs))),
                0.5,
            )

            with self.assertRaisesRegex(
                SimulationRunError, "water_normal_gradient_mode"
            ):
                SandWaterRunAdapter(
                    exchange,
                    run,
                    water_normal_gradient_mode="unknown",
                )

    def test_out_of_range_time_and_nonzero_sand_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exchange, run, _ = self.make_exchange_and_run(Path(temporary))
            adapter = SandWaterRunAdapter(exchange, run)
            with self.assertRaises(SimulationRunError):
                adapter.sample(0.3)

        with tempfile.TemporaryDirectory() as temporary:
            exchange, run, _ = self.make_exchange_and_run(
                Path(temporary), sand_height=0.1
            )
            with self.assertRaises(SimulationRunError):
                SandWaterRunAdapter(exchange, run)

    def test_pure_water_case_uses_absolute_surface_level(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange, _, terrain = self.make_exchange_and_run(root)
            output = root / "case"
            metadata = prepare_pure_water_case(
                exchange_bundle=exchange,
                output=output,
                initial_surface_level=0.25,
                water_dt=0.001,
                steps=2,
                frame_stride=1,
                target_cfl=0.45,
                duration_seconds=0.04,
            )
            water = np.load(output / "total_water.npy")
            np.testing.assert_allclose(
                water, np.maximum(0.25 - terrain, 0.0), atol=1e-12
            )
            self.assertEqual(water.dtype, np.float64)
            with (output / "scene.yaml").open(
                "r", encoding="utf-8"
            ) as stream:
                scene = yaml.safe_load(stream)
            self.assertEqual(scene["grid"]["resolution"], [6, 5])
            self.assertEqual(scene["initial"]["sand"], 0.0)
            self.assertEqual(scene["run"]["duration"], 0.04)
            self.assertEqual(
                metadata["initialization"]["kind"],
                "absolute_surface_level",
            )

    def test_sand_frame_geometry_and_material_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exchange, run, terrain = self.make_exchange_and_run(
                Path(temporary), sand_height=0.2
            )
            adapter = SandWaterRunAdapter(
                exchange,
                run,
                require_pure_water=False,
                sand_epsilon=1.0e-6,
            )
            frame = adapter.sample(0.1)
            np.testing.assert_allclose(frame.sand_surface, terrain + 0.2)
            np.testing.assert_allclose(
                frame.saturation,
                0.1 / (0.2 + 0.1),
                atol=1e-7,
            )
            # The fixture stores 0.1 as *pure* water and no pore water. P4 must
            # not darken the sand merely because a transparent layer covers it.
            np.testing.assert_allclose(frame.mixed_water_wetness, 0.0)
            self.assertTrue(frame.sand_mask.all())

            geometry = adapter.sand_geometry(
                frame,
                coverage_factor=0.55,
                normal_thickness_ratio=0.1,
                full_coverage_height=0.2,
            )
            self.assertEqual(geometry.xyz_gs.shape, (12, 3))
            self.assertEqual(geometry.rotations_wxyz.shape, (12, 4))
            np.testing.assert_allclose(geometry.coverage, 1.0, atol=1e-7)
            np.testing.assert_allclose(geometry.mixed_water_wetness, 0.0)
            np.testing.assert_allclose(
                np.linalg.norm(geometry.normals_gs, axis=1), 1.0, atol=1e-7
            )
            np.testing.assert_allclose(
                np.linalg.norm(geometry.rotations_wxyz, axis=1),
                1.0,
                atol=1e-7,
            )
            # terrain = 0.1*x_index + 0.01*y_index on a unit grid.
            expected_normal = np.asarray([-0.1, -0.01, 1.0])
            expected_normal /= np.linalg.norm(expected_normal)
            np.testing.assert_allclose(
                geometry.normals_gs[0], expected_normal, atol=1e-7
            )
            np.testing.assert_allclose(
                geometry.scales_gs[:, 2],
                0.1
                * np.minimum(
                    geometry.scales_gs[:, 0], geometry.scales_gs[:, 1]
                ),
                atol=1e-7,
            )

            # The quaternion must orient the local thin z-axis along the
            # analytic sand normal. This catches wxyz/xyzw convention swaps.
            w, x, y, z = geometry.rotations_wxyz[0]
            rotation = np.asarray(
                [
                    [
                        1 - 2 * (y * y + z * z),
                        2 * (x * y - z * w),
                        2 * (x * z + y * w),
                    ],
                    [
                        2 * (x * y + z * w),
                        1 - 2 * (x * x + z * z),
                        2 * (y * z - x * w),
                    ],
                    [
                        2 * (x * z - y * w),
                        2 * (y * z + x * w),
                        1 - 2 * (x * x + y * y),
                    ],
                ]
            )
            np.testing.assert_allclose(
                rotation[:, 2], geometry.normals_gs[0], atol=1e-6
            )

    def test_p4_mixed_water_controls_wet_sand_without_free_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exchange, run, _ = self.make_exchange_and_run(
                Path(temporary), sand_height=0.2, mixed_water=True
            )
            adapter = SandWaterRunAdapter(
                exchange,
                run,
                require_pure_water=False,
                sand_epsilon=1.0e-6,
            )
            frame = adapter.sample(0.1)
            np.testing.assert_allclose(frame.mixed_water_height, 0.1)
            np.testing.assert_allclose(frame.pure_water_height, 0.0)
            np.testing.assert_allclose(frame.mixed_water_wetness, 0.5)
            self.assertFalse(frame.free_water_mask.any())
            geometry = adapter.sand_geometry(
                frame, full_coverage_height=0.2
            )
            np.testing.assert_allclose(geometry.mixed_water_wetness, 0.5)

    def test_terrain_reference_uses_complete_fixed_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exchange, run, _ = self.make_exchange_and_run(
                Path(temporary), sand_height=0.2
            )
            adapter = SandWaterRunAdapter(
                exchange,
                run,
                require_pure_water=False,
                sand_epsilon=1.0e-6,
            )
            frame = adapter.sample(0.1)
            triangles = adapter.terrain_reference_triangles(frame)
            self.assertEqual(triangles.shape, (12, 3, 3))
            self.assertTrue(np.isfinite(triangles).all())

    def test_water_reference_keeps_only_all_wet_triangles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exchange, run, _ = self.make_exchange_and_run(Path(temporary))
            adapter = SandWaterRunAdapter(
                exchange,
                run,
                free_water_epsilon=0.05,
            )
            frame = adapter.sample(0.1)
            all_wet = adapter.water_reference_triangles(frame)
            self.assertEqual(all_wet.shape, (12, 3, 3))
            self.assertTrue(np.isfinite(all_wet).all())

            mask = frame.free_water_mask.copy()
            mask[0, 0] = False
            partial = replace(frame, free_water_mask=mask)
            triangles = adapter.water_reference_triangles(partial)
            self.assertLess(triangles.shape[0], all_wet.shape[0])
            dry_point = adapter.geometry(frame).xyz_gs[0]
            self.assertFalse(np.any(np.all(
                np.isclose(triangles, dry_point[None, None, :]), axis=2
            )))

    def test_t1_triangle_geometry_is_fixed_shared_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exchange, run, _ = self.make_exchange_and_run(
                Path(temporary), sand_height=0.2
            )
            adapter = SandWaterRunAdapter(
                exchange,
                run,
                require_pure_water=False,
                sand_epsilon=1.0e-6,
            )
            frame = adapter.sample(0.1)
            geometry = adapter.build_sand_geometry(
                frame,
                backend="triangle_games",
                full_coverage_height=0.2,
                triangle_coverage_gamma=1.0,
            )

            # A 4x3 cell-center grid owns 3x2 quads and exactly two fixed
            # checkerboard triangles per quad. No subdivision is active in T1.
            self.assertEqual(geometry.geometry_backend, "triangle_games")
            self.assertEqual(geometry.base_triangle_count, 12)
            self.assertEqual(geometry.candidate_triangle_count, 12)
            self.assertEqual(geometry.rejected_triangle_count, 0)
            self.assertEqual(geometry.xyz_gs.shape, (12, 3))
            self.assertEqual(geometry.triangle_vertices_gs.shape, (12, 3, 3))
            np.testing.assert_allclose(
                geometry.triangle_vertex_sand_height, 0.2
            )
            np.testing.assert_allclose(
                geometry.triangle_slope, np.hypot(0.1, 0.01), atol=1.0e-7
            )
            self.assertAlmostEqual(geometry.sand_epsilon, 1.0e-6)
            self.assertAlmostEqual(geometry.full_coverage_height, 0.2)
            np.testing.assert_array_equal(geometry.boundary_class, 0)
            np.testing.assert_allclose(geometry.coverage, 1.0, atol=1e-7)
            np.testing.assert_allclose(
                np.linalg.norm(geometry.normals_gs, axis=1), 1.0, atol=1e-7
            )
            np.testing.assert_allclose(
                np.linalg.norm(geometry.rotations_wxyz, axis=1),
                1.0,
                atol=1e-7,
            )
            self.assertTrue(np.all(geometry.scales_gs > 0.0))

            # The first even-parity quad uses p00,p10,p11. Its centroid is
            # fixed by topology and terrain+sand, not by a fitted optimizer.
            np.testing.assert_allclose(
                geometry.xyz_gs[0],
                [-5.0 / 6.0, -2.0 / 3.0, 0.27],
                atol=1e-7,
            )
            repeated = adapter.build_sand_geometry(
                frame,
                backend="triangle_games",
                full_coverage_height=0.2,
            )
            np.testing.assert_array_equal(
                repeated.active_cell_indices, geometry.active_cell_indices
            )
            np.testing.assert_allclose(repeated.xyz_gs, geometry.xyz_gs)

    def test_t2_uniform_subdivision_has_four_deterministic_children(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exchange, run, _ = self.make_exchange_and_run(
                Path(temporary), sand_height=0.2
            )
            adapter = SandWaterRunAdapter(
                exchange,
                run,
                require_pure_water=False,
                sand_epsilon=1.0e-6,
            )
            frame = adapter.sample(0.1)
            x = np.arange(4, dtype=np.float32)[:, None]
            y = np.arange(3, dtype=np.float32)[None, :]
            frame = replace(
                frame,
                # Nonuniform fields verify that child material values are
                # interpolated at shared midpoints, not copied per parent.
                saturation=0.10 * x + 0.05 * y,
                mixed_water_wetness=0.20 + 0.20 * x + 0.05 * y,
            )
            parent = adapter.build_sand_geometry(
                frame,
                backend="triangle_games",
                full_coverage_height=0.2,
                triangle_subdivision_level=0,
            )
            subdivided = adapter.build_sand_geometry(
                frame,
                backend="triangle_games",
                full_coverage_height=0.2,
                triangle_subdivision_level=1,
            )

            self.assertEqual(subdivided.subdivision_level, 1)
            self.assertEqual(subdivided.pre_subdivision_triangle_count, 12)
            self.assertEqual(subdivided.xyz_gs.shape, (48, 3))
            self.assertEqual(subdivided.active_cell_indices.shape, (48, 3))
            np.testing.assert_array_equal(
                subdivided.active_cell_indices[:4, :2],
                np.broadcast_to(parent.active_cell_indices[0], (4, 2)),
            )
            np.testing.assert_array_equal(
                subdivided.active_cell_indices[:4, 2], [0, 1, 2, 3]
            )
            # Child 0 uses [v0, midpoint(v0,v1), midpoint(v2,v0)].
            np.testing.assert_allclose(
                subdivided.xyz_gs[0],
                [-7.0 / 6.0, -5.0 / 6.0, 0.235],
                atol=1.0e-7,
            )
            self.assertAlmostEqual(subdivided.saturation[0], 1.0 / 24.0)
            self.assertAlmostEqual(subdivided.mixed_water_wetness[0], 0.275)
            np.testing.assert_allclose(subdivided.coverage, 1.0, atol=1e-7)
            np.testing.assert_allclose(
                np.linalg.norm(subdivided.normals_gs, axis=1),
                1.0,
                atol=1e-7,
            )
            self.assertEqual(
                subdivided.rejected_triangle_count,
                parent.rejected_triangle_count,
            )

            repeated = adapter.build_sand_geometry(
                frame,
                backend="triangle_games",
                full_coverage_height=0.2,
                triangle_subdivision_level=1,
            )
            np.testing.assert_array_equal(
                repeated.active_cell_indices,
                subdivided.active_cell_indices,
            )
            np.testing.assert_allclose(repeated.xyz_gs, subdivided.xyz_gs)

    def test_t1_triangle_boundary_is_clipped_without_crossing_dry_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exchange, run, _ = self.make_exchange_and_run(
                Path(temporary), sand_height=0.2
            )
            adapter = SandWaterRunAdapter(
                exchange,
                run,
                require_pure_water=False,
                sand_epsilon=1.0e-6,
            )
            frame = adapter.sample(0.1)
            sand = np.zeros_like(frame.sand_height)
            sand[0, 0] = 0.2
            sparse = replace(
                frame,
                sand_height=sand,
                sand_surface=frame.terrain + sand,
                sand_mask=sand > adapter.sand_epsilon,
                saturation=np.zeros_like(sand),
                mixed_water_wetness=np.zeros_like(sand),
            )
            clipped = adapter.build_sand_geometry(
                sparse,
                backend="triangle_games",
                full_coverage_height=0.2,
                triangle_boundary_mode="clipped",
            )
            # The corner belongs to the two base triangles of the first quad.
            # Each is clipped at h_s=epsilon rather than discarded or expanded
            # into unrelated dry quads.
            self.assertEqual(clipped.candidate_triangle_count, 2)
            self.assertEqual(clipped.xyz_gs.shape[0], 2)
            np.testing.assert_array_equal(clipped.boundary_class, 1)
            self.assertEqual(
                clipped.triangle_vertex_sand_height.shape, (2, 3)
            )
            self.assertEqual(clipped.triangle_slope.shape, (2,))
            self.assertTrue(np.all(np.isfinite(clipped.triangle_slope)))
            self.assertTrue(np.all(clipped.triangle_slope >= 0.0))
            np.testing.assert_allclose(
                clipped.triangle_vertex_sand_height.min(axis=1),
                adapter.sand_epsilon,
                atol=1.0e-8,
            )
            np.testing.assert_allclose(
                clipped.triangle_vertex_sand_height.max(axis=1), 0.2
            )
            self.assertTrue(np.all(clipped.coverage > 0.0))
            self.assertTrue(np.all(clipped.coverage < 1.0))

            clipped_t2 = adapter.build_sand_geometry(
                sparse,
                backend="triangle_games",
                full_coverage_height=0.2,
                triangle_boundary_mode="clipped",
                triangle_subdivision_level=1,
            )
            self.assertEqual(clipped_t2.pre_subdivision_triangle_count, 2)
            self.assertEqual(clipped_t2.xyz_gs.shape[0], 8)
            np.testing.assert_array_equal(clipped_t2.boundary_class, 1)
            np.testing.assert_array_equal(
                clipped_t2.active_cell_indices[:, 2],
                np.tile(np.arange(4), 2),
            )

            fully_active = adapter.build_sand_geometry(
                sparse,
                backend="triangle_games",
                full_coverage_height=0.2,
                triangle_boundary_mode="fully_active",
            )
            self.assertEqual(fully_active.xyz_gs.shape[0], 0)

            dry = replace(
                sparse,
                sand_height=np.zeros_like(sand),
                sand_surface=frame.terrain.copy(),
                sand_mask=np.zeros_like(sand, dtype=np.bool_),
            )
            empty = adapter.build_sand_geometry(
                dry,
                backend="triangle_games",
                full_coverage_height=0.2,
            )
            self.assertEqual(empty.candidate_triangle_count, 0)
            self.assertEqual(empty.xyz_gs.shape, (0, 3))
            self.assertEqual(empty.rotations_wxyz.shape, (0, 4))

    def test_t1_connectivity_keeps_steep_sand_on_regular_terrain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exchange, run, _ = self.make_exchange_and_run(
                Path(temporary), sand_height=0.2
            )
            adapter = SandWaterRunAdapter(
                exchange,
                run,
                require_pure_water=False,
                sand_epsilon=1.0e-6,
            )
            frame = adapter.sample(0.1)
            terrain = np.zeros((4, 3), dtype=np.float32)
            sand = np.full((4, 3), 0.2, dtype=np.float32)
            sand[2:, :] = 10.0
            steep = self.replace_sand_scene(frame, adapter, terrain, sand)

            geometry = adapter.build_sand_geometry(
                steep,
                backend="triangle_games",
                full_coverage_height=0.2,
                max_tangent_scale_ratio=4.0,
                triangle_terrain_max_edge_ratio=4.0,
            )

            # The old T1 rule rejected these triangles from the total surface
            # slope. T1.1 keeps them because the underlying terrain is flat,
            # and records them only as candidates for later T2 subdivision.
            self.assertEqual(geometry.rejected_triangle_count, 0)
            self.assertEqual(geometry.xyz_gs.shape[0], 12)
            self.assertGreater(geometry.steep_sand_triangle_count, 0)

    def test_t1_connectivity_distinguishes_contact_from_exposed_cliff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exchange, run, _ = self.make_exchange_and_run(
                Path(temporary), sand_height=0.2
            )
            adapter = SandWaterRunAdapter(
                exchange,
                run,
                require_pure_water=False,
                sand_epsilon=1.0e-6,
            )
            frame = adapter.sample(0.1)
            low_to_high = np.zeros((4, 3), dtype=np.float32)
            low_to_high[2:, :] = 2.0
            sand_on_low_side = np.zeros((4, 3), dtype=np.float32)
            sand_on_low_side[:2, :] = 0.2
            contact = self.replace_sand_scene(
                frame, adapter, low_to_high, sand_on_low_side
            )
            contact_geometry = adapter.build_sand_geometry(
                contact,
                backend="triangle_games",
                full_coverage_height=0.2,
                triangle_terrain_max_edge_ratio=1.1,
                triangle_contact_tolerance=0.0,
            )
            self.assertGreater(
                contact_geometry.rejected_high_terrain_contact_count, 0
            )
            self.assertEqual(contact_geometry.rejected_terrain_cliff_count, 0)

            # Reverse the terrain step: sand on a high platform does not enter
            # the low dry cell, but the exposed cliff still must not become an
            # artificial diagonal Gaussian film.
            high_to_low = 2.0 - low_to_high
            exposed = self.replace_sand_scene(
                frame, adapter, high_to_low, sand_on_low_side
            )
            exposed_geometry = adapter.build_sand_geometry(
                exposed,
                backend="triangle_games",
                full_coverage_height=0.2,
                triangle_terrain_max_edge_ratio=1.1,
                triangle_contact_tolerance=0.0,
            )
            self.assertEqual(
                exposed_geometry.rejected_high_terrain_contact_count, 0
            )
            self.assertGreater(
                exposed_geometry.rejected_terrain_cliff_count, 0
            )

    def test_t1_connectivity_allows_a_fully_buried_terrain_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exchange, run, _ = self.make_exchange_and_run(
                Path(temporary), sand_height=0.2
            )
            adapter = SandWaterRunAdapter(
                exchange,
                run,
                require_pure_water=False,
                sand_epsilon=1.0e-6,
            )
            frame = adapter.sample(0.1)
            terrain = np.zeros((4, 3), dtype=np.float32)
            terrain[2:, :] = 2.0
            sand = np.full((4, 3), 2.2, dtype=np.float32)
            sand[2:, :] = 0.3
            buried = self.replace_sand_scene(frame, adapter, terrain, sand)

            geometry = adapter.build_sand_geometry(
                buried,
                backend="triangle_games",
                full_coverage_height=0.2,
                triangle_terrain_max_edge_ratio=1.1,
                triangle_contact_tolerance=0.0,
            )

            # Terrain alone is a discontinuity, but both sand columns rise
            # above the higher side. The visible sand surface is connected.
            self.assertEqual(geometry.rejected_triangle_count, 0)
            self.assertEqual(geometry.xyz_gs.shape[0], 12)
            self.assertGreater(geometry.accepted_buried_step_count, 0)

    def test_sand_geometry_backend_selector_preserves_legacy_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exchange, run, _ = self.make_exchange_and_run(
                Path(temporary), sand_height=0.2
            )
            adapter = SandWaterRunAdapter(
                exchange, run, require_pure_water=False
            )
            frame = adapter.sample(0.1)
            direct = adapter.sand_geometry(frame)
            dispatched = adapter.build_sand_geometry(frame)
            np.testing.assert_array_equal(direct.xyz_gs, dispatched.xyz_gs)
            np.testing.assert_array_equal(
                direct.rotations_wxyz, dispatched.rotations_wxyz
            )
            with self.assertRaisesRegex(
                SimulationRunError, "zero or one"
            ):
                adapter.build_sand_geometry(
                    frame,
                    backend="triangle_games",
                    triangle_subdivision_level=2,
                )
            with self.assertRaisesRegex(
                SimulationRunError, "cell_tangent or triangle_games"
            ):
                adapter.build_sand_geometry(frame, backend="unknown")

    def test_p3_case_samples_configured_sand_at_cell_centers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange, _, _ = self.make_exchange_and_run(root)
            output = root / "p3-case"
            profile = {
                "schema_version": 1,
                "profile_id": "unit-p3",
                "initial": {
                    "sand": {
                        "type": "gaussian",
                        "center": [0.0, 0.0],
                        "sigma": 0.5,
                        "height": 0.25,
                    },
                    "total_water": 0.0,
                },
                "physics": {
                    "sand_terrain_contact_tolerance": 0.005,
                },
                "solver": {
                    "water_dt": 0.001,
                    "sand_substeps": 2,
                    "steps": 2,
                    "frame_stride": 1,
                    "target_cfl": 0.45,
                },
            }
            metadata = prepare_sand_water_case(
                exchange_bundle=exchange,
                output=output,
                profile=profile,
            )
            grid = GridSpec((4, 3), (-2.0, -1.5), (2.0, 1.5))
            expected = sample_height_spec(
                profile["initial"]["sand"], grid, name="sand"
            )
            np.testing.assert_allclose(np.load(output / "sand.npy"), expected)
            self.assertGreater(metadata["initialization"]["sand_volume"], 0.0)
            with (output / "scene.yaml").open(
                "r", encoding="utf-8"
            ) as stream:
                scene = yaml.safe_load(stream)
            self.assertEqual(scene["initial"]["sand"]["file"], "sand.npy")
            self.assertTrue(scene["physics"]["enable_sand_internal_force"])
            self.assertEqual(
                scene["physics"]["sand_terrain_contact_tolerance"],
                0.005,
            )
            self.assertEqual(
                metadata["initialization"]["sand_kind"],
                "terrain_relative_thickness",
            )

    def test_p3_absolute_surface_initialization_subtracts_terrain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange, _, terrain = self.make_exchange_and_run(root)
            output = root / "surface-case"
            profile = {
                "schema_version": 1,
                "initial": {
                    "sand_surface": {
                        "type": "gaussian",
                        "center": [0.0, 0.0],
                        "sigma": 2.0,
                        "height": 0.5,
                    },
                    "total_water": 0.0,
                },
                "solver": {
                    "water_dt": 0.001,
                    "steps": 1,
                    "frame_stride": 1,
                    "target_cfl": 0.45,
                },
            }
            metadata = prepare_sand_water_case(
                exchange_bundle=exchange,
                output=output,
                profile=profile,
            )
            grid = GridSpec((4, 3), (-2.0, -1.5), (2.0, 1.5))
            target = sample_height_spec(
                profile["initial"]["sand_surface"],
                grid,
                name="sand_surface",
            )
            np.testing.assert_allclose(
                np.load(output / "target_sand_surface.npy"), target
            )
            np.testing.assert_allclose(
                np.load(output / "sand.npy"),
                np.maximum(target - terrain, 0.0),
            )
            self.assertEqual(
                metadata["initialization"]["sand_kind"],
                "absolute_surface",
            )
            self.assertAlmostEqual(
                metadata["initialization"]["target_surface_max"],
                float(target.max()),
            )

    def test_p3_rejects_ambiguous_sand_initializers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange, _, _ = self.make_exchange_and_run(root)
            profile = {
                "schema_version": 1,
                "initial": {
                    "sand": 0.1,
                    "sand_surface": 0.2,
                    "total_water": 0.0,
                },
                "solver": {
                    "water_dt": 0.001,
                    "steps": 1,
                    "frame_stride": 1,
                    "target_cfl": 0.45,
                },
            }
            with self.assertRaisesRegex(ValueError, "exactly one"):
                prepare_sand_water_case(
                    exchange_bundle=exchange,
                    output=root / "ambiguous-case",
                    profile=profile,
                )

    def test_p3_saturation_initialization_matches_solver_definition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange, _, _ = self.make_exchange_and_run(root)
            output = root / "wet-case"
            profile = {
                "schema_version": 1,
                "initial": {
                    "sand": {
                        "type": "cylinder",
                        "radius": 1.0,
                        "height": 0.3,
                    },
                    "saturation": 0.25,
                },
                "solver": {
                    "water_dt": 0.001,
                    "steps": 1,
                    "frame_stride": 1,
                    "target_cfl": 0.45,
                },
            }
            prepare_sand_water_case(
                exchange_bundle=exchange,
                output=output,
                profile=profile,
            )
            sand = np.load(output / "sand.npy")
            water = np.load(output / "total_water.npy")
            np.testing.assert_allclose(water, sand / 3.0)
            material = sand + water
            recovered = np.divide(
                water,
                material,
                out=np.zeros_like(water),
                where=material > 0.0,
            )
            np.testing.assert_allclose(recovered[sand > 0.0], 0.25)

    def test_p3_sand_saturation_override_preserves_exterior_water(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange, _, terrain = self.make_exchange_and_run(root)
            output = root / "uniform-sand-saturation-case"
            profile = {
                "schema_version": 1,
                "initial": {
                    "sand": {
                        "type": "box",
                        "min": [-1.0, -1.0],
                        "max": [0.0, 1.0],
                        "height": 0.3,
                    },
                    "water_surface": 0.4,
                    "sand_saturation_override": {
                        "value": 0.25,
                        "sand_height_epsilon": 1.0e-8,
                    },
                },
                "solver": {
                    "water_dt": 0.001,
                    "steps": 1,
                    "frame_stride": 1,
                    "target_cfl": 0.45,
                },
            }
            metadata = prepare_sand_water_case(
                exchange_bundle=exchange,
                output=output,
                profile=profile,
            )
            sand = np.load(output / "sand.npy")
            water = np.load(output / "total_water.npy")
            base_water = np.load(
                output / "base_total_water_before_saturation_override.npy"
            )
            mask = np.load(output / "sand_saturation_override_mask.npy") > 0.5

            np.testing.assert_allclose(base_water, np.maximum(0.4 - terrain, 0.0))
            np.testing.assert_allclose(water[mask], sand[mask] / 3.0)
            np.testing.assert_allclose(water[~mask], base_water[~mask])
            realized = water[mask] / (water[mask] + sand[mask])
            np.testing.assert_allclose(realized, 0.25)
            self.assertEqual(
                metadata["initialization"]["sand_saturation_override"],
                0.25,
            )
            self.assertAlmostEqual(
                metadata["initialization"][
                    "sand_saturation_override_realized_min"
                ],
                0.25,
            )
            self.assertIn(
                "uniform_sand_saturation_override",
                metadata["initialization"]["water_kind"],
            )

    def test_saturation_override_boundary_feather_is_monotone(self) -> None:
        support = np.zeros((9, 9), dtype=bool)
        support[1:8, 1:8] = True
        weight = _interior_feather_weight(support, 2)

        # The outer support ring retains the base hydraulic state; the next
        # ring is blended and the sufficiently deep core is fully overridden.
        self.assertEqual(weight[1, 4], 0.0)
        self.assertAlmostEqual(weight[2, 4], 0.5)
        self.assertEqual(weight[3, 4], 1.0)
        self.assertEqual(weight[0, 4], 0.0)
        self.assertTrue(np.all((weight >= 0.0) & (weight <= 1.0)))

    def test_subtract_shape_creates_a_non_negative_breach(self) -> None:
        grid = GridSpec((8, 8), (-2.0, -2.0), (2.0, 2.0))
        field = sample_height_spec(
            {
                "combine": "subtract",
                "shapes": [
                    {
                        "type": "box",
                        "min": [-1.5, -0.5],
                        "max": [1.5, 0.5],
                        "height": 0.4,
                    },
                    {
                        "type": "box",
                        "min": [-0.25, -0.75],
                        "max": [0.25, 0.75],
                        "height": 0.4,
                    },
                ],
            },
            grid,
            name="breached_ridge",
        )
        self.assertTrue(np.all(field >= 0.0))
        self.assertGreater(field[1, 3], 0.0)
        self.assertEqual(field[3, 3], 0.0)

    def test_water_surface_field_is_independent_from_sand(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange, _, terrain = self.make_exchange_and_run(root)
            output = root / "water-surface-case"
            profile = {
                "schema_version": 1,
                "initial": {
                    "sand": 0.1,
                    "water_surface": {
                        "combine": "max",
                        "shapes": [
                            0.25,
                            {
                                "type": "box",
                                "min": [-2.0, -1.5],
                                "max": [0.0, 1.5],
                                "height": 0.45,
                            },
                        ],
                    },
                },
                "solver": {
                    "water_dt": 0.001,
                    "steps": 1,
                    "frame_stride": 1,
                    "target_cfl": 0.45,
                },
            }
            metadata = prepare_sand_water_case(
                exchange_bundle=exchange,
                output=output,
                profile=profile,
            )
            target = np.load(output / "target_water_surface.npy")
            water = np.load(output / "total_water.npy")
            np.testing.assert_allclose(
                water, np.maximum(target - terrain, 0.0), atol=1.0e-12
            )
            self.assertEqual(
                metadata["initialization"]["water_kind"],
                "absolute_surface_field",
            )
            self.assertEqual(
                metadata["initialization"][
                    "target_water_surface_wet_error_max"
                ],
                0.0,
            )

    def test_constrained_water_surface_caps_depth_and_applies_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange, _, _ = self.make_exchange_and_run(root)
            output = root / "constrained-water-case"
            profile = {
                "schema_version": 1,
                "initial": {
                    "sand": 0.0,
                    "water_surface": {
                        "surface": 2.0,
                        "max_depth": 0.25,
                        "mask": {
                            "type": "box",
                            "min": [-2.0, -1.5],
                            "max": [0.0, 1.5],
                            "height": 1.0,
                        },
                    },
                },
                "solver": {
                    "water_dt": 0.001,
                    "steps": 1,
                    "frame_stride": 1,
                    "target_cfl": 0.45,
                },
            }

            metadata = prepare_sand_water_case(
                exchange_bundle=exchange,
                output=output,
                profile=profile,
            )
            water = np.load(output / "total_water.npy")
            depth_limit = np.load(output / "water_depth_limit.npy")
            mask = np.load(output / "water_initial_mask.npy")

            self.assertEqual(
                metadata["initialization"]["water_kind"],
                "constrained_absolute_surface_field",
            )
            np.testing.assert_allclose(depth_limit, 0.25)
            np.testing.assert_allclose(water, 0.25 * mask)
            self.assertLessEqual(float(water.max()), 0.25)
            self.assertGreater(
                metadata["initialization"][
                    "target_water_surface_wet_residual_max"
                ],
                0.0,
            )
            self.assertNotIn(
                "target_water_surface_wet_error_max",
                metadata["initialization"],
            )


if __name__ == "__main__":
    unittest.main()
