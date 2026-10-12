from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.pile_benchmark import (  # noqa: E402
    _sand_triangles,
    load_experiment_config,
    materialize_initial_perturbation,
    resolve_solver_scene,
    select_experiment_phase,
)


class PileBenchmarkTests(unittest.TestCase):
    def test_current_reconstruction_baseline_parameters_are_explicit(self) -> None:
        path = REPO_ROOT / "configs/physics/fig4-piles-preflight.yaml"
        source = load_experiment_config(path)
        scene = source["solver_scene"]

        self.assertEqual(
            scene["physics"],
            {
                "gravity": 9.81,
                "diffusion_rate": 0.0,
                "momentum_exchange_rate": 1.0,
                "ground_friction": 0.42,
                "sand_terrain_contact_tolerance": 0.0,
                "young_modulus": 75000.0,
                "poisson": 0.25,
                "drucker_prager_friction": 0.45,
                "cohesion": {
                    "c0": 400.0,
                    "c1": 2000.0,
                    "c2": 2500.0,
                    "phi1": 0.20,
                    "phi2": 0.25,
                    "phi3": 0.33,
                },
                "enable_sand_internal_force": True,
            },
        )
        self.assertEqual(
            scene["time"],
            {
                "water_dt": 0.0025,
                "sand_substeps": 3,
                "adaptive": True,
                "target_cfl": 0.45,
                "internal_force_height_fraction": 0.0004,
            },
        )
        self.assertEqual(scene["grid"]["resolution"], [160, 160])
        self.assertAlmostEqual(scene["grid"]["spacing"], 0.015)
        self.assertEqual(
            scene["initial"]["sand"],
            {
                "type": "cylinder",
                "center": [0.0, 0.0],
                "radius": 0.30,
                "height": 0.21,
                "edge_width": 0.04,
            },
        )

    def test_preflight_config_expands_saturation_without_changing_sand(self) -> None:
        path = REPO_ROOT / "configs/physics/fig4-piles-preflight.yaml"
        config = load_experiment_config(path)
        scene, phi = resolve_solver_scene(config)
        sand = scene["initial"]["sand"]
        water = scene["initial"]["total_water"]

        self.assertAlmostEqual(phi, 0.25)
        self.assertEqual(sand["radius"], water["radius"])
        self.assertEqual(sand["edge_width"], water["edge_width"])
        self.assertAlmostEqual(water["height"], sand["height"] / 3.0)
        self.assertNotIn("saturation", scene["initial"])

    def test_range_phase_overrides_only_dynamic_range_fields(self) -> None:
        path = REPO_ROOT / "configs/physics/fig4-piles-preflight.yaml"
        source = load_experiment_config(path)
        ranged = select_experiment_phase(source, "range")

        expected_scene = deepcopy(source["solver_scene"])
        expected_scene["name"] = "fig4_piles_range_phi045"
        expected_scene["initial"]["saturation"] = 0.45
        expected_scene["run"].update({"duration": 0.5, "frame_stride": 10})
        expected_render = deepcopy(source["render"])
        expected_render["crop_half_extent"] = 1.10

        self.assertEqual(ranged["solver_scene"], expected_scene)
        self.assertEqual(ranged["render"], expected_render)
        self.assertEqual(ranged["selected_phase"], "range")

    def test_sensitivity_phases_preserve_physical_snapshot_times(self) -> None:
        path = REPO_ROOT / "configs/physics/fig4-piles-preflight.yaml"
        source = load_experiment_config(path)
        baseline = select_experiment_phase(source, "range")
        half_dt = select_experiment_phase(source, "range_half_dt")
        friction = select_experiment_phase(source, "range_friction_042")

        base_scene = baseline["solver_scene"]
        half_scene = half_dt["solver_scene"]
        friction_scene = friction["solver_scene"]
        expected_half = deepcopy(base_scene)
        expected_half["name"] = "fig4_piles_range_phi045_half_dt"
        expected_half["time"]["water_dt"] = 0.00125
        expected_half["run"]["frame_stride"] = 20
        self.assertEqual(half_scene, expected_half)
        self.assertEqual(half_dt["render"], baseline["render"])
        self.assertAlmostEqual(
            half_scene["time"]["water_dt"] * half_scene["run"]["frame_stride"],
            base_scene["time"]["water_dt"] * base_scene["run"]["frame_stride"],
        )

        expected_friction = deepcopy(base_scene)
        expected_friction["name"] = "fig4_piles_range_phi045_friction_042"
        self.assertEqual(friction_scene, expected_friction)
        self.assertEqual(friction["render"], baseline["render"])
        self.assertEqual(friction["lifecycle"]["status"], "historical")
        self.assertAlmostEqual(base_scene["physics"]["ground_friction"], 0.42)

    def test_sweep_phase_freezes_accepted_dt_friction_and_geometry(self) -> None:
        path = REPO_ROOT / "configs/physics/fig4-piles-preflight.yaml"
        source = load_experiment_config(path)
        sweep = select_experiment_phase(source, "sweep")
        scene = sweep["solver_scene"]

        expected_scene = deepcopy(source["solver_scene"])
        expected_scene["name"] = "fig4_piles_sweep"
        expected_scene["grid"]["resolution"] = [224, 224]
        expected_scene["run"].update({"duration": 2.0, "frame_stride": 20})
        expected_render = deepcopy(source["render"])
        expected_render["crop_half_extent"] = 1.50

        self.assertEqual(scene, expected_scene)
        self.assertEqual(sweep["render"], expected_render)
        self.assertAlmostEqual(scene["physics"]["ground_friction"], 0.42)

    def test_triangle_builder_keeps_boundary_ramp_but_not_empty_cells(self) -> None:
        x = np.asarray([-1.0, 0.0, 1.0])
        y = np.asarray([-1.0, 0.0, 1.0])
        sand = np.zeros((3, 3), dtype=np.float64)
        sand[1, 1] = 0.2
        triangles, normals = _sand_triangles(x, y, sand, sand, 1.0e-8)

        self.assertEqual(triangles.shape, (6, 3, 3))
        self.assertEqual(normals.shape, (6, 3))
        self.assertTrue(np.all(normals[:, 2] >= 0.0))

    def test_optional_free_water_surface_can_be_empty(self) -> None:
        x = np.asarray([-1.0, 0.0, 1.0])
        y = np.asarray([-1.0, 0.0, 1.0])
        surface = np.zeros((3, 3), dtype=np.float64)
        pure_water = np.zeros_like(surface)

        triangles, normals = _sand_triangles(
            x,
            y,
            surface,
            pure_water,
            1.0e-8,
            allow_empty=True,
        )

        self.assertEqual(triangles.shape, (0, 3, 3))
        self.assertEqual(normals.shape, (0, 3))

    def test_sinusoidal_initializers_isolate_height_and_saturation(self) -> None:
        path = REPO_ROOT / "configs/physics/flat-long-dam-preflight.yaml"
        source = load_experiment_config(path)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for phase in (
                "perturbation_control",
                "perturbation_saturation",
                "perturbation_height",
            ):
                selected = select_experiment_phase(source, phase)
                scene, phi = resolve_solver_scene(selected)
                _, report = materialize_initial_perturbation(
                    scene,
                    phi,
                    selected["initial_perturbation"],
                    root / phase,
                )
                sand = np.load(root / phase / "sand.npy", allow_pickle=False)
                water = np.load(
                    root / phase / "total_water.npy", allow_pickle=False
                )
                fields[phase] = (sand, water)
                self.assertAlmostEqual(report["wavelength_in_cells"], 16.0)

            control_sand, control_water = fields["perturbation_control"]
            saturation_sand, saturation_water = fields["perturbation_saturation"]
            height_sand, height_water = fields["perturbation_height"]
            active = control_sand > 0.0

            np.testing.assert_array_equal(saturation_sand, control_sand)
            self.assertGreater(
                float(np.max(np.abs(saturation_water - control_water))), 0.0
            )
            self.assertGreater(float(np.max(np.abs(height_sand - control_sand))), 0.0)
            height_phi = height_water[active] / (
                height_water[active] + height_sand[active]
            )
            np.testing.assert_allclose(height_phi, 0.25, rtol=0.0, atol=1.0e-14)


if __name__ == "__main__":
    unittest.main()
