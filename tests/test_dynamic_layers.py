from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scene.gaussian_model import GaussianModel


class DynamicLayerDispatchTests(unittest.TestCase):
    def test_sand_water_dispatch_uses_physical_time(self) -> None:
        model = SimpleNamespace(
            args=SimpleNamespace(render_sand=True),
            _add_sand_water_gaussians=Mock(return_value="sand-water"),
            _add_sand_water_fluid_gaussian=Mock(),
        )

        result = GaussianModel.update_dynamic_layers(model, time_seconds=0.125)

        self.assertEqual(result, "sand-water")
        model._add_sand_water_gaussians.assert_called_once_with(0.125)
        model._add_sand_water_fluid_gaussian.assert_not_called()

    def test_pure_water_dispatch_preserves_physical_time(self) -> None:
        model = SimpleNamespace(
            args=SimpleNamespace(render_sand=False),
            _add_sand_water_gaussians=Mock(),
            _add_sand_water_fluid_gaussian=Mock(return_value="water"),
        )

        result = GaussianModel.update_dynamic_layers(model, time_seconds=0.25)

        self.assertEqual(result, "water")
        model._add_sand_water_fluid_gaussian.assert_called_once_with(0.25)
        model._add_sand_water_gaussians.assert_not_called()

    def test_physical_time_is_required(self) -> None:
        sand_water = SimpleNamespace(
            args=SimpleNamespace(render_sand=True),
        )

        with self.assertRaisesRegex(TypeError, "time_seconds"):
            GaussianModel.update_dynamic_layers(sand_water)
        with self.assertRaisesRegex(ValueError, "physical time"):
            GaussianModel.update_dynamic_layers(sand_water, time_seconds=None)


if __name__ == "__main__":
    unittest.main()
