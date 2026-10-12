from pathlib import Path
import tempfile
import unittest

import yaml

from sandygs.configuration import ConfigError
from scripts.run_p2_pipeline import load_profile


REPO_ROOT = Path(__file__).resolve().parents[1]


class P2PipelineTests(unittest.TestCase):
    def test_repository_profiles_select_sandygs_p4(self) -> None:
        for name in ("garden-p2-pure-water.yaml", "garden-p2-pure-water-sequence.yaml"):
            profile = load_profile(REPO_ROOT / "configs/simulations" / name)
            self.assertEqual(
                profile["render"]["water_rendering_backend"], "sandygs_p4"
            )

    def test_missing_backend_defaults_to_sandygs_p4(self) -> None:
        profile = {
            "schema_version": 1,
            "solver": {},
            "render": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.yaml"
            path.write_text(yaml.safe_dump(profile), encoding="utf-8")
            resolved = load_profile(path)
        self.assertEqual(
            resolved["render"]["water_rendering_backend"], "sandygs_p4"
        )

    def test_explicit_legacy_backend_reports_migration_error(self) -> None:
        profile = {
            "schema_version": 1,
            "solver": {},
            "render": {"water_rendering_backend": "rainygs_legacy"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.yaml"
            path.write_text(yaml.safe_dump(profile), encoding="utf-8")
            with self.assertRaisesRegex(
                ConfigError, "pre-refactor-reconstruction-v1"
            ):
                load_profile(path)


if __name__ == "__main__":
    unittest.main()
