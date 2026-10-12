from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from sandygs.configuration import (
    ConfigError,
    build_scene_manifest,
    load_scene_config,
    normalize_scene_config,
    renderer_arguments_from_manifest,
    resolve_water_rendering_backend,
    validate_current_runtime_config,
    validate_sand_water_resources,
)
from sandygs.exchange import GridSpec
from sandygs.prepare import base_metadata, write_field
from scripts.migrate_scene_config import migrate_scene_file


REPO_ROOT = Path(__file__).resolve().parents[1]
GARDEN = REPO_ROOT / "configs/scenes/garden.yaml"


class ConfigurationTests(unittest.TestCase):
    def _renderer(self) -> dict:
        scene, _ = load_scene_config(GARDEN)
        return dict(scene["renderer"])

    def _scene(self, paths: dict, *, version: int = 2) -> dict:
        scene = {
            "schema_version": version,
            "scene_id": "unit-scene",
            "paths": paths,
        }
        key = "renderer" if version == 2 else "rainygs_arguments"
        scene[key] = self._renderer()
        return scene

    def _write_bundle(self, directory: Path) -> None:
        directory.mkdir(parents=True)
        grid = GridSpec((4, 3), (-2.0, -1.5), (2.0, 1.5))
        metadata = base_metadata(
            scene_id="unit-scene",
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
        (directory / "metadata.yaml").write_text(
            yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
        )

    def test_repository_garden_is_v2_without_legacy_runtime_fields(self) -> None:
        scene, migration = load_scene_config(GARDEN)
        self.assertEqual(scene["schema_version"], 2)
        self.assertFalse(migration["converted"])
        self.assertNotIn("fluid_frames", scene["paths"])
        self.assertNotIn("diffuse_raindrop", scene["renderer"])
        self.assertEqual(
            scene["renderer"]["height_map_range"], [-6.0, 6.0, -6.0, 6.0]
        )
        self.assertEqual(
            scene["renderer"]["align_rotation_angles"],
            [241.209, 0.298943, 70.0],
        )
        self.assertEqual(scene["renderer"]["diffuse_fluid"], [59, 158, 215])
        renderer_bytes = yaml.safe_dump(
            scene["renderer"], sort_keys=True
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(renderer_bytes).hexdigest(),
            "504c70a3d537ac1242b7f45db0fa06b571c60be61eb5fb26fe309983296d9251",
        )

    def test_v1_conversion_lists_and_removes_historical_fields(self) -> None:
        legacy = self._scene(
            {
                "dataset": {"root": "datasets", "path": "scene"},
                "artifacts": {"root": "artifacts", "path": "runs"},
                "fluid_frames": {"root": "derived", "path": "simrain"},
            },
            version=1,
        )
        legacy["rainygs_arguments"]["diffuse_raindrop"] = [255, 255, 255]
        normalized, migration = normalize_scene_config(legacy)

        self.assertEqual(normalized["schema_version"], 2)
        self.assertEqual(normalized["renderer"]["diffuse_fluid"], [59, 158, 215])
        self.assertNotIn("diffuse_raindrop", normalized["renderer"])
        self.assertNotIn("fluid_frames", normalized["paths"])
        self.assertIn(
            "rainygs_arguments.diffuse_raindrop", migration["removed_fields"]
        )
        self.assertEqual(migration["removed_paths"], ["paths.fluid_frames"])

    def test_v2_rejects_mixed_removed_and_legacy_backend_fields(self) -> None:
        scene = self._scene({}, version=2)
        scene["rainygs_arguments"] = dict(scene["renderer"])
        with self.assertRaisesRegex(ConfigError, "both"):
            normalize_scene_config(scene)

        scene.pop("rainygs_arguments")
        scene["renderer"]["fluid_root"] = "/legacy"
        with self.assertRaisesRegex(ConfigError, "removed field"):
            normalize_scene_config(scene)

        scene["renderer"].pop("fluid_root")
        scene["renderer"]["water_rendering_backend"] = "rainygs_legacy"
        with self.assertRaisesRegex(ConfigError, "pre-refactor-reconstruction-v1"):
            normalize_scene_config(scene)

        scene["scene_id"] = ""
        scene["renderer"].pop("water_rendering_backend")
        with self.assertRaisesRegex(ConfigError, "non-empty scene_id"):
            normalize_scene_config(scene)

    def test_current_water_backend_defaults_to_p4_and_rejects_legacy(self) -> None:
        self.assertEqual(
            resolve_water_rendering_backend({}, source="unit profile"),
            "sandygs_p4",
        )
        self.assertEqual(
            resolve_water_rendering_backend(
                {"water_rendering_backend": "sandygs_p4"},
                source="unit profile",
            ),
            "sandygs_p4",
        )
        with self.assertRaisesRegex(
            ConfigError, "pre-refactor-reconstruction-v1"
        ):
            resolve_water_rendering_backend(
                {"water_rendering_backend": "rainygs_legacy"},
                source="unit profile",
            )
        with self.assertRaisesRegex(ConfigError, "unsupported"):
            resolve_water_rendering_backend(
                {"water_rendering_backend": "unknown"},
                source="unit profile",
            )

    def test_current_runtime_rejects_retired_backends_and_fields(self) -> None:
        self.assertEqual(
            validate_current_runtime_config({}, source="unit profile"),
            ("sand_water", "sandygs_p4"),
        )
        with self.assertRaisesRegex(ConfigError, "removed simulation backend"):
            validate_current_runtime_config(
                {"simulation_backend": "simrain"}, source="unit profile"
            )
        with self.assertRaisesRegex(ConfigError, "removed legacy water"):
            validate_current_runtime_config(
                {"fluid_root": "/legacy"}, source="unit profile"
            )

    def test_production_loader_rejects_v1_scene(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.yaml"
            path.write_text(
                yaml.safe_dump(self._scene({}, version=1), sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "retired scene schema v1"):
                load_scene_config(path)

    def test_operation_validation_does_not_require_unconsumed_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange = root / "derived/exchange"
            exchange.mkdir(parents=True)
            paths_file = root / "paths.yaml"
            paths_file.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "roots": {
                            "datasets": str(root / "datasets"),
                            "checkpoints": str(root / "checkpoints"),
                            "derived": str(root / "derived"),
                            "artifacts": str(root / "artifacts"),
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            scene_file = root / "scene.yaml"
            scene_file.write_text(
                yaml.safe_dump(
                    self._scene(
                        {
                            "dataset": {"root": "datasets", "path": "missing"},
                            "checkpoint": {
                                "root": "checkpoints",
                                "path": "missing",
                            },
                            "p1_exchange": {
                                "root": "derived",
                                "path": "exchange",
                            },
                            "artifacts": {
                                "root": "artifacts",
                                "path": "runs",
                                "required": False,
                            },
                        }
                    ),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            manifest, arguments, _ = build_scene_manifest(
                paths_file,
                scene_file,
                check_inputs=True,
                run_id="test",
                operation="simulation",
                repo_root=REPO_ROOT,
            )
            self.assertEqual(
                manifest["input_status"]["dataset"],
                "not required for simulation",
            )
            self.assertNotIn("fluid_root", arguments)
            with self.assertRaisesRegex(ConfigError, "required dataset"):
                build_scene_manifest(
                    paths_file,
                    scene_file,
                    check_inputs=True,
                    run_id="test",
                    operation="sand_water_render",
                    repo_root=REPO_ROOT,
                )

    def test_deep_resource_validation_matches_simulation_and_render_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exchange = root / "exchange"
            self._write_bundle(exchange)
            paths = {
                "p1_exchange": str(exchange),
                "dataset": str(root / "dataset"),
                "checkpoint": str(root / "checkpoint"),
            }
            simulation = validate_sand_water_resources(
                paths,
                self._renderer(),
                None,
                include_render=False,
                expected_scene_id="unit-scene",
            )
            self.assertNotIn("render", simulation)
            with self.assertRaisesRegex(ConfigError, "does not match"):
                validate_sand_water_resources(
                    paths,
                    self._renderer(),
                    None,
                    include_render=False,
                    expected_scene_id="another-scene",
                )

            dataset = Path(paths["dataset"])
            checkpoint = Path(paths["checkpoint"])
            for path in (
                dataset / "sparse/0/cameras.bin",
                dataset / "sparse/0/images.bin",
                dataset / "sparse/0/points3D.bin",
                checkpoint / "cfg_args",
                checkpoint / "point_cloud/iteration_7/point_cloud.ply",
                checkpoint
                / "brdf_mlp/iteration_7/overcast_soil_puresky_1k_3.png",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            rendered = validate_sand_water_resources(
                paths,
                self._renderer(),
                {},
                include_render=True,
                expected_scene_id="unit-scene",
            )
            self.assertEqual(rendered["render"]["checkpoint_iteration"], 7)

    def test_offline_migration_refuses_overwrite_and_manifest_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = self._scene(
                {
                    "artifacts": {
                        "root": "artifacts",
                        "path": "runs",
                        "required": False,
                    }
                },
                version=1,
            )
            legacy["rainygs_arguments"]["diffuse_moisture"] = [1, 2, 3]
            source = root / "legacy.yaml"
            output = root / "v2.yaml"
            source.write_text(
                yaml.safe_dump(legacy, sort_keys=False), encoding="utf-8"
            )
            migration = migrate_scene_file(source, output)
            self.assertIn(
                "rainygs_arguments.diffuse_moisture",
                migration["removed_fields"],
            )
            self.assertEqual(yaml.safe_load(output.read_text())["schema_version"], 2)
            with self.assertRaisesRegex(ConfigError, "overwrite"):
                migrate_scene_file(source, output)

            with self.assertRaisesRegex(ConfigError, "both"):
                renderer_arguments_from_manifest(
                    {
                        "schema_version": 2,
                        "renderer_arguments": {},
                        "rainygs_arguments": {},
                    }
                )
            with self.assertRaisesRegex(ConfigError, "removed field"):
                renderer_arguments_from_manifest(
                    {
                        "schema_version": 2,
                        "rainygs_arguments": self._renderer(),
                    }
                )
            with self.assertRaisesRegex(ConfigError, "removed renderer field"):
                renderer_arguments_from_manifest(
                    {
                        "schema_version": 2,
                        "renderer_arguments": {"fluid_root": "/legacy"},
                    }
                )
            with self.assertRaisesRegex(ConfigError, "schema_version 2"):
                renderer_arguments_from_manifest(
                    {
                        "schema_version": 1,
                        "rainygs_arguments": self._renderer(),
                    }
                )


if __name__ == "__main__":
    unittest.main()
