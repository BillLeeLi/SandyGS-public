from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from sandygs.configuration import ConfigError
from sandygs.solver import (
    IMPORTED_SOLVER_REVISION,
    resolve_sand_water_solver,
    solver_source_digest,
    write_solver_build_manifest,
)


class SolverResolutionTests(unittest.TestCase):
    def _fixture(self, root: Path, build_root: Path | None = None) -> tuple[Path, Path]:
        source = root / "solvers/sand_water"
        source.mkdir(parents=True)
        (source / "xmake.lua").write_text('set_project("fixture")\n', encoding="utf-8")
        (source / "main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
        build = build_root or source / "build/linux/x86_64/release"
        build.mkdir(parents=True)
        for name in ("demo", "tests"):
            binary = build / name
            binary.write_bytes(f"fixture-{name}".encode("ascii"))
            binary.chmod(0o755)
        write_solver_build_manifest(
            source_root=source,
            build_directory=build,
            mode="release",
            jobs=2,
            configure_command=["xmake", "f", "-m", "release"],
            build_commands=[
                ["xmake", "build", "-j", "2", "demo"],
                ["xmake", "build", "-j", "2", "tests"],
            ],
            compiler="c++",
            compiler_version="fixture compiler",
            xmake_version="fixture xmake",
            tests_command=[str(build / "tests")],
            tests_executed=38,
        )
        paths = root / "paths.yaml"
        paths.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "roots": {"artifacts": str(root / "artifacts")},
                }
            ),
            encoding="utf-8",
        )
        return paths, build / "demo"

    def test_source_digest_tracks_compilation_inputs_not_docs_or_builds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, _ = self._fixture(root)
            source = root / "solvers/sand_water"
            initial, files = solver_source_digest(source)
            self.assertEqual(files, ["main.cpp", "xmake.lua"])
            (source / "notes.md").write_text("documentation\n", encoding="utf-8")
            (source / "build/cache.txt").write_text("derived\n", encoding="utf-8")
            self.assertEqual(solver_source_digest(source)[0], initial)
            (source / "main.cpp").write_text("int main() { return 1; }\n", encoding="utf-8")
            self.assertNotEqual(solver_source_digest(source)[0], initial)
            self.assertTrue(paths.is_file())

    def test_default_resolution_verifies_manifest_source_and_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, binary = self._fixture(root)
            resolved = resolve_sand_water_solver(
                cli_path=None, paths_config=paths, repo_root=root
            )
            self.assertEqual(resolved.path, binary)
            self.assertEqual(resolved.selection, "in_tree_default")
            self.assertEqual(
                resolved.audit()["source_import_revision"], IMPORTED_SOLVER_REVISION
            )

            binary.write_bytes(b"changed")
            binary.chmod(0o755)
            with self.assertRaisesRegex(ConfigError, "binary hash"):
                resolve_sand_water_solver(
                    cli_path=None, paths_config=paths, repo_root=root
                )

    def test_changed_source_and_missing_manifest_require_explicit_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, binary = self._fixture(root)
            (root / "solvers/sand_water/main.cpp").write_text(
                "int main() { return 2; }\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ConfigError, "sources changed"):
                resolve_sand_water_solver(
                    cli_path=None, paths_config=paths, repo_root=root
                )

            manifest = binary.parent / "sandygs-build.json"
            manifest.unlink()
            with self.assertRaisesRegex(ConfigError, "build manifest is missing"):
                resolve_sand_water_solver(
                    cli_path=binary, paths_config=paths, repo_root=root
                )

    def test_priority_is_cli_then_local_config_then_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local_build = root / "local-build"
            paths, default_binary = self._fixture(root, build_root=local_build)
            value = yaml.safe_load(paths.read_text(encoding="utf-8"))
            value["executables"] = {"sand_water_solver": str(default_binary)}
            paths.write_text(yaml.safe_dump(value), encoding="utf-8")
            local = resolve_sand_water_solver(
                cli_path=None, paths_config=paths, repo_root=root
            )
            explicit = resolve_sand_water_solver(
                cli_path=default_binary, paths_config=paths, repo_root=root
            )
            self.assertEqual(local.selection, "local_config")
            self.assertEqual(explicit.selection, "command_line")

            manifest_path = default_binary.parent / "sandygs-build.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source"]["import_revision"] = "0" * 40
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "source revision"):
                resolve_sand_water_solver(
                    cli_path=default_binary, paths_config=paths, repo_root=root
                )


if __name__ == "__main__":
    unittest.main()
