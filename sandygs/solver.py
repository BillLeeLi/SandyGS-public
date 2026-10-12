"""Resolve and audit the explicitly built in-tree sand-water solver."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from sandygs.configuration import ConfigError, VARIABLE_PATTERN, load_paths_config


IMPORTED_SOLVER_REVISION = "1ea4c352995788517317a9b522853e34fe3c2f1b"
BUILD_MANIFEST_NAME = "sandygs-build.json"
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def solver_source_files(source_root: Path) -> list[Path]:
    """Return files that can affect a compiled solver binary."""

    source_root = source_root.resolve()
    files = [source_root / "xmake.lua"]
    files.extend(
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SOURCE_SUFFIXES
        and "build" not in path.relative_to(source_root).parts
    )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise ConfigError(f"solver source is incomplete: {missing[0]}")
    return sorted(set(files), key=lambda path: path.relative_to(source_root).as_posix())


def solver_source_digest(source_root: Path) -> tuple[str, list[str]]:
    source_root = source_root.resolve()
    digest = hashlib.sha256()
    relative_files: list[str] = []
    for path in solver_source_files(source_root):
        relative = path.relative_to(source_root).as_posix()
        relative_files.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest(), relative_files


def write_solver_build_manifest(
    *,
    source_root: Path,
    build_directory: Path,
    mode: str,
    jobs: int,
    configure_command: list[str],
    build_commands: list[list[str]],
    compiler: str,
    compiler_version: str,
    xmake_version: str,
    tests_command: list[str],
    tests_executed: int,
) -> Path:
    """Record a successful explicit build and test run atomically."""

    source_root = source_root.resolve()
    build_directory = build_directory.resolve()
    source_sha256, source_files = solver_source_digest(source_root)
    binaries: dict[str, Any] = {}
    for name in ("demo", "tests", "benchmark"):
        path = build_directory / name
        if path.is_file():
            binaries[name] = {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    if "demo" not in binaries or "tests" not in binaries:
        raise ConfigError(
            f"solver build is missing demo or tests binary in {build_directory}"
        )
    manifest = {
        "schema_version": 1,
        "source": {
            "import_revision": IMPORTED_SOLVER_REVISION,
            "sha256": source_sha256,
            "file_count": len(source_files),
            "files": source_files,
        },
        "build": {
            "mode": mode,
            "jobs": jobs,
            "configure_command": configure_command,
            "build_commands": build_commands,
            "compiler": compiler,
            "compiler_version": compiler_version,
            "xmake_version": xmake_version,
            "options": {"language": "cxx20"},
        },
        "tests": {
            "status": "passed",
            "command": tests_command,
            "executed": tests_executed,
            "failures": 0,
        },
        "binaries": binaries,
    }
    build_directory.mkdir(parents=True, exist_ok=True)
    destination = build_directory / BUILD_MANIFEST_NAME
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=build_directory, delete=False
    ) as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(destination)
    return destination


def _load_build_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(
            f"solver build manifest is missing: {path}; run "
            "python scripts/build_sand_water_solver.py"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid solver build manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ConfigError(f"unsupported solver build manifest: {path}")
    return value


def _expand_executable(value: Any, paths_config: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("executables.sand_water_solver must be a non-empty path")
    missing = sorted(
        {
            match.group(1) or match.group(2)
            for match in VARIABLE_PATTERN.finditer(value)
            if (match.group(1) or match.group(2)) not in os.environ
        }
    )
    if missing:
        raise ConfigError(
            "executables.sand_water_solver uses unset environment variable(s): "
            + ", ".join(missing)
        )
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = paths_config.parent / path
    return path.resolve()


@dataclass(frozen=True)
class SolverExecutable:
    path: Path
    selection: str
    binary_sha256: str
    manifest_path: Path
    manifest: dict[str, Any]

    def audit(self) -> dict[str, Any]:
        source = self.manifest["source"]
        build = self.manifest["build"]
        return {
            "selection": self.selection,
            "path": str(self.path),
            "binary_sha256": self.binary_sha256,
            "build_manifest": str(self.manifest_path),
            "source_import_revision": source["import_revision"],
            "source_sha256": source["sha256"],
            "source_file_count": source["file_count"],
            "build_mode": build["mode"],
            "build_options": build.get("options", {}),
            "compiler": build["compiler"],
            "compiler_version": build["compiler_version"],
            "xmake_version": build["xmake_version"],
            "tests": self.manifest["tests"],
        }


def resolve_sand_water_solver(
    *,
    cli_path: Path | None,
    paths_config: Path,
    repo_root: Path,
) -> SolverExecutable:
    """Apply CLI → local config → in-tree default priority and verify identity."""

    paths_config = paths_config.expanduser().resolve()
    repo_root = repo_root.resolve()
    local = load_paths_config(paths_config)
    executables = local.get("executables", {})
    if not isinstance(executables, dict):
        raise ConfigError(f"{paths_config} executables must be a mapping")
    configured = executables.get("sand_water_solver")
    if cli_path is not None:
        selected = cli_path.expanduser().resolve()
        selection = "command_line"
    elif configured is not None:
        selected = _expand_executable(configured, paths_config)
        selection = "local_config"
    else:
        selected = (
            repo_root
            / "solvers/sand_water/build/linux/x86_64/release/demo"
        ).resolve()
        selection = "in_tree_default"

    if not selected.is_file() or not os.access(selected, os.X_OK):
        raise ConfigError(
            f"sand-water solver executable is missing or not executable: {selected}; "
            "run python scripts/build_sand_water_solver.py"
        )
    manifest_path = selected.parent / BUILD_MANIFEST_NAME
    manifest = _load_build_manifest(manifest_path)
    source = manifest.get("source")
    build = manifest.get("build")
    tests = manifest.get("tests")
    binaries = manifest.get("binaries")
    if not all(isinstance(value, dict) for value in (source, build, tests, binaries)):
        raise ConfigError(f"incomplete solver build manifest: {manifest_path}")
    if source.get("import_revision") != IMPORTED_SOLVER_REVISION:
        raise ConfigError(
            f"solver build source revision does not match imported snapshot; rebuild "
            f"with python scripts/build_sand_water_solver.py"
        )
    current_source_sha256, current_files = solver_source_digest(
        repo_root / "solvers/sand_water"
    )
    if (
        source.get("sha256") != current_source_sha256
        or source.get("files") != current_files
    ):
        raise ConfigError(
            "solver sources changed after the selected binary was built; run "
            "python scripts/build_sand_water_solver.py"
        )
    executed = tests.get("executed")
    if (
        tests.get("status") != "passed"
        or not isinstance(executed, int)
        or isinstance(executed, bool)
        or executed <= 0
        or tests.get("failures") != 0
    ):
        raise ConfigError(f"solver build has no passing test record: {manifest_path}")
    binary = binaries.get(selected.name)
    if not isinstance(binary, dict):
        raise ConfigError(
            f"solver build manifest has no binary record for {selected.name}: "
            f"{manifest_path}"
        )
    actual_sha256 = sha256_file(selected)
    if binary.get("sha256") != actual_sha256:
        raise ConfigError(
            f"solver binary hash does not match {manifest_path}; rebuild explicitly"
        )
    return SolverExecutable(
        path=selected,
        selection=selection,
        binary_sha256=actual_sha256,
        manifest_path=manifest_path,
        manifest=manifest,
    )
