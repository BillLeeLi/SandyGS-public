#!/usr/bin/env python3
"""Explicitly configure, build, test and record the in-tree CPU solver."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.solver import write_solver_build_manifest  # noqa: E402


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def capture(command: list[str], *, cwd: Path, environment: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ANSI_ESCAPE.sub("", completed.stdout).strip()


def first_nonempty_line(value: str) -> str:
    return next((line.strip() for line in value.splitlines() if line.strip()), "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("release", "debug"), default="release")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")

    source = (REPO_ROOT / "solvers/sand_water").resolve()
    if not (source / "xmake.lua").is_file():
        parser.error(f"solver source has no xmake.lua: {source}")
    environment = os.environ.copy()
    # CUDA activation exports GCC 11 for extension builds. The CPU solver's
    # audited reference compiler is the ordinary system C++20 toolchain.
    environment.pop("CC", None)
    environment.pop("CXX", None)
    # First-time package installation needs explicit confirmation in CI.
    configure_command = ["xmake", "f", "-y", "-m", args.mode]
    build_commands = [
        ["xmake", "build", "-j", str(args.jobs), target]
        for target in ("demo", "tests")
    ]
    print("+", " ".join(configure_command), flush=True)
    subprocess.run(configure_command, cwd=source, env=environment, check=True)
    for build_command in build_commands:
        print("+", " ".join(build_command), flush=True)
        subprocess.run(build_command, cwd=source, env=environment, check=True)

    build_directory = (
        source
        / "build"
        / platform.system().lower()
        / platform.machine()
        / args.mode
    )
    tests = build_directory / "tests"
    tests_command = [str(tests)]
    if not tests.is_file():
        raise FileNotFoundError(f"solver test binary was not built: {tests}")
    print("+", str(tests), flush=True)
    test_output = capture(tests_command, cwd=source, environment=environment)
    print(test_output, flush=True)
    summary = re.search(
        r"(?m)^(\d+) tests executed, (\d+) failures$", test_output
    )
    if summary is None or int(summary.group(2)) != 0:
        raise RuntimeError("could not verify the C++ test summary")
    tests_executed = int(summary.group(1))

    compiler = environment.get("CXX", "c++")
    manifest = write_solver_build_manifest(
        source_root=source,
        build_directory=build_directory,
        mode=args.mode,
        jobs=args.jobs,
        configure_command=configure_command,
        build_commands=build_commands,
        compiler=compiler,
        compiler_version=first_nonempty_line(
            capture([compiler, "--version"], cwd=source, environment=environment)
        ),
        xmake_version=first_nonempty_line(
            capture(["xmake", "--version"], cwd=source, environment=environment)
        ),
        tests_command=tests_command,
        tests_executed=tests_executed,
    )
    print(f"solver build and tests passed; manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
