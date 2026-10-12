"""Assemble and verify the immutable pre-refactor reconstruction baseline."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any
import warnings

import numpy as np
import yaml

from sandygs.render_baseline import (
    RenderBaselineError,
    _schedule_index,
    _solver_samples,
    freeze_render_baseline,
    verify_render_baseline,
)


class ReconstructionFreezeError(RuntimeError):
    """Raised when a reconstruction freeze input or payload is invalid."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except OSError as exc:
        raise ReconstructionFreezeError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReconstructionFreezeError(f"{path} must contain a YAML mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: Any, *, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ReconstructionFreezeError(f"{label} must be a safe relative path: {value}")
    return path


def _run(arguments: list[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        output = getattr(exc, "stdout", None) or getattr(exc, "output", None) or ""
        raise ReconstructionFreezeError(
            f"command failed: {' '.join(arguments)}\n{output.strip()}"
        ) from exc
    return completed.stdout.strip()


def _git_revision(repo: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=repo)


def _git_status(repo: Path) -> list[str]:
    output = _run(["git", "status", "--short"], cwd=repo)
    return output.splitlines() if output else []


def _copy_file(
    source: Path,
    destination: Path,
    *,
    root: Path,
    entries: list[dict[str, Any]],
) -> None:
    if not source.is_file():
        raise ReconstructionFreezeError(f"required freeze input is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_digest = _sha256(source)
    destination_digest = _sha256(destination)
    if source_digest != destination_digest:
        raise ReconstructionFreezeError(f"copy verification failed for {source}")
    entries.append(
        {
            "path": destination.relative_to(root).as_posix(),
            "size_bytes": destination.stat().st_size,
            "sha256": destination_digest,
        }
    )


def _copy_declared_path(
    source: Path,
    destination: Path,
    *,
    root: Path,
    entries: list[dict[str, Any]],
) -> None:
    if source.is_file():
        _copy_file(source, destination, root=root, entries=entries)
        return
    if not source.is_dir():
        raise ReconstructionFreezeError(f"required freeze input is missing: {source}")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise ReconstructionFreezeError(f"freeze input directory is empty: {source}")
    for path in files:
        _copy_file(
            path,
            destination / path.relative_to(source),
            root=root,
            entries=entries,
        )


def _resource_group_files(
    group: dict[str, Any], roots: dict[str, Path]
) -> tuple[str, str, Path, list[Path]]:
    identifier = str(group.get("id", ""))
    root_name = str(group.get("root", ""))
    if not identifier or root_name not in roots:
        raise ReconstructionFreezeError(f"invalid resource group {identifier!r}")
    base_relative = _safe_relative(group.get("base"), label=f"resource {identifier} base")
    base = (roots[root_name] / base_relative).resolve()
    root = roots[root_name].resolve()
    try:
        base.relative_to(root)
    except ValueError as exc:
        raise ReconstructionFreezeError(
            f"resource {identifier} escapes root {root_name}"
        ) from exc

    if group.get("tree") is True:
        if not base.is_dir():
            raise ReconstructionFreezeError(f"resource tree is missing: {base}")
        files = sorted(path for path in base.rglob("*") if path.is_file())
    else:
        declared = group.get("paths")
        if not isinstance(declared, list) or not declared:
            raise ReconstructionFreezeError(
                f"resource {identifier} needs non-empty paths or tree=true"
            )
        files = []
        for value in declared:
            relative = _safe_relative(value, label=f"resource {identifier} path")
            path = (base / relative).resolve()
            try:
                path.relative_to(base)
            except ValueError as exc:
                raise ReconstructionFreezeError(
                    f"resource {identifier} path escapes its base"
                ) from exc
            if not path.is_file():
                raise ReconstructionFreezeError(f"resource file is missing: {path}")
            files.append(path)
    if not files:
        raise ReconstructionFreezeError(f"resource {identifier} contains no files")
    return identifier, root_name, base_relative, files


def resource_group_manifest(group: dict[str, Any], roots: dict[str, Path]) -> dict[str, Any]:
    """Hash one declared external resource group without copying its contents."""

    identifier, root_name, base_relative, files = _resource_group_files(group, roots)
    base = (roots[root_name] / base_relative).resolve()

    members = []
    aggregate = hashlib.sha256()
    for path in files:
        relative = path.relative_to(base).as_posix()
        digest = _sha256(path)
        size = path.stat().st_size
        aggregate.update(f"{digest}  {size}  {relative}\n".encode("utf-8"))
        members.append({"path": relative, "size_bytes": size, "sha256": digest})
    return {
        "id": identifier,
        "logical_root": root_name,
        "logical_base": base_relative.as_posix(),
        "file_count": len(members),
        "total_size_bytes": sum(item["size_bytes"] for item in members),
        "sha256": aggregate.hexdigest(),
        "members": members,
    }


def _preflight_render_spec(
    spec_path: Path, artifacts_root: Path, repo_root: Path
) -> int:
    spec = _load_yaml(spec_path)
    source = spec.get("source")
    products = spec.get("products")
    cases = spec.get("cases")
    if not isinstance(source, dict) or not isinstance(products, dict) or not products:
        raise ReconstructionFreezeError(f"invalid render baseline spec: {spec_path}")
    if not isinstance(cases, list) or not cases:
        raise ReconstructionFreezeError(f"render baseline spec has no cases: {spec_path}")
    source_root = artifacts_root / _safe_relative(
        source.get("artifact_relative"), label="source artifact"
    )
    solver_source_root = artifacts_root / _safe_relative(
        source.get("solver_artifact_relative", source.get("artifact_relative")),
        label="solver source artifact",
    )
    release_root = source_root / _safe_relative(
        source.get("release_relative"), label="release root"
    )
    schedule = _schedule_index(
        source_root
        / _safe_relative(source.get("schedule_relative"), label="render schedule")
    )
    requested_times = []
    total = 0
    for case in cases:
        if not isinstance(case, dict):
            raise ReconstructionFreezeError(f"invalid case in {spec_path}")
        output_group = _safe_relative(case["output_group"], label="output group")
        output_name = str(case["output_name"])
        key = (output_group.as_posix(), output_name)
        if key not in schedule:
            raise ReconstructionFreezeError(f"case {case['id']} is absent from {spec_path}")
        frame = schedule[key]
        expected_time = float(case["time_seconds"])
        if int(frame.get("camera_index")) != int(case["camera_index"]):
            raise ReconstructionFreezeError(
                f"case {case['id']} camera does not match its render schedule"
            )
        if abs(float(frame.get("time_seconds")) - expected_time) > 1e-12:
            raise ReconstructionFreezeError(
                f"case {case['id']} time does not match its render schedule"
            )
        if frame.get("save_debug") is not True:
            raise ReconstructionFreezeError(
                f"case {case['id']} has no raw debug capture"
            )
        requested_times.append(expected_time)
        for template in products.values():
            path = release_root / output_group / str(template).format(output_name=output_name)
            if not path.is_file():
                raise ReconstructionFreezeError(f"render payload is missing: {path}")
            total += path.stat().st_size

    run_root = solver_source_root / _safe_relative(
        source.get("solver_run_relative"), label="solver run"
    )
    _, steps = _solver_samples(run_root, requested_times)
    fields = spec.get("solver_fields")
    if not isinstance(fields, list) or not fields:
        raise ReconstructionFreezeError(f"render baseline spec has no solver fields: {spec_path}")
    for step in steps:
        for field in fields:
            path = run_root / "frames" / f"{step:06d}" / f"{field}.npy"
            if not path.is_file():
                raise ReconstructionFreezeError(f"solver payload is missing: {path}")
            total += path.stat().st_size
    for value in spec.get("provenance_files", []):
        path = source_root / _safe_relative(value, label="provenance file")
        if not path.is_file():
            raise ReconstructionFreezeError(f"provenance file is missing: {path}")
        total += path.stat().st_size
    for value in spec.get("solver_provenance_files", []):
        path = solver_source_root / _safe_relative(value, label="solver provenance file")
        if not path.is_file():
            raise ReconstructionFreezeError(f"solver provenance file is missing: {path}")
        total += path.stat().st_size
    for value in spec.get("reference_implementation_files", []):
        path = repo_root / _safe_relative(value, label="reference implementation file")
        if not path.is_file():
            raise ReconstructionFreezeError(
                f"reference implementation file is missing: {path}"
            )
        total += path.stat().st_size
    return total


def _capture_environment(
    *, repo_root: Path, solver_repo: Path, solver_binary: Path
) -> dict[str, Any]:
    try:
        import torch

        torch_record = {
            # ``torch.__version__`` is a ``TorchVersion`` subclass in recent
            # releases.  Convert version-like values to plain scalars before
            # passing the environment record to PyYAML.
            "version": str(torch.__version__),
            "cuda_runtime": (
                str(torch.version.cuda) if torch.version.cuda is not None else None
            ),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_names": (
                [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
                if torch.cuda.is_available()
                else []
            ),
            "compute_capabilities": (
                [list(torch.cuda.get_device_capability(index)) for index in range(torch.cuda.device_count())]
                if torch.cuda.is_available()
                else []
            ),
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        torch_record = {"error": str(exc)}

    def optional(arguments: list[str], cwd: Path | None = None) -> str | None:
        try:
            return _run(arguments, cwd=cwd)
        except ReconstructionFreezeError:
            return None

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch_record,
        "nvidia_smi": optional(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "nvcc": optional(["/usr/local/cuda/bin/nvcc", "--version"]),
        "cuda_host_compiler": optional(["/usr/bin/g++-11", "--version"]),
        "solver_compiler": optional(["/usr/bin/g++", "--version"]),
        "xmake": optional(["xmake", "--version"]),
        "solver_build": {
            "mode": "release",
            "language": "cxx20",
            "parallel_runtime": "TBB automatic concurrency",
            "logical_cpu_count": os.cpu_count(),
            "TBB_NUM_THREADS": os.environ.get("TBB_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "binary_sha256": _sha256(solver_binary),
            "binary_file": optional(["file", str(solver_binary)]),
            "dynamic_libraries": optional(["ldd", str(solver_binary)]),
        },
        "repository": {
            "revision": _git_revision(repo_root),
            "status": _git_status(repo_root),
        },
        "solver_repository": {
            "revision": _git_revision(solver_repo),
            "status": _git_status(solver_repo),
            "remotes": optional(["git", "remote", "-v"], cwd=solver_repo) or "",
            "submodules": optional(["git", "submodule", "status"], cwd=solver_repo) or "",
        },
    }


def _camera_contract_arrays(config: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    """Reconstruct the exact generated-orbit extrinsic/intrinsic contract on CPU."""

    import torch

    from scene.colmap_loader import (
        qvec2rotmat,
        read_extrinsics_binary,
        read_intrinsics_binary,
    )
    from utils.camera_utils import (
        generate_camera_poses_on_hemisphere,
        pad_poses,
        transform_poses_pca,
    )
    from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2

    sparse = dataset_root / "sparse" / "0"
    extrinsics = read_extrinsics_binary(str(sparse / "images.bin"))
    intrinsics = read_intrinsics_binary(str(sparse / "cameras.bin"))
    records = []
    for value in extrinsics.values():
        intrinsic = intrinsics[value.camera_id]
        if intrinsic.model == "SIMPLE_PINHOLE":
            focal_x = focal_y = float(intrinsic.params[0])
        elif intrinsic.model == "PINHOLE":
            focal_x, focal_y = map(float, intrinsic.params[:2])
        else:
            raise ReconstructionFreezeError(
                f"unsupported camera model in contract: {intrinsic.model}"
            )
        records.append(
            {
                "name": Path(value.name).stem,
                "R": np.transpose(qvec2rotmat(value.qvec)),
                "T": np.asarray(value.tvec),
                "fov_x": focal2fov(focal_x, intrinsic.width),
                "fov_y": focal2fov(focal_y, intrinsic.height),
            }
        )
    records.sort(key=lambda item: item["name"])
    if bool(config.get("eval")):
        hold = int(config.get("llffhold", 8))
        records = [value for index, value in enumerate(records) if index % hold != 0]
    if not records:
        raise ReconstructionFreezeError("camera contract has no training cameras")

    training_world_view = np.stack(
        [getWorld2View2(value["R"], value["T"]).T for value in records]
    )
    c2ws = np.stack([np.linalg.inv(value.T) for value in training_world_view])
    pose = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
    _, colmap_to_world = transform_poses_pca(pose)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Using torch.cross without specifying the dim arg is deprecated.*",
            category=UserWarning,
        )
        generated = generate_camera_poses_on_hemisphere(
            str(config["dataset_name"]),
            int(config["camera_path_frames"]),
        ).cpu().numpy()
    generated = np.linalg.inv(colmap_to_world) @ pad_poses(generated)
    axis_flip = np.diag([1, -1, -1, 1])
    world_view = np.stack(
        [np.linalg.inv(c2w @ axis_flip).T.astype(np.float32) for c2w in generated]
    )
    camera_center = np.stack([np.linalg.inv(value)[3, :3] for value in world_view]).astype(
        np.float32
    )
    width, height = map(int, config["resolution"])
    first = records[0]
    projection = (
        getProjectionMatrix(
            znear=0.01,
            zfar=100.0,
            fovX=float(first["fov_x"]),
            fovY=float(first["fov_y"]),
        )
        .transpose(0, 1)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    fx = width / (2.0 * np.tan(float(first["fov_x"]) * 0.5))
    fy = height / (2.0 * np.tan(float(first["fov_y"]) * 0.5))
    intrinsic = np.asarray(
        [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return {
        "world_view_transform": world_view,
        "camera_center": camera_center,
        "projection_matrix": projection,
        "intrinsic_matrix": intrinsic,
        "resolution": np.asarray([height, width], dtype=np.int32),
        "training_camera_names": [value["name"] for value in records],
    }


def _write_camera_contract(
    config: dict[str, Any],
    roots: dict[str, Path],
    destination: Path,
    *,
    root: Path,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_root = roots[str(config["dataset_root"])] / _safe_relative(
        config["dataset_path"], label="camera dataset path"
    )
    arrays = _camera_contract_arrays(config, dataset_root)
    destination.mkdir(parents=True, exist_ok=True)
    array_records = []
    for name in (
        "world_view_transform",
        "camera_center",
        "projection_matrix",
        "intrinsic_matrix",
        "resolution",
    ):
        path = destination / f"{name}.npy"
        np.save(path, arrays[name], allow_pickle=False)
        digest = _sha256(path)
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
        array_records.append(
            {
                "name": name,
                "path": path.name,
                "dtype": str(arrays[name].dtype),
                "shape": list(arrays[name].shape),
                "sha256": digest,
            }
        )
    selected = [int(value) for value in config["selected_indices"]]
    if any(value < 0 or value >= arrays["world_view_transform"].shape[0] for value in selected):
        raise ReconstructionFreezeError("camera contract selected index is out of range")
    manifest = {
        "schema_version": 1,
        "camera_contract_id": str(config["id"]),
        "generation": {
            "dataset_name": str(config["dataset_name"]),
            "camera_path": "generated_orbit",
            "camera_path_frames": int(config["camera_path_frames"]),
            "eval": bool(config["eval"]),
            "llffhold": int(config["llffhold"]),
            "training_camera_names": arrays["training_camera_names"],
        },
        "selected_indices": selected,
        "arrays": array_records,
    }
    manifest_path = destination / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)
    entries.append(
        {
            "path": manifest_path.relative_to(root).as_posix(),
            "size_bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        }
    )
    return manifest


def preflight_reconstruction_freeze(
    *,
    spec_path: Path,
    paths_file: Path,
    repo_root: Path,
    solver_repo: Path,
    solver_binary: Path,
    require_clean: bool,
) -> dict[str, Any]:
    spec = _load_yaml(spec_path)
    if spec.get("schema_version") != 1:
        raise ReconstructionFreezeError("freeze spec schema_version must be 1")
    paths = _load_yaml(paths_file)
    try:
        roots = {
            name: Path(value).expanduser().resolve()
            for name, value in paths["roots"].items()
        }
    except (KeyError, AttributeError) as exc:
        raise ReconstructionFreezeError("paths file must define roots") from exc
    artifacts_root = roots["artifacts"]
    output_relative = _safe_relative(
        spec.get("output_artifact_relative"), label="output artifact"
    )
    output_root = artifacts_root / output_relative
    if output_root.exists():
        raise ReconstructionFreezeError(f"frozen output already exists: {output_root}")

    behavior_revision = str(spec.get("behavior_revision", ""))
    solver_revision = str(spec.get("solver_revision", ""))
    if len(behavior_revision) != 40 or len(solver_revision) != 40:
        raise ReconstructionFreezeError("behavior and solver revisions must be full commits")
    current_revision = _git_revision(repo_root)
    _run(["git", "merge-base", "--is-ancestor", behavior_revision, current_revision], cwd=repo_root)
    repo_status = _git_status(repo_root)
    solver_status = _git_status(solver_repo)
    if _git_revision(solver_repo) != solver_revision:
        raise ReconstructionFreezeError("solver repository revision does not match freeze spec")
    if solver_status:
        raise ReconstructionFreezeError("solver repository must be clean before freezing")
    if require_clean and repo_status:
        raise ReconstructionFreezeError("SandyGS repository must be clean before freezing")
    if not solver_binary.is_file():
        raise ReconstructionFreezeError(f"solver binary is missing: {solver_binary}")

    render_total = 0
    render_specs = spec.get("render_baseline_specs")
    if not isinstance(render_specs, list) or not render_specs:
        raise ReconstructionFreezeError("freeze spec needs render_baseline_specs")
    for value in render_specs:
        baseline_spec = repo_root / _safe_relative(value, label="render baseline spec")
        child = _load_yaml(baseline_spec)
        declared = _safe_relative(child["output"]["artifact_relative"], label="child output")
        if declared.parent != output_relative:
            raise ReconstructionFreezeError(
                f"render baseline output must be directly below {output_relative}: {baseline_spec}"
            )
        render_total += _preflight_render_spec(
            baseline_spec, artifacts_root, repo_root
        )

    reports = spec.get("determinism_reports")
    if not isinstance(reports, dict) or not reports:
        raise ReconstructionFreezeError("freeze spec needs determinism reports")
    for identifier, value in reports.items():
        path = artifacts_root / _safe_relative(value, label=f"report {identifier}")
        report = _load_yaml(path)
        if report.get("all_pairs_exact") is not True:
            raise ReconstructionFreezeError(f"determinism report is not exact: {path}")

    for identifier, value in spec.get("fig4_runs", {}).items():
        source = artifacts_root / _safe_relative(value, label=f"Fig.4 run {identifier}")
        for relative_value in spec.get("fig4_payload_paths", []):
            relative = _safe_relative(relative_value, label="Fig.4 payload")
            if not (source / relative).exists():
                raise ReconstructionFreezeError(f"Fig.4 payload is missing: {source / relative}")

    for relative_value in spec.get("preserved_render_baselines", []):
        verify_render_baseline(
            artifacts_root / _safe_relative(relative_value, label="preserved baseline")
        )

    resource_groups = spec.get("resource_groups")
    if not isinstance(resource_groups, list) or not resource_groups:
        raise ReconstructionFreezeError("freeze spec needs resource_groups")
    for group in resource_groups:
        if not isinstance(group, dict):
            raise ReconstructionFreezeError("each resource group must be a mapping")
        _resource_group_files(group, roots)

    camera = spec.get("camera_contract")
    if not isinstance(camera, dict):
        raise ReconstructionFreezeError("freeze spec needs camera_contract")
    dataset = roots[str(camera["dataset_root"])] / _safe_relative(
        camera["dataset_path"], label="camera dataset path"
    )
    for name in ("cameras.bin", "images.bin"):
        if not (dataset / "sparse" / "0" / name).is_file():
            raise ReconstructionFreezeError(f"camera input is missing: {name}")
    arrays = _camera_contract_arrays(camera, dataset)
    selected = [int(value) for value in camera.get("selected_indices", [])]
    if not selected or any(
        value < 0 or value >= arrays["world_view_transform"].shape[0]
        for value in selected
    ):
        raise ReconstructionFreezeError("camera contract selected index is out of range")

    return {
        "spec": spec,
        "roots": roots,
        "artifacts_root": artifacts_root,
        "output_root": output_root,
        "repository_revision": current_revision,
        "repository_status": repo_status,
        "estimated_render_payload_bytes": render_total,
    }


def freeze_reconstruction_baseline(
    *,
    spec_path: Path,
    paths_file: Path,
    repo_root: Path,
    solver_repo: Path,
    solver_binary: Path,
) -> Path:
    """Create the complete immutable reconstruction baseline."""

    state = preflight_reconstruction_freeze(
        spec_path=spec_path,
        paths_file=paths_file,
        repo_root=repo_root,
        solver_repo=solver_repo,
        solver_binary=solver_binary,
        require_clean=True,
    )
    spec = state["spec"]
    roots = state["roots"]
    artifacts_root = state["artifacts_root"]
    output_root = state["output_root"]
    output_root.mkdir(parents=True)
    marker = output_root / "INCOMPLETE"
    marker.write_text(
        "Freeze did not complete. Do not use this directory as a baseline.\n",
        encoding="utf-8",
    )
    support_entries: list[dict[str, Any]] = []
    try:
        render_records = []
        for value in spec["render_baseline_specs"]:
            baseline_spec = repo_root / _safe_relative(value, label="render baseline spec")
            child_root = freeze_render_baseline(
                spec_path=baseline_spec,
                artifacts_root=artifacts_root,
                repo_root=repo_root,
            )
            # The child freezer hashes both sides of every copy.  Defer the
            # full payload reread to the single top-level verification pass.
            child_manifest = _load_yaml(child_root / "manifest.yaml")
            render_records.append(
                {
                    "baseline_id": child_manifest["baseline_id"],
                    "path": child_root.relative_to(output_root).as_posix(),
                    "manifest_sha256": _sha256(child_root / "manifest.yaml"),
                    "payload_count": len(child_manifest["files"]),
                }
            )

        provenance = output_root / "provenance"
        repo_files = [
            spec_path,
            repo_root / _safe_relative(spec["matrix_spec"], label="matrix spec"),
            repo_root / "environment.yml",
            repo_root / "requirements.txt",
            repo_root / "requirements-test.txt",
        ]
        repo_files.extend(
            repo_root / _safe_relative(value, label="render baseline spec")
            for value in spec["render_baseline_specs"]
        )
        for source in repo_files:
            _copy_file(
                source,
                provenance / "repository" / source.relative_to(repo_root),
                root=output_root,
                entries=support_entries,
            )

        materialized = artifacts_root / _safe_relative(
            spec["materialized_matrix_artifact_relative"], label="materialized matrix"
        )
        _copy_declared_path(
            materialized,
            provenance / "materialized_matrix",
            root=output_root,
            entries=support_entries,
        )
        for identifier, value in spec["determinism_reports"].items():
            _copy_file(
                artifacts_root / _safe_relative(value, label=f"report {identifier}"),
                provenance / "determinism" / f"{identifier}.yaml",
                root=output_root,
                entries=support_entries,
            )

        fig4_root = output_root / "fig4"
        for identifier, value in spec["fig4_runs"].items():
            source_root = artifacts_root / _safe_relative(value, label=f"Fig.4 {identifier}")
            for relative_value in spec["fig4_payload_paths"]:
                relative = _safe_relative(relative_value, label="Fig.4 payload")
                _copy_declared_path(
                    source_root / relative,
                    fig4_root / identifier / relative,
                    root=output_root,
                    entries=support_entries,
                )

        camera_manifest = _write_camera_contract(
            spec["camera_contract"],
            roots,
            provenance / "camera_contract",
            root=output_root,
            entries=support_entries,
        )

        resource_records = [
            resource_group_manifest(value, roots) for value in spec["resource_groups"]
        ]
        resource_records.append(
            {
                "id": "solver_binary",
                "logical_root": "external_solver_build",
                "logical_base": "build/linux/x86_64/release/demo",
                "file_count": 1,
                "total_size_bytes": solver_binary.stat().st_size,
                "sha256": _sha256(solver_binary),
                "members": [
                    {
                        "path": "demo",
                        "size_bytes": solver_binary.stat().st_size,
                        "sha256": _sha256(solver_binary),
                    }
                ],
            }
        )
        resources = {
            "schema_version": 1,
            "resources": resource_records,
            "unused_resources": spec.get("unused_resources", {}),
        }
        resources_path = provenance / "resources.yaml"
        with resources_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(resources, stream, sort_keys=False, allow_unicode=True)
        support_entries.append(
            {
                "path": resources_path.relative_to(output_root).as_posix(),
                "size_bytes": resources_path.stat().st_size,
                "sha256": _sha256(resources_path),
            }
        )

        environment = _capture_environment(
            repo_root=repo_root,
            solver_repo=solver_repo,
            solver_binary=solver_binary,
        )
        environment_path = provenance / "environment.yaml"
        with environment_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(environment, stream, sort_keys=False, allow_unicode=True)
        support_entries.append(
            {
                "path": environment_path.relative_to(output_root).as_posix(),
                "size_bytes": environment_path.stat().st_size,
                "sha256": _sha256(environment_path),
            }
        )

        bundle_path = provenance / f"sand-water-mixtures-{spec['solver_revision']}.bundle"
        _run(["git", "bundle", "create", str(bundle_path), "--all"], cwd=solver_repo)
        _run(["git", "bundle", "verify", str(bundle_path)], cwd=solver_repo)
        support_entries.append(
            {
                "path": bundle_path.relative_to(output_root).as_posix(),
                "size_bytes": bundle_path.stat().st_size,
                "sha256": _sha256(bundle_path),
            }
        )

        preserved = []
        for value in spec["preserved_render_baselines"]:
            relative = _safe_relative(value, label="preserved baseline")
            baseline_root = artifacts_root / relative
            baseline = verify_render_baseline(baseline_root)
            preserved.append(
                {
                    "baseline_id": baseline["baseline_id"],
                    "artifact_relative": relative.as_posix(),
                    "manifest_sha256": _sha256(baseline_root / "manifest.yaml"),
                }
            )

        manifest = {
            "schema_version": 1,
            "baseline_id": str(spec["baseline_id"]),
            "status": "frozen",
            "overwrite_policy": "forbidden",
            "behavior_revision": str(spec["behavior_revision"]),
            "freeze_tool_revision": state["repository_revision"],
            "solver_revision": str(spec["solver_revision"]),
            "solver_long_run_repeat_count": 1,
            "p5_server_render_repeat_count": 3,
            "limitations": [
                "P5 server long-run solver determinism was not repeated; one canonical run is frozen.",
                "The three P5 server render trees were pairwise exact while reusing that solver run.",
            ],
            "render_baselines": render_records,
            "fig4_cases": list(spec["fig4_runs"]),
            "determinism_reports": list(spec["determinism_reports"]),
            "camera_contract": {
                "camera_contract_id": camera_manifest["camera_contract_id"],
                "manifest": "provenance/camera_contract/manifest.yaml",
                "manifest_sha256": _sha256(
                    provenance / "camera_contract" / "manifest.yaml"
                ),
            },
            "resources_manifest": {
                "path": "provenance/resources.yaml",
                "sha256": _sha256(resources_path),
            },
            "environment_manifest": {
                "path": "provenance/environment.yaml",
                "sha256": _sha256(environment_path),
            },
            "solver_bundle": {
                "path": bundle_path.relative_to(output_root).as_posix(),
                "sha256": _sha256(bundle_path),
            },
            "preserved_render_baselines": preserved,
            "support_files": sorted(support_entries, key=lambda item: item["path"]),
        }
        manifest_path = output_root / "manifest.yaml"
        with manifest_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)
        with (output_root / "SHA256SUMS").open("w", encoding="utf-8") as stream:
            for entry in manifest["support_files"]:
                stream.write(f"{entry['sha256']}  {entry['path']}\n")
        marker.unlink()
    except Exception:
        raise
    verify_reconstruction_baseline(output_root)
    return output_root


def verify_reconstruction_baseline(output_root: Path) -> dict[str, Any]:
    """Verify the complete frozen payload without requiring external resources."""

    output_root = output_root.expanduser().resolve()
    if (output_root / "INCOMPLETE").exists():
        raise ReconstructionFreezeError(f"freeze is incomplete: {output_root}")
    manifest = _load_yaml(output_root / "manifest.yaml")
    if manifest.get("status") != "frozen":
        raise ReconstructionFreezeError(f"baseline is not marked frozen: {output_root}")
    support_files = manifest.get("support_files")
    render_baselines = manifest.get("render_baselines")
    if not isinstance(support_files, list) or not support_files:
        raise ReconstructionFreezeError("baseline manifest has no support file list")
    if not isinstance(render_baselines, list) or not render_baselines:
        raise ReconstructionFreezeError("baseline manifest has no render baselines")
    for entry in support_files:
        relative = _safe_relative(entry.get("path"), label="support file")
        path = output_root / relative
        if not path.is_file():
            raise ReconstructionFreezeError(f"frozen support file is missing: {path}")
        digest = _sha256(path)
        if digest != entry.get("sha256"):
            raise ReconstructionFreezeError(
                f"frozen support hash mismatch: {relative}"
            )
    checksum_path = output_root / "SHA256SUMS"
    try:
        checksum_contents = checksum_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReconstructionFreezeError(
            f"cannot read frozen checksum index: {checksum_path}"
        ) from exc
    expected_checksums = "".join(
        f"{entry['sha256']}  {entry['path']}\n" for entry in support_files
    )
    if checksum_contents != expected_checksums:
        raise ReconstructionFreezeError("SHA256SUMS does not match the manifest")
    for record in render_baselines:
        child = output_root / _safe_relative(record.get("path"), label="child baseline")
        try:
            child_manifest = verify_render_baseline(child)
        except RenderBaselineError as exc:
            raise ReconstructionFreezeError(str(exc)) from exc
        if _sha256(child / "manifest.yaml") != record.get("manifest_sha256"):
            raise ReconstructionFreezeError(f"child manifest hash mismatch: {child}")
        if child_manifest.get("baseline_id") != record.get("baseline_id"):
            raise ReconstructionFreezeError(f"child baseline id mismatch: {child}")
    return manifest
