"""Turn a reviewed E4 height-field candidate into exchange and solver cases."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile

import numpy as np
import yaml

from sandygs.exchange import GridSpec, sha256_file, validate_bundle
from sandygs.prepare import base_metadata, prepare_sand_water_case, sample_height_spec, write_field

from .e4_local_fit import _verified_input
from .workspace import ExistingSandError, _load_yaml, _stage_location, record_stage


def validate_e4_fields(terrain_xy: np.ndarray, sand_xy: np.ndarray,
                       support_yx: np.ndarray, terrain_yx: np.ndarray,
                       sand_yx: np.ndarray) -> np.ndarray:
    """Check value and axis contracts before writing a physical cell grid."""
    if (terrain_xy.ndim != 2 or terrain_xy.shape != sand_xy.shape
            or any(a.shape != terrain_xy.T.shape for a in (support_yx, terrain_yx, sand_yx))):
        raise ExistingSandError("E4 terrain, sand and support shapes are inconsistent")
    if (terrain_xy.dtype != np.float32 or sand_xy.dtype != np.float32
            or terrain_yx.dtype != np.float32 or sand_yx.dtype != np.float32):
        raise ExistingSandError("E4 candidate heights must be float32")
    if not (np.array_equal(terrain_xy, terrain_yx.T)
            and np.array_equal(sand_xy, sand_yx.T)):
        raise ExistingSandError("E4 [x,y] arrays do not match direct [row,col] transposition")
    if (not np.isfinite(terrain_xy).all() or not np.isfinite(sand_xy).all()
            or np.any(sand_xy < 0)):
        raise ExistingSandError("E4 terrain and sand must be finite; sand must be nonnegative")
    support_xy = np.ascontiguousarray(support_yx.T.astype(bool))
    if not np.any(support_xy) or np.any(sand_xy[~support_xy] != 0):
        raise ExistingSandError("E4 support is empty or sand extends beyond reviewed support")
    return support_xy


def validate_water_patch(water_xy: np.ndarray, sand_xy: np.ndarray,
                         source_valid_xy: np.ndarray, source_alpha_xy: np.ndarray,
                         *, boundary_cells: int = 8, alpha_min: float = .98) -> dict:
    wet = water_xy > 0
    if water_xy.shape != sand_xy.shape or not wet.any():
        raise ExistingSandError("water patch must occupy at least one E4 cell")
    if not np.isfinite(water_xy).all() or np.any(water_xy < 0):
        raise ExistingSandError("water patch must be finite and nonnegative")
    if np.any(wet & (sand_xy > 0)):
        raise ExistingSandError("initial water patch overlaps reviewed sand")
    if not np.all(source_valid_xy[wet]) or np.min(source_alpha_xy[wet]) < alpha_min:
        raise ExistingSandError("initial water patch leaves high-confidence observed tabletop")
    indices = np.argwhere(wet)
    clearance = int(np.min(np.minimum(indices, np.asarray(water_xy.shape) - 1 - indices)))
    if clearance < boundary_cells:
        raise ExistingSandError("initial water patch is too close to the solver boundary")
    return {"wet_cells": int(wet.sum()), "sand_overlap_cells": 0,
            "minimum_source_alpha": float(source_alpha_xy[wet].min()),
            "minimum_boundary_clearance_cells": clearance}


def water_patch_validation_options(config: dict) -> dict:
    """Read explicit water-region review thresholds without changing defaults."""
    raw = config.get("water_patch_validation", {})
    if not isinstance(raw, dict):
        raise ExistingSandError("water_patch_validation must be a mapping")
    unknown = set(raw) - {"boundary_cells", "alpha_min"}
    if unknown:
        raise ExistingSandError(
            "water_patch_validation has unknown keys: "
            + ", ".join(sorted(unknown))
        )
    boundary_cells = raw.get("boundary_cells", 8)
    alpha_min = raw.get("alpha_min", .98)
    if (isinstance(boundary_cells, bool) or not isinstance(boundary_cells, int)
            or boundary_cells < 0):
        raise ExistingSandError(
            "water_patch_validation.boundary_cells must be a nonnegative integer"
        )
    if (isinstance(alpha_min, bool) or not isinstance(alpha_min, (int, float))
            or not np.isfinite(alpha_min) or not 0.0 <= float(alpha_min) <= 1.0):
        raise ExistingSandError(
            "water_patch_validation.alpha_min must be finite and within [0,1]"
        )
    return {"boundary_cells": boundary_cells, "alpha_min": float(alpha_min)}


def _overview(path: Path, terrain: np.ndarray, sand: np.ndarray, water: np.ndarray,
              grid: GridSpec) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [grid.domain_min[0], grid.domain_max[0],
              grid.domain_min[1], grid.domain_max[1]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, field, title, cmap in zip(
        axes, (terrain, sand, water),
        ("terrain elevation", "initial sand thickness", "initial water thickness"),
        ("viridis", "magma", "Blues"),
    ):
        plotted = ax.imshow(field.T, origin="lower", extent=extent, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("simulation x")
        ax.set_ylabel("simulation y")
        fig.colorbar(plotted, ax=ax, shrink=.75)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_e4_initialization(run_root: Path, repair_id: str, stage_id: str,
                          config_path: Path, *, repo_root: Path,
                          command: list[str] | None = None) -> dict:
    run_root = Path(run_root).resolve()
    config_path = Path(config_path).resolve()
    source = _stage_location(run_root, "modeling", repair_id)
    target = _stage_location(run_root, "modeling", stage_id)
    if target.exists():
        raise ExistingSandError(f"refusing to overwrite: {target}")
    run = _load_yaml(run_root / "run.yaml", "run")
    manifest = _load_yaml(source / "manifest.yaml", "E4 repair manifest")
    record = _load_yaml(source / "record.yaml", "E4 repair record")
    if manifest.get("stage_id") != repair_id or record.get("prompt", {}).get("type") != "reviewed_single_background_invalid_repair":
        raise ExistingSandError("source must be a registered, reviewed E4 point-repair stage")
    entries = {item["path"]: item for item in manifest["artifacts"]}
    def checked(name: str) -> Path:
        return _verified_input(source, name, entries)

    terrain = np.load(checked("candidate_terrain_xy.npy"), allow_pickle=False)
    sand = np.load(checked("candidate_sand_height_xy.npy"), allow_pickle=False)
    terrain_yx = np.load(checked("candidate_terrain_yx.npy"), allow_pickle=False)
    sand_yx = np.load(checked("candidate_sand_height_yx.npy"), allow_pickle=False)
    parent_ortho = record["parameters"]["orthographic_stage_id"]
    ortho = _stage_location(run_root, "modeling", parent_ortho)
    ortho_manifest = _load_yaml(ortho / "manifest.yaml", "E4 orthographic manifest")
    ortho_entries = {item["path"]: item for item in ortho_manifest["artifacts"]}
    def ortho_checked(name: str) -> Path:
        return _verified_input(ortho, name, ortho_entries)
    support_yx = np.load(ortho_checked("sand_valid_yx.npy"), allow_pickle=False)
    source_valid_yx = np.load(ortho_checked("source_valid_yx.npy"), allow_pickle=False)
    source_alpha_yx = np.load(ortho_checked("source_alpha.npy"), allow_pickle=False)
    camera = _load_yaml(ortho_checked("camera.yaml"), "E4 orthographic camera")
    support = validate_e4_fields(terrain, sand, support_yx, terrain_yx, sand_yx)
    bounds = camera["bounds_simulation_xy"]
    grid = GridSpec(terrain.shape, (float(bounds[0]), float(bounds[2])),
                    (float(bounds[1]), float(bounds[3])))
    if (list(camera["image_size_wh"]) != list(terrain.shape)
            or not np.isclose(grid.spacing[0], grid.spacing[1], rtol=0, atol=1e-12)
            or not np.isclose(grid.spacing[0], float(camera["cell_spacing_simulation"]), rtol=0, atol=1e-12)):
        raise ExistingSandError("E4 camera and exchange grid geometry disagree")
    sim_to_gs = np.asarray(run["coordinate"]["sim_to_gs"], dtype=np.float64)
    if not np.allclose(sim_to_gs, np.asarray(camera["sim_to_gs"]), rtol=0, atol=1e-9):
        raise ExistingSandError("E4 camera and run simulation transforms disagree")
    config = _load_yaml(config_path, "E4 initialization config")
    profile = config.get("profile")
    if (config.get("schema_version") != 1 or config.get("scene_id") != run["scene_id"]
            or not isinstance(profile, dict) or profile.get("initial") != {"sand": {"exchange_field": "sand_height"}}
            or not isinstance(config.get("water_patch"), dict)):
        raise ExistingSandError("E4 config must identify this scene, exchange sand and a water patch")
    water = sample_height_spec(config["water_patch"], grid, name="water_patch")
    water_validation = water_patch_validation_options(config)
    water_audit = validate_water_patch(
        water,
        sand,
        source_valid_yx.T,
        source_alpha_yx.T,
        **water_validation,
    )
    water_audit["required_boundary_clearance_cells"] = water_validation[
        "boundary_cells"
    ]
    water_audit["required_source_alpha_min"] = water_validation["alpha_min"]
    cell_area = grid.spacing[0] * grid.spacing[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = Path(tempfile.mkdtemp(prefix=f".e4_init_{stage_id}_", dir=run_root))
    try:
        exchange = payload / "exchange"
        exchange.mkdir()
        metadata = base_metadata(
            scene_id=run["scene_id"], grid=grid, sim_to_gs=sim_to_gs,
            scale_status="unverified", meters_per_simulation_unit=None,
            scale_source="indoor02 PGSR-to-simulation identity; metric scale unmeasured",
        )
        fields = {}
        for name, array, semantic, nonnegative in (
            ("terrain", terrain, "static_terrain_elevation", False),
            ("sand_height", sand, "sand_thickness", True),
            ("sand_support", support.astype(np.float32), "reviewed_sand_support_mask", True),
        ):
            descriptor = write_field(exchange, f"frames/000000/{name}.npy", array)
            descriptor["semantic"] = semantic
            if nonnegative:
                descriptor["constraints"].append("nonnegative")
            fields[name] = descriptor
        metadata["frames"] = [{"frame_index": 0, "step_index": 0,
                                "time_seconds": 0.0, "fields": fields}]
        metadata["provenance"] = {
            "source_kind": "reviewed_existing_sand_e4_cell_centers",
            "source_stage_id": repair_id,
            "source_manifest_sha256": sha256_file(source / "manifest.yaml"),
            "source_terrain_sha256": sha256_file(checked("candidate_terrain_xy.npy")),
            "source_sand_sha256": sha256_file(checked("candidate_sand_height_xy.npy")),
            "axis_conversion": "PGSR raster [row,col] direct transpose to [x,y]",
        }
        with (exchange / "metadata.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(metadata, stream, sort_keys=False)
        validate_bundle(exchange)
        case_summaries = {}
        for case_name, water_spec in (("dry", 0.0), ("water", config["water_patch"])):
            case_profile = deepcopy(profile)
            case_profile["initial"]["total_water"] = water_spec
            case_profile["profile_id"] = f"{profile['profile_id']}-{case_name}"
            case_root = payload / "cases" / case_name
            case_root.mkdir(parents=True)
            with (case_root / "profile.yaml").open("x", encoding="utf-8") as stream:
                yaml.safe_dump(case_profile, stream, sort_keys=False)
            case = prepare_sand_water_case(exchange_bundle=exchange,
                                           output=case_root / "input", profile=case_profile)
            initial = case["initialization"]
            if initial["sand_kind"] != "exchange_sand_height" or not np.array_equal(
                np.load(case_root / "input" / "sand.npy"), sand.astype(np.float64)
            ):
                raise ExistingSandError("solver sand initial state differs from E4 candidate")
            case_summaries[case_name] = initial
        if case_summaries["dry"]["water_cell_count"] != 0:
            raise ExistingSandError("dry solver case unexpectedly contains water")
        if not np.allclose(np.load(payload / "cases/water/input/total_water.npy"), water, rtol=0, atol=1e-12):
            raise ExistingSandError("wet solver case differs from sampled water patch")
        shutil.copyfile(config_path, payload / "config.yaml")
        _overview(payload / "initial_overview.png", terrain, sand, water, grid)
        summary = {
            "status": "exchange and dry/water solver initial cases prepared; solver not run",
            "grid_resolution_xy": list(grid.resolution), "bounds_xy": [float(v) for v in bounds],
            "cell_spacing": float(grid.spacing[0]),
            "scale_status": "unverified", "sand_support_cells": int(support.sum()),
            "sand_positive_cells": int(np.count_nonzero(sand > 0)),
            "sand_volume_simulation_units_cubed": float(sand.sum() * cell_area),
            "water_volume_simulation_units_cubed": float(water.sum() * cell_area),
            "water_patch_check": water_audit, "cases": case_summaries,
        }
        with (payload / "summary.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(summary, stream, sort_keys=False)
        record_out = {
            "schema_version": 1, "stage": "modeling", "stage_id": stage_id,
            "camera_ids": [], "prompt": {"type": "e4_reviewed_heightfield_solver_initialization"},
            "seed": None, "models": [],
            "parameters": {"repair_stage_id": repair_id, "orthographic_stage_id": parent_ortho,
                           "config_sha256": sha256_file(config_path), "cases": ["dry", "water"]},
            "commands": [command] if command else [], "command_cwd": str(Path(repo_root).resolve()),
            "parents": [{"stage": "modeling", "stage_id": repair_id},
                        {"stage": "modeling", "stage_id": parent_ortho}],
            "artifacts": [{"path": path.relative_to(payload).as_posix(), "role": "e4_solver_initialization"}
                          for path in sorted(payload.rglob("*")) if path.is_file()],
            "code_files": ["sandygs/exchange.py", "sandygs/prepare.py",
                           "sandygs/existing_sand/e4_initialization.py",
                           "scripts/experiments/existing_sand_e4_initialize.py"],
        }
        with (payload / "record.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(record_out, stream, sort_keys=False, allow_unicode=True)
        payload.rename(target)
        try:
            # The input cases were prepared inside the temporary stage. Update
            # their audit path only after the atomic move to the final stage.
            for case_name in ("dry", "water"):
                case_path = target / "cases" / case_name / "input" / "case.yaml"
                case_meta = _load_yaml(case_path, f"{case_name} solver case")
                case_meta["exchange_bundle"] = str(target / "exchange")
                with case_path.open("w", encoding="utf-8") as stream:
                    yaml.safe_dump(case_meta, stream, sort_keys=False)
            validate_bundle(target / "exchange")
            record_stage(run_root, target / "record.yaml", repo_root=repo_root)
        except BaseException:
            shutil.rmtree(target)
            raise
        return {"stage_dir": str(target), **summary}
    finally:
        if payload.exists():
            shutil.rmtree(payload)
