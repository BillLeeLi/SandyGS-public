"""Safety and provenance checks for the offline existing-sand workspace."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from sandygs.existing_sand import (
    ExistingSandError,
    initialize_run,
    record_stage,
    verify_run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "scripts/experiments/existing_sand.py"


def _source(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    dataset = tmp_path / "dataset"
    camera = dataset / "sparse/0/cameras.bin"
    camera.parent.mkdir(parents=True)
    camera.write_bytes(b"camera calibration")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    ply = checkpoint / "point_cloud.ply"
    ply.write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 3\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"end_header\n0 0 0\n1 0 0\n0 1 0\n"
    )
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump({
        "schema_version": 1,
        "scene_id": "sand_scene",
        "source": {
            "dataset": str(dataset),
            "checkpoint": str(checkpoint),
            "point_cloud": "point_cloud.ply",
            "dataset_files": ["sparse/0/cameras.bin"],
        },
        "camera_ids": ["camera_0"],
        "coordinate": {"sim_to_gs": [
            [1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        ]},
    }), encoding="utf-8")
    return config, checkpoint, ply, camera


def _stage(run_root: Path, stage: str, stage_id: str,
           parents: list[dict[str, str]] | None = None) -> tuple[Path, Path]:
    directory = run_root / "stages" / stage / stage_id
    directory.mkdir(parents=True)
    artifact = directory / "result.npy"
    artifact.write_bytes(b"synthetic result")
    record = directory / "record.yaml"
    record.write_text(yaml.safe_dump({
        "schema_version": 1, "stage": stage, "stage_id": stage_id,
        "camera_ids": ["camera_0"], "prompt": {"points": [[4, 5]]}, "seed": 7,
        "models": [], "parameters": {"threshold": 0.8},
        "commands": [["python", "stage.py"]],
        "parents": parents or [],
        "artifacts": [{"path": "result.npy", "role": "result"}],
    }), encoding="utf-8")
    return record, artifact


def test_run_cli_records_source_without_modifying_it(tmp_path: Path) -> None:
    config, checkpoint, ply, camera = _source(tmp_path)
    original = ply.read_bytes()
    run_root = tmp_path / "artifacts" / "run_001"
    result = subprocess.run(
        [sys.executable, "-B", str(CLI), "init", "--config", str(config),
         "--output", str(run_root), "--run-id", "run_001"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = yaml.safe_load((run_root / "run.yaml").read_text())
    assert manifest["source"]["gaussian_count"] == 3
    assert manifest["source"]["point_cloud"] == "point_cloud.ply"
    assert manifest["source"]["checkpoint_files"][0]["sha256"]
    assert manifest["coordinate"]["gs_to_sim"][0][3] == -2
    assert ply.read_bytes() == original
    assert verify_run(run_root)["stages"] == []
    assert subprocess.run(
        [sys.executable, "-B", str(CLI), "verify", "--run", str(run_root)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    ).returncode == 0
    with pytest.raises(ExistingSandError, match="overwrite"):
        initialize_run(config, run_root, run_id="run_001", repo_root=REPO_ROOT)
    camera.write_bytes(b"changed camera")
    with pytest.raises(ExistingSandError, match="dataset file has changed"):
        verify_run(run_root)


def test_stage_chain_and_tamper_detection(tmp_path: Path) -> None:
    config, _, _, _ = _source(tmp_path)
    run_root = tmp_path / "run_001"
    initialize_run(config, run_root, run_id="run_001", repo_root=REPO_ROOT)
    seg_record, seg_artifact = _stage(run_root, "segmentation", "manual_001")
    model_file = tmp_path / "small_model.bin"
    model_file.write_bytes(b"model version one")
    descriptor = yaml.safe_load(seg_record.read_text())
    descriptor["models"] = [{"name": "toy_mask", "version": "v1",
                              "checkpoint": str(model_file)}]
    seg_record.write_text(yaml.safe_dump(descriptor))
    seg = record_stage(run_root, seg_record, repo_root=REPO_ROOT)
    assert seg["source_gaussian_count"] == 3
    assert seg["models"][0]["sha256"]
    background_record, _ = _stage(
        run_root, "background", "local_001",
        [{"stage": "segmentation", "stage_id": "manual_001"}],
    )
    record_stage(run_root, background_record, repo_root=REPO_ROOT)
    assert len(verify_run(run_root)["stages"]) == 2
    model_file.write_bytes(b"model version two")
    with pytest.raises(ExistingSandError, match="model checkpoint has changed"):
        verify_run(run_root)
    model_file.write_bytes(b"model version one")
    with pytest.raises(ExistingSandError, match="overwrite"):
        record_stage(run_root, seg_record, repo_root=REPO_ROOT)
    seg_artifact.write_bytes(b"tampered")
    with pytest.raises(ExistingSandError, match="artifact has changed"):
        verify_run(run_root)
    seg_artifact.write_bytes(b"synthetic result")
    original_record = seg_record.read_text()
    seg_record.write_text(original_record + "\n# changed after registration\n")
    with pytest.raises(ExistingSandError, match="record has changed"):
        verify_run(run_root)
    seg_record.write_text(original_record)
    seg_manifest = seg_record.parent / "manifest.yaml"
    seg_manifest.write_text(seg_manifest.read_text() + "\n# changed parent\n")
    with pytest.raises(ExistingSandError, match="parent stage has changed"):
        verify_run(run_root)


def test_rejects_source_overlap_and_escaped_artifacts(tmp_path: Path) -> None:
    config, checkpoint, _, _ = _source(tmp_path)
    with pytest.raises(ExistingSandError, match="overlaps checkpoint"):
        initialize_run(config, checkpoint / "output", run_id="run_001",
                       repo_root=REPO_ROOT)
    run_root = tmp_path / "run_001"
    initialize_run(config, run_root, run_id="run_001", repo_root=REPO_ROOT)
    record, _ = _stage(run_root, "segmentation", "manual_001")
    descriptor = yaml.safe_load(record.read_text())
    descriptor["artifacts"][0]["path"] = "../../outside.npy"
    record.write_text(yaml.safe_dump(descriptor))
    with pytest.raises(ExistingSandError, match="inside"):
        record_stage(run_root, record, repo_root=REPO_ROOT)
    assert not (record.parent / "manifest.yaml").exists()
    descriptor["artifacts"][0]["path"] = "result.npy"
    descriptor["camera_ids"] = ["unregistered_camera"]
    record.write_text(yaml.safe_dump(descriptor))
    with pytest.raises(ExistingSandError, match="unknown camera_id"):
        record_stage(run_root, record, repo_root=REPO_ROOT)
    assert not (record.parent / "manifest.yaml").exists()
