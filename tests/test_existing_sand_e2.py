"""Check E2 occlusion conservatism, source index binding and feature gating."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import pytest
import yaml

from sandygs.existing_sand.e2 import select_seeds, select_frequency
from sandygs.existing_sand.e2_affinity import expand_affinity
from sandygs.existing_sand.e2_grounding import normalized_cxcywh_to_xyxy
from sandygs.existing_sand.e2_projection import (combine_votes, view_votes,
                                                 view_votes_frequency, frequency_labels)
from sandygs.existing_sand.workspace import initialize_run, record_stage, ExistingSandError

ROOT = Path(__file__).resolve().parents[1]


def _scene(tmp_path):
    ds = tmp_path / "dataset"
    (ds / "sparse/0").mkdir(parents=True)
    (ds / "sparse/0/cameras.bin").write_bytes(b"camera")
    ck = tmp_path / "checkpoint"
    ck.mkdir()
    xyz = np.array([[0, 0, 2], [-0.4, 0, 2], [0.4, 0, 2],
                    [0, 0, 3], [0.16, 0, 2]], dtype=np.float32)
    vertices = np.empty(5, dtype=[(k, "f4") for k in
                                  ("x", "y", "z", "scale_0", "scale_1")])
    for i, key in enumerate(("x", "y", "z")):
        vertices[key] = xyz[:, i]
    vertices["scale_0"] = vertices["scale_1"] = np.log(0.01)
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(ck / "source.ply"))
    config = tmp_path / "run_config.yaml"
    config.write_text(yaml.safe_dump({
        "schema_version": 1, "scene_id": "sand_scene",
        "source": {"dataset": str(ds), "checkpoint": str(ck),
                   "point_cloud": "source.ply",
                   "dataset_files": ["sparse/0/cameras.bin"]},
        "camera_ids": ["a.png", "b.png", "c.png"],
        "coordinate": {"sim_to_gs": np.eye(4).tolist()},
    }))
    run = tmp_path / "run_001"
    initialize_run(config, run, run_id="run_001", repo_root=ROOT)
    return run, (ck / "source.ply").read_bytes()


def _register(run: Path, kind: str, name: str, cameras: list[str], files: list[str],
              *, prompt=None, params=None, parents=None):
    stage = run / "stages" / kind / name
    record = {"schema_version": 1, "stage": kind, "stage_id": name,
              "camera_ids": cameras, "prompt": prompt or {}, "seed": None,
              "models": [], "parameters": params or {}, "commands": [],
              "parents": parents or [],
              "artifacts": [{"path": file, "role": "test"} for file in files]}
    (stage / "record.yaml").write_text(yaml.safe_dump(record))
    record_stage(run, stage / "record.yaml", repo_root=ROOT)
    return stage


def test_conservative_depth_mask_and_tristate_votes():
    h = w = 41
    depth = np.full((h, w), 2, dtype=np.float32)
    valid = np.ones_like(depth, dtype=bool)
    mask = np.zeros_like(valid)
    mask[16:25, 16:25] = True
    xyz = np.array([[0, 0, 2], [-0.4, 0, 2], [0.4, 0, 2],
                    [0, 0, 3], [0.16, 0, 2]], dtype=np.float32)
    scale = np.full((5, 2), np.log(0.01))
    k = np.array([[50, 0, 20], [0, 50, 20], [0, 0, 1]])
    votes = view_votes(xyz, scale, mask, depth, valid, k, np.eye(4))
    np.testing.assert_array_equal(votes, [1, 0, 0, 2, 2])
    np.testing.assert_array_equal(combine_votes((votes == 1).astype(np.uint8) * 2,
                                               (votes == 0).astype(np.uint8) * 2, 2), votes)
    depth[20, 20] = 9  # one depth jump makes a center/foreground vote unsafe
    assert view_votes(xyz, scale, mask, depth, valid, k, np.eye(4))[0] == 2
    assert normalized_cxcywh_to_xyxy([0.5, 0.5, 0.2, 0.2], 100, 100) == pytest.approx(
        [40, 40, 60, 60])


def test_e2_source_order_and_heldout_votes(tmp_path):
    run, source = _scene(tmp_path)
    obs = run / "stages/observations/observed"
    files = []
    for index, name in enumerate(("a.png", "b.png", "c.png")):
        view = obs / f"view_{index:04d}"
        view.mkdir(parents=True)
        camera = {"camera_id": name, "render_size_wh": [41, 41],
                  "pgsr_K": [[50, 0, 20], [0, 50, 20], [0, 0, 1]],
                  "gs_w2c": np.eye(4).tolist()}
        (view / "camera.yaml").write_text(yaml.safe_dump(camera))
        np.save(view / "plane_depth.npy", np.full((41, 41), 2, np.float32))
        np.save(view / "depth_valid.npy", np.ones((41, 41), dtype=bool))
        files += [f"{view.name}/{f}" for f in ("camera.yaml", "plane_depth.npy", "depth_valid.npy")]
    _register(run, "observations", "observed", ["a.png", "b.png", "c.png"], files)
    masks = run / "stages/segmentation/manual_masks"
    entries = []
    mask_files = []
    for index, name in enumerate(("a.png", "b.png", "c.png")):
        view = masks / f"view_{index:04d}"
        view.mkdir(parents=True)
        mask = np.zeros((41, 41), dtype=np.uint8)
        if index < 2:
            mask[16:25, 16:25] = 255
        else:
            mask[27:35, 27:35] = 255  # held-out SAM disagrees with center selection
        Image.fromarray(mask).save(view / "mask.png")
        mask_files.append(f"{view.name}/mask.png")
        entries.append({"camera_id": name, "directory": view.name,
                        "role": "holdout" if index == 2 else "seed"})
    (masks / "quality.yaml").write_text(yaml.safe_dump({"views": entries}))
    mask_files.append("quality.yaml")
    _register(run, "segmentation", "manual_masks", ["a.png", "b.png", "c.png"],
              mask_files, prompt={"type": "manual_sam_points_box"},
              params={"capture_id": "observed"},
              parents=[{"stage": "observations", "stage_id": "observed"}])
    result = select_seeds(run, "manual_masks", "seeds", repo_root=ROOT)
    assert result["counts"] == {"negative": 2, "positive": 1, "unknown": 2}
    np.testing.assert_array_equal(np.load(run / "stages/segmentation/seeds/gaussian_labels.npy"),
                                  [1, 0, 0, 2, 2])
    assert result["holdout_center_checks"][0]["positive_outside"] == 1
    assert (tmp_path / "checkpoint/source.ply").read_bytes() == source
    with pytest.raises(ExistingSandError, match="overwrite"):
        select_seeds(run, "manual_masks", "seeds", repo_root=ROOT)

    affinity = run / "stages/segmentation/affinity"
    affinity.mkdir()
    features = np.zeros((5, 32), dtype=np.float16)
    features[0, 0] = features[4, 0] = 1
    features[1:3, 1] = 1
    np.save(affinity / "features.npy", features)
    np.save(affinity / "feature_valid.npy", np.array([True, True, True, False, True]))
    np.savez(affinity / "scale_gate.npz", weight=np.zeros((32, 1)),
             bias=np.zeros(32), bounds=np.array([0.1, 1.]))
    _register(run, "segmentation", "affinity", ["a.png", "b.png"],
              ["features.npy", "feature_valid.npy", "scale_gate.npz"],
              prompt={"type": "scene_sam_affinity_training"},
              params={"capture_id": "observed"},
              parents=[{"stage": "observations", "stage_id": "observed"}])
    # The boundary unknown has zero positive votes, so even a perfect
    # affinity match cannot silently promote it to a deletion candidate.
    expanded = expand_affinity(run, "seeds", "affinity", "expanded",
                               repo_root=ROOT, scale=0.5, prototypes=1)
    assert expanded["added_positive"] == 0
    one_vote = run / "stages/segmentation/seeds_onevote"
    one_vote.mkdir()
    labels = np.load(run / "stages/segmentation/seeds/gaussian_labels.npy")
    pos = np.load(run / "stages/segmentation/seeds/positive_votes.npy")
    neg = np.load(run / "stages/segmentation/seeds/negative_votes.npy")
    pos[4] = 1  # visible once, with no opposing vote
    for filename, values in (("gaussian_labels.npy", labels),
                             ("positive_votes.npy", pos), ("negative_votes.npy", neg)):
        np.save(one_vote / filename, values)
    _register(run, "segmentation", "seeds_onevote", ["a.png", "b.png"],
              ["gaussian_labels.npy", "positive_votes.npy", "negative_votes.npy"],
              prompt={"type": "multiview_sam_depth_seeds", "mask_id": "manual_masks"},
              params={"capture_id": "observed"},
              parents=[{"stage": "segmentation", "stage_id": "manual_masks"}])
    added = expand_affinity(run, "seeds_onevote", "affinity", "expanded_onevote",
                            repo_root=ROOT, scale=0.5, prototypes=1)
    assert added["added_positive"] == 1
    np.testing.assert_array_equal(np.load(
        run / "stages/segmentation/expanded_onevote/gaussian_labels.npy"),
        [1, 0, 0, 2, 1])


def test_frequency_visibility_and_zero_vote_rule():
    mask = np.zeros((41, 41), dtype=bool)
    mask[20, 20] = True
    depth = np.full((41, 41), 2, dtype=np.float32)
    valid = np.ones_like(mask)
    k = np.array([[50, 0, 20], [0, 50, 20], [0, 0, 1]])
    xyz = np.array([[0, 0, 2], [0, 0, 3], [0, 0, 1.5],
                    [0.4, 0, 2], [0, 0, -1]], dtype=np.float32)
    votes = view_votes_frequency(xyz, mask, depth, valid, k, np.eye(4))
    np.testing.assert_array_equal(votes, [1, 2, 1, 0, 2])
    valid[20, 20] = False
    assert view_votes_frequency(xyz, mask, depth, valid, k, np.eye(4))[0] == 2
    labels, fraction = frequency_labels(np.array([1, 1, 0, 0], np.uint16),
                                        np.array([2, 1, 1, 0], np.uint16), 0.65)
    np.testing.assert_array_equal(labels, [0, 1, 0, 2])
    np.testing.assert_allclose(fraction[:3], [0.5, 1, 0])
    assert np.isnan(fraction[3])
    with pytest.raises(ExistingSandError, match="threshold"):
        frequency_labels(np.array([1]), np.array([1]), 0)


def test_e2_frequency_stage_threshold_holdout_and_source_order(tmp_path):
    run, source = _scene(tmp_path)
    obs = run / "stages/observations/observed"
    obs_files = []
    mask_stage = run / "stages/segmentation/manual_masks"
    mask_files = []
    entries = []
    for index, name in enumerate(("a.png", "b.png", "c.png")):
        view = obs / f"view_{index:04d}"
        view.mkdir(parents=True)
        camera = {"camera_id": name, "render_size_wh": [41, 41],
                  "pgsr_K": [[50, 0, 20], [0, 50, 20], [0, 0, 1]],
                  "gs_w2c": np.eye(4).tolist()}
        (view / "camera.yaml").write_text(yaml.safe_dump(camera))
        np.save(view / "plane_depth.npy", np.full((41, 41), 2, np.float32))
        np.save(view / "depth_valid.npy", np.ones((41, 41), dtype=bool))
        obs_files += [f"{view.name}/{f}" for f in
                      ("camera.yaml", "plane_depth.npy", "depth_valid.npy")]
        folder = mask_stage / f"view_{index:04d}"
        folder.mkdir(parents=True)
        mask = np.zeros((41, 41), dtype=np.uint8)
        if index != 1:
            mask[20, 20] = 255
        else:
            mask[20, 30] = 255
        Image.fromarray(mask).save(folder / "mask.png")
        mask_files.append(f"{folder.name}/mask.png")
        entries.append({"camera_id": name, "directory": folder.name,
                        "role": "holdout" if index == 2 else "seed"})
    _register(run, "observations", "observed", ["a.png", "b.png", "c.png"], obs_files)
    (mask_stage / "quality.yaml").write_text(yaml.safe_dump({"views": entries}))
    mask_files.append("quality.yaml")
    _register(run, "segmentation", "manual_masks", ["a.png", "b.png", "c.png"],
              mask_files, prompt={"type": "manual_sam_points_box"},
              params={"capture_id": "observed"},
              parents=[{"stage": "observations", "stage_id": "observed"}])
    high = select_frequency(run, "manual_masks", "frequency_high",
                            repo_root=ROOT, sand_fraction=0.65)
    low = select_frequency(run, "manual_masks", "frequency_low",
                           repo_root=ROOT, sand_fraction=0.5)
    folder = run / "stages/segmentation/frequency_high"
    np.testing.assert_array_equal(np.load(folder / "gaussian_labels.npy"),
                                  [0, 0, 0, 2, 0])
    np.testing.assert_array_equal(np.load(folder / "visible_votes.npy"),
                                  [2, 2, 2, 0, 2])
    np.testing.assert_array_equal(np.load(folder / "positive_votes.npy"),
                                  [1, 0, 1, 0, 0])
    assert np.load(folder / "sand_fraction.npy")[0] == 0.5
    assert high["counts"]["unknown"] == 1
    assert low["counts"]["positive"] == 2
    assert high["holdout_center_checks"][0]["negative_inside"] == 1
    assert (tmp_path / "checkpoint/source.ply").read_bytes() == source
    with pytest.raises(ExistingSandError, match="overwrite"):
        select_frequency(run, "manual_masks", "frequency_high", repo_root=ROOT)
