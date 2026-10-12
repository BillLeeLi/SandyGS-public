from pathlib import Path
import tempfile

import numpy as np
import pytest
import yaml

from sandygs.exchange import GridSpec
from sandygs.prepare import base_metadata, prepare_sand_water_case, write_field
from sandygs.existing_sand.e4_initialization import (
    validate_e4_fields,
    validate_water_patch,
    water_patch_validation_options,
)
from sandygs.existing_sand.workspace import ExistingSandError


def test_e4_direct_transpose_and_reviewed_water_region():
    terrain_yx = np.arange(48, dtype=np.float32).reshape(6, 8) * .01
    sand_yx = np.zeros((6, 8), dtype=np.float32)
    sand_yx[2:4, 3:5] = .12
    support_yx = sand_yx > 0
    support_xy = validate_e4_fields(terrain_yx.T.copy(), sand_yx.T.copy(),
                                    support_yx, terrain_yx, sand_yx)
    np.testing.assert_array_equal(support_xy, support_yx.T)
    with pytest.raises(ExistingSandError, match="transposition"):
        validate_e4_fields(terrain_yx[::-1].T.copy(), sand_yx.T.copy(),
                           support_yx, terrain_yx, sand_yx)
    water = np.zeros((8, 6), dtype=np.float64)
    water[6, 2] = .02
    audit = validate_water_patch(water, sand_yx.T, np.ones((8, 6), bool),
                                 np.ones((8, 6)), boundary_cells=1)
    assert audit["wet_cells"] == 1
    water[3, 2] = .01
    with pytest.raises(ExistingSandError, match="overlaps"):
        validate_water_patch(water, sand_yx.T, np.ones((8, 6), bool),
                             np.ones((8, 6)), boundary_cells=1)


def test_water_patch_validation_can_explicitly_allow_domain_boundary():
    defaults = water_patch_validation_options({})
    assert defaults == {"boundary_cells": 8, "alpha_min": .98}
    options = water_patch_validation_options({
        "water_patch_validation": {"boundary_cells": 0, "alpha_min": .99}
    })
    assert options == {"boundary_cells": 0, "alpha_min": .99}

    water = np.zeros((4, 4), dtype=np.float64)
    water[0, 0] = .06
    with pytest.raises(ExistingSandError, match="solver boundary"):
        validate_water_patch(
            water,
            np.zeros_like(water),
            np.ones_like(water, dtype=bool),
            np.ones_like(water),
            **defaults,
        )
    audit = validate_water_patch(
        water,
        np.zeros_like(water),
        np.ones_like(water, dtype=bool),
        np.ones_like(water),
        **options,
    )
    assert audit["minimum_boundary_clearance_cells"] == 0
    assert audit["sand_overlap_cells"] == 0


def test_water_patch_validation_rejects_unknown_or_invalid_options():
    with pytest.raises(ExistingSandError, match="unknown keys"):
        water_patch_validation_options({
            "water_patch_validation": {"allow_overlap": True}
        })
    with pytest.raises(ExistingSandError, match="nonnegative integer"):
        water_patch_validation_options({
            "water_patch_validation": {"boundary_cells": -1}
        })
    with pytest.raises(ExistingSandError, match=r"within \[0,1\]"):
        water_patch_validation_options({
            "water_patch_validation": {"alpha_min": 1.1}
        })


def test_solver_case_reads_checked_sand_height_from_exchange():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        exchange = root / "exchange"
        exchange.mkdir()
        grid = GridSpec((4, 3), (-.4, -1.2), (1.2, 0.0))
        terrain = np.full(grid.resolution, -2.6, dtype=np.float32)
        sand = np.arange(12, dtype=np.float32).reshape(grid.resolution) * .01
        metadata = base_metadata(scene_id="test", grid=grid, sim_to_gs=np.eye(4),
                                 scale_status="unverified", meters_per_simulation_unit=None,
                                 scale_source="test")
        fields = {}
        for name, values, semantic in (("terrain", terrain, "static_terrain_elevation"),
                                        ("sand_height", sand, "sand_thickness")):
            field = write_field(exchange, f"frames/000000/{name}.npy", values)
            field["semantic"] = semantic
            if name == "sand_height":
                field["constraints"].append("nonnegative")
            fields[name] = field
        metadata["frames"] = [{"frame_index": 0, "step_index": 0,
                                "time_seconds": 0.0, "fields": fields}]
        with (exchange / "metadata.yaml").open("w") as stream:
            yaml.safe_dump(metadata, stream)
        profile = {"schema_version": 1, "profile_id": "e4-test",
                   "initial": {"sand": {"exchange_field": "sand_height"}, "total_water": 0.0},
                   "solver": {"water_dt": .001, "steps": 1, "frame_stride": 1,
                              "target_cfl": .45}}
        case = prepare_sand_water_case(exchange_bundle=exchange, output=root / "case",
                                       profile=profile)
        np.testing.assert_array_equal(np.load(root / "case/sand.npy"), sand)
        assert case["initialization"]["sand_kind"] == "exchange_sand_height"
        assert case["initialization"]["water_volume"] == 0.0
