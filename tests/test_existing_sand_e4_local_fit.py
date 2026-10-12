import numpy as np
from scipy.ndimage import distance_transform_edt

from sandygs.existing_sand.e4_local_fit import local_tabletop_candidates


def test_visible_tabletop_and_stable_center_replace_edge_bump():
    row, col = np.indices((64, 64))
    x = (col + .5) / 64
    y = (row + .5) / 64
    source = (1 + .1 * x - .05 * y).astype(np.float32)
    sand_mask = (row - 32) ** 2 + (col - 32) ** 2 < 20 ** 2
    inner_distance = distance_transform_edt(sand_mask)
    base = (source - .01).copy()
    base[sand_mask & (inner_distance <= 8)] += .20
    sand = source + .1
    valid = np.ones(source.shape, dtype=bool)
    alpha = np.ones(source.shape, dtype=np.float32)

    result = local_tabletop_candidates(
        source, valid, alpha, sand, sand_mask, base, valid, alpha,
        (0, 1, 0, 1), outer_min=2, outer_max=8, inner_min=8,
        residual_max=.02, local_deviation_max=.005,
    )

    assert result["outer"].sum() > 100
    assert result["inner"].sum() > 100
    assert np.allclose(result["plane"], source, atol=1e-6)
    assert np.array_equal(result["local"][~sand_mask], source[~sand_mask])
    assert np.allclose(result["local"][result["inner"]], base[result["inner"]])
    edge_spike = sand_mask & (inner_distance <= 8)
    assert not result["inner"][edge_spike].any()
    assert np.max(np.abs(result["local"][edge_spike] - source[edge_spike])) < .02
    assert np.mean(np.abs(result["local"][edge_spike] - source[edge_spike])) < .01
