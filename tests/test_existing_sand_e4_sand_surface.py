import numpy as np
from scipy.ndimage import distance_transform_edt

from sandygs.existing_sand.e4_sand_surface import compare_sand_surfaces


def test_selected_core_full_scene_edge_and_isolated_spike():
    row, col = np.indices((32, 32))
    support = (row - 16) ** 2 + (col - 16) ** 2 < 11 ** 2
    source = np.full((32, 32), -2.0, dtype=np.float32)
    source[support] = -1.8
    edge = support & (distance_transform_edt(support) <= 2)
    source[edge] = -2.0
    selected = source.copy()
    selected[support] = -1.8
    alpha = np.ones_like(source)
    alpha[edge] = .6
    # The selected-only renderer sees a background plane at a low-alpha edge.
    selected[16, 26] = -2.1
    alpha[16, 26] = .6
    # Full-scene rasterization has one independently corrupted top pixel.
    source[16, 24] = -1.0
    result = compare_sand_surfaces(source, selected, alpha, support,
                                   alpha_min=.98, core_distance=3,
                                   spike_max=.05)
    assert result["trusted_selected"][16, 16]
    assert result["hybrid"][16, 16] == selected[16, 16]
    assert not result["trusted_selected"][16, 26]
    assert result["hybrid"][16, 26] == source[16, 26]
    assert result["source_spike"][16, 24]
    assert not result["trusted_selected"][16, 24]
    assert abs(result["hybrid"][16, 24] - selected[16, 24]) < 1e-6
    assert result["selected"][16, 26] == -2.1
    assert result["full"][16, 24] == -1.0
