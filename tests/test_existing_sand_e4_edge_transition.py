import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from sandygs.existing_sand.e4_edge_transition import harmonic_edge_transition
from sandygs.existing_sand.workspace import ExistingSandError


def test_harmonic_edge_keeps_core_and_replaces_negative_rim():
    support = np.zeros((24, 24), dtype=bool)
    support[5:19, 5:19] = True
    distance = distance_transform_edt(support)
    raw = np.full(support.shape, np.nan, dtype=np.float32)
    raw[support] = .10
    raw[support & (distance <= 3)] = -.01
    result, band = harmonic_edge_transition(raw, support, 3)
    assert np.array_equal(band, support & (distance <= 3))
    assert np.array_equal(result[support & ~band], raw[support & ~band])
    assert (result[support] > 0).all()
    assert np.isnan(result[~support]).all()
    assert np.mean(result[support & (distance == 1)]) < np.mean(result[support & (distance == 3)])


def test_negative_core_requires_a_wider_band_and_internal_invalid_stays_visible():
    support = np.zeros((24, 24), dtype=bool)
    support[5:19, 5:19] = True
    distance = distance_transform_edt(support)
    raw = np.full(support.shape, np.nan, dtype=np.float32)
    raw[support] = .10
    raw[7, 7] = -.01
    with pytest.raises(ExistingSandError, match="negative or empty core"):
        harmonic_edge_transition(raw, support, 1)
    raw[7, 7] = .10
    raw[12, 12] = np.nan
    result, _ = harmonic_edge_transition(raw, support, 3)
    assert np.isnan(result[12, 12])
    assert np.array_equal(result[support & (distance > 3) & np.isfinite(raw)],
                          raw[support & (distance > 3) & np.isfinite(raw)])
