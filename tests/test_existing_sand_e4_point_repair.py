import numpy as np
import pytest

from sandygs.existing_sand.e4_point_repair import repair_one_background_invalid
from sandygs.existing_sand.workspace import ExistingSandError


def test_reviewed_internal_pixel_uses_local_base_and_preserves_every_other_cell():
    support = np.zeros((7, 7), dtype=bool)
    support[1:6, 1:6] = True
    thickness = np.full((7, 7), np.nan, dtype=np.float32)
    thickness[support] = .1
    thickness[3, 3] = np.nan
    top = np.full((7, 7), -1.9, dtype=np.float32)
    base = np.full((7, 7), -2.0, dtype=np.float32)
    original_valid = np.ones((7, 7), dtype=bool)
    original_valid[3, 3] = False
    result, audit = repair_one_background_invalid(
        thickness, top, base, support, original_valid, 3, 3,
    )
    assert result[3, 3] == pytest.approx(.1)
    assert np.array_equal(result[support & np.isfinite(thickness)],
                          thickness[support & np.isfinite(thickness)])
    assert np.isnan(result[~support]).all()
    assert audit["other_supported_pixels_bitwise_unchanged"] is True
    assert audit["difference_from_neighbor_median"] == pytest.approx(0, abs=1e-6)


def test_wrong_location_or_disagreeing_fit_is_rejected():
    support = np.ones((5, 5), dtype=bool)
    thickness = np.full((5, 5), .1, dtype=np.float32)
    thickness[2, 2] = np.nan
    top = np.full((5, 5), -1.9, dtype=np.float32)
    base = np.full((5, 5), -2.0, dtype=np.float32)
    original_valid = np.ones((5, 5), dtype=bool)
    original_valid[2, 2] = False
    with pytest.raises(ExistingSandError, match="reviewed internal invalid pixel"):
        repair_one_background_invalid(thickness, top, base, support, original_valid, 2, 1)
    base[2, 2] = -2.1
    with pytest.raises(ExistingSandError, match="disagrees"):
        repair_one_background_invalid(thickness, top, base, support, original_valid, 2, 2)
