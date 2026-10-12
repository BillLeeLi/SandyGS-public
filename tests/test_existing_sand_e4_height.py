import numpy as np
import pytest

from sandygs.existing_sand.e4_height import (
    image_yx_to_exchange_xy, orthographic_grid, similarity_from_run,
    surface_comparison, valid_height,
)
from sandygs.existing_sand.workspace import ExistingSandError


def test_orthographic_pixels_and_exchange_axes():
    (xx, yy), spacing = orthographic_grid((0.0, 4.0, 10.0, 12.0), 32, 16)
    assert spacing == 0.125
    assert xx.shape == yy.shape == (16, 32)
    assert np.isclose(xx[0, 0], 0.0625)
    assert np.isclose(yy[0, 0], 10.0625)
    assert np.isclose(yy[-1, 0], 11.9375)
    image = xx + 10 * yy
    exchange = image_yx_to_exchange_xy(image)
    assert exchange.shape == (32, 16)
    assert np.isclose(exchange[0, 0], .0625 + 10 * 10.0625)
    assert np.isclose(exchange[-1, -1], 3.9375 + 10 * 11.9375)
    assert np.array_equal(exchange, image.T)


def test_camera_z_conversion_preserves_invalid_and_signed_thickness():
    depth = np.array([[2.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    alpha = np.array([[1.0, 1.0], [0.4, 0.8]], dtype=np.float32)
    clean, height, valid = valid_height(depth, alpha, camera_z=5.0, scale=2.0, alpha_min=.5)
    assert valid.tolist() == [[True, False], [False, True]]
    assert np.isnan(clean[0, 1]) and np.isnan(height[1, 0])
    assert np.isclose(height[0, 0], 4.0)
    assert np.isclose(height[1, 1], 3.0)
    sand = np.array([[4.0, 4.0], [4.0, 2.0]], dtype=np.float32)
    base = np.full((2, 2), 3.0, dtype=np.float32)
    delta, both = surface_comparison(sand, valid, base, np.ones((2, 2), bool))
    assert np.isclose(delta[0, 0], 1.0)
    assert np.isclose(delta[1, 1], -1.0)
    assert np.isnan(delta[0, 1])
    assert both.sum() == 2


def test_similarity_rejects_shear_or_anisotropic_scale():
    identity = {"coordinate": {"sim_to_gs": np.eye(4).tolist()}}
    matrix, scale = similarity_from_run(identity)
    assert scale == 1.0
    assert np.array_equal(matrix, np.eye(4))
    bad = np.diag([2.0, 1.0, 1.0, 1.0])
    with pytest.raises(ExistingSandError, match="uniform"):
        similarity_from_run({"coordinate": {"sim_to_gs": bad.tolist()}})
    with pytest.raises(ExistingSandError, match="square"):
        orthographic_grid((0, 4, 0, 4), 32, 16)
