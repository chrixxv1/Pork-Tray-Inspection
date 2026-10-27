"""Tests of the performance helpers."""

import numpy as np
import pytest
from PIL import Image

from porkqc.features import hog_features
from porkqc.perf import (arithmetic_intensity, gd_iteration, hessian_formation, load_image_draft,
                         newton_iteration, randomized_svd_flops, time_best)


def test_flop_counts_match_hand_calculation():
    # n = 10, p = 3: GD = 4*10*3 = 120 FLOPs, reading A twice = 2*10*3*8 = 480 bytes.
    assert gd_iteration(10, 3) == (120.0, 480.0)
    # Hessian: 2*10*9 + 30 = 210 FLOPs.
    assert hessian_formation(10, 3)[0] == 210.0
    # Newton adds LU (2*27/3 = 18) and two solves (2*9 = 18) to GD + Hessian.
    assert newton_iteration(10, 3)[0] == 120 + 210 + 18 + 18
    # Randomized SVD, q = 0: two passes of 2*n*d*r plus one n x r QR.
    assert randomized_svd_flops(4, 5, 2, 0) == 2 * 4 * 5 * 2 * 2 + 4 * 4 * 4


def test_gd_is_memory_bound_and_hessian_intensity_grows_with_p():
    assert np.isclose(arithmetic_intensity(*gd_iteration(1000, 51)), 0.25)
    small = arithmetic_intensity(*hessian_formation(1000, 11))
    large = arithmetic_intensity(*hessian_formation(1000, 401))
    assert large > 10 * small
    with pytest.raises(ValueError):
        arithmetic_intensity(1.0, 0.0)


def test_time_best_is_positive_and_validates():
    assert time_best(lambda: sum(range(1000)), repeats=3) > 0
    with pytest.raises(ValueError):
        time_best(lambda: None, repeats=0)


def test_draft_decoding_halves_resolution_and_keeps_feature_length(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "frame.jpg"
    Image.fromarray(rng.integers(0, 255, (540, 720, 3), dtype=np.uint8)).save(path)
    rgb = load_image_draft(path)
    assert rgb.shape == (270, 360, 3)
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0
    assert hog_features(rgb).shape == (5040,)
    with pytest.raises(FileNotFoundError):
        load_image_draft(tmp_path / "missing.jpg")

