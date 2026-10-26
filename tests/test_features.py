"""Tests of image loading and feature extraction."""

import numpy as np
import pytest
from PIL import Image

from porkqc.data import load_split
from porkqc.features import extract, hog_features, load_image, pixel_features


@pytest.fixture
def textured_image(tmp_path):
    """Save a 720x540 random-noise RGB image and return its path.

    Random texture gives HOG plenty of gradients to describe, unlike the
    uniform grey images in the tiny_coco fixture.
    """
    rng = np.random.default_rng(0)
    path = tmp_path / "noise.jpg"
    Image.fromarray(rng.integers(0, 255, (540, 720, 3), dtype=np.uint8)).save(path)
    return path


def test_load_image_range_and_shape(textured_image):
    rgb = load_image(textured_image)
    assert rgb.shape == (540, 720, 3)
    assert rgb.dtype == np.float64
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0


def test_exposure_scales_and_clips(textured_image):
    base = load_image(textured_image)
    darker = load_image(textured_image, exposure=0.5)
    brighter = load_image(textured_image, exposure=3.0)
    np.testing.assert_allclose(darker, base * 0.5)
    assert brighter.max() == 1.0  # clipped, not above 1


def test_feature_lengths(textured_image):
    rgb = load_image(textured_image)
    assert pixel_features(rgb).shape == (48 * 36 * 3,)
    assert hog_features(rgb).shape == (5040,)


def test_features_are_deterministic(textured_image):
    rgb = load_image(textured_image)
    np.testing.assert_array_equal(hog_features(rgb), hog_features(rgb))


def test_hog_is_more_robust_to_exposure_than_pixels(textured_image):
    base = load_image(textured_image)
    dark = load_image(textured_image, exposure=0.8)

    def relative_change(f):
        return np.linalg.norm(f(dark) - f(base)) / np.linalg.norm(f(base))

    assert relative_change(hog_features) < 0.5 * relative_change(pixel_features)


def test_extract_aligns_features_with_labels_and_caches(tiny_coco, tmp_path):
    records = load_split(tiny_coco)
    X, y = extract(records, "pixels", cache_dir=tmp_path / "cache")
    assert X.shape == (3, 48 * 36 * 3)
    assert y.tolist() == [0, 1, 1]
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
    X2, _ = extract(records, "pixels", cache_dir=tmp_path / "cache")  # from cache
    np.testing.assert_array_equal(X, X2)


def test_wrong_image_size_warns(tmp_path):
    Image.new("RGB", (100, 100)).save(tmp_path / "small.jpg")
    with pytest.warns(UserWarning, match="expected"):
        load_image(tmp_path / "small.jpg")


@pytest.mark.parametrize("bad", [np.zeros((10, 10)), np.full((10, 10, 3), 2.0)])
def test_invalid_arrays_rejected(bad):
    with pytest.raises(ValueError):
        hog_features(bad)


def test_invalid_arguments_rejected(tiny_coco):
    with pytest.raises(KeyError):
        extract(load_split(tiny_coco), "sift")
    with pytest.raises(ValueError):
        load_image(tiny_coco / "frame_0.jpg", exposure=0)
    with pytest.raises(FileNotFoundError):
        load_image(tiny_coco / "missing.jpg")
