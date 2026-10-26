"""Feature extraction from tray images.

Two feature extractors are provided so that they can be compared:

* ``"pixels"`` - the image downsampled to a small RGB thumbnail. A simple
  baseline that keeps colour (meat is red, background is grey) but is
  sensitive to where the tray sits in the frame.
* ``"hog"`` - a Histogram of Oriented Gradients on the greyscale image. It
  describes local edge structure (tray rim, film creases, meat edges), which is
  where packaging defects show up, and is largely invariant to exposure.

Both return a 1-D float64 vector of fixed length for any input image.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from skimage.feature import hog

from .data import EXPECTED_SIZE, ImageRecord

#: Thumbnail size (width, height) for the pixel features. Keeps the 4:3 aspect.
PIXEL_SIZE = (48, 36)

#: Working size (width, height) for HOG. 16-px cells on 240x180 give 15x11 cells.
HOG_SIZE = (240, 180)


def load_image(path: str | Path, exposure: float = 1.0) -> np.ndarray:
    """Load an image as an RGB float array in [0, 1].

    Args:
        path: Path to an image file readable by Pillow.
        exposure: Multiplicative brightness factor, used for robustness tests.
            1.0 leaves the image unchanged. Values are clipped to [0, 1].

    Returns:
        Array of shape ``(H, W, 3)`` and dtype float64.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``exposure`` is not positive.
    """
    if exposure <= 0:
        raise ValueError(f"exposure must be positive, got {exposure}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        if im.size != EXPECTED_SIZE:
            # The method still works (everything is resized below) but the
            # input is outside the domain the model was validated on.
            warnings.warn(f"{path.name} is {im.size}, expected {EXPECTED_SIZE}", stacklevel=2)
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    return np.clip(rgb * exposure, 0.0, 1.0)


def _resize(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an RGB float image with area averaging (anti-aliased)."""
    im = Image.fromarray((rgb * 255).round().astype(np.uint8))
    return np.asarray(im.resize(size, Image.Resampling.BOX), dtype=np.float64) / 255.0


def pixel_features(rgb: np.ndarray) -> np.ndarray:
    """Downsampled RGB thumbnail, flattened.

    Args:
        rgb: Image of shape ``(H, W, 3)`` with values in [0, 1].

    Returns:
        Vector of length ``48 * 36 * 3 = 5184``.
    """
    _check_rgb(rgb)
    return _resize(rgb, PIXEL_SIZE).ravel()


def hog_features(rgb: np.ndarray) -> np.ndarray:
    """Histogram of Oriented Gradients of the greyscale image.

    Uses 9 orientation bins, 16x16-pixel cells and 2x2-cell blocks with L2-Hys
    normalisation (Dalal & Triggs, 2005). Block normalisation is what makes the
    descriptor robust to exposure changes.

    Args:
        rgb: Image of shape ``(H, W, 3)`` with values in [0, 1].

    Returns:
        Vector of length ``10 * 14 * 2 * 2 * 9 = 5040``.
    """
    _check_rgb(rgb)
    grey = _resize(rgb, HOG_SIZE) @ np.array([0.299, 0.587, 0.114])
    return hog(grey, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True)


EXTRACTORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "pixels": pixel_features,
    "hog": hog_features,
}


def extract(records: list[ImageRecord], method: str, exposure: float = 1.0,
            cache_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels for a list of records.

    Args:
        records: Images to process.
        method: Key of :data:`EXTRACTORS` (``"pixels"`` or ``"hog"``).
        exposure: Brightness factor passed to :func:`load_image`.
        cache_dir: If given, results are cached as ``.npz`` files keyed by the
            method, exposure and the list of file names, so repeated runs skip
            image decoding.

    Returns:
        ``(X, y)`` with ``X`` of shape ``(n_images, n_features)`` and ``y`` of
        shape ``(n_images,)`` containing 0/1 labels.

    Raises:
        KeyError: If ``method`` is not a known extractor.
    """
    if method not in EXTRACTORS:
        raise KeyError(f"Unknown feature method '{method}'; choose from {list(EXTRACTORS)}")
    y = np.array([r.label for r in records], dtype=np.float64)

    cache_file = None
    if cache_dir is not None:
        # hashlib (not hash()) so the key is identical across Python sessions.
        names = "\n".join(r.path.name for r in records).encode()
        key = hashlib.sha1(names).hexdigest()[:12]
        cache_file = Path(cache_dir) / f"{method}_{exposure:g}_{key}.npz"
        if cache_file.is_file():
            return np.load(cache_file)["X"], y

    fn = EXTRACTORS[method]
    X = np.stack([fn(load_image(r.path, exposure)) for r in records])

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, X=X)
    return X, y


def _check_rgb(rgb: np.ndarray) -> None:
    """Raise ValueError unless ``rgb`` is an ``(H, W, 3)`` array in [0, 1]."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) image, got shape {rgb.shape}")
    if rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Image values must lie in [0, 1]")
