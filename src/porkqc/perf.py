"""Performance helpers: timing, FLOP and arithmetic-intensity estimates, fast decoding.

FLOP counts are analytic estimates that count a multiply and an add as two
floating-point operations. Arithmetic intensity (AI) is FLOPs per byte moved
between memory and the processor, as in a roofline analysis: kernels with low AI
are limited by memory bandwidth, kernels with high AI by peak arithmetic rate.
All byte counts assume float64 (8 bytes) and that each array is read from
memory once per pass, i.e. no reuse from cache.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

BYTES_PER_FLOAT = 8


def time_best(fn: Callable[[], object], repeats: int = 5) -> float:
    """Best (minimum) wall-clock time of ``fn()`` over ``repeats`` runs, in seconds.

    The minimum is used, as ``timeit`` does, because other processes can only
    make a run slower, never faster.

    Args:
        fn: Function with no arguments.
        repeats: Number of runs.

    Returns:
        The shortest run time in seconds.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def gd_iteration(n: int, p: int) -> tuple[float, float]:
    """FLOPs and bytes of one gradient-descent iteration.

    Two matrix-vector products with the ``n x p`` matrix ``A`` (``A theta`` and
    ``A^T r``), each ``2 n p`` FLOPs, each reading ``A`` once.

    Args:
        n: Number of samples.
        p: Number of parameters (features + 1).

    Returns:
        ``(flops, bytes)``.
    """
    return 4.0 * n * p, 2.0 * n * p * BYTES_PER_FLOAT


def hessian_formation(n: int, p: int) -> tuple[float, float]:
    """FLOPs and bytes of forming ``H = A^T diag(s) A`` for an ``n x p`` matrix.

    Scaling the rows costs ``n p``; the product ``(A * s)^T A`` costs about
    ``2 n p^2``. ``A`` is read twice and the scaled copy written once.

    Args:
        n: Number of samples.
        p: Number of parameters.

    Returns:
        ``(flops, bytes)``.
    """
    return 2.0 * n * p**2 + n * p, 3.0 * n * p * BYTES_PER_FLOAT


def newton_iteration(n: int, p: int) -> tuple[float, float]:
    """FLOPs and bytes of one Newton iteration.

    Gradient (``4 n p``), Hessian formation (``2 n p^2 + n p``), LU
    factorisation (``2 p^3 / 3``) and the two triangular solves (``2 p^2``).

    Args:
        n: Number of samples.
        p: Number of parameters.

    Returns:
        ``(flops, bytes)``.
    """
    gf, gb = gd_iteration(n, p)
    hf, hb = hessian_formation(n, p)
    return gf + hf + 2.0 * p**3 / 3 + 2.0 * p**2, gb + hb


def randomized_svd_flops(n: int, d: int, r: int, q: int) -> float:
    """FLOPs of the randomized SVD of an ``n x d`` matrix with ``r`` columns and ``q`` power iterations.

    There are ``2q + 2`` products with the data matrix (``2 n d r`` each) plus
    Householder QR factorisations of tall ``n x r`` and ``d x r`` matrices
    (about ``4 m r^2`` each): ``q + 1`` of the first kind and ``q`` of the second.

    Args:
        n: Rows of the data matrix.
        d: Columns of the data matrix.
        r: Number of sampled directions (components + oversampling).
        q: Number of power iterations.

    Returns:
        Estimated FLOPs.
    """
    return 2.0 * n * d * r * (2 * q + 2) + 4.0 * ((q + 1) * n + q * d) * r**2


def full_svd_flops(n: int, d: int) -> float:
    """Lower-bound FLOPs of a thin SVD (singular values and right vectors).

    Uses ``4 m k^2 + 8 k^3`` with ``m = max(n, d)`` and ``k = min(n, d)``
    (Golub & Van Loan, 2013, ch. 8); forming the left vectors as well, as
    ``numpy.linalg.svd`` does, costs more.

    Args:
        n: Rows.
        d: Columns.

    Returns:
        Estimated FLOPs.
    """
    m, k = max(n, d), min(n, d)
    return 4.0 * m * k**2 + 8.0 * k**3


def arithmetic_intensity(flops: float, nbytes: float) -> float:
    """Arithmetic intensity in FLOP per byte.

    Args:
        flops: Floating-point operations.
        nbytes: Bytes moved.

    Returns:
        ``flops / nbytes``.
    """
    if nbytes <= 0:
        raise ValueError("nbytes must be positive")
    return flops / nbytes


def load_image_draft(path: str | Path, size: tuple[int, int] = (360, 270)) -> np.ndarray:
    """Load a JPEG at reduced resolution using the decoder's DCT scaling.

    ``Image.draft`` asks the JPEG decoder to skip detail it would only throw
    away: decoding at 1/2 scale (or 1/4, 1/8) is much cheaper than decoding at
    full size and resizing afterwards. HOG features are computed at 240 x 180,
    so decoding at 360 x 270 discards nothing that the features use.

    Args:
        path: Path to a JPEG file.
        size: Requested (width, height); the decoder picks the nearest scale
            that is at least this large.

    Returns:
        RGB array in [0, 1] of shape ``(H, W, 3)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        im.draft("RGB", size)
        return np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0

