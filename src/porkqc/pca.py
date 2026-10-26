"""Principal Component Analysis computed with the thin singular value decomposition.

Given a centred data matrix ``Xc = X - mean`` of shape ``(n, d)``, the thin SVD
``Xc = U S V^T`` gives the principal directions as the rows of ``V^T`` and the
variance along each as ``S**2 / (n - 1)``.

Why SVD rather than an eigendecomposition of the covariance ``Xc^T Xc``:
forming ``Xc^T Xc`` squares the condition number of the problem, so small
singular values lose about twice as many significant digits. The SVD works on
``Xc`` directly and is backward stable (Golub & Van Loan, 2013, ch. 8).

Two solvers are provided:

* ``"full"`` - LAPACK thin SVD (``numpy.linalg.svd``). For an ``m x k`` matrix
  (``m >= k``) this costs on the order of ``m k**2`` flops, with a constant
  between about 4 and 14 depending on which factors are formed (Golub & Van
  Loan, 2013, ch. 8). Here ``k = min(n, d) = 3349``, so all ~3349 components
  are computed even though only a few hundred are kept.
* ``"randomized"`` - the randomized range finder of Halko, Martinsson & Tropp
  (2011). It computes only the leading ``r = n_components + oversample``
  components at a cost of order ``n d r`` flops per pass over the data, which
  is much cheaper when ``r << min(n, d)``. Its error is controlled by the
  oversampling and the number of power iterations, and it is checked against
  the full solver in the tests.
"""

from __future__ import annotations

import numpy as np


class PCA:
    """Principal Component Analysis with optional whitening.

    Args:
        n_components: Number of principal components to keep.
        whiten: If True, scale each component score to unit variance. This
            puts all features on the same scale for the downstream classifier,
            which improves the conditioning of its Hessian.
        solver: ``"full"`` (exact thin SVD) or ``"randomized"`` (Halko et al.).
        oversample: Extra random directions used by the randomized solver.
        n_power_iter: Power iterations used by the randomized solver. The
            error decays like ``(s[k+p] / s[k]) ** (2q + 1)``, so a slowly
            decaying spectrum needs more iterations (examined in Phase 7).
        seed: Seed for the randomized solver's test matrix.

    Attributes:
        mean_: Per-feature mean of the training data, shape ``(d,)``.
        components_: Principal directions, shape ``(n_components, d)``, with
            orthonormal rows.
        singular_values_: Singular values of the centred data, shape
            ``(n_components,)``.
        explained_variance_ratio_: Fraction of total variance captured by each
            kept component.
    """

    def __init__(self, n_components: int, whiten: bool = True, solver: str = "full",
                 oversample: int = 10, n_power_iter: int = 10, seed: int = 0) -> None:
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if solver not in ("full", "randomized"):
            raise ValueError("solver must be 'full' or 'randomized'")
        self.n_components = n_components
        self.whiten = whiten
        self.solver = solver
        self.oversample = oversample
        self.n_power_iter = n_power_iter
        self.seed = seed

    def fit(self, X: np.ndarray) -> "PCA":
        """Compute the principal components of ``X``.

        Args:
            X: Training data, shape ``(n_samples, n_features)``.

        Returns:
            The fitted instance (to allow chaining).

        Raises:
            ValueError: If ``X`` is not 2-D, contains non-finite values, or has
                fewer samples or features than ``n_components``.
        """
        X = _check_matrix(X)
        n, d = X.shape
        if self.n_components > min(n, d):
            raise ValueError(f"n_components={self.n_components} exceeds min(n, d)={min(n, d)}")

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # Total variance from the Frobenius norm, so it is exact for both
        # solvers (the randomized solver only returns the leading values).
        total_var = np.einsum("ij,ij->", Xc, Xc)
        if self.solver == "full":
            # full_matrices=False is the thin SVD: Vt is (min(n, d), d).
            _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        else:
            s, vt = randomized_svd(Xc, self.n_components, self.oversample,
                                   self.n_power_iter, self.seed)
        # Fix the sign of each component (SVD is only unique up to sign) so
        # results are reproducible across LAPACK builds.
        signs = np.sign(vt[np.arange(vt.shape[0]), np.abs(vt).argmax(axis=1)])
        vt *= signs[:, None]

        k = self.n_components
        self.components_ = vt[:k]
        self.singular_values_ = s[:k]
        self.explained_variance_ = s[:k] ** 2 / (n - 1)
        self.explained_variance_ratio_ = s[:k] ** 2 / total_var
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto the principal components.

        Args:
            X: Data of shape ``(n_samples, n_features)``.

        Returns:
            Component scores of shape ``(n_samples, n_components)``.
        """
        X = _check_matrix(X)
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"Expected {self.mean_.shape[0]} features, got {X.shape[1]}")
        Z = (X - self.mean_) @ self.components_.T
        if self.whiten:
            Z /= np.sqrt(self.explained_variance_)
        return Z

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map component scores back to feature space (approximate reconstruction).

        Args:
            Z: Scores of shape ``(n_samples, n_components)``.

        Returns:
            Reconstructed data of shape ``(n_samples, n_features)``.
        """
        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_)
        return Z @ self.components_ + self.mean_


def randomized_svd(A: np.ndarray, rank: int, oversample: int = 10,
                   n_power_iter: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Leading singular values and right singular vectors by random projection.

    Implements Algorithms 4.4 and 5.1 of Halko, Martinsson & Tropp (2011):
    sample the range of ``A`` with a Gaussian test matrix, refine it with power
    iterations (re-orthonormalising after every multiplication to avoid the
    loss of accuracy from repeated powering), then take the SVD of the small
    projected matrix.

    Args:
        A: Matrix of shape ``(m, n)``.
        rank: Number of singular triplets wanted.
        oversample: Extra sample directions; 5-10 is the usual recommendation.
        n_power_iter: Number of power iterations.
        seed: Random seed for the test matrix.

    Returns:
        ``(s, vt)`` with ``s`` of shape ``(rank,)`` and ``vt`` of shape
        ``(rank, n)``.
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    r = min(rank + oversample, m, n)
    Q, _ = np.linalg.qr(A @ rng.standard_normal((n, r)))
    for _ in range(n_power_iter):
        Q, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Q)
    B = Q.T @ A  # (r, n): small enough for a dense SVD
    _, s, vt = np.linalg.svd(B, full_matrices=False)
    return s[:rank], vt[:rank]


def _check_matrix(X: np.ndarray) -> np.ndarray:
    """Return ``X`` as a float64 2-D array, raising ValueError if invalid."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got {X.ndim}-D")
    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinite values")
    return X
