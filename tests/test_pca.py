"""Verification of PCA: algebraic identities, reference implementation, randomized solver."""

import numpy as np
import pytest
from sklearn.decomposition import PCA as SkPCA

from porkqc.pca import PCA, randomized_svd


@pytest.fixture
def low_rank():
    """60x200 matrix of rank 8 plus small noise, with decaying spectrum."""
    rng = np.random.default_rng(0)
    U = np.linalg.qr(rng.standard_normal((60, 8)))[0]
    V = np.linalg.qr(rng.standard_normal((200, 8)))[0]
    s = 10.0 ** -np.arange(0, 8) * 0.5 + 1
    return U * s * 20 @ V.T + 0.01 * rng.standard_normal((60, 200))


def test_components_are_orthonormal(low_rank):
    p = PCA(10).fit(low_rank)
    np.testing.assert_allclose(p.components_ @ p.components_.T, np.eye(10), atol=1e-12)


def test_full_rank_reconstruction_is_exact(low_rank):
    p = PCA(60, whiten=False).fit(low_rank)
    np.testing.assert_allclose(p.inverse_transform(p.transform(low_rank)), low_rank, atol=1e-10)


def test_whitened_scores_have_unit_variance(low_rank):
    Z = PCA(5, whiten=True).fit(low_rank).transform(low_rank)
    np.testing.assert_allclose(Z.var(axis=0, ddof=1), 1.0, rtol=1e-10)


def test_matches_scikit_learn(low_rank):
    ours = PCA(6, whiten=False).fit(low_rank)
    ref = SkPCA(6, svd_solver="full").fit(low_rank)
    np.testing.assert_allclose(ours.explained_variance_ratio_, ref.explained_variance_ratio_, rtol=1e-10)
    # Components agree up to sign.
    np.testing.assert_allclose(np.abs(ours.components_ @ ref.components_.T), np.eye(6), atol=1e-8)


def test_randomized_solver_recovers_leading_subspace(low_rank):
    full = PCA(8, solver="full").fit(low_rank)
    rand = PCA(8, solver="randomized").fit(low_rank)
    np.testing.assert_allclose(rand.singular_values_, full.singular_values_, rtol=1e-6)
    # Cosines of principal angles between the two subspaces are all ~1.
    cosines = np.linalg.svd(full.components_ @ rand.components_.T, compute_uv=False)
    assert cosines.min() > 1 - 1e-8


def test_randomized_svd_without_power_iterations_is_less_accurate():
    # A slowly decaying spectrum is the hard case the power iterations address.
    rng = np.random.default_rng(1)
    A = rng.standard_normal((300, 300)) @ np.diag(1 / np.sqrt(np.arange(1, 301)))
    exact = np.linalg.svd(A, compute_uv=False)[:10]
    err0 = np.abs(randomized_svd(A, 10, n_power_iter=0)[0] - exact).max()
    err4 = np.abs(randomized_svd(A, 10, n_power_iter=4)[0] - exact).max()
    assert err4 < err0


def test_invalid_inputs_are_rejected(low_rank):
    with pytest.raises(ValueError):
        PCA(0)
    with pytest.raises(ValueError):
        PCA(61).fit(low_rank)
    with pytest.raises(ValueError):
        PCA(2).fit(np.array([[np.nan, 1.0], [1.0, 2.0], [0.0, 0.0]]))
    p = PCA(2).fit(low_rank)
    with pytest.raises(ValueError, match="features"):
        p.transform(low_rank[:, :10])
