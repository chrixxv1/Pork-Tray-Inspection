"""Tests of the PCA + logistic-regression pipeline."""

import numpy as np
from sklearn.metrics import roc_auc_score

from porkqc.model import TrayClassifier


def _signal_data(signal_scale):
    """300 samples, 50 features; the label depends only on feature 0."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 50))
    X[:, 0] *= signal_scale
    y = (X[:, 0] + 0.3 * signal_scale * rng.standard_normal(300) > 0).astype(float)
    return X, y


def test_pipeline_learns_a_high_variance_signal():
    X, y = _signal_data(signal_scale=5.0)
    clf = TrayClassifier(n_components=10, l2=1e-3).fit(X[:200], y[:200])
    assert roc_auc_score(y[200:], clf.predict_proba(X[200:])) > 0.9


def test_pca_can_discard_a_low_variance_signal():
    """Documents a known limitation: PCA is unsupervised.

    When the label depends on a direction with no more variance than the
    noise, truncating to a few components can discard it, and the classifier
    falls to near-chance. A supervised projection or more components would be
    needed in that case.
    """
    X, y = _signal_data(signal_scale=1.0)
    truncated = TrayClassifier(n_components=10, l2=1e-3).fit(X[:200], y[:200])
    complete = TrayClassifier(n_components=50, l2=1e-3).fit(X[:200], y[:200])
    assert roc_auc_score(y[200:], complete.predict_proba(X[200:])) > 0.9
    assert roc_auc_score(y[200:], truncated.predict_proba(X[200:])) < 0.75


def test_newton_and_gd_pipelines_agree():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 30))
    y = (rng.random(200) < 0.5).astype(float)
    a = TrayClassifier(5, l2=1e-1, solver="newton").fit(X, y).predict_proba(X)
    b = TrayClassifier(5, l2=1e-1, solver="gd", max_iter=5000).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b, atol=1e-4)
