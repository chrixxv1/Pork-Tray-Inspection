"""Tests of the evaluation metrics against hand calculations and scikit-learn."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from porkqc.evaluate import binary_metrics, bootstrap_ci, roc_auc, threshold_for_recall


def test_auc_limits_and_ties():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.full(4, 0.5)) == 0.5
    assert np.isnan(roc_auc(np.ones(3), np.ones(3)))


def test_auc_matches_scikit_learn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200).round(2)  # rounding creates ties
    assert np.isclose(roc_auc(y, s), roc_auc_score(y, s))


def test_binary_metrics_hand_example():
    y = np.array([1, 1, 1, 0, 0])
    p = np.array([0.9, 0.6, 0.2, 0.7, 0.1])
    m = binary_metrics(y, p, threshold=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 1)
    assert np.isclose(m["recall"], 2 / 3) and np.isclose(m["precision"], 2 / 3)


def test_threshold_for_recall_meets_target():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 100)
    p = rng.random(100)
    for target in (0.5, 0.9, 1.0):
        t = threshold_for_recall(y, p, target)
        assert binary_metrics(y, p, t)["recall"] >= target
    with pytest.raises(ValueError):
        threshold_for_recall(y, p, 0.0)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 80)
    p = np.clip(y * 0.4 + rng.random(80) * 0.6, 0, 1)
    lo, hi = bootstrap_ci(y, p, roc_auc, n_boot=500)
    assert lo <= roc_auc(y, p) <= hi
