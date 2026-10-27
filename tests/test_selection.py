"""Tests of frame grouping and grouped cross-validation."""

from pathlib import Path

import numpy as np
import pytest

from porkqc.data import ImageRecord
from porkqc.evaluate import roc_auc
from porkqc.selection import frame_block_groups, frame_number, grouped_cv_scores


def _rec(name):
    return ImageRecord(Path(name), name, 0, ())


def test_frame_number_and_blocks():
    recs = [_rec("frame_with_red_8214.jpg"), _rec("frame_with_red_8299.jpg"),
            _rec("frame_with_red_8300.jpg"), _rec("frame_with_red_3.jpg")]
    assert frame_number(recs[0]) == 8214
    assert frame_block_groups(recs, block=100).tolist() == [82, 82, 83, 0]


def test_bad_names_and_block_rejected():
    with pytest.raises(ValueError):
        frame_number(_rec("no_number_here.jpg"))
    with pytest.raises(ValueError):
        frame_block_groups([_rec("frame_1.jpg")], block=0)


def test_no_group_is_split_across_folds():
    rng = np.random.default_rng(0)
    n = 300
    groups = np.repeat(np.arange(30), 10)
    X, y = rng.standard_normal((n, 3)), rng.integers(0, 2, n).astype(float)
    seen = []

    def fit_predict(Xtr, ytr, Xte):
        seen.append(len(Xte))
        return Xte[:, 0]

    scores = grouped_cv_scores(X, y, groups, fit_predict, roc_auc, n_splits=5)
    assert len(scores) == 5 and sum(seen) == n

    from sklearn.model_selection import GroupKFold
    for tr, te in GroupKFold(5).split(X, y, groups):
        assert not set(groups[tr]) & set(groups[te])


def test_grouping_removes_leakage_from_duplicates():
    """With duplicated rows, random folds reward memorisation; grouped folds do not."""
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(1)
    X0 = rng.standard_normal((100, 5))
    y0 = rng.integers(0, 2, 100).astype(float)  # labels are pure noise
    X, y = np.repeat(X0, 3, axis=0), np.repeat(y0, 3)  # three copies, like augmentation
    groups = np.repeat(np.arange(100), 3)

    def fit_predict(Xtr, ytr, Xte):
        return KNeighborsClassifier(1).fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    grouped = grouped_cv_scores(X, y, groups, fit_predict, roc_auc).mean()
    random = np.mean([roc_auc(y[te], fit_predict(X[tr], y[tr], X[te]))
                      for tr, te in KFold(5, shuffle=True, random_state=0).split(X)])
    assert random > 0.9          # leakage: copies of the test row are in training
    assert abs(grouped - 0.5) < 0.15   # honest: noise labels give chance-level AUC
