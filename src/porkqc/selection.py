"""Grouped cross-validation for model selection.

The training images are frames from conveyor video, and each frame appears
three times (exposure-augmented copies). A random K-fold split would put copies
of the same frame - and neighbouring frames showing the same tray - on both
sides of a fold, so validation scores would be optimistic (target leakage).

Following the grouped-split practice for correlated records, frames are
grouped into blocks of consecutive frame numbers and whole blocks are assigned
to folds (``sklearn.model_selection.GroupKFold``). Augmented copies share a
frame number, so they always fall in the same group. Different videos reuse
frame numbers, so a block may contain frames from several videos; that only
makes the grouping more conservative.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
from sklearn.model_selection import GroupKFold

from .data import ImageRecord


def frame_number(record: ImageRecord) -> int:
    """Frame number parsed from the source name, e.g. ``frame_with_red_8214.jpg`` -> 8214.

    Args:
        record: Image record.

    Returns:
        The frame number.

    Raises:
        ValueError: If the source name contains no number.
    """
    m = re.search(r"(\d+)(?!.*\d)", record.source_name)
    if m is None:
        raise ValueError(f"No frame number in {record.source_name!r}")
    return int(m.group(1))


def frame_block_groups(records: list[ImageRecord], block: int = 100) -> np.ndarray:
    """Group label for each record: its frame number divided by ``block``.

    Args:
        records: Image records.
        block: Number of consecutive frame numbers per group.

    Returns:
        Integer array of shape ``(len(records),)``.

    Raises:
        ValueError: If ``block`` is less than 1.
    """
    if block < 1:
        raise ValueError("block must be at least 1")
    return np.array([frame_number(r) // block for r in records])


def grouped_cv_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                      score: Callable[[np.ndarray, np.ndarray], float],
                      n_splits: int = 5) -> np.ndarray:
    """Score a model with grouped K-fold cross-validation.

    Args:
        X: Features, shape ``(n, d)``.
        y: Labels, shape ``(n,)``.
        groups: Group label per row; no group is split across folds.
        fit_predict: ``fit_predict(X_train, y_train, X_test) -> scores`` fits a
            model on the training rows and returns scores for the test rows.
        score: Metric ``score(y_true, scores) -> float``.
        n_splits: Number of folds.

    Returns:
        Array of ``n_splits`` fold scores.
    """
    out = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        out.append(score(y[te], fit_predict(X[tr], y[tr], X[te])))
    return np.array(out)
