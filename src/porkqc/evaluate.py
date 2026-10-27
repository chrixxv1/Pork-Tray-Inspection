"""Evaluation metrics for the pass/fail decision, with bootstrap uncertainty.

Conventions: the positive class is FAIL (a defective tray), so

* **recall** is the fraction of defective trays that are caught, and
* **precision** is the fraction of rejected trays that really are defective.

On a production line a missed defect (false negative) reaches a customer,
while a false reject costs a manual re-check, so recall is the primary
operating constraint and the threshold is chosen to meet a target recall.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import rankdata


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    AUC equals the probability that a randomly chosen FAIL image scores higher
    than a randomly chosen PASS image; ties count one half.

    Args:
        y: Labels in {0, 1}.
        score: Any score that increases with the probability of FAIL.

    Returns:
        AUC in [0, 1], or NaN if only one class is present.
    """
    y = np.asarray(y)
    score = np.asarray(score)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Midranks (average rank for tied scores) handle ties exactly.
    ranks = rankdata(np.concatenate([pos, neg]))
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def binary_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Confusion-matrix metrics at a given decision threshold.

    Args:
        y: Labels in {0, 1}.
        prob: Predicted ``P(FAIL)``.
        threshold: Images with ``prob >= threshold`` are rejected (FAIL).

    Returns:
        Dictionary with accuracy, balanced accuracy, precision, recall,
        specificity, F1, AUC and the raw counts tp, fp, tn, fn.
    """
    y = np.asarray(y).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": roc_auc(y, np.asarray(prob)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def threshold_for_recall(y: np.ndarray, prob: np.ndarray, target_recall: float) -> float:
    """Highest threshold whose recall on ``(y, prob)`` is at least ``target_recall``.

    Intended for the validation split only; the chosen threshold is then
    applied unchanged to the test split.

    Args:
        y: Labels in {0, 1}.
        prob: Predicted ``P(FAIL)``.
        target_recall: Required recall in (0, 1].

    Returns:
        The decision threshold.

    Raises:
        ValueError: If ``target_recall`` is outside (0, 1].
    """
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must lie in (0, 1]")
    pos = np.sort(np.asarray(prob)[np.asarray(y) == 1])[::-1]
    k = int(np.ceil(target_recall * len(pos)))  # need the top-k positives above threshold
    return float(pos[k - 1])


def bootstrap_ci(y: np.ndarray, prob: np.ndarray,
                 metric: Callable[[np.ndarray, np.ndarray], float],
                 n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a metric.

    Resamples test images with replacement. This captures sampling
    uncertainty from the small test set, but not correlation between nearby
    video frames, so the interval is likely optimistic.

    Args:
        y: Labels.
        prob: Predicted probabilities.
        metric: Function ``metric(y, prob) -> float``.
        n_boot: Number of bootstrap resamples.
        level: Confidence level.
        seed: Random seed.

    Returns:
        ``(lower, upper)`` bounds; resamples where the metric is NaN are skipped.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = metric(y[idx], prob[idx])
        if np.isfinite(v):
            vals.append(v)
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))
