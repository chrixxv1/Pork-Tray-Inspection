"""End-to-end classifier: PCA (with whitening), then logistic regression."""

from __future__ import annotations

import numpy as np

from .logistic import LogisticRegression
from .pca import PCA


class TrayClassifier:
    """PCA + logistic-regression pipeline for tray-level pass/fail.

    Args:
        n_components: Number of principal components kept.
        l2: Ridge penalty of the logistic regression.
        solver: Logistic-regression solver, ``"newton"`` or ``"gd"``.
        pca_solver: PCA solver, ``"full"`` or ``"randomized"``.
        max_iter: Maximum solver iterations.

    Attributes:
        pca_: Fitted :class:`~porkqc.pca.PCA`.
        clf_: Fitted :class:`~porkqc.logistic.LogisticRegression`.
    """

    def __init__(self, n_components: int = 100, l2: float = 1e-2, solver: str = "newton",
                 pca_solver: str = "full", max_iter: int = 100) -> None:
        self.n_components = n_components
        self.l2 = l2
        self.solver = solver
        self.pca_solver = pca_solver
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrayClassifier":
        """Fit PCA on ``X`` and the classifier on the whitened PCA scores.

        Args:
            X: Raw features, shape ``(n_samples, n_features)``.
            y: Labels in {0, 1}.

        Returns:
            The fitted instance.
        """
        self.pca_ = PCA(self.n_components, whiten=True, solver=self.pca_solver).fit(X)
        self.clf_ = LogisticRegression(self.l2, self.solver, max_iter=self.max_iter)
        self.clf_.fit(self.pca_.transform(X), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``P(FAIL)`` for each row of ``X``."""
        return self.clf_.predict_proba(self.pca_.transform(X))
