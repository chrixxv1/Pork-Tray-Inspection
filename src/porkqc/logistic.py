"""L2-regularised logistic regression trained with Newton's method or gradient descent.

The model predicts ``P(FAIL | x) = sigmoid(w . x + b)``. Training minimises

    f(theta) = (1/n) sum_i [ log(1 + exp(z_i)) - y_i z_i ] + (lambda/2) ||w||^2,
    z_i = w . x_i + b,   theta = [w, b],

the mean negative log-likelihood plus a ridge penalty (the intercept ``b`` is
not penalised). ``f`` is convex, and strictly convex for ``lambda > 0``, so its
minimiser is the unique root of the gradient, ``grad f(theta) = 0``.

With ``p = sigmoid(z)`` and the augmented matrix ``A = [X, 1]``:

    gradient  g(theta) = A^T (p - y) / n + lambda [w, 0]
    Hessian   H(theta) = A^T diag(p (1 - p)) A / n + lambda diag(1, ..., 1, 0)

Solvers:

* ``"newton"`` - Newton's method for the root-finding problem ``g(theta) = 0``.
  It is the multivariate form of the scalar update ``s <- s - F(s) / F'(s)``
  used in the shooting method: here ``F`` is the gradient and ``F'`` is the
  Hessian, so each iteration solves the linear system ``H dtheta = -g`` by LU
  factorisation (``scipy.linalg.lu_factor`` / ``lu_solve``). Near the root the
  convergence is quadratic. As a safeguard against overshooting far from the
  root, a step that increases ``f`` is halved until it does not.
* ``"gd"`` - gradient descent, ``theta <- theta - h g(theta)``. This is the
  explicit (forward) Euler method applied to the gradient-flow ODE
  ``d theta / dt = -g(theta)``. Near the minimum ``g`` is approximately
  ``H (theta - theta*)``, so the error is multiplied by ``(I - h H)`` each step
  and the iteration is stable only if ``|1 - h lambda_i| < 1`` for every
  eigenvalue ``lambda_i`` of ``H``, i.e. ``0 < h < 2 / lambda_max`` - the same
  limit as explicit Euler for ``y' = -lambda y``. Convergence is linear, at a
  rate set by the condition number of ``H``. It is the baseline solver.

Both solvers stop when the residual ``||g(theta)||_inf <= tol``, the same kind
of stopping rule as the shooting method (``|F(s)| < tol``) and the relaxation
methods (``||A T - b||_inf``). For a strongly convex ``f`` this residual also
bounds the parameter error: ``||theta - theta*||_2 <= ||g||_2 / mu``, where
``mu`` is the smallest eigenvalue of ``H``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import lu_factor, lu_solve


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    The naive ``1 / (1 + exp(-z))`` overflows ``exp`` for large negative ``z``
    (float64 overflows above about 1.8e308, i.e. ``z < -709``). Evaluating
    separately on each sign keeps every ``exp`` argument non-positive, so it
    can only underflow harmlessly to 0.

    Args:
        z: Array of logits.

    Returns:
        Array of the same shape with values in [0, 1].
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_loss(y: np.ndarray, z: np.ndarray) -> float:
    """Mean negative log-likelihood computed from logits.

    Uses ``log(1 + exp(z)) = logaddexp(0, z)``, which never overflows, instead
    of ``log(sigmoid(z))``, which returns ``-inf`` once ``sigmoid`` rounds to 0.

    Args:
        y: Labels in {0, 1}.
        z: Logits, same shape as ``y``.

    Returns:
        The mean loss (a non-negative float).
    """
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


@dataclass
class FitHistory:
    """Per-iteration record of an optimisation run.

    Attributes:
        loss: Objective value after each iteration (index 0 is the start).
        grad_norm: Residual ``||g||_inf`` after each iteration.
        time_s: Wall-clock time since the start of ``fit`` at each iteration.
        converged: Whether ``||g||_inf <= tol`` was reached within ``max_iter``.
    """

    loss: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    time_s: list[float] = field(default_factory=list)
    converged: bool = False


class LogisticRegression:
    """Binary L2-regularised logistic regression.

    Args:
        l2: Ridge penalty ``lambda`` (must be positive).
        solver: ``"newton"`` or ``"gd"``.
        tol: Stopping tolerance on the residual ``||grad f||_inf``.
        max_iter: Maximum number of iterations.
        step: Gradient-descent step size ``h``. If None, ``h = 1 / lambda_max``
            of the Hessian at the starting point, half the stability limit.
            Ignored by Newton's method.

    Attributes:
        coef_: Weights ``w``, shape ``(n_features,)``.
        intercept_: Bias ``b``.
        history_: :class:`FitHistory` of the last call to :meth:`fit`.
        hessian_cond_: Condition number ``sigma_max / sigma_min`` of the
            Hessian at the solution (Newton only).
        stability_limit_: ``2 / lambda_max`` of the starting Hessian, the
            largest stable gradient-descent step (gradient descent only).
        step_: Step size actually used (gradient descent only).
    """

    def __init__(self, l2: float = 1e-3, solver: str = "newton", tol: float = 1e-8,
                 max_iter: int = 100, step: float | None = None) -> None:
        if l2 <= 0:
            raise ValueError("l2 must be positive so the problem is strictly convex")
        if solver not in ("newton", "gd"):
            raise ValueError("solver must be 'newton' or 'gd'")
        if step is not None and step <= 0:
            raise ValueError("step must be positive")
        self.l2 = l2
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.step = step

    # ------------------------------------------------------------------ helpers
    def _objective(self, A: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        w = theta[:-1]
        return log_loss(y, A @ theta) + 0.5 * self.l2 * float(w @ w)

    def _gradient(self, A: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        n = A.shape[0]
        g = A.T @ (sigmoid(A @ theta) - y) / n
        g[:-1] += self.l2 * theta[:-1]
        return g

    def _hessian(self, A: np.ndarray, theta: np.ndarray) -> np.ndarray:
        n = A.shape[0]
        p = sigmoid(A @ theta)
        s = p * (1.0 - p)
        H = (A * s[:, None]).T @ A / n  # A^T diag(s) A without forming diag(s)
        H[np.diag_indices(A.shape[1] - 1)] += self.l2
        return H

    # --------------------------------------------------------------------- API
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit the model.

        Args:
            X: Features, shape ``(n_samples, n_features)``.
            y: Labels in {0, 1}, shape ``(n_samples,)``.

        Returns:
            The fitted instance.

        Raises:
            ValueError: If shapes disagree, labels are not 0/1, or inputs
                contain non-finite values.
        """
        X, y = _check_xy(X, y)
        A = np.hstack([X, np.ones((X.shape[0], 1))])
        theta = np.zeros(A.shape[1])
        self.history_ = FitHistory()
        t0 = time.perf_counter()

        def record(theta: np.ndarray, g: np.ndarray) -> None:
            self.history_.loss.append(self._objective(A, y, theta))
            self.history_.grad_norm.append(float(np.abs(g).max()))
            self.history_.time_s.append(time.perf_counter() - t0)

        g = self._gradient(A, y, theta)
        record(theta, g)
        if self.solver == "newton":
            theta = self._fit_newton(A, y, theta, g, record)
        else:
            theta = self._fit_gd(A, y, theta, g, record)

        self.coef_ = theta[:-1]
        self.intercept_ = float(theta[-1])
        return self

    def _fit_newton(self, A, y, theta, g, record):
        for _ in range(self.max_iter):
            if np.abs(g).max() <= self.tol:
                self.history_.converged = True
                break
            # Newton update: solve H dtheta = -g by LU factorisation.
            H = self._hessian(A, theta)
            dtheta = lu_solve(lu_factor(H), -g)
            # Safeguard: halve the step while it would increase f.
            f0, t = self._objective(A, y, theta), 1.0
            while self._objective(A, y, theta + t * dtheta) > f0 and t > 1e-10:
                t *= 0.5
            theta = theta + t * dtheta
            g = self._gradient(A, y, theta)
            record(theta, g)
        # Condition number sigma_max / sigma_min (computed via the SVD).
        self.hessian_cond_ = float(np.linalg.cond(self._hessian(A, theta)))
        return theta

    def _fit_gd(self, A, y, theta, g, record):
        # At theta = 0 every p = 1/2, where p(1 - p) takes its maximum of 1/4,
        # so the starting Hessian has the largest curvature anywhere and its
        # top eigenvalue gives a stability limit that holds for every theta.
        lam_max = float(np.linalg.eigvalsh(self._hessian(A, theta))[-1])
        self.stability_limit_ = 2.0 / lam_max
        self.step_ = self.step if self.step is not None else 1.0 / lam_max
        for _ in range(self.max_iter):
            if np.abs(g).max() <= self.tol:
                self.history_.converged = True
                break
            theta = theta - self.step_ * g
            g = self._gradient(A, y, theta)
            record(theta, g)
            if not np.isfinite(self.history_.loss[-1]):
                break  # diverged (unstable step size)
        return theta

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the logits ``w . x + b``, shape ``(n_samples,)``."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.coef_.shape[0]:
            raise ValueError(f"Expected shape (n, {self.coef_.shape[0]}), got {X.shape}")
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``P(FAIL | x)`` for each row of ``X``, shape ``(n_samples,)``."""
        return sigmoid(self.decision_function(X))


def _check_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert training inputs."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if y.shape != (X.shape[0],):
        raise ValueError(f"y must have shape ({X.shape[0]},), got {y.shape}")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("Inputs contain NaN or infinite values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y must contain only 0 and 1")
    return X, y
