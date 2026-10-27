"""Verification of the logistic-regression solvers.

Four kinds of evidence:
1. the gradient and Hessian are checked against central finite differences;
2. the Newton solution is checked against an established implementation
   (scikit-learn) on the same objective;
3. convergence behaviour matches theory (quadratic for Newton, linear for
   gradient descent, both reaching the same solution);
4. gradient descent obeys the explicit-Euler stability limit h < 2 / lambda_max.
"""

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SkLR

from porkqc.logistic import LogisticRegression, log_loss, sigmoid


def _augmented(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])


# ---------------------------------------------------------------- stability of the formulas
def test_sigmoid_is_stable_for_extreme_logits():
    z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any overflow warning fails the test
        p = sigmoid(z)
    assert np.all(np.isfinite(p))
    assert p[0] == 0.0 and p[-1] == 1.0 and p[2] == 0.5


def test_sigmoid_matches_naive_formula_in_safe_range():
    z = np.linspace(-20, 20, 101)
    np.testing.assert_allclose(sigmoid(z), 1 / (1 + np.exp(-z)), rtol=1e-12)


def test_log_loss_is_finite_for_confident_wrong_predictions():
    # Naive log(sigmoid(z)) would give -inf here.
    assert np.isclose(log_loss(np.array([1.0]), np.array([-800.0])), 800.0)


# ---------------------------------------------------------------- derivatives
# Central differences have truncation error O(h^2) and round-off error
# O(u / h), with u ~ 1.1e-16 for float64, so the total error is smallest near
# h ~ u^(1/3) ~ 5e-6. h = 1e-6 is close to that optimum.
H_FD = 1e-6


def test_gradient_matches_central_finite_differences(logistic_data):
    X, y = logistic_data
    m = LogisticRegression(l2=0.1)
    A = _augmented(X)
    theta = np.random.default_rng(2).standard_normal(A.shape[1]) * 0.5
    fd = np.array([(m._objective(A, y, theta + H_FD * e) - m._objective(A, y, theta - H_FD * e))
                   / (2 * H_FD) for e in np.eye(len(theta))])
    np.testing.assert_allclose(m._gradient(A, y, theta), fd, rtol=1e-6, atol=1e-9)


def test_hessian_matches_central_finite_differences_of_gradient(logistic_data):
    X, y = logistic_data
    m = LogisticRegression(l2=0.1)
    A = _augmented(X)
    theta = np.random.default_rng(3).standard_normal(A.shape[1]) * 0.5
    fd = np.stack([(m._gradient(A, y, theta + H_FD * e) - m._gradient(A, y, theta - H_FD * e))
                   / (2 * H_FD) for e in np.eye(len(theta))])
    np.testing.assert_allclose(m._hessian(A, theta), fd, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------- Newton's method
def test_newton_matches_scikit_learn(logistic_data):
    X, y = logistic_data
    C = 1.0
    # sklearn minimises C*sum(loss) + ||w||^2/2, i.e. our objective with l2 = 1/(C n).
    ours = LogisticRegression(l2=1 / (C * len(y)), tol=1e-12).fit(X, y)
    ref = SkLR(C=C, tol=1e-12, max_iter=10_000).fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_.ravel(), atol=1e-5)
    assert abs(ours.intercept_ - ref.intercept_[0]) < 1e-5


def test_newton_converges_in_few_iterations_and_monotonically(logistic_data):
    X, y = logistic_data
    m = LogisticRegression(l2=1e-2, tol=1e-10).fit(X, y)
    loss = np.array(m.history_.loss)
    assert m.history_.converged
    assert len(loss) <= 12  # quadratic convergence: a handful of steps
    assert np.all(np.diff(loss) <= 1e-15)  # the step-halving safeguard prevents increases


def test_newton_residual_decreases_quadratically(logistic_data):
    X, y = logistic_data
    m = LogisticRegression(l2=1e-2, tol=1e-14).fit(X, y)
    g = np.array(m.history_.grad_norm)
    g = g[g > 1e-13]  # ignore iterations already at round-off level
    # Quadratic convergence: r_{k+1} ~ C r_k^2, so log(r_{k+1}) / log(r_k) -> 2.
    ratios = np.log(g[2:]) / np.log(g[1:-1])
    assert ratios[-1] > 1.5


# ---------------------------------------------------------------- gradient descent
def test_gradient_descent_reaches_newton_solution(logistic_data):
    X, y = logistic_data
    newton = LogisticRegression(l2=1e-1, tol=1e-14).fit(X, y)
    gd = LogisticRegression(l2=1e-1, solver="gd", tol=1e-9, max_iter=20_000).fit(X, y)
    assert gd.history_.converged
    assert np.all(np.diff(gd.history_.loss) <= 1e-15)
    # ||theta - theta*|| <= ||g|| / mu with mu >= l2 = 0.1, so a 1e-9 residual
    # pins the parameters to well within 1e-6.
    np.testing.assert_allclose(gd.coef_, newton.coef_, atol=1e-6)
    assert len(gd.history_.loss) > 3 * len(newton.history_.loss)


def test_gradient_descent_obeys_explicit_euler_stability_limit(logistic_data):
    X, y = logistic_data
    # Just below the stability limit of the starting Hessian: converges.
    limit = LogisticRegression(l2=0.1, solver="gd", max_iter=0).fit(X, y).stability_limit_
    stable = LogisticRegression(l2=0.1, solver="gd", tol=1e-8, max_iter=5000,
                                step=0.95 * limit).fit(X, y)
    assert stable.history_.converged

    # Just above the limit at the solution, 2 / lambda_max(H*): the error is
    # amplified by |1 - h lambda_max| > 1 near the minimum, so it never converges.
    newton = LogisticRegression(l2=0.1, tol=1e-14).fit(X, y)
    A = _augmented(X)
    theta_star = np.r_[newton.coef_, newton.intercept_]
    lam_max_star = np.linalg.eigvalsh(newton._hessian(A, theta_star))[-1]
    unstable = LogisticRegression(l2=0.1, solver="gd", tol=1e-8, max_iter=5000,
                                  step=1.1 * 2 / lam_max_star).fit(X, y)
    assert not unstable.history_.converged
    assert np.any(np.diff(unstable.history_.loss) > 0)  # loss is not monotone


# ---------------------------------------------------------------- robustness and inputs
def test_separable_data_gives_finite_solution():
    # Without regularisation the solution would diverge to infinity.
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    m = LogisticRegression(l2=1e-3).fit(X, y)
    assert np.isfinite(m.coef_).all() and m.history_.converged


@pytest.mark.parametrize("X, y, msg", [
    (np.zeros(5), np.zeros(5), "2-D"),
    (np.zeros((5, 2)), np.zeros(4), "shape"),
    (np.zeros((3, 2)), np.array([0.0, 1.0, 2.0]), "only 0 and 1"),
    (np.full((2, 2), np.nan), np.zeros(2), "NaN"),
])
def test_invalid_inputs_are_rejected(X, y, msg):
    with pytest.raises(ValueError, match=msg):
        LogisticRegression().fit(X, y)


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        LogisticRegression(l2=0)
    with pytest.raises(ValueError):
        LogisticRegression(solver="adam")
    with pytest.raises(ValueError):
        LogisticRegression(solver="gd", step=-1.0)
