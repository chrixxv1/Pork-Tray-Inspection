# Logistic regression with Newton's method and gradient descent

Notebook: `notebooks/5_logistic.ipynb`. Code: `src/porkqc/logistic.py`, `src/porkqc/model.py`.

## Method

Logistic regression estimates P(FAIL) from the PCA scores of each image. Training
minimises the mean negative log-likelihood plus an L2 penalty, which is posed as
finding the root of the gradient, ∇f(θ) = 0. Two solvers are compared:

- **Newton's method**: the shooting-method update s ← s − F/F′ applied to ∇f = 0.
  Each iteration solves H Δθ = −∇f by LU factorisation.
- **Gradient descent**: θ ← θ − h∇f, which is explicit Euler applied to
  dθ/dt = −∇f, so it is stable only for h < 2/λ_max(H).

Both stop when the residual ‖∇f‖∞ ≤ tol.

## Results (HOG features, k = 50 PCA components, λ = 1e-4)

| Measure | Newton | Gradient descent |
|---|---|---|
| Iterations to converge | 6 (tol 1e-10) | 104 (tol 1e-8) |
| Convergence rate | Quadratic | Linear |
| Agreement with scikit-learn (max coefficient difference) | 3e-8 | 3e-7 vs Newton |

Validation AUC: 0.784.

## Verification

- Gradient and Hessian match central finite differences (h = 1e-6, near the
  optimal step u^(1/3) for float64).
- Newton's solution matches scikit-learn to ~3e-8 on the real data.
- Newton's residual shows quadratic convergence; gradient descent reaches the
  same solution.
- The sigmoid and loss are evaluated in overflow-safe forms.
- 44 pytest tests pass.

## Stability of gradient descent

The stability limit is 2/λ_max ≈ 8.0 (λ_max ≈ 1/4 because the PCA scores are
whitened). Gradient descent converged in 217, 104 and 45 iterations at 0.25,
0.5 and 0.95 × the limit, and did not converge at 1.25 × the limit: the
residual stalled near 1e-2 as the iterates oscillated about the minimum. This
matches the explicit-Euler limit for y′ = −λy.

## Effect of the number of PCA components k

| k | Newton iterations | GD iterations | Hessian condition number | Validation AUC |
|---|---|---|---|---|
| 10 | 5 | 27 | 2.6 | 0.651 |
| 50 | 6 | 70 | 7.5 | 0.784 |
| 200 | 7 | 237 | 21.4 | 0.702 |
| 400 | 9 | 1906 | 166.4 | 0.638 |

Conditioning worsens as k grows, which slows gradient descent sharply while
Newton is barely affected. Validation AUC peaks near k = 50.

## Findings

- Newton's method is selected as the main solver: it needs far fewer iterations
  and is insensitive to conditioning.
- The residual cannot be reduced below about 1e-12, because round-off from
  summing over 3,349 training images sets a floor.
- Hyperparameters here are provisional; they are selected properly in Phase 6.
