"""Drift simulation verifying the stationary bias bound of ALGORITHM §4.3:

    |E[theta_hat_n] - p_n| <= delta * lambda / (1 - lambda)   (stationary)

Simulation: p_k drifts linearly with per-period step delta; the decayed
estimator A/T with lambda = 1 - 1/N_max tracks it within the bound.
"""

import numpy as np
import pytest

from psyche.update import lambda_from_n_max, stationary_bias_bound


def _decayed_estimate(samples: np.ndarray, lam: float) -> tuple[float, float]:
    """Zero-prior-strength decayed estimator A_n / T_n (§4.3)."""
    A = 0.0
    T = 0.0
    for x in samples:
        A = lam * A + x
        T = lam * T + 1.0
    return A / T, T


@pytest.mark.parametrize("lam", [0.95, 0.99, 0.995])
def test_stationary_bias_bound(lam):
    delta = 0.002          # per-period drift step
    p_start, p_end = 0.2, 0.8
    n = 2000               # burn-in well past T_inf = 1/(1-lam)
    ramp = min(n, int((p_end - p_start) / delta))
    p_path = np.minimum(p_start + delta * np.arange(n), p_end)
    # keep drifting for the whole horizon by cycling the ramp
    if ramp < n:
        p_path = np.concatenate([p_path[:ramp],
                                 p_end - delta * np.arange(n - ramp) % (p_end - p_start)])
        p_path = np.clip(p_path, 0.05, 0.95)

    rng = np.random.default_rng(123)
    n_trials = 40
    errors = []
    for _ in range(n_trials):
        samples = rng.binomial(1, p_path)
        theta_hat, _ = _decayed_estimate(samples, lam)
        errors.append(abs(theta_hat - p_path[-1]))

    bound = stationary_bias_bound(delta, lam)
    mean_err = float(np.mean(errors))
    # E|err| <= |bias| + sd; stationary sd bound (§4.3b) + MC slack
    var_bound = (1 - lam) / (4 * (1 + lam))
    assert mean_err <= bound + 3 * np.sqrt(var_bound) + 0.01


def test_n_max_equivalence():
    """T_n = sum lambda^j <= T_inf = 1/(1-lam) = N_max (§4.1)."""
    lam = lambda_from_n_max(200.0)
    assert lam == pytest.approx(0.995)
    T = 0.0
    for _ in range(10000):
        T = lam * T + 1.0
    assert pytest.approx(200.0, rel=1e-3) == T


def test_decay_tracks_drift_better_than_no_decay():
    """Under drift, the decayed estimator beats the non-decayed one (§4.3c)."""
    delta = 0.003
    n = 1500
    p_path = np.clip(0.3 + delta * np.arange(n), 0, 0.95)
    rng = np.random.default_rng(9)
    lam = lambda_from_n_max(100.0)
    errs_decay, errs_flat = [], []
    for _ in range(30):
        samples = rng.binomial(1, p_path)
        th, _ = _decayed_estimate(samples, lam)
        errs_decay.append(abs(th - p_path[-1]))
        th_flat, _ = _decayed_estimate(samples, 1.0)
        errs_flat.append(abs(th_flat - p_path[-1]))
    assert np.mean(errs_decay) < np.mean(errs_flat)
