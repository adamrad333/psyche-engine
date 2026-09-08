"""Convergence simulation (ALGORITHM §4.2): posterior mean -> p a.s."""

import numpy as np
import pytest

from psyche.models import BetaAttribute, DirichletAttribute


@pytest.mark.parametrize("p", [0.15, 0.5, 0.85])
def test_beta_posterior_mean_converges(p):
    """Stream n Bernoulli(p) confirmations; posterior mean -> p (SLLN, §4.2)."""
    rng = np.random.default_rng(42)
    a = BetaAttribute(alpha0=6.0, beta0=14.0)  # kappa=20 prior at 0.3
    n = 4000
    for x in rng.binomial(1, p, size=n):
        a.observe(int(x), weight=1.0)
    assert a.mean == pytest.approx(p, abs=0.03)
    # posterior concentrates: Var = O(1/n)
    assert a.var < 1.0 / n


def test_dirichlet_posterior_mean_converges():
    probs = np.array([0.5, 0.3, 0.2])
    rng = np.random.default_rng(7)
    d = DirichletAttribute(levels=["a", "b", "c"], alphas0=[10.0, 6.0, 4.0])
    draws = rng.choice(3, size=4000, p=probs)
    for k in draws:
        d.observe(d.levels[k], weight=1.0)
    means = d.mean
    for k, lev in enumerate(d.levels):
        assert means[lev] == pytest.approx(probs[k], abs=0.03)
