"""Entropy and sign edge cases (ALGORITHM §5.3, Appendix B).

Differential entropy of concentrated Beta posteriors is NEGATIVE - personas
must be compared relative to the prior entropy, never to zero.
"""

import pytest
from scipy.integrate import quad
from scipy.stats import beta as beta_dist
from scipy.stats import dirichlet as dir_dist

from psyche.models import BetaAttribute, DirichletAttribute, Persona


def test_beta_entropy_matches_scipy():
    for a, b in [(6.0, 14.0), (46.0, 74.0), (0.5, 0.5), (1.0, 1.0), (2.5, 3.5)]:
        ours = BetaAttribute(alpha0=a, beta0=b).entropy()
        assert ours == pytest.approx(beta_dist(a, b).entropy(), rel=1e-10)


def test_dirichlet_entropy_matches_scipy():
    for alphas in ([1.0, 1.0, 1.0], [2.0, 3.0, 4.0], [10.0, 6.0, 4.0]):
        ours = DirichletAttribute(levels=["a", "b", "c"], alphas0=list(alphas)).entropy()
        assert ours == pytest.approx(dir_dist(alphas).entropy(), rel=1e-10)


def test_concentrated_beta_has_negative_entropy():
    """Beta(46,74) ~ -1.70 nats (Appendix B) - concentrated => negative."""
    a = BetaAttribute(alpha0=6.0, beta0=14.0, A=40.0, B=60.0)
    assert (a.alpha, a.beta) == (46.0, 74.0)
    assert a.entropy() == pytest.approx(-1.70, abs=0.01)
    assert a.entropy() < 0.0


def test_prior_entropy_reference():
    """Beta(6,14) ~ -0.90 nats (Appendix B) - the comparison baseline."""
    prior = BetaAttribute(alpha0=6.0, beta0=14.0)
    assert prior.entropy() == pytest.approx(-0.90, abs=0.01)


def test_entropy_decreases_with_concentration():
    """More data (fixed mean) => more concentrated => lower entropy."""
    entropies = []
    for scale in (1.0, 5.0, 25.0):
        a = BetaAttribute(alpha0=6.0, beta0=14.0, A=40.0 * scale, B=60.0 * scale)
        entropies.append(a.entropy())
    assert entropies[0] > entropies[1] > entropies[2]


def test_uniform_beta_entropy_is_zero():
    assert BetaAttribute(alpha0=1.0, beta0=1.0).entropy() == pytest.approx(0.0)


def test_dirichlet_entropy_vs_numerical_integration():
    """Sanity-check the closed form against direct -E[log p] integration."""
    d = DirichletAttribute(levels=["x", "y"], alphas0=[3.0, 5.0])
    a, b = d.alphas

    def integrand(p):
        return -dir_dist([a, b]).pdf([p, 1 - p]) * \
            (beta_dist(a, b).logpdf(p))
    numeric, _ = quad(integrand, 0, 1)
    assert d.entropy() == pytest.approx(numeric, rel=1e-6)


def test_mean_entropy_average():
    p = Persona.new({
        "r": BetaAttribute(6.0, 14.0, A=40.0, B=60.0),
        "ch": DirichletAttribute(["a", "b"], [10.0, 10.0]),
    })
    expected = (p.attributes["r"].entropy() + p.attributes["ch"].entropy()) / 2
    assert p.mean_entropy() == pytest.approx(expected)


def test_split_uses_relative_entropy_sign():
    """A persona at the prior is less concentrated than a trained one, so
    mean entropy ordering (not sign vs zero) drives split decisions."""
    prior = Persona.new({"r": BetaAttribute(6.0, 14.0)})
    trained = Persona.new({"r": BetaAttribute(6.0, 14.0, A=400.0, B=600.0)})
    assert trained.mean_entropy() < prior.mean_entropy()  # both negative
