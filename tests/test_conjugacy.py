"""Beta/Dirichlet conjugacy unit tests (ALGORITHM §1, Appendix B)."""

import math

import numpy as np
import pytest
from scipy.special import betaln, digamma

from psyche.models import BetaAttribute, DirichletAttribute, Persona, PersonaStore
from psyche.update import e_step, em_batch


class TestBetaConjugacy:
    def test_confirm_increments_alpha(self):
        a = BetaAttribute(alpha0=6.0, beta0=14.0)
        a.observe(1, weight=1.0)
        assert a.alpha == pytest.approx(7.0)
        assert a.beta == pytest.approx(14.0)

    def test_refute_increments_beta(self):
        a = BetaAttribute(alpha0=6.0, beta0=14.0)
        a.observe(0, weight=1.0)
        assert a.alpha == pytest.approx(6.0)
        assert a.beta == pytest.approx(15.0)

    def test_weighted_observe(self):
        a = BetaAttribute(alpha0=6.0, beta0=14.0)
        a.observe(1, weight=0.554)  # Appendix B combined weight rho_i1
        assert pytest.approx(0.554) == a.A
        assert pytest.approx(0.0) == a.B

    def test_estimators(self):
        a = BetaAttribute(alpha0=46.0, beta0=74.0)
        assert a.mean == pytest.approx(46 / 120)
        n0 = 120.0
        assert a.var == pytest.approx(46 * 74 / (n0**2 * (n0 + 1)))
        assert a.map_estimate == pytest.approx(45 / 118)

    def test_map_boundary(self):
        a = BetaAttribute(alpha0=0.5, beta0=0.5)
        assert a.map_estimate in (0.0, 1.0)  # boundary, not the interior formula

    def test_posterior_predictive(self):
        a = BetaAttribute(alpha0=6.0, beta0=14.0)
        assert a.predictive(1) == pytest.approx(0.3)
        assert a.predictive(0) == pytest.approx(0.7)
        assert a.predictive(0) + a.predictive(1) == pytest.approx(1.0)

    def test_log_predictive_matches(self):
        a = BetaAttribute(alpha0=3.0, beta0=7.0)
        assert a.log_predictive(1) == pytest.approx(math.log(a.predictive(1)))

    def test_decay_touches_counts_only(self):
        a = BetaAttribute(alpha0=6.0, beta0=14.0, A=40.0, B=60.0)
        a.decay(0.995**3)
        assert pytest.approx(40 * 0.995**3) == a.A   # 39.40 per Appendix B
        assert pytest.approx(60 * 0.995**3) == a.B   # 59.11
        assert a.alpha0 == 6.0 and a.beta0 == 14.0   # priors immutable

    def test_decay_floors_at_zero(self):
        a = BetaAttribute(alpha0=1.0, beta0=1.0, A=1e-12, B=0.0)
        a.decay(0.0)
        assert a.A == 0.0 and a.B == 0.0


class TestDirichletConjugacy:
    def test_observe_increments_level(self):
        d = DirichletAttribute(levels=["a", "b", "c"], alphas0=[2.0, 2.0, 2.0])
        d.observe("b", weight=1.0)
        assert d.alphas == pytest.approx([2.0, 3.0, 2.0])

    def test_estimators(self):
        d = DirichletAttribute(levels=["a", "b"], alphas0=[3.0, 7.0])
        assert d.mean["a"] == pytest.approx(0.3)
        a0 = 10.0
        assert d.var("a") == pytest.approx(3 * 7 / (a0**2 * (a0 + 1)))

    def test_predictive_sums_to_one(self):
        d = DirichletAttribute(levels=["a", "b", "c"], alphas0=[1.5, 2.5, 6.0])
        assert sum(d.predictive(lev) for lev in d.levels) == pytest.approx(1.0)

    def test_decay_counts_only(self):
        d = DirichletAttribute(levels=["a", "b"], alphas0=[2.0, 3.0], A=[10.0, 5.0])
        d.decay(0.5)
        assert pytest.approx([5.0, 2.5]) == d.A
        assert d.alphas0 == [2.0, 3.0]

    def test_entropy_formula(self):
        d = DirichletAttribute(levels=["a", "b", "c"], alphas0=[2.0, 3.0, 4.0])
        a = np.array(d.alphas)
        a0 = a.sum()
        log_b = sum(float(x) for x in
                    [betaln(a[0], a[1] + a[2]) + betaln(a[1], a[2])])  # ln B via chain
        expected = (log_b + (a0 - 3) * digamma(a0)
                    - sum((a[k] - 1) * digamma(a[k]) for k in range(3)))
        assert d.entropy() == pytest.approx(expected)


class TestAppendixBMicroExample:
    """The worked example of ALGORITHM.md Appendix B end-to-end."""

    def test_decay_estep_mstep(self):
        p1 = Persona.new({"r": BetaAttribute(6.0, 14.0, A=40.0, B=60.0)}, pi=0.5)
        p2 = Persona.new({"r": BetaAttribute(6.0, 14.0, A=5.0, B=40.0)}, pi=0.5)
        store = PersonaStore([p1, p2])

        # decay with lambda^3, lambda = 0.995
        store.personas[0].decay_counts(3.0)
        store.personas[1].decay_counts(3.0)
        a1 = store.personas[0].attributes["r"]
        assert pytest.approx(39.40, abs=0.01) == a1.A
        assert pytest.approx(59.11, abs=0.01) == a1.B
        assert a1.mean == pytest.approx(0.383, abs=0.001)

        # one EM iteration with w_tilde = 0.8, x = 1
        r1 = 0.383 / (0.383 + 0.170)
        rho1 = 0.8 * r1
        em_batch(store, [{"r": 1}], np.array([0.8]), dt=0.0, max_iter=1)
        assert pytest.approx(39.40 + rho1, abs=0.02) == a1.A
        # decay cap: alpha + beta <= kappa + N_max (Section 4.1)
        assert a1.alpha + a1.beta <= 20 + 200 + 1.0

    def test_responsibilities_two_personas(self):
        p1 = Persona.new({"r": BetaAttribute(6.0, 14.0, A=40.0, B=60.0)}, pi=0.5)
        p2 = Persona.new({"r": BetaAttribute(6.0, 14.0, A=5.0, B=40.0)}, pi=0.5)
        R = e_step([p1, p2], [{"r": 1}])
        assert R[0, 0] == pytest.approx(0.693, abs=0.01)
        assert R[0, 1] == pytest.approx(0.307, abs=0.01)
        assert R.sum() == pytest.approx(1.0)
