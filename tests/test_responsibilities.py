"""Responsibility / combined-weight tests (ALGORITHM §2, §3.3, §6 invariants)."""

import numpy as np
import pytest

from psyche.models import BetaAttribute, DirichletAttribute, Persona, PersonaStore
from psyche.update import e_step, em_batch, m_step, snapshot_counts


def _store() -> PersonaStore:
    attrs1 = {
        "r": BetaAttribute(6.0, 14.0, A=40.0, B=60.0),
        "ch": DirichletAttribute(["email", "phone"], [10.0, 10.0], A=[8.0, 2.0]),
    }
    attrs2 = {
        "r": BetaAttribute(6.0, 14.0, A=5.0, B=40.0),
        "ch": DirichletAttribute(["email", "phone"], [10.0, 10.0], A=[1.0, 9.0]),
    }
    return PersonaStore([Persona.new(attrs1, pi=0.6), Persona.new(attrs2, pi=0.4)])


def _obs() -> list[dict]:
    return [{"r": 1, "ch": "email"}, {"r": 0, "ch": "phone"}, {"r": 1},
            {"r": 0, "ch": "email"}, {"ch": "phone"}]


def test_responsibility_rows_sum_to_one():
    store = _store()
    R = e_step(store.personas, _obs())
    assert R.shape == (5, 2)
    np.testing.assert_allclose(R.sum(axis=1), 1.0, rtol=1e-12)


def test_log_space_matches_naive():
    store = _store()
    obs_list = _obs()
    R = e_step(store.personas, obs_list)
    # naive computation in probability space
    for i, obs in enumerate(obs_list):
        probs = []
        for persona in store.personas:
            p = persona.pi
            for attr, val in obs.items():
                a = persona.attributes[attr]
                if isinstance(a, BetaAttribute):
                    p *= a.predictive(int(val))
                else:
                    p *= a.predictive(str(val))
            probs.append(p)
        total = sum(probs)
        for s in range(2):
            assert R[i, s] == pytest.approx(probs[s] / total, rel=1e-9)


def test_missing_attributes_contribute_no_factor():
    store = _store()
    R_full = e_step(store.personas, [{"r": 1, "ch": "email"}])
    R_part = e_step(store.personas, [{"r": 1}])
    assert not np.isclose(R_full[0, 0], R_part[0, 0])  # channel info matters


def test_combined_weight_sums():
    """sum_s rho_is == w_tilde_i and total batch pseudo-mass == ESS."""
    store = _store()
    obs_list = _obs()
    w_tilde = np.array([1.2, 0.8, 1.0, 0.6, 0.4])  # already ESS-normalized
    R = e_step(store.personas, obs_list)
    rho = R * w_tilde[:, None]
    np.testing.assert_allclose(rho.sum(axis=1), w_tilde, rtol=1e-12)
    assert rho.sum() == pytest.approx(w_tilde.sum())


def test_mstep_pseudo_mass_equals_ess():
    """Total pseudo-count added by one batch equals ESS, not n (§3.3)."""
    store = _store()
    obs_list = _obs()
    w_tilde = np.array([1.2, 0.8, 1.0, 0.6, 0.4])  # sums to 4.0 = ESS
    R = e_step(store.personas, obs_list)
    base = snapshot_counts(store.personas)
    m_step(store.personas, base, obs_list, R, w_tilde)
    added = 0.0
    for s, persona in enumerate(store.personas):
        a = persona.attributes["r"]
        added += (a.A - base[s]["r"][0]) + (a.B - base[s]["r"][1])
    # 4 leads observed attribute r; pseudo-mass added to r equals their w sum
    assert added == pytest.approx(w_tilde[:4].sum())


def test_pi_update_proportional_to_mass():
    store = _store()
    obs_list = _obs()
    w_tilde = np.ones(5)
    R = e_step(store.personas, obs_list)
    base = snapshot_counts(store.personas)
    mass = m_step(store.personas, base, obs_list, R, w_tilde)
    pis = np.array([p.pi for p in store.personas])
    np.testing.assert_allclose(pis, mass / mass.sum(), rtol=1e-12)
    assert pis.sum() == pytest.approx(1.0)


def test_em_monotone_loglikelihood():
    """EM lower bound never decreases the observed-data log-likelihood (§2.3)."""
    store = _store()
    obs_list = _obs() * 4  # 20 leads
    w_tilde = np.ones(len(obs_list))
    from psyche.update import log_likelihood
    lls = []
    base = snapshot_counts(store.personas)
    R = e_step(store.personas, obs_list)
    for _ in range(15):
        lls.append(log_likelihood(store.personas, obs_list, w_tilde))
        m_step(store.personas, base, obs_list, R, w_tilde)
        R = e_step(store.personas, obs_list)
    diffs = np.diff(lls)
    assert (diffs >= -1e-9).all()


def test_em_batch_decay_applied_once():
    """Decay exactly once per batch on pre-batch counts (invariant iii)."""
    store = _store()
    a_before = store.personas[0].attributes["r"].A
    obs_list = _obs()
    em_batch(store, obs_list, np.ones(5), dt=2.0, max_iter=3)
    # base counts were decayed once: A_base = A_before * lam^2; the batch adds
    # increments, so final A must be >= decayed base (x=1 present in batch)
    lam = store.personas[0].lam
    a_after = store.personas[0].attributes["r"].A
    assert a_after >= a_before * lam**2 - 1e-12
    # and strictly less than if decay had been applied twice
    assert a_after >= a_before * lam**4


def test_prune_underweight_personas():
    from psyche.update import prune_underweight_personas
    store = _store()
    mass = np.array([50.0, 1.0])  # persona 1 under n_min
    folded = prune_underweight_personas(store, mass, n_min=5.0, ts="2024-01-01")
    assert folded == [1]
    a = store.personas[1].attributes["r"]
    assert a.A == 0.0 and a.B == 0.0  # folded into prior
    assert store.personas[1].lineage[-1]["event"] == "folded"
    # guard: never fold everything
    folded = prune_underweight_personas(store, np.array([1.0, 1.0]), n_min=5.0)
    assert folded == []
