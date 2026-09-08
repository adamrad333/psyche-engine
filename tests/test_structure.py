"""Split/merge structural adaptation tests (ALGORITHM §5.2-5.3).

Both operations must be BIC-gated and logged in persona lineage.

Note: a mixture of Bernoullis on a *single* binary attribute is
unidentifiable (the marginal p(x) has one degree of freedom), so a split
there can never reduce BIC - correctly. Split tests therefore use two
correlated binary attributes, where separated clusters genuinely raise the
marginal likelihood.
"""

import numpy as np
import pytest

from psyche.models import BetaAttribute, Persona, PersonaStore
from psyche.update import (
    bic,
    e_step,
    m_step,
    snapshot_counts,
    sym_kl,
    try_merge,
    try_split,
)


def _obs_bimodal() -> list[dict]:
    """Two cleanly separated clusters over two correlated attributes."""
    return [{"r": 1, "s": 1}] * 30 + [{"r": 0, "s": 0}] * 30


def _obs_homogeneous() -> list[dict]:
    """One cluster: r nearly always 1, s an independent coin flip."""
    rng = np.random.default_rng(3)
    return [{"r": 1 if i < 55 else 0, "s": int(rng.random() < 0.5)}
            for i in range(60)]


def _store_single(obs: list[dict]) -> PersonaStore:
    """One persona trained on the observations (unit weights, no decay)."""
    persona = Persona.new({"r": BetaAttribute(3.0, 3.0),
                           "s": BetaAttribute(3.0, 3.0)}, pi=1.0)
    store = PersonaStore([persona])
    store.config["tau_split"] = -100.0   # force entropy gate open (H is negative)
    store.config["tau_merge"] = 0.05
    store.config["n_min"] = 5.0
    w = np.ones(len(obs))
    R = e_step(store.personas, obs)
    base = snapshot_counts(store.personas)  # pre-M-step snapshot (zero counts)
    m_step(store.personas, base, obs, R, w)
    store.test_base = base  # keep for try_split(base=...)
    return store


class TestSplit:
    def test_bimodal_split_accepted_and_logged(self):
        obs = _obs_bimodal()
        store = _store_single(obs)
        parent = store.personas[0]
        parent_counts = sum(a.A + a.B for a in parent.attributes.values())
        w = np.ones(len(obs))
        R = e_step(store.personas, obs)

        new_bases = try_split(store, 0, obs, R, w, n_eff=float(len(obs)),
                              ts="2024-01-02", base=store.test_base)
        assert new_bases is not None
        assert len(store.personas) == 2
        # BIC actually decreased
        fresh = _store_single(obs)
        assert bic(store.personas, obs, w, n_eff=float(len(obs))) < \
            bic(fresh.personas, obs, w, n_eff=float(len(obs)))
        # pseudo-mass preserved across children
        total = sum(a.A + a.B for p in store.personas
                    for a in p.attributes.values())
        assert total == pytest.approx(parent_counts, rel=1e-6)
        # lineage logged on both children
        for child in store.personas:
            events = [e["event"] for e in child.lineage]
            assert "split_child" in events
            detail = child.lineage[-1]["detail"]
            assert detail["parent"] == parent.id
            assert detail["bic_after"] < detail["bic_before"]
        # children separated: one high-q, one low-q on attribute r
        qs = sorted(p.attributes["r"].mean for p in store.personas)
        assert qs[0] < 0.3 < qs[1]

    def test_homogeneous_split_rejected_by_bic(self):
        obs = _obs_homogeneous()
        store = _store_single(obs)
        w = np.ones(len(obs))
        R = e_step(store.personas, obs)
        assert try_split(store, 0, obs, R, w, n_eff=float(len(obs)),
                         ts="2024-01-02", base=store.test_base) is None
        assert len(store.personas) == 1

    def test_split_blocked_below_n_min(self):
        obs = _obs_bimodal()
        store = _store_single(obs)
        store.config["n_min"] = 1e9
        w = np.ones(len(obs))
        R = e_step(store.personas, obs)
        assert try_split(store, 0, obs, R, w, n_eff=float(len(obs)),
                         base=store.test_base) is None

    def test_split_blocked_by_entropy_gate(self):
        obs = _obs_bimodal()
        store = _store_single(obs)
        store.config["tau_split"] = 0.0  # trained-persona entropy is below this
        w = np.ones(len(obs))
        R = e_step(store.personas, obs)
        assert try_split(store, 0, obs, R, w, n_eff=float(len(obs)),
                         base=store.test_base) is None

    def test_split_refused_without_base_snapshot(self):
        obs = _obs_bimodal()
        store = _store_single(obs)
        w = np.ones(len(obs))
        R = e_step(store.personas, obs)
        assert try_split(store, 0, obs, R, w, n_eff=float(len(obs)),
                         base=None) is None


class TestMerge:
    def _two_similar(self) -> PersonaStore:
        p1 = Persona.new({"r": BetaAttribute(6.0, 14.0, A=10.0, B=20.0)}, pi=0.5)
        p2 = Persona.new({"r": BetaAttribute(6.0, 14.0, A=11.0, B=21.0)}, pi=0.5)
        store = PersonaStore([p1, p2])
        store.config["tau_merge"] = 0.5
        store.config["tau_split"] = 10.0
        return store

    def test_similar_personas_merge(self):
        store = self._two_similar()
        obs = [{"r": 1}] * 10 + [{"r": 0}] * 10
        w = np.ones(len(obs))
        counts_before = sum(p.attributes["r"].A + p.attributes["r"].B
                            for p in store.personas)
        new_bases = try_merge(store, 0, 1, obs, w, n_eff=20.0, ts="2024-01-03")
        assert new_bases is not None
        assert len(store.personas) == 1
        merged = store.personas[0]
        a = merged.attributes["r"]
        # counts add
        assert a.A + a.B == pytest.approx(counts_before)
        assert a.A == pytest.approx(21.0) and a.B == pytest.approx(41.0)
        # lineage logged with both parents
        events = [e["event"] for e in merged.lineage]
        assert "merge" in events
        detail = merged.lineage[-1]["detail"]
        assert len(detail["parents"]) == 2
        assert detail["bic_after"] < detail["bic_before"]

    def test_distinct_personas_do_not_merge(self):
        p1 = Persona.new({"r": BetaAttribute(6.0, 14.0, A=100.0, B=5.0)}, pi=0.5)
        p2 = Persona.new({"r": BetaAttribute(6.0, 14.0, A=5.0, B=100.0)}, pi=0.5)
        store = PersonaStore([p1, p2])
        store.config["tau_merge"] = 0.01
        obs = [{"r": 1}] * 10 + [{"r": 0}] * 10
        w = np.ones(len(obs))
        assert try_merge(store, 0, 1, obs, w, n_eff=20.0) is None
        assert len(store.personas) == 2


def test_sym_kl():
    p1 = Persona.new({"r": BetaAttribute(6.0, 14.0)})
    p2 = Persona.new({"r": BetaAttribute(6.2, 14.1)})
    p3 = Persona.new({"r": BetaAttribute(60.0, 6.0)})
    assert sym_kl(p1, p1) == pytest.approx(0.0, abs=1e-12)
    assert sym_kl(p1, p2) < sym_kl(p1, p3)
    assert sym_kl(p1, p3) == pytest.approx(sym_kl(p3, p1))
