"""Persona store tests: persistence, canonical ordering, privacy, invariants."""

import json
import re

import pytest

from psyche.metrics import brier_binary, mixture_predictive, score_batch
from psyche.models import BetaAttribute, DirichletAttribute, Persona, PersonaStore
from psyche.update import Confirmation, lambda_from_n_max


def _persona(pi: float, seed_counts: float) -> Persona:
    return Persona.new({
        "r": BetaAttribute(6.0, 14.0, A=seed_counts, B=seed_counts / 2),
        "ch": DirichletAttribute(["a", "b"], [10.0, 10.0], A=[seed_counts, 0.0]),
    }, pi=pi)


def test_json_roundtrip(tmp_path):
    store = PersonaStore([_persona(0.6, 40.0), _persona(0.4, 5.0)],
                         last_update_ts="2024-01-01")
    store.config["attributes"] = {"r": {"type": "binary", "base_rate": 0.3}}
    store.save(tmp_path)
    loaded = PersonaStore.load(tmp_path)
    assert len(loaded.personas) == 2
    assert loaded.last_update_ts == "2024-01-01"
    a = loaded.personas[0].attributes["r"]
    assert pytest.approx(40.0) == a.A
    assert a.alpha0 == pytest.approx(6.0)
    assert loaded.personas[0].lineage  # lineage preserved


def test_canonical_ordering_by_pi(tmp_path):
    store = PersonaStore([_persona(0.2, 1.0), _persona(0.7, 2.0), _persona(0.1, 3.0)])
    store.save(tmp_path)
    pis = [p.pi for p in store.personas]
    assert pis == sorted(pis, reverse=True)
    # filenames reflect rank order for stable git diffs
    files = sorted(p.name for p in (tmp_path / "personas").glob("persona_*.json"))
    assert files == ["persona_00.json", "persona_01.json", "persona_02.json"]
    first = json.loads((tmp_path / "personas" / "persona_00.json").read_text())
    assert first["pi"] == pytest.approx(0.7)


def test_stable_diffs_on_resave(tmp_path):
    store = PersonaStore([_persona(0.5, 40.0), _persona(0.5, 5.0)])
    store.save(tmp_path)
    snap1 = (tmp_path / "personas" / "persona_00.json").read_text()
    store.save(tmp_path)
    snap2 = (tmp_path / "personas" / "persona_00.json").read_text()
    assert snap1 == snap2  # deterministic serialization


def test_no_pii_in_store(tmp_path):
    """Serialized personas must never contain names/emails/NPIs."""
    store = PersonaStore([_persona(1.0, 10.0)])
    store.save(tmp_path)
    blob = "\n".join(p.read_text() for p in tmp_path.rglob("*.json"))
    assert not re.search(r"\b\d{10}\b", blob)        # no raw NPIs
    assert "@" not in blob                            # no emails
    for forbidden in ("name", "email", "npi"):
        assert f'"{forbidden}"' not in blob


def test_lambda_n_max_stored_and_consistent(tmp_path):
    p = _persona(1.0, 0.0)
    store = PersonaStore([p])
    store.save(tmp_path)
    d = json.loads((tmp_path / "personas" / "persona_00.json").read_text())
    lam, n_max = d["decay"]["lambda"], d["decay"]["n_max"]
    assert lam == pytest.approx(lambda_from_n_max(n_max))


def test_priors_immutable_across_decay(tmp_path):
    store = PersonaStore([_persona(1.0, 40.0)])
    before = json.loads(json.dumps(store.personas[0].to_dict()))
    store.personas[0].decay_counts(5.0)
    after = store.personas[0].to_dict()
    for attr in before["attributes"].values():
        if attr["type"] == "binary":
            assert attr["alpha0"] == after["attributes"]["r"]["alpha0"]
            assert attr["beta0"] == after["attributes"]["r"]["beta0"]
        else:
            assert attr["alphas0"] == after["attributes"]["ch"]["alphas0"]


def test_out_of_sample_brier_ordering():
    """Brier computed before consumption differs from post-consumption scoring."""
    persona = _persona(1.0, 0.0)  # prior predictive q=0.3 for 'r'
    confirmations = [Confirmation(lead_hash=f"h{i}", attribute="r", value=1,
                                  ts="2024-01-01") for i in range(10)]
    pre = score_batch([persona], confirmations)[0]["brier"]
    assert pre == pytest.approx((0.3 - 1.0) ** 2)  # prior predictive, out-of-sample
    for c in confirmations:
        persona.attributes["r"].observe(int(c.value), weight=1.0)
    post = score_batch([persona], confirmations)[0]["brier"]
    assert post < pre  # consuming first would inflate skill


def test_mixture_predictive():
    p1 = _persona(0.75, 0.0)
    p2 = _persona(0.25, 0.0)
    p2.attributes["r"] = BetaAttribute(1.0, 9.0)  # q2 = 0.1
    q = mixture_predictive([p1, p2], "r")
    assert q == pytest.approx(0.75 * 0.3 + 0.25 * 0.1)


def test_brier_binary_math():
    assert brier_binary(0.3, 1) == pytest.approx(0.49)
    assert brier_binary(0.3, 0) == pytest.approx(0.09)
