"""End-to-end pipeline tests: ingest -> update -> show, fully offline."""

import json
from pathlib import Path

import pandas as pd
import pytest

from psyche.enrich import NPPESClient
from psyche.models import PersonaStore
from psyche.pipeline import ingest, show, update

DATA = Path(__file__).parent.parent / "data"


@pytest.fixture()
def store_dir(tmp_path):
    return tmp_path / "store"


def test_ingest_persists_only_hashes(store_dir):
    enricher = NPPESClient(stub={
        "1234567890": {"taxonomy": "207Q00000X", "state": "CA",
                       "enumeration_date": "2010-05-01", "stale": False},
    })
    store = ingest(DATA / "leads.csv", store_dir,
                   margins_csv=DATA / "population_margins.csv",
                   enricher=enricher)
    leads = pd.read_csv(store_dir / "leads.csv", dtype=str)
    assert "lead_hash" in leads.columns
    blob = "\n".join(p.read_text() for p in Path(store_dir).rglob("*")
                     if p.is_file() and p.suffix in (".json", ".csv"))
    # no raw identifiers anywhere in the store
    for raw in ("alice", "@example.com", "1234567890"):
        assert raw not in blob
    assert len(store.personas) >= 1
    # raking produced ESS-normalized weights
    w = leads["w_tilde"].astype(float)
    assert (w > 0).all()
    assert w.sum() <= len(leads) + 1e-6


def test_update_cycle_and_idempotence(store_dir):
    ingest(DATA / "leads.csv", store_dir, margins_csv=DATA / "population_margins.csv",
           enricher=NPPESClient(stub={}))
    summary = update(DATA / "confirmations.csv", store_dir)
    assert summary["status"] == "updated"
    assert summary["ess"] <= summary["n_leads"] + 1e-9

    store = PersonaStore.load(store_dir)
    assert store.last_update_ts is not None
    # counts were updated beyond the prior
    a = store.personas[0].attributes["responds_email"]
    assert a.A + a.B > 0

    # metrics history was written out-of-sample
    history = (store_dir / "metrics" / "history.jsonl").read_text().splitlines()
    assert len(history) >= 1
    rec = json.loads(history[0])
    assert {"ts", "attribute", "brier", "n"} <= set(rec)

    # re-running the same batch is a no-op (all events already consumed)
    again = update(DATA / "confirmations.csv", store_dir)
    assert again["status"] == "no_new_events"


def test_decay_applied_once_across_batches(store_dir):
    ingest(DATA / "leads.csv", store_dir, margins_csv=DATA / "population_margins.csv",
           enricher=NPPESClient(stub={}))
    update(DATA / "confirmations.csv", store_dir)
    store = PersonaStore.load(store_dir)
    top = store.personas[0]
    a = top.attributes["responds_email"]
    counts_after_first = a.A + a.B
    dt_days = (pd.Timestamp("2024-02-01")
               - pd.Timestamp(store.last_update_ts)).total_seconds() / 86400.0

    # a second batch much later with a single confirmation
    df = pd.DataFrame([{"lead_id": "L001", "attribute": "responds_email",
                        "value": "1", "ts": "2024-02-01", "source": "crm"}])
    path = store_dir / "batch2.csv"
    df.to_csv(path, index=False)
    update(path, store_dir)
    store2 = PersonaStore.load(store_dir)
    same = next(p for p in store2.personas if p.id == top.id)
    a2 = same.attributes["responds_email"]
    lam = same.lam
    # counts = decayed old counts + one weighted increment (decay applied once)
    expected_min = counts_after_first * lam**dt_days
    assert expected_min - 1e-9 <= a2.A + a2.B
    # but decay did happen (less than undecayed counts + 1)
    assert counts_after_first + 1.0 + 1e-9 > a2.A + a2.B


def test_show_output(store_dir):
    ingest(DATA / "leads.csv", store_dir, enricher=NPPESClient(stub={}))
    update(DATA / "confirmations.csv", store_dir)
    text = show(store_dir)
    assert "persona" in text and "responds_email" in text


def test_quarantine_invalid_rows(store_dir, tmp_path):
    df = pd.DataFrame([
        {"lead_id": "OK1", "name": "x", "email": "x@y.z", "npi": "",
         "taxonomy": "207Q00000X", "state": "CA"},
        {"lead_id": "", "name": "", "email": "", "npi": "",
         "taxonomy": "207Q00000X", "state": "CA"},          # no identifiers
        {"lead_id": "BAD", "name": "y", "email": "y@y.z", "npi": "123",
         "taxonomy": "207Q00000X", "state": "CA"},          # bad NPI
    ])
    path = tmp_path / "leads.csv"
    df.to_csv(path, index=False)
    ingest(path, store_dir, enricher=NPPESClient(stub={}))
    leads = pd.read_csv(store_dir / "leads.csv")
    assert len(leads) == 1
    quarantine = pd.read_csv(store_dir / "quarantine.csv")
    assert len(quarantine) == 2
    assert set(quarantine["reason"]) == {"missing_identifiers", "invalid_npi"}


def test_quarantine_never_persists_pii(store_dir, tmp_path):
    """Regression: quarantined rows must hold only hashed id + reason + ts."""
    from psyche.enrich import hash_id

    df = pd.DataFrame([
        {"lead_id": "OK1", "name": "alice", "email": "alice@example.com",
         "npi": "1234567890", "taxonomy": "207Q00000X", "state": "CA"},
        {"lead_id": "BAD1", "name": "bob smith", "email": "bob@example.com",
         "npi": "99887", "taxonomy": "207Q00000X", "state": "CA"},  # bad NPI
    ])
    path = tmp_path / "leads.csv"
    df.to_csv(path, index=False)
    ingest(path, store_dir, enricher=NPPESClient(stub={}))

    raw = (store_dir / "quarantine.csv").read_text()
    # no raw identifiers (name / email / NPI / lead_id) anywhere in the file
    for pii in ("bob smith", "bob", "bob@example.com", "example.com",
                "99887", "BAD1"):
        assert pii not in raw
    quarantine = pd.read_csv(store_dir / "quarantine.csv")
    assert len(quarantine) == 1
    row = quarantine.iloc[0]
    # same sha256 primary-key scheme as the leads store (NPI hash first)
    assert row["lead_hash"] == hash_id("npi:99887")
    assert row["reason"] == "invalid_npi"
    assert isinstance(row["ts"], str) and row["ts"]
