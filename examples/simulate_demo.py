"""End-to-end synthetic demo for psyche-engine (no network).

Generates a synthetic lead panel with two latent segments whose response
rates drift over time, runs ingest + a sequence of nightly update batches
(stub enrichment), and prints the out-of-sample Brier trend and final
persona summary.

Run:  python examples/simulate_demo.py [workdir]
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from psyche.enrich import NPPESClient
from psyche.pipeline import ingest, show, update
from psyche.update import lambda_from_n_max

TAXONOMIES = ["207Q00000X", "207R00000X", "208D00000X"]
STATES = ["CA", "TX", "NY", "FL"]


def synth_leads(rng: np.random.Generator, n: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "lead_id": f"D{i:04d}",
            "name": f"synthetic lead {i}",
            "email": f"lead{i}@synthetic.example",
            "npi": str(1900000000 + i),
            "taxonomy": rng.choice(TAXONOMIES, p=[0.4, 0.35, 0.25]),
            "state": rng.choice(STATES, p=[0.4, 0.3, 0.2, 0.1]),
        })
    return pd.DataFrame(rows)


def synth_margins() -> pd.DataFrame:
    rows = [{"dimension": "taxonomy", "level": t, "population": p}
            for t, p in zip(TAXONOMIES, [5200, 4100, 2700], strict=True)]
    rows += [{"dimension": "state", "level": s, "population": p}
             for s, p in zip(STATES, [4800, 3600, 2400, 1200], strict=True)]
    return pd.DataFrame(rows)


def synth_batch(rng: np.random.Generator, leads: pd.DataFrame, day: int,
                drift: float) -> pd.DataFrame:
    """One daily batch. Two latent segments whose response propensity drifts
    upward by ``drift`` per day (concept drift for the engine to track)."""
    rows = []
    p_high = min(0.55 + drift * day, 0.92)
    p_low = min(0.15 + drift * day / 2, 0.6)
    for i, row in leads.iterrows():
        high = i % 2 == 0  # deterministic latent segment
        if rng.random() < 0.7:
            p = p_high if high else p_low
            rows.append({"lead_id": row.lead_id, "attribute": "responds_email",
                         "value": int(rng.random() < p),
                         "ts": f"2024-01-{day + 1:02d}", "source": "campaign"})
        if rng.random() < 0.3:
            probs = [0.6, 0.3, 0.1] if high else [0.15, 0.65, 0.2]
            channel = ["email", "phone", "sms"][rng.choice(3, p=probs)]
            rows.append({"lead_id": row.lead_id, "attribute": "preferred_channel",
                         "value": channel,
                         "ts": f"2024-01-{day + 1:02d}", "source": "crm"})
    return pd.DataFrame(rows)


def main() -> None:
    workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="psyche_demo_"))
        cleanup = True
    else:
        cleanup = False
    workdir.mkdir(parents=True, exist_ok=True)
    store_dir = workdir / "store"
    rng = np.random.default_rng(2024)

    print(f"demo workdir: {workdir}")
    leads = synth_leads(rng)
    leads_csv = workdir / "leads.csv"
    margins_csv = workdir / "population_margins.csv"
    leads.to_csv(leads_csv, index=False)
    synth_margins().to_csv(margins_csv, index=False)

    stub = {row.npi: {"taxonomy": row.taxonomy, "state": row.state,
                      "enumeration_date": "2015-01-01", "stale": False}
            for row in leads.itertuples()}

    with NPPESClient(stub=stub) as enricher:
        store = ingest(leads_csv, store_dir, margins_csv=margins_csv,
                       enricher=enricher)
    print(f"ingested {len(leads)} leads; seeded {len(store.personas)} persona(s) "
          f"(lambda={lambda_from_n_max(200.0):.3f}, N_max=200)")

    drift = 0.004  # per-day drift of the true response rates
    for day in range(14):
        batch = synth_batch(rng, leads, day, drift)
        batch_csv = workdir / f"confirmations_day{day:02d}.csv"
        batch.to_csv(batch_csv, index=False)
        summary = update(batch_csv, store_dir)
        brier = {r["attribute"]: r["brier"] for r in summary.get("brier", [])}
        print(f"day {day + 1:02d}: n={summary.get('n_confirmations', 0):3d} "
              f"ess={summary.get('ess', 0):5.1f} "
              f"personas={summary.get('n_personas', '-')} "
              f"brier(responds_email)={brier.get('responds_email', float('nan')):.4f} "
              f"actions={summary.get('actions')}")

    print()
    print(show(store_dir))
    print()
    print("note: response rates drifted +0.4%/day; decayed personas track the")
    print("drift within bias <= delta*lambda/(1-lambda) "
          f"= {drift * lambda_from_n_max(200.0) / (1 - lambda_from_n_max(200.0)):.4f}")
    if cleanup:
        shutil.rmtree(workdir, ignore_errors=True)
        print(f"(cleaned up {workdir})")


if __name__ == "__main__":
    main()
