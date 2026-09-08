"""End-to-end pipeline: ingest, nightly update, show (WORKFLOWS.md §1-§2).

Privacy hard requirement: only hashed lead IDs, cell assignments and
aggregate counts are persisted. Raw names/emails/NPIs are hashed in memory
immediately after use and never written to the store.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from psyche.enrich import NPPESClient, hash_id
from psyche.metrics import append_history, read_history, score_batch
from psyche.models import BetaAttribute, DirichletAttribute, Persona, PersonaStore
from psyche.raking import kish_ess, load_margins_csv, rake
from psyche.update import (
    Confirmation,
    em_batch,
    group_by_lead,
    prune_underweight_personas,
    structural_adaptation,
)

LEADS_FILENAME = "leads.csv"
QUARANTINE_FILENAME = "quarantine.csv"

LEAD_COLUMNS = ["lead_id", "name", "email", "npi", "taxonomy", "state"]
CONFIRMATION_COLUMNS = ["lead_id", "attribute", "value", "ts", "source"]

DEFAULT_ATTRIBUTES: dict = {
    "responds_email": {"type": "binary", "base_rate": 0.30},
    "preferred_channel": {
        "type": "categorical",
        "levels": ["email", "phone", "sms"],
        "base_shares": [0.5, 0.3, 0.2],
    },
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _lead_hash_for(npi: str | None, email: str | None, lead_id: str) -> str:
    """Primary key: NPI hash when present, else email hash, else id hash."""
    if npi:
        return hash_id(f"npi:{npi}")
    if email:
        return hash_id(f"email:{email}")
    return hash_id(f"id:{lead_id}")


def _confirmation_lead_hash(lead_id: str) -> str:
    """Hash used to join raw confirmation lead_ids against stored leads."""
    return hash_id(f"id:{lead_id}")


def _quarantine_record(row: pd.Series, reason: str) -> dict:
    """Privacy-safe quarantine row: hashed lead ID + reason + timestamp only.

    Raw names/emails/NPIs are never persisted (SPEC privacy requirement), not
    even in the quarantine bucket; the hashed ID keeps rows auditable.
    """
    lead_id = row.get("lead_id", "").strip()
    email = row.get("email", "").strip()
    npi = row.get("npi", "").strip()
    return {
        "lead_hash": _lead_hash_for(npi or None, email or None, lead_id),
        "reason": reason,
        "ts": datetime.now(UTC).isoformat(),
    }


def _seed_persona(attributes_spec: dict, kappa: float, n_max: float,
                  ts: str | None = None) -> Persona:
    """Seed a persona from the global prior: alpha0 = kappa * base_rate (§5.1)."""
    from psyche.update import lambda_from_n_max

    attrs = {}
    for name, spec in attributes_spec.items():
        if spec["type"] == "binary":
            p = float(spec["base_rate"])
            attrs[name] = BetaAttribute(alpha0=kappa * p, beta0=kappa * (1.0 - p))
        else:
            shares = spec.get("base_shares") or [1.0 / len(spec["levels"])] * len(spec["levels"])
            attrs[name] = DirichletAttribute(levels=list(spec["levels"]),
                                             alphas0=[kappa * s for s in shares])
    return Persona.new(attrs, pi=1.0, lam=lambda_from_n_max(n_max), n_max=n_max,
                       event="seeded", ts=ts)


def _register_attribute(store: PersonaStore, name: str, kind: str,
                        observed_values: list, ts: str | None) -> None:
    """Register a previously unseen attribute on every persona (cold start §5.1)."""
    kappa = float(store.config["kappa"])
    registry = store.config.setdefault("attributes", {})
    if name in registry:
        return
    if kind == "binary":
        registry[name] = {"type": "binary", "base_rate": 0.5}
        new_attr = BetaAttribute(alpha0=kappa * 0.5, beta0=kappa * 0.5)
    else:
        levels = sorted({str(v) for v in observed_values})
        share = 1.0 / len(levels)
        registry[name] = {"type": "categorical", "levels": levels,
                          "base_shares": [share] * len(levels)}
        new_attr = DirichletAttribute(levels=levels, alphas0=[kappa * share] * len(levels))
    for persona in store.personas:
        if name not in persona.attributes:
            persona.attributes[name] = new_attr.__class__.from_dict(new_attr.to_dict())
            persona.log_lineage("attribute_registered", ts, {"attribute": name})


# ---------------------------------------------------------------------------
# ingest (WORKFLOWS §1)
# ---------------------------------------------------------------------------


def ingest(leads_csv: str | Path, store_dir: str | Path,
           margins_csv: str | Path | None = None,
           enricher: NPPESClient | None = None,
           attributes: dict | None = None,
           seed: bool = True) -> PersonaStore:
    """Validate, dedupe, enrich, rake and persist leads (hashed ids only).

    Seeds the persona store from global priors when empty and ``seed`` is set.
    """
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    store = PersonaStore.load(store_dir)
    if attributes is not None:
        store.config["attributes"] = attributes
    elif not store.config.get("attributes"):
        store.config["attributes"] = dict(DEFAULT_ATTRIBUTES)

    df = pd.read_csv(leads_csv, dtype=str).fillna("")
    valid_rows: list[dict] = []
    quarantine: list[dict] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        lead_id = row.get("lead_id", "").strip()
        email = row.get("email", "").strip()
        npi = row.get("npi", "").strip()
        if not lead_id and not email and not npi:
            quarantine.append(_quarantine_record(row, "missing_identifiers"))
            continue
        if npi and not (npi.isdigit() and len(npi) == 10):
            quarantine.append(_quarantine_record(row, "invalid_npi"))
            continue
        taxonomy, state = row.get("taxonomy", "").strip(), row.get("state", "").strip()
        low_confidence = not npi
        enrichment: dict = {}
        if enricher is not None and npi:
            try:
                enrichment = enricher.lookup(npi) or {}
            except ConnectionError:
                enrichment = {}
            taxonomy = enrichment.get("taxonomy") or taxonomy
            state = enrichment.get("state") or state
        lead_hash = _lead_hash_for(npi or None, email or None, lead_id)
        if lead_hash in seen:  # dedupe: NPI hash first, then email hash
            quarantine.append(_quarantine_record(row, "duplicate"))
            continue
        seen.add(lead_hash)
        valid_rows.append({
            "lead_hash": lead_hash,
            "lead_id_hash": _confirmation_lead_hash(lead_id) if lead_id else "",
            "npi_hash": hash_id(f"npi:{npi}") if npi else "",
            "taxonomy": taxonomy,
            "state": state,
            "low_confidence": low_confidence,
            "enrichment_stale": bool(enrichment.get("stale", False)),
        })

    leads = pd.DataFrame(valid_rows)
    if quarantine:
        pd.DataFrame(quarantine).to_csv(store_dir / QUARANTINE_FILENAME, index=False)

    # raking against population margins (ALGORITHM §3); weights frozen per §3.3
    if margins_csv is not None and len(leads):
        margins = load_margins_csv(str(margins_csv))
        result = rake(leads[["taxonomy", "state"]], margins,
                      eps=float(store.config["eps_rake"]))
        leads["w"] = result.w
        leads["w_tilde"] = result.w_tilde
        leads["weight_flag"] = [f or "" for f in result.weight_flags]
    elif len(leads):
        leads["w"] = 1.0
        leads["w_tilde"] = 1.0
        leads["weight_flag"] = ""

    if len(leads):
        out = store_dir / LEADS_FILENAME
        if out.exists():
            previous = pd.read_csv(out, dtype=str).fillna("")
            leads = pd.concat([previous, leads], ignore_index=True)
            leads = leads.drop_duplicates(subset="lead_hash", keep="last")
        leads.to_csv(out, index=False)

    # cold start: seed personas from global priors (ALGORITHM §5.1)
    if seed and len(store.personas) == 0 and store.config.get("attributes"):
        persona = _seed_persona(store.config["attributes"],
                                float(store.config["kappa"]),
                                float(store.config["n_max"]))
        store.personas = [persona]

    store.save(store_dir)
    return store


# ---------------------------------------------------------------------------
# nightly update (WORKFLOWS §2, ALGORITHM §6)
# ---------------------------------------------------------------------------


def _load_confirmations(confirmations_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(confirmations_csv, dtype=str).fillna("")
    missing = [c for c in CONFIRMATION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"confirmations CSV missing columns: {missing}")
    return df


def update(confirmations_csv: str | Path, store_dir: str | Path,
           margins_csv: str | Path | None = None) -> dict:
    """Run one nightly update cycle (ALGORITHM §6 pseudocode, invariants i-v)."""
    store_dir = Path(store_dir)
    store = PersonaStore.load(store_dir)
    df = _load_confirmations(confirmations_csv)

    # fetch only events newer than last_update_ts; exit clean when none
    if store.last_update_ts is not None:
        df = df[df["ts"] > store.last_update_ts]
    if df.empty:
        return {"status": "no_new_events", "store_dir": str(store_dir)}
    batch_ts = str(df["ts"].max())

    # register unseen attributes (typed events: binary | categorical)
    for attr, group in df.groupby("attribute"):
        if attr not in store.config.get("attributes", {}):
            kind = group["kind"].iloc[0] if "kind" in group.columns else ""
            if kind not in ("binary", "categorical"):
                kind = "binary" if set(group["value"].unique()) <= {"0", "1"} \
                    else "categorical"
            _register_attribute(store, attr, kind, list(group["value"]), ts=batch_ts)

    if len(store.personas) == 0:
        store.personas = [_seed_persona(store.config["attributes"],
                                        float(store.config["kappa"]),
                                        float(store.config["n_max"]), ts=batch_ts)]

    # resolve leads: raw lead_id -> stored hash; unknown leads get median weight
    leads_path = store_dir / LEADS_FILENAME
    if leads_path.exists():
        leads = pd.read_csv(leads_path, dtype=str).fillna("")
        if margins_csv is not None and len(leads):
            # leads changed since ingestion: re-rake the known panel (§3)
            margins = load_margins_csv(str(margins_csv))
            result = rake(leads[["taxonomy", "state"]], margins,
                          eps=float(store.config["eps_rake"]))
            leads["w"] = result.w
            leads["w_tilde"] = result.w_tilde
            leads.to_csv(leads_path, index=False)
    else:
        leads = pd.DataFrame(columns=["lead_id_hash", "w", "w_tilde"])
    id_to_w = dict(zip(leads.get("lead_id_hash", []),
                       pd.to_numeric(leads.get("w", pd.Series(dtype=float)),
                                     errors="coerce").fillna(1.0), strict=False))
    median_w = float(np.median(list(id_to_w.values()))) if id_to_w else 1.0

    confirmations: list[Confirmation] = []
    for _, row in df.iterrows():
        raw_value = row["value"]
        attr = row["attribute"]
        spec = store.config["attributes"].get(attr, {})
        value: int | str = int(raw_value) if spec.get("type") == "binary" else raw_value
        confirmations.append(Confirmation(lead_hash=_confirmation_lead_hash(row["lead_id"]),
                                          attribute=attr, value=value, ts=row["ts"],
                                          source=row.get("source", "unknown")))

    # out-of-sample Brier BEFORE any decay / EM consumes the batch (invariant v)
    records = score_batch(store.personas, confirmations, batch_ts=batch_ts)
    append_history(records, store_dir / "metrics")

    # group per lead; frozen raw weights -> batch ESS normalization (§3.3)
    lead_ids, obs_list = group_by_lead(confirmations)
    first_w: dict[str, float] = {}
    for _, row in df.iterrows():
        h = _confirmation_lead_hash(row["lead_id"])
        first_w.setdefault(h, id_to_w.get(h, median_w))
    w_raw = np.array([first_w[h] for h in lead_ids], dtype=float)

    ess = kish_ess(w_raw)
    w_tilde = w_raw * ess / w_raw.sum() if w_raw.sum() > 0 else w_raw

    # decay once, pre-E-step (invariants i, iii); dt in days
    dt = 0.0
    if store.last_update_ts is not None:
        dt = max(0.0, (pd.Timestamp(batch_ts)
                       - pd.Timestamp(store.last_update_ts)).total_seconds() / 86400.0)

    result = em_batch(store, obs_list, w_tilde, dt=dt,
                      tol=float(store.config["em_tol"]),
                      max_iter=int(store.config["em_max_iter"]))

    prune_underweight_personas(store, result.mass, float(store.config["n_min"]),
                               ts=batch_ts)
    R = result.R
    if R.shape[1] != len(store.personas):  # defensive; prune never drops personas
        from psyche.update import e_step
        R = e_step(store.personas, obs_list)

    actions = structural_adaptation(store, obs_list, R, w_tilde, n_eff=ess,
                                    ts=batch_ts, base=result.base_counts)

    store.last_update_ts = batch_ts
    store.save(store_dir)

    return {
        "status": "updated",
        "batch_ts": batch_ts,
        "n_confirmations": len(confirmations),
        "n_leads": len(lead_ids),
        "ess": ess,
        "log_likelihood": result.log_likelihood,
        "em_iterations": result.iterations,
        "em_converged": result.converged,
        "brier": records,
        "actions": actions,
        "n_personas": len(store.personas),
    }


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def show(store_dir: str | Path) -> str:
    """Human-readable summary of the persona store and metrics trend."""
    store_dir = Path(store_dir)
    store = PersonaStore.load(store_dir)
    lines = [f"persona store: {store_dir}  (last update: {store.last_update_ts})",
             f"personas: {len(store.personas)}", ""]
    for rank, p in enumerate(store.personas):
        lines.append(f"[{rank:02d}] persona {p.id}  pi={p.pi:.4f}  "
                     f"mean_entropy={p.mean_entropy():.3f} nats  "
                     f"lambda={p.lam} (N_max={p.n_max:g})")
        for name, a in p.attributes.items():
            if isinstance(a, BetaAttribute):
                lines.append(f"     {name}: q={a.mean:.3f} "
                             f"(alpha={a.alpha:.2f}, beta={a.beta:.2f}, "
                             f"H={a.entropy():.3f})")
            else:
                means = ", ".join(f"{k}={v:.3f}" for k, v in a.mean.items())
                lines.append(f"     {name}: {means} "
                             f"(alpha0={a.alpha_total:.2f}, H={a.entropy():.3f})")
        if p.lineage:
            events = ", ".join(e["event"] for e in p.lineage)
            lines.append(f"     lineage: {events}")
    history = read_history(store_dir / "metrics")
    if history:
        lines.append("")
        lines.append("metrics (recent out-of-sample Brier):")
        for rec in history[-5:]:
            lines.append(f"     {rec['ts']}  {rec['attribute']}: "
                         f"brier={rec['brier']:.4f} (n={rec['n']})")
    return "\n".join(lines)
