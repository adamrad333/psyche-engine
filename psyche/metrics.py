"""Out-of-sample predictive metrics (ALGORITHM §6 invariant v, WORKFLOWS §3).

Brier scores are computed on each confirmation with the persona mixture
posterior predictive *before* the confirmation is consumed by any update,
so the metric tracks genuine predictive skill. History is appended to
``metrics/history.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path

from psyche.models import BetaAttribute, Persona
from psyche.update import Confirmation


def brier_binary(q: float, x: int) -> float:
    """Brier score for a binary event: (q - x)^2 with q = P(x = 1)."""
    return (q - x) ** 2


def brier_categorical(qs: dict[str, float], observed: str) -> float:
    """Multiclass Brier score: sum_k (q_k - 1{k == observed})^2."""
    return sum((q - (1.0 if lev == observed else 0.0)) ** 2 for lev, q in qs.items())


def mixture_predictive(personas: list[Persona], attribute: str):
    """Mixture posterior predictive for one attribute under current pi.

    Returns q = P(x=1) for binary attributes, or {level: P(level)} for
    categorical attributes. Returns None if no persona has the attribute.
    """
    total_pi = sum(p.pi for p in personas if attribute in p.attributes)
    if total_pi <= 0:
        return None
    first = next(p.attributes[attribute] for p in personas if attribute in p.attributes)
    if isinstance(first, BetaAttribute):
        q = sum(p.pi * p.attributes[attribute].mean  # type: ignore[attr-defined]
                for p in personas if attribute in p.attributes)
        return q / total_pi
    levels = first.levels  # type: ignore[attr-defined]
    out = dict.fromkeys(levels, 0.0)
    for p in personas:
        if attribute not in p.attributes:
            continue
        means = p.attributes[attribute].mean  # type: ignore[attr-defined]
        for lev in levels:
            out[lev] += p.pi * means[lev]
    return {lev: v / total_pi for lev, v in out.items()}


def score_batch(personas: list[Persona], confirmations: list[Confirmation],
                batch_ts: str | None = None) -> list[dict]:
    """Score every confirmation out-of-sample, BEFORE it is consumed.

    Call this strictly before decay / EM. Returns one record per attribute:
    {"ts", "attribute", "brier", "n"}.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for c in confirmations:
        pred = mixture_predictive(personas, c.attribute)
        if pred is None:
            continue
        if isinstance(pred, float):
            bs = brier_binary(pred, int(c.value))
        else:
            bs = brier_categorical(pred, str(c.value))
        sums[c.attribute] = sums.get(c.attribute, 0.0) + bs
        counts[c.attribute] = counts.get(c.attribute, 0) + 1
    ts = batch_ts or (confirmations[-1].ts if confirmations else None)
    return [{"ts": ts, "attribute": attr,
             "brier": sums[attr] / counts[attr], "n": counts[attr]}
            for attr in sorted(sums)]


def append_history(records: list[dict], metrics_dir: str | Path) -> Path:
    """Append records to metrics/history.jsonl (append-only)."""
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "history.jsonl"
    with path.open("a") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return path


def read_history(metrics_dir: str | Path) -> list[dict]:
    path = Path(metrics_dir) / "history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
