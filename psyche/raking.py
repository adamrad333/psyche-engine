"""Post-stratification / raking against population margins (ALGORITHM §3).

Iterative proportional fitting (IPF) toward population margins (e.g.
NPPES provider taxonomy x state), Kish effective sample size, and
ESS-normalized design weights ``w_tilde`` with ``sum(w_tilde) == ESS <= n``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RakeResult:
    w: np.ndarray                    # raked weights after trimming
    w_tilde: np.ndarray              # ESS-normalized weights
    ess: float                       # Kish effective sample size
    deff: float                      # design effect n / ESS
    weight_flags: list[str | None]   # per-lead flags, e.g. "cell_not_in_margins"
    iterations: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"w": self.w, "w_tilde": self.w_tilde,
                             "weight_flag": self.weight_flags})


def kish_ess(w: np.ndarray) -> float:
    """Kish ESS = (sum w)^2 / sum w^2 (ALGORITHM §3.3)."""
    w = np.asarray(w, dtype=float)
    denom = float((w**2).sum())
    return float(w.sum() ** 2 / denom) if denom > 0 else 0.0


def rake(cells: pd.DataFrame,
         margins: dict[str, dict[str, float]],
         eps: float = 1e-6,
         max_iter: int = 1000,
         trim_quantile: float = 0.99) -> RakeResult:
    """Rake unit base weights toward population margins via IPF (§3.2).

    ``cells`` has one row per lead and one column per calibration dimension
    (e.g. ``taxonomy``, ``state``). ``margins[dim][level]`` is the population
    total for that level. Leads whose cell level is absent from a margin are
    frozen at the median valid weight and flagged (§3.3 edge cases). Weights
    are trimmed at ``trim_quantile`` before ESS normalization (§3.3).
    """
    n = len(cells)
    if n == 0:
        raise ValueError("no leads to rake")
    dims = [d for d in margins if d in cells.columns]
    if not dims:
        raise ValueError("no calibration dimension present in cells")

    # mask of leads whose levels are all present in the margins
    known = np.ones(n, dtype=bool)
    for d in dims:
        levels = cells[d].astype(str).to_numpy()
        known &= np.isin(levels, list(margins[d].keys()))

    w = np.where(known, 1.0, np.nan)
    iterations = 0

    def margin_error(weights: np.ndarray) -> float:
        err = 0.0
        for d in dims:
            levels = cells[d].astype(str).to_numpy()
            for level, target in margins[d].items():
                mask = (levels == level) & known
                current = float(weights[mask].sum())
                err = max(err, abs(current - target) / max(abs(target), 1e-12))
        return err

    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        for d in dims:
            levels = cells[d].astype(str).to_numpy()
            for level, target in margins[d].items():
                mask = (levels == level) & known
                current = float(np.nansum(w[mask]))
                if current > 0:
                    w[mask] *= target / current
        if margin_error(np.nan_to_num(w, nan=0.0)) < eps:
            break

    # freeze unknown-cell leads at the median valid weight, flag them (§3.3)
    flags: list[str | None] = [None] * n
    if not known.all():
        median_w = float(np.nanmedian(w)) if known.any() else 1.0
        w[~known] = median_w
        for i in np.flatnonzero(~known):
            flags[i] = "cell_not_in_margins"

    # trim leverage at the quantile BEFORE ESS normalization (§3.3)
    cap = float(np.quantile(w, trim_quantile))
    if cap > 0:
        w = np.minimum(w, cap)

    ess = kish_ess(w)
    total = float(w.sum())
    w_tilde = w * ess / total if total > 0 else w.copy()
    deff = n / ess if ess > 0 else float("inf")
    return RakeResult(w=w, w_tilde=w_tilde, ess=ess, deff=deff,
                      weight_flags=flags, iterations=iterations)


def load_margins_csv(path: str) -> dict[str, dict[str, float]]:
    """Load margins from CSV with columns ``dimension, level, population``."""
    df = pd.read_csv(path)
    margins: dict[str, dict[str, float]] = {}
    for dim, group in df.groupby("dimension"):
        margins[str(dim)] = {str(k): float(v)
                             for k, v in zip(group["level"], group["population"], strict=True)}
    return margins
