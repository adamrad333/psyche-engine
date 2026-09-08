"""Update engine: EM-style soft updates, recency decay, structural adaptation.

Implements ALGORITHM.md sections 2 (E/M steps), 4 (recency decay) and 5.2-5.3
(BIC, split/merge). Hard invariants (ALGORITHM §6):

(i)   priors are immutable after seeding - decay applies only to counts;
(ii)  every count increment carries weight rho_is = w_tilde_i * r_is;
(iii)  decay is applied exactly once per batch to pre-batch counts, and the
      first E-step runs on post-decay, pre-M-step counts;
(iv)  lambda = 1 - 1/N_max is stored per persona;
(v)   Brier scores are computed before consumption (see psyche.metrics).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from psyche.models import PROB_FLOOR, BetaAttribute, Persona, PersonaStore

# ---------------------------------------------------------------------------
# Confirmations
# ---------------------------------------------------------------------------


@dataclass
class Confirmation:
    """A typed observation of one persona attribute for one lead.

    ``value`` is 0/1 for binary attributes, a level string for categorical.
    ``lead_hash`` is a sha256 hash - never a raw identifier.
    """

    lead_hash: str
    attribute: str
    value: int | str
    ts: str
    source: str = "unknown"


def group_by_lead(confirmations: list[Confirmation]) -> tuple[list[str], list[dict]]:
    """Group a batch of confirmations into per-lead observation dicts.

    Returns (lead_hashes, obs_list) with stable first-seen ordering.
    """
    leads: list[str] = []
    obs: dict[str, dict] = {}
    for c in confirmations:
        if c.lead_hash not in obs:
            obs[c.lead_hash] = {}
            leads.append(c.lead_hash)
        obs[c.lead_hash][c.attribute] = c.value
    return leads, [obs[h] for h in leads]


# ---------------------------------------------------------------------------
# Decay (ALGORITHM §4)
# ---------------------------------------------------------------------------


def lambda_from_n_max(n_max: float) -> float:
    """lambda = 1 - 1/N_max (ALGORITHM §4.1)."""
    if n_max <= 1.0:
        raise ValueError("n_max must be > 1")
    return 1.0 - 1.0 / n_max


def n_max_from_lambda(lam: float) -> float:
    return 1.0 / (1.0 - lam)


def stationary_bias_bound(delta: float, lam: float) -> float:
    """Stationary drift bias bound delta * lambda / (1 - lambda) (§4.3a)."""
    return delta * lam / (1.0 - lam)


def decay_counts(store: PersonaStore, dt: float) -> None:
    """Decay all personas' data counts by lam**dt. Priors untouched.

    Must be called exactly once per batch, before the E-step.
    """
    if dt < 0:
        raise ValueError("dt must be non-negative")
    for persona in store.personas:
        persona.decay_counts(dt)


# ---------------------------------------------------------------------------
# E-step (ALGORITHM §2.1)
# ---------------------------------------------------------------------------


def e_step(personas: list[Persona], obs_list: list[dict]) -> np.ndarray:
    """Responsibilities r_is in log space from posterior predictives.

    Returns an (n, S) matrix whose rows sum to 1.
    """
    n, s_count = len(obs_list), len(personas)
    if n == 0:
        return np.zeros((0, s_count))
    log_r = np.empty((n, s_count), dtype=float)
    pis = np.array([max(p.pi, PROB_FLOOR) for p in personas])
    for i, obs in enumerate(obs_list):
        for s, persona in enumerate(personas):
            log_r[i, s] = math.log(pis[s]) + persona.predictive_log_likelihood(obs)
    log_r -= logsumexp(log_r, axis=1, keepdims=True)
    return np.exp(log_r)


def log_likelihood(personas: list[Persona], obs_list: list[dict],
                   w_tilde: np.ndarray | None = None) -> float:
    """(Weighted) observed-data log-likelihood sum_i w_i log p(x_i | Theta)."""
    n, s_count = len(obs_list), len(personas)
    if n == 0:
        return 0.0
    log_p = np.empty((n, s_count), dtype=float)
    pis = np.array([max(p.pi, PROB_FLOOR) for p in personas])
    for i, obs in enumerate(obs_list):
        for s, persona in enumerate(personas):
            log_p[i, s] = math.log(pis[s]) + persona.predictive_log_likelihood(obs)
    ll = logsumexp(log_p, axis=1)
    if w_tilde is None:
        return float(ll.sum())
    return float(np.dot(np.asarray(w_tilde, dtype=float), ll))


# ---------------------------------------------------------------------------
# M-step (ALGORITHM §2.2)
# ---------------------------------------------------------------------------

CountsSnapshot = list[dict[str, tuple[float, float] | list[float]]]


def snapshot_counts(personas: list[Persona]) -> CountsSnapshot:
    snap: CountsSnapshot = []
    for persona in personas:
        d: dict[str, tuple[float, float] | list[float]] = {}
        for name, a in persona.attributes.items():
            if isinstance(a, BetaAttribute):
                d[name] = (a.A, a.B)
            else:
                d[name] = list(a.A)
        snap.append(d)
    return snap


def m_step(personas: list[Persona], base: CountsSnapshot, obs_list: list[dict],
           R: np.ndarray, w_tilde: np.ndarray) -> np.ndarray:
    """Weighted conjugate M-step.

    Counts are recomputed from the post-decay snapshot ``base`` plus this
    batch's increments, each weighted by rho_is = w_tilde_i * r_is. Mixing
    weights become pi_s proportional to sum_i rho_is.

    Returns the per-persona responsibility mass n_s = sum_i rho_is.
    """
    w_tilde = np.asarray(w_tilde, dtype=float)
    rho = R * w_tilde[:, None]  # (n, S)
    mass = rho.sum(axis=0)
    for s, persona in enumerate(personas):
        for name, a in persona.attributes.items():
            if isinstance(a, BetaAttribute):
                a0, b0 = base[s][name]
                hits = sum(rho[i, s] * int(obs.get(name, 0))
                           for i, obs in enumerate(obs_list) if name in obs)
                total = sum(rho[i, s] for i, obs in enumerate(obs_list) if name in obs)
                a.A = a0 + hits
                a.B = b0 + (total - hits)
            else:
                base_a = base[s][name]
                new_a = list(base_a)
                for i, obs in enumerate(obs_list):
                    if name in obs:
                        new_a[a.levels.index(str(obs[name]))] += rho[i, s]
                a.A = new_a
    total_mass = mass.sum()
    if total_mass > 0:
        for s, persona in enumerate(personas):
            persona.pi = float(mass[s] / total_mass)
    return mass


@dataclass
class EMResult:
    R: np.ndarray
    mass: np.ndarray          # n_s = sum_i rho_is per persona
    log_likelihood: float
    iterations: int
    converged: bool
    base_counts: CountsSnapshot = field(repr=False, default=None)  # type: ignore[assignment]


def em_batch(store: PersonaStore, obs_list: list[dict], w_tilde: np.ndarray,
             dt: float = 0.0, tol: float = 1e-6, max_iter: int = 100,
             apply_decay: bool = True) -> EMResult:
    """One nightly batch: decay once, then EM to convergence (ALGORITHM §6).

    The M-step always rebuilds counts from the post-decay snapshot, so EM
    iterations never double-count the batch.
    """
    if apply_decay and dt > 0:
        decay_counts(store, dt)
    base = snapshot_counts(store.personas)
    w_tilde = np.asarray(w_tilde, dtype=float)
    R = e_step(store.personas, obs_list)
    prev_ll = -np.inf
    ll = log_likelihood(store.personas, obs_list, w_tilde)
    mass = R.sum(axis=0) * 0.0
    iterations = 0
    converged = False
    for _ in range(max_iter):
        iterations += 1
        mass = m_step(store.personas, base, obs_list, R, w_tilde)
        R_new = e_step(store.personas, obs_list)
        ll = log_likelihood(store.personas, obs_list, w_tilde)
        d_r = float(np.abs(R_new - R).max()) if R.size else 0.0
        R = R_new
        if ll - prev_ll < tol * max(1.0, abs(ll)) and d_r < math.sqrt(tol):
            converged = True
            break
        prev_ll = ll
    return EMResult(R=R, mass=mass, log_likelihood=ll, iterations=iterations,
                    converged=converged, base_counts=base)


def prune_underweight_personas(store: PersonaStore, mass: np.ndarray,
                               n_min: float, ts: str | None = None) -> list[int]:
    """Fold personas with n_s <= n_min back into the prior (ALGORITHM §2).

    Guarded so at least one persona always retains its counts.
    """
    folded: list[int] = []
    survivors = [s for s in range(len(store.personas)) if mass[s] > n_min]
    if not survivors:
        return folded  # never fold the whole store
    for s, persona in enumerate(store.personas):
        if mass[s] <= n_min and any(a_has_counts(a) for a in persona.attributes.values()):
            persona.reset_counts()
            persona.log_lineage("folded", ts, {"mass": float(mass[s]), "n_min": n_min})
            folded.append(s)
    return folded


def a_has_counts(a) -> bool:
    if isinstance(a, BetaAttribute):
        return a.A > 0 or a.B > 0
    return any(x > 0 for x in a.A)


# ---------------------------------------------------------------------------
# BIC (ALGORITHM §5.2)
# ---------------------------------------------------------------------------


def persona_dof(personas: list[Persona]) -> int:
    """d_S = S * (|J_B| + sum_j (K_j - 1)) + (S - 1)."""
    s_count = len(personas)
    if s_count == 0:
        return 0
    per = 0
    for a in personas[0].attributes.values():
        per += 1 if isinstance(a, BetaAttribute) else a.k - 1
    return s_count * per + (s_count - 1)


def bic(personas: list[Persona], obs_list: list[dict], w_tilde: np.ndarray,
        n_eff: float) -> float:
    ll = log_likelihood(personas, obs_list, w_tilde)
    d = persona_dof(personas)
    return -2.0 * ll + d * math.log(max(n_eff, 1.0))


# ---------------------------------------------------------------------------
# Split / merge (ALGORITHM §5.3)
# ---------------------------------------------------------------------------


def sym_kl(p: Persona, q: Persona) -> float:
    """Symmetrized KL between predictive vectors, averaged per attribute."""
    pv, qv = p.predictive_vector(), q.predictive_vector()
    total, count = 0.0, 0
    for name in pv:
        if name not in qv:
            continue
        for lev in pv[name]:
            a = max(pv[name][lev], PROB_FLOOR)
            b = max(qv[name].get(lev, 0.0), PROB_FLOOR)
            total += a * math.log(a / b) + b * math.log(b / a)
        count += 1
    return total / max(count, 1)


def _feature_matrix(persona: Persona, obs_list: list[dict]) -> np.ndarray:
    """Confirmation vectors: binary attrs -> 1 dim, categorical -> one-hot.

    Missing entries are filled with the persona's posterior predictive means
    so clustering runs on a complete matrix.
    """
    dims: list[tuple[str, int]] = []  # (attr, level index) where index -1 = binary
    for name, a in persona.attributes.items():
        if isinstance(a, BetaAttribute):
            dims.append((name, -1))
        else:
            dims.extend((name, k) for k in range(a.k))
    X = np.zeros((len(obs_list), len(dims)))
    for i, obs in enumerate(obs_list):
        for j, (name, lev) in enumerate(dims):
            a = persona.attributes[name]
            if isinstance(a, BetaAttribute):
                X[i, j] = float(obs[name]) if name in obs else a.mean
            else:
                if name in obs:
                    X[i, j] = 1.0 if a.levels[lev] == str(obs[name]) else 0.0
                else:
                    X[i, j] = a.alphas[lev] / a.alpha_total
    return X


def _weighted_kmeans2(X: np.ndarray, weights: np.ndarray,
                      max_iter: int = 50) -> np.ndarray:
    """Deterministic weighted k-means with k=2. Returns cluster labels {0,1}."""
    n = len(X)
    if n == 1:
        return np.zeros(1, dtype=int)
    # init: heaviest lead and the lead farthest from it (deterministic)
    first = int(np.argmax(weights))
    d2 = ((X - X[first]) ** 2).sum(axis=1)
    second = int(np.argmax(d2))
    if d2[second] == 0.0:  # all points identical: split by weight rank
        order = np.argsort(-weights)
        labels = np.zeros(n, dtype=int)
        labels[order[n // 2:]] = 1
        return labels
    centroids = np.stack([X[first], X[second]])
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dist = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        # tie-break toward cluster 0 by construction of argmin
        if np.array_equal(new_labels, labels) and new_labels.any() and not new_labels.all():
            labels = new_labels
            break
        labels = new_labels
        for c in (0, 1):
            mask = labels == c
            if not mask.any():  # empty cluster: reseed at farthest point
                farthest = int(np.argmax(dist.min(axis=1)))
                labels[farthest] = c
                mask = labels == c
            w = weights[mask]
            centroids[c] = (X[mask] * w[:, None]).sum(axis=0) / w.sum()
    return labels


def _refit(personas: list[Persona], bases: CountsSnapshot, obs_list: list[dict],
           w_tilde: np.ndarray, iters: int = 25, tol: float = 1e-8) -> None:
    """Mini-EM refit of a candidate persona set against fixed historical bases.

    BIC (ALGORITHM §5.2) is defined at EM convergence, so split/merge
    candidates must be re-tuned before the likelihood comparison.
    """
    R = e_step(personas, obs_list)
    prev = -np.inf
    for _ in range(iters):
        m_step(personas, bases, obs_list, R, w_tilde)
        R = e_step(personas, obs_list)
        ll = log_likelihood(personas, obs_list, w_tilde)
        if ll - prev < tol * max(1.0, abs(ll)):
            break
        prev = ll


def try_split(store: PersonaStore, s_idx: int, obs_list: list[dict], R: np.ndarray,
              w_tilde: np.ndarray, n_eff: float, ts: str | None = None,
              base: CountsSnapshot | None = None) -> CountsSnapshot | None:
    """Split persona s into two children if entropy/mass thresholds hold and
    BIC decreases (ALGORITHM §5.3).

    Children are seeded with the parent's post-decay base counts plus
    cluster-conditional weighted batch counts (total pseudo-mass preserved),
    then the candidate model is re-tuned via EM before the BIC comparison
    (§5.2: BIC is computed at EM convergence). ``base`` must be the
    post-decay, pre-M-step snapshot aligned with ``store.personas``; without
    it the batch would be double-counted, so the split is refused.

    Returns the candidate bases (aligned with the new store.personas) on
    acceptance, None on rejection.
    """
    cfg = store.config
    persona = store.personas[s_idx]
    mass_s = float((R[:, s_idx] * w_tilde).sum())
    if persona.mean_entropy() <= cfg["tau_split"] or mass_s <= cfg["n_min"]:
        return None
    if len(store.personas) >= cfg["s_max"]:
        return None
    if base is None:
        return None  # need the pre-M-step snapshot for an honest refit

    bic_before = bic(store.personas, obs_list, w_tilde, n_eff)

    X = _feature_matrix(persona, obs_list)
    rho_s = R[:, s_idx] * w_tilde
    labels = _weighted_kmeans2(X, rho_s + 1e-12)
    if labels.min() == labels.max():
        return None  # could not find two clusters
    mass_c = np.array([rho_s[labels == c].sum() for c in (0, 1)])
    if mass_c.min() <= 0:
        return None

    base_s: dict = base[s_idx]
    children = []
    child_bases: CountsSnapshot = []
    for c in (0, 1):
        child = persona.copy_empty(pi=persona.pi * mass_c[c] / mass_s, ts=ts, event="split")
        child.lineage = list(persona.lineage)[:-1]  # inherit parent history, drop "split" seed
        share = mass_c[c] / mass_s
        cbase: dict[str, tuple[float, float] | list[float]] = {}
        for name, a in child.attributes.items():
            if isinstance(a, BetaAttribute):
                base_a, base_b = base_s[name]
                a.A = float(base_a) * share
                a.B = float(base_b) * share
                cbase[name] = (a.A, a.B)
                for i, obs in enumerate(obs_list):
                    if labels[i] == c and name in obs:
                        x = int(obs[name])
                        a.A += rho_s[i] * x
                        a.B += rho_s[i] * (1 - x)
            else:
                a.A = [x * share for x in base_s[name]]
                cbase[name] = list(a.A)
                for i, obs in enumerate(obs_list):
                    if labels[i] == c and name in obs:
                        a.A[a.levels.index(str(obs[name]))] += rho_s[i]
        child.log_lineage("split_child", ts,
                          {"parent": persona.id, "cluster_mass": float(mass_c[c]),
                           "parent_mean_entropy": persona.mean_entropy()})
        children.append(child)
        child_bases.append(cbase)

    candidate = [p for k, p in enumerate(store.personas) if k != s_idx] + children
    cand_bases = [base[k] for k in range(len(store.personas)) if k != s_idx] + child_bases
    _refit(candidate, cand_bases, obs_list, w_tilde)
    bic_after = bic(candidate, obs_list, w_tilde, n_eff)
    if not bic_after < bic_before:
        return None

    for child in children:
        child.lineage[-1]["detail"]["bic_before"] = bic_before
        child.lineage[-1]["detail"]["bic_after"] = bic_after
    store.personas = candidate
    return cand_bases


def try_merge(store: PersonaStore, s_idx: int, t_idx: int, obs_list: list[dict],
              w_tilde: np.ndarray, n_eff: float, ts: str | None = None,
              base: CountsSnapshot | None = None) -> CountsSnapshot | None:
    """Merge personas s,t if their predictive symmetrized KL is below
    tau_merge, the merged entropy stays below tau_split, and BIC decreases
    (ALGORITHM §5.3). Counts add. With ``base`` given, the merged candidate
    is re-tuned via EM before the BIC comparison (§5.2).

    Returns the candidate bases (aligned with the new store.personas) on
    acceptance, None on rejection."""
    cfg = store.config
    ps, pt = store.personas[s_idx], store.personas[t_idx]
    kl = sym_kl(ps, pt)
    if kl >= cfg["tau_merge"]:
        return None

    bic_before = bic(store.personas, obs_list, w_tilde, n_eff)

    merged = ps.copy_empty(pi=ps.pi + pt.pi, ts=ts, event="merge")
    merged.lineage = list(ps.lineage) + list(pt.lineage)  # both parents' history
    for name, a in merged.attributes.items():
        as_, at = ps.attributes[name], pt.attributes[name]
        if isinstance(a, BetaAttribute):
            a.A = as_.A + at.A  # type: ignore[union-attr]
            a.B = as_.B + at.B  # type: ignore[union-attr]
        else:
            a.A = [x + y for x, y in zip(as_.A, at.A, strict=True)]  # type: ignore[union-attr]
    if merged.mean_entropy() >= cfg["tau_split"]:
        return None

    candidate = [p for k, p in enumerate(store.personas) if k not in (s_idx, t_idx)]
    candidate.append(merged)
    cand_bases: CountsSnapshot | None = None
    if base is not None:
        merged_base: dict[str, tuple[float, float] | list[float]] = {}
        for name, a in merged.attributes.items():
            bs, bt = base[s_idx][name], base[t_idx][name]
            if isinstance(a, BetaAttribute):
                merged_base[name] = (bs[0] + bt[0], bs[1] + bt[1])  # type: ignore[index]
            else:
                merged_base[name] = [x + y for x, y in zip(bs, bt, strict=True)]  # type: ignore[arg-type]
        cand_bases = [base[k] for k in range(len(store.personas))
                      if k not in (s_idx, t_idx)] + [merged_base]
        _refit(candidate, cand_bases, obs_list, w_tilde)
    bic_after = bic(candidate, obs_list, w_tilde, n_eff)
    if not bic_after < bic_before:
        return None

    merged.log_lineage("merge", ts,
                       {"parents": [ps.id, pt.id], "sym_kl": kl,
                        "bic_before": bic_before, "bic_after": bic_after})
    store.personas = candidate
    if cand_bases is None:
        cand_bases = snapshot_counts(candidate)
    return cand_bases


def structural_adaptation(store: PersonaStore, obs_list: list[dict], R: np.ndarray,
                          w_tilde: np.ndarray, n_eff: float,
                          ts: str | None = None,
                          base: CountsSnapshot | None = None) -> dict:
    """Split pass then merge pass, each operation BIC-gated and lineage-logged.

    ``base`` is the post-decay, pre-M-step snapshot; it is kept aligned with
    ``store.personas`` as operations are accepted so later operations never
    double-count the batch.
    """
    actions: dict[str, list] = {"splits": [], "merges": []}
    bases = base
    # splits
    idx = 0
    while idx < len(store.personas):
        persona = store.personas[idx]
        # recompute responsibilities if the store changed shape
        if R.shape[1] != len(store.personas):
            R = e_step(store.personas, obs_list)
        new_bases = try_split(store, idx, obs_list, R, w_tilde, n_eff, ts=ts,
                              base=bases)
        if new_bases is not None:
            actions["splits"].append(persona.id)
            bases = new_bases
            R = e_step(store.personas, obs_list)
        idx += 1
    # merges
    changed = True
    while changed and len(store.personas) > 1:
        changed = False
        for s in range(len(store.personas)):
            for t in range(s + 1, len(store.personas)):
                if s >= len(store.personas) or t >= len(store.personas):
                    break
                ids = (store.personas[s].id, store.personas[t].id)
                new_bases = try_merge(store, s, t, obs_list, w_tilde, n_eff,
                                      ts=ts, base=bases)
                if new_bases is not None:
                    actions["merges"].append(ids)
                    bases = new_bases
                    changed = True
                    break
            if changed:
                break
    store.normalize_pis()
    return actions
