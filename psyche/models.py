"""Data model: conjugate attribute posteriors, personas, and the persona store.

Implements ALGORITHM.md sections 0-1 and 5.3. Only sufficient statistics
(counts ``A``/``B``) and immutable priors (``alpha0``/``beta0``/``alphas0``)
are stored; posterior hyperparameters are always derived as prior + counts.

Privacy: nothing in this module ever sees raw PII. Lead identifiers are
sha256 hashes produced upstream (``psyche.enrich.hash_id``).
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import betaln, digamma, gammaln

PROB_FLOOR = 1e-9  # numerical safety floor for entropy / KL (ALGORITHM §7.3)


# ---------------------------------------------------------------------------
# Attribute posteriors (ALGORITHM §1)
# ---------------------------------------------------------------------------


@dataclass
class BetaAttribute:
    """Beta-Bernoulli attribute posterior (ALGORITHM §1.1).

    Stores immutable prior ``(alpha0, beta0)`` and decayed data counts
    ``(A, B)``; the posterior is ``Beta(alpha0 + A, beta0 + B)``.
    """

    alpha0: float
    beta0: float
    A: float = 0.0
    B: float = 0.0

    # -- derived hyperparameters ------------------------------------------
    @property
    def alpha(self) -> float:
        return self.alpha0 + self.A

    @property
    def beta(self) -> float:
        return self.beta0 + self.B

    # -- estimators (Appendix A) ------------------------------------------
    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def var(self) -> float:
        n0 = self.alpha + self.beta
        return self.alpha * self.beta / (n0**2 * (n0 + 1.0))

    @property
    def map_estimate(self) -> float:
        if self.alpha > 1.0 and self.beta > 1.0:
            return (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        return 1.0 if self.alpha > self.beta else 0.0

    # -- posterior predictive ----------------------------------------------
    def predictive(self, x: int) -> float:
        """P(x) for the next event, x in {0, 1} (ALGORITHM §1.1)."""
        p1 = self.mean
        return p1 if x == 1 else 1.0 - p1

    def log_predictive(self, x: int) -> float:
        return math.log(max(self.predictive(x), PROB_FLOOR))

    # -- exact differential entropy (ALGORITHM §5.3) ------------------------
    def entropy(self) -> float:
        a, b = self.alpha, self.beta
        return (
            betaln(a, b)
            - (a - 1.0) * digamma(a)
            - (b - 1.0) * digamma(b)
            + (a + b - 2.0) * digamma(a + b)
        )

    # -- mutation -----------------------------------------------------------
    def observe(self, x: int, weight: float) -> None:
        """Weighted conjugate increment (ALGORITHM §2.2)."""
        if x == 1:
            self.A += weight
        else:
            self.B += weight

    def decay(self, factor: float) -> None:
        """Decay data counts only; priors are immutable (ALGORITHM §4)."""
        self.A = max(0.0, self.A * factor)
        self.B = max(0.0, self.B * factor)

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "type": "binary",
            "alpha0": self.alpha0,
            "beta0": self.beta0,
            "A": self.A,
            "B": self.B,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BetaAttribute:
        return cls(alpha0=float(d["alpha0"]), beta0=float(d["beta0"]),
                   A=float(d.get("A", 0.0)), B=float(d.get("B", 0.0)))


@dataclass
class DirichletAttribute:
    """Dirichlet-Categorical attribute posterior (ALGORITHM §1.2).

    Stores immutable prior ``alphas0`` and decayed counts ``A[k]``; the
    posterior is ``Dirichlet(alphas0 + A)``.
    """

    levels: list[str]
    alphas0: list[float]
    A: list[float] | None = None

    def __post_init__(self) -> None:
        if self.A is None:
            self.A = [0.0] * len(self.levels)
        if not (len(self.levels) == len(self.alphas0) == len(self.A)):
            raise ValueError("levels, alphas0 and A must have equal length")

    @property
    def k(self) -> int:
        return len(self.levels)

    @property
    def alphas(self) -> list[float]:
        return [a0 + a for a0, a in zip(self.alphas0, self.A, strict=True)]

    @property
    def alpha_total(self) -> float:
        return sum(self.alphas)

    @property
    def prior_total(self) -> float:
        return sum(self.alphas0)

    # -- estimators (Appendix A) ------------------------------------------
    @property
    def mean(self) -> dict[str, float]:
        total = self.alpha_total
        return dict(zip(self.levels, [a / total for a in self.alphas], strict=True))

    def var(self, level: str) -> float:
        a = self.alphas[self.levels.index(level)]
        a0 = self.alpha_total
        return a * (a0 - a) / (a0**2 * (a0 + 1.0))

    @property
    def map_estimate(self) -> dict[str, float]:
        denom = self.alpha_total - self.k
        out = {}
        for lev, a in zip(self.levels, self.alphas, strict=True):
            out[lev] = (a - 1.0) / denom if a > 1.0 and denom > 0 else 0.0
        return out

    # -- posterior predictive ----------------------------------------------
    def predictive(self, level: str) -> float:
        return self.mean[level]

    def log_predictive(self, level: str) -> float:
        return math.log(max(self.predictive(level), PROB_FLOOR))

    # -- exact differential entropy (ALGORITHM §5.3) ------------------------
    def entropy(self) -> float:
        a = np.asarray(self.alphas, dtype=float)
        a0 = float(a.sum())
        log_b = float(gammaln(a).sum() - gammaln(a0))  # ln B(alpha)
        return float(log_b + (a0 - self.k) * digamma(a0)
                     - float(((a - 1.0) * digamma(a)).sum()))

    # -- mutation -----------------------------------------------------------
    def observe(self, level: str, weight: float) -> None:
        self.A[self.levels.index(level)] += weight

    def decay(self, factor: float) -> None:
        self.A = [max(0.0, a * factor) for a in self.A]

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "type": "categorical",
            "levels": list(self.levels),
            "alphas0": list(self.alphas0),
            "A": list(self.A),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DirichletAttribute:
        return cls(levels=list(d["levels"]), alphas0=[float(x) for x in d["alphas0"]],
                   A=[float(x) for x in d.get("A", [0.0] * len(d["levels"]))])


Attribute = BetaAttribute | DirichletAttribute


def attribute_from_dict(d: dict) -> Attribute:
    if d["type"] == "binary":
        return BetaAttribute.from_dict(d)
    if d["type"] == "categorical":
        return DirichletAttribute.from_dict(d)
    raise ValueError(f"unknown attribute type: {d['type']!r}")


# ---------------------------------------------------------------------------
# Persona (ALGORITHM §0, §5.3)
# ---------------------------------------------------------------------------


@dataclass
class Persona:
    """A persona: mixing weight plus per-attribute conjugate posteriors."""

    id: str
    pi: float
    attributes: dict[str, Attribute]
    lam: float = 0.995
    n_max: float = 200.0
    lineage: list[dict] = field(default_factory=list)

    @classmethod
    def new(cls, attributes: dict[str, Attribute], pi: float = 1.0,
            lam: float = 0.995, n_max: float = 200.0,
            event: str = "seeded", ts: str | None = None) -> Persona:
        persona = cls(id=uuid.uuid4().hex[:12], pi=pi, attributes=attributes,
                      lam=lam, n_max=n_max)
        persona.log_lineage(event, ts, {})
        return persona

    def log_lineage(self, event: str, ts: str | None, detail: dict) -> None:
        self.lineage.append({"event": event, "ts": ts, "detail": detail})

    # -- E-step support ------------------------------------------------------
    def predictive_log_likelihood(self, obs: dict[str, int | str]) -> float:
        """log p(x_i | s): sum of log posterior predictives over observed
        attributes; missing attributes contribute nothing (ALGORITHM §2.1)."""
        total = 0.0
        for attr, value in obs.items():
            a = self.attributes.get(attr)
            if a is None:
                continue
            if isinstance(a, BetaAttribute):
                total += a.log_predictive(int(value))
            else:
                total += a.log_predictive(str(value))
        return total

    # -- structural adaptation support (ALGORITHM §5.3) ----------------------
    def mean_entropy(self) -> float:
        """Average attribute entropy H_bar_s = (1/J) sum_j H_js."""
        if not self.attributes:
            return 0.0
        return float(np.mean([a.entropy() for a in self.attributes.values()]))

    def predictive_vector(self) -> dict[str, dict[str, float]]:
        """Posterior predictive probabilities per attribute (for KL)."""
        out: dict[str, dict[str, float]] = {}
        for name, a in self.attributes.items():
            if isinstance(a, BetaAttribute):
                out[name] = {"0": 1.0 - a.mean, "1": a.mean}
            else:
                out[name] = a.mean
        return out

    def decay_counts(self, dt: float) -> None:
        """Decay all data counts by lam**dt, floored at 0 (ALGORITHM §4)."""
        factor = self.lam ** dt
        for a in self.attributes.values():
            a.decay(factor)

    def reset_counts(self) -> None:
        """Fold back into the prior (zero the data counts)."""
        for a in self.attributes.values():
            if isinstance(a, BetaAttribute):
                a.A = a.B = 0.0
            else:
                a.A = [0.0] * a.k

    def copy_empty(self, pi: float | None = None, ts: str | None = None,
                   event: str = "split") -> Persona:
        """New persona with identical priors/levels and zeroed counts."""
        attrs: dict[str, Attribute] = {}
        for name, a in self.attributes.items():
            if isinstance(a, BetaAttribute):
                attrs[name] = BetaAttribute(a.alpha0, a.beta0)
            else:
                attrs[name] = DirichletAttribute(list(a.levels), list(a.alphas0))
        child = Persona.new(attrs, pi=self.pi if pi is None else pi,
                            lam=self.lam, n_max=self.n_max, event=event, ts=ts)
        return child

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pi": self.pi,
            "attributes": {k: v.to_dict() for k, v in self.attributes.items()},
            "decay": {"lambda": self.lam, "n_max": self.n_max},
            "lineage": self.lineage,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Persona:
        decay = d.get("decay", {})
        return cls(
            id=d["id"],
            pi=float(d["pi"]),
            attributes={k: attribute_from_dict(v) for k, v in d["attributes"].items()},
            lam=float(decay.get("lambda", 0.995)),
            n_max=float(decay.get("n_max", 200.0)),
            lineage=list(d.get("lineage", [])),
        )


# ---------------------------------------------------------------------------
# PersonaStore
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "kappa": 20.0,          # prior strength (ALGORITHM §5.1)
    "n_max": 200.0,         # effective sample cap -> lambda = 1 - 1/n_max (§4.1)
    "tau_split": 0.0,       # split if mean attribute entropy exceeds this (§5.3)
    "tau_merge": 0.01,      # merge if symmetrized predictive KL below this (§5.3)
    "n_min": 5.0,           # minimum responsibility mass (§2, §5.3)
    "s_max": 12,            # persona count cap (§5.2)
    "eps_rake": 1e-6,       # IPF convergence tolerance (§3.2)
    "em_tol": 1e-6,
    "em_max_iter": 100,
    # attribute registry: name -> {type, base_rate | levels + base_shares}
    "attributes": {},
}


class PersonaStore:
    """Persona collection persisted as one JSON per persona in ``personas/``.

    Canonical ordering (sorted by pi descending, ties broken by the first
    binary predictive) keeps git diffs stable (ALGORITHM §2, identifiability).
    """

    META_FILENAME = "_meta.json"

    def __init__(self, personas: list[Persona] | None = None,
                 last_update_ts: str | None = None, config: dict | None = None):
        self.personas: list[Persona] = list(personas or [])
        self.last_update_ts = last_update_ts
        self.config: dict = {**DEFAULT_CONFIG, **(config or {})}

    # -- persistence --------------------------------------------------------
    @classmethod
    def load(cls, store_dir: str | Path) -> PersonaStore:
        store_dir = Path(store_dir)
        personas_dir = store_dir / "personas"
        config: dict = {}
        last_update_ts = None
        meta_path = personas_dir / cls.META_FILENAME
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            config = meta.get("config", {})
            last_update_ts = meta.get("last_update_ts")
        personas = []
        if personas_dir.exists():
            for path in sorted(personas_dir.glob("persona_*.json")):
                personas.append(Persona.from_dict(json.loads(path.read_text())))
        return cls(personas=personas, last_update_ts=last_update_ts, config=config)

    def save(self, store_dir: str | Path) -> None:
        store_dir = Path(store_dir)
        personas_dir = store_dir / "personas"
        personas_dir.mkdir(parents=True, exist_ok=True)
        self.canonical_sort()
        # remove stale persona files, then write in canonical rank order
        for path in personas_dir.glob("persona_*.json"):
            path.unlink()
        for rank, persona in enumerate(self.personas):
            path = personas_dir / f"persona_{rank:02d}.json"
            path.write_text(json.dumps(persona.to_dict(), indent=2, sort_keys=True) + "\n")
        meta = {"last_update_ts": self.last_update_ts, "config": self.config}
        (personas_dir / self.META_FILENAME).write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n")

    # -- canonical ordering ---------------------------------------------------
    def _sort_key(self, persona: Persona) -> tuple:
        first_binary_q = 0.0
        for a in persona.attributes.values():
            if isinstance(a, BetaAttribute):
                first_binary_q = a.mean
                break
        return (-persona.pi, first_binary_q, persona.id)

    def canonical_sort(self) -> None:
        self.personas.sort(key=self._sort_key)

    # -- convenience -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.personas)

    def pis(self) -> np.ndarray:
        return np.array([p.pi for p in self.personas], dtype=float)

    def normalize_pis(self) -> None:
        total = sum(p.pi for p in self.personas)
        if total <= 0:
            n = len(self.personas)
            for p in self.personas:
                p.pi = 1.0 / n
        else:
            for p in self.personas:
                p.pi /= total
