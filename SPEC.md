# psyche-engine — Implementation SPEC

Derived from `docs/ALGORITHM.md` and `docs/WORKFLOWS.md` (single source of truth).
This file pins the exact module interfaces before coding. Python 3.11+.

Package: `psyche` (distribution name `psyche-engine`).
Dependencies: numpy, scipy, pandas, requests. Dev: pytest, ruff.

## Notation mapping

- Stored per persona per attribute: priors `alpha0/beta0` (immutable) and data counts
  `A`/`B` (binary) or `A[k]` (categorical), decayed by `lambda**dt` once per batch.
- Posterior hyperparameters are derived: `alpha = alpha0 + A`, `beta = beta0 + B`,
  `alphas[k] = alphas0[k] + A[k]`.
- `rho_is = w_tilde_i * r_is`; `sum_i w_tilde_i == ESS <= n`.
- `lambda = 1 - 1/N_max`; stationary drift bias bound `delta * lambda / (1 - lambda)`.

## Store layout (on disk)

```
personas/
  _meta.json            # store meta: last_update_ts, config snapshot, attribute registry
  persona_00.json       # canonical order: sorted by pi desc, ties by first binary predictive
  persona_01.json
leads.csv               # hashed lead ids ONLY: lead_hash, npi_hash, taxonomy, state, w, w_tilde, weight_flag
metrics/history.jsonl   # append-only Brier history
```

Persona JSON schema (per file):

```json
{
  "id": "uuid4-hex",
  "pi": 0.5,
  "attributes": {
    "<attr>": {"type": "binary", "alpha0": 6.0, "beta0": 14.0, "A": 0.0, "B": 0.0},
    "<attr>": {"type": "categorical", "levels": ["..."], "alphas0": [..], "A": [..]}
  },
  "decay": {"lambda": 0.995, "n_max": 200.0},
  "lineage": [{"event": "seeded|split|merge|folded", "ts": "...", "detail": {}}]
}
```

Privacy hard requirement: persona JSONs and leads.csv never contain names, emails, or
raw NPIs — only sha256 hashes, cell assignments, aggregate counts.

## psyche/models.py

```python
@dataclass BetaAttribute:
    alpha0: float; beta0: float; A: float = 0.0; B: float = 0.0
    alpha -> float            # alpha0 + A
    beta -> float             # beta0 + B
    mean -> float             # alpha/(alpha+beta)
    var -> float
    map_estimate -> float     # (alpha-1)/(n0-2), boundary 0/1 if alpha,beta<=1
    predictive(x: int) -> float          # P(x), x in {0,1}
    log_predictive(x: int) -> float
    entropy() -> float                   # exact Beta differential entropy (ALGORITHM §5.3)
    observe(x: int, weight: float)       # A += w*x; B += w*(1-x)
    decay(factor: float)                 # A,B *= factor, floored at 0; priors untouched

@dataclass DirichletAttribute:
    levels: list[str]; alphas0: list[float]; A: list[float] (zeros init)
    alphas -> list[float]     # alphas0 + A
    alpha_total -> float
    mean -> dict[str, float]
    predictive(level) -> float
    log_predictive(level) -> float
    entropy() -> float        # exact Dirichlet entropy (ALGORITHM §5.3)
    observe(level, weight)
    decay(factor)

Attribute = BetaAttribute | DirichletAttribute

@dataclass Persona:
    id: str; pi: float; attributes: dict[str, Attribute]
    lam: float = 0.995; n_max: float = 200.0
    lineage: list[dict]
    predictive_log_likelihood(obs: dict[str, value]) -> float   # sum of log predictives
    mean_entropy() -> float                                      # (1/J) sum_j H_j
    predictive_vector() -> dict[str, dict]                       # q_js / q_jsk for KL
    to_dict() / from_dict()

class PersonaStore:
    personas: list[Persona]
    last_update_ts: str | None
    config: dict                # kappa, tau_split, tau_merge, n_min, s_max, eps_rake, base_rates
    @classmethod load(dir) -> PersonaStore
    save(dir)                   # canonical ordering: sort by pi desc, tie by first binary
                                # predictive; files persona_{rank:02d}.json + _meta.json
    canonical_sort()
```

## psyche/raking.py

```python
@dataclass RakeResult:
    w: np.ndarray               # raw raked weights (trimmed)
    w_tilde: np.ndarray         # ESS-normalized, sum == ESS
    ess: float                  # (sum w)^2 / sum w^2
    deff: float                 # n / ESS
    weight_flags: list[str | None]  # "cell_not_in_margins" etc.
    iterations: int

def rake(cells: pd.DataFrame,                   # one row per lead, cols = calibration dims
         margins: dict[str, dict[str, float]],  # dim -> level -> population total
         eps: float = 1e-6, max_iter: int = 1000,
         trim_quantile: float = 0.99) -> RakeResult
    # IPF (ALGORITHM §3.2): cycle dims, rescale cells to margins until rel. error < eps.
    # Leads in cells absent from margins: weight frozen at median valid weight + flag.
    # Trim at trim_quantile BEFORE ESS normalization; then w_tilde = w*ESS/sum(w).
```

## psyche/update.py

```python
@dataclass Confirmation:
    lead_hash: str; attribute: str; value: int | str  # 0/1 binary, str level categorical
    ts: str; source: str = "unknown"

def lambda_from_n_max(n_max) -> float            # 1 - 1/n_max
def stationary_bias_bound(delta, lam) -> float   # delta*lam/(1-lam)  (ALGORITHM §4.3a)
def decay_counts(store, dt)                      # counts *= lam**dt, once, pre-E-step
def e_step(personas, obs_by_lead: dict[str, dict]) -> np.ndarray  # n x S log-space, rows sum to 1
def log_likelihood(personas, obs_by_lead, w_tilde) -> float       # weighted observed-data LL
def m_step(personas, base_counts, obs_by_lead, R, w_tilde)
    # counts = base (post-decay snapshot) + sum_i rho_is x_ij ; pi_s prop. sum_i rho_is
def em_batch(store, obs_by_lead, w_tilde, tol=1e-6, max_iter=100) -> EMResult(R, ess, loglik)
    # decay once -> EM to convergence -> prune under-mass personas (n_s <= n_min folded
    # to prior, lineage-logged) -> returns responsibilities
def bic(personas, obs_by_lead, w_tilde, n_eff) -> float
    # -2*ll + d*log(n_eff); d = S*(|J_B| + sum(K_j-1)) + (S-1)
def try_split(store, s_idx, obs_by_lead, R, w_tilde, n_eff, ts=None, base=None)
    #   -> CountsSnapshot | None (truthy = accepted; returns candidate bases aligned
    #      with the new store.personas so follow-up ops never double-count the batch)
    # weighted k-means k=2 on member confirmation vectors; children seeded with the
    # parent's post-decay base share + cluster-conditional weighted counts (mass
    # preserved); candidate re-tuned via mini-EM (BIC is defined at EM convergence,
    # §5.2); accepted iff BIC decreases; lineage-logged. base=None -> refused.
def try_merge(store, s_idx, t_idx, obs_by_lead, w_tilde, n_eff, ts=None, base=None)
    #   -> CountsSnapshot | None
    # symmetrized KL on predictive vectors < tau_merge; counts add; merged entropy
    # stays below tau_split; BIC-gated (with mini-EM refit when base given);
    # lineage-logged
def sym_kl(p: Persona, q: Persona) -> float  # per-attribute averaged sym KL, 1e-9 floor
def structural_adaptation(store, obs_by_lead, R, w_tilde, cfg)
    # split pass (mean_entropy > tau_split and mass > n_min) then merge pass, both BIC-gated
```

## psyche/enrich.py

```python
NPPES_BASE_URL = "https://npiregistry.cms.hhs.gov/api/"
NPPES_VERSION = "2.1"

def hash_id(value: str) -> str    # sha256 hex (used for lead ids, emails, NPIs)

class NPPESClient:
    def __init__(cache_path, ttl_days=90, stub: dict | None = None,
                 session: requests.Session | None = None,
                 max_retries=3, backoff=0.5, timeout=10.0)
        # stub mode: serve from dict, NEVER touch network (test mode)
    def lookup(npi: str) -> dict | None
        # -> {"taxonomy": str|None, "state": str|None, "enumeration_date": str|None,
        #     "stale": bool}
        # SQLite cache keyed by NPI with TTL; polite retry w/ exponential backoff on
        # 429/5xx; offline fallback to last cached snapshot (stale=True)
    def close()
```

## psyche/metrics.py

```python
def brier_binary(q: float, x: int) -> float           # (q - x)^2, q = P(x=1)
def brier_categorical(qs: dict, observed) -> float    # sum_k (q_k - 1{k==obs})^2
def score_batch(personas, confirmations) -> list[dict]
    # OUT-OF-SAMPLE: predictive computed from pre-update mixture
    # (sum_s pi_s * q_js(x)) BEFORE any decay/EM consumes the batch.
    # Returns records {"ts", "attribute", "brier", "n", "ess"} per attribute.
def append_history(records, metrics_dir)   # append JSONL to metrics/history.jsonl
```

## psyche/pipeline.py

```python
def ingest(leads_csv, store_dir, margins_csv=None, enricher=None, seed_personas=1) -> PersonaStore
    # validate schema, dedupe (email hash then NPI hash), quarantine invalid rows,
    # enrich via NPPESClient (hash NPI immediately), extract calibration cells,
    # rake -> w_tilde persisted in leads.csv (hashed ids only),
    # seed personas from global priors (kappa * base rates) if store empty.
def update(confirmations_csv, store_dir) -> dict
    # nightly cycle (ALGORITHM §6, invariants i-v):
    #   load store + leads; drop already-consumed events (ts <= last_update_ts);
    #   Brier score batch OUT-OF-SAMPLE first (metrics.history);
    #   rake batch leads -> w_tilde; decay counts once (dt from last_update_ts);
    #   EM to convergence; pi update; structural adaptation (BIC-gated, lineage);
    #   persist personas/*.json (canonical order) + metrics; update last_update_ts.
def show(store_dir) -> str   # human-readable summary: pi, predictives, ESS, entropy
```

## psyche/cli.py

`psyche ingest leads.csv [--store DIR] [--margins CSV] [--offline]`
`psyche update confirmations.csv [--store DIR] [--margins CSV]`
`psyche show [--store DIR]`
argparse-based; entry point `psyche = psyche.cli:main`.

## tests/ (pytest, no network)

1. `test_conjugacy.py` — Beta/Dirichlet update increments, estimators, Appendix-B micro-example.
2. `test_convergence.py` — Bernoulli(p) stream, posterior mean -> p within tol (SLLN §4.2).
3. `test_drift.py` — drifting p_k (|d|<=delta), stationary bias <= delta*lam/(1-lam) + MC slack (§4.3a).
4. `test_raking.py` — IPF reproduces margins exactly, ESS/deff math, sum(w_tilde)==ESS,
   missing-cell flag + median freeze, trimming.
5. `test_responsibilities.py` — rows of R sum to 1, sum_s rho_is == w_tilde_i,
   batch pseudo-mass == ESS, log-space == naive-space up to fp error.
6. `test_entropy.py` — exact Beta/Dirichlet entropy formulas; concentrated Beta has
   NEGATIVE differential entropy (Beta(46,74) ~ -1.70, Beta(6,14) ~ -0.90 per App. B).
7. `test_store.py` — JSON roundtrip, canonical ordering by pi, priors immutable across
   decay, no-PII check on serialized output, N_max <-> lambda equivalence.

## examples/simulate_demo.py

End-to-end synthetic demo: generate leads + drifting confirmations, ingest, run several
update batches, print Brier trend + persona summary. No network (stub enricher).
