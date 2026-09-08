# psyche-engine

[![CI](https://github.com/OWNER/psyche-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/psyche-engine/actions/workflows/ci.yml)
[![Nightly update](https://github.com/OWNER/psyche-engine/actions/workflows/update.yml/badge.svg)](https://github.com/OWNER/psyche-engine/actions/workflows/update.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**psyche-engine** is a Bayesian persona engine for lead/panel segmentation.
It maintains a mixture of *personas* whose attributes (binary propensities,
categorical preferences) are tracked with exact conjugate updates — Beta–Bernoulli
and Dirichlet–Categorical — corrected for sample-selection bias by raking
against population margins, and kept adaptive under concept drift by bounded
geometric forgetting with a provable stationary bias bound
`bias ≤ δλ/(1−λ)`.

Every claim above is pinned to a theorem in
[docs/ALGORITHM.md](docs/ALGORITHM.md) — conjugacy (§1), EM coordinate ascent
(§2.3), IPF convergence (§3.2), Kish ESS (§3.3), the N_max = (1−λ)⁻¹
equivalence (§4.1), and the drift bias/variance bounds (§4.3).

## Highlights

- **Exact conjugate posteriors** — the store keeps only sufficient statistics
  `(A, B)` plus immutable priors; posteriors are derived, never stored.
- **Soft assignment** — leads carry log-space responsibilities `r_is`; every
  count increment is weighted by `ρ_is = w̃_i · r_is` where `w̃` is
  ESS-normalized so each batch adds exactly `ESS ≤ n` pseudo-counts.
- **Raking** — iterative proportional fitting toward population margins
  (e.g. NPPES provider taxonomy × state) with Kish ESS and leverage trimming.
- **Bounded memory** — recency decay on counts only, applied once per batch
  pre-E-step; `λ = 1 − 1/N_max` gives an exact sliding geometric window.
- **Self-evaluating** — every confirmation is scored out-of-sample (Brier)
  *before* it is consumed; a rising Brier trend is the drift alarm.
- **Structural adaptation** — personas split/merge on posterior entropy and
  symmetrized KL, only when BIC decreases, all logged in `lineage`.
- **Auditable** — personas live as canonical-ordered JSON in git; the nightly
  GitHub Action commits them with `[skip ci]`.

## Install

```bash
pip install -e .          # runtime: numpy, scipy, pandas, requests
pip install -e .[dev]     # + pytest, ruff
```

## Quickstart

```bash
# ingest a leads CSV (validation, dedupe, NPPES enrichment, raking)
psyche ingest data/leads.csv --store store --margins data/population_margins.csv --offline

# run one update cycle over a confirmations CSV
psyche update data/confirmations.csv --store store

# inspect personas and metrics
psyche show --store store
```

Or fully programmatically / offline (no network):

```bash
python examples/simulate_demo.py   # synthetic two-segment drift demo
```

## Workflows

### 1. Ingestion & enrichment

```mermaid
flowchart TD
    A["Raw leads CSV / CRM export"] --> B["Validation and dedupe<br/>schema checks, email hash + NPI dedupe"]
    B -->|"invalid rows quarantined"| BQ["Quarantine bucket + error report"]
    B --> C{"NPI present?"}
    C -->|"yes"| D["NPPES public API lookup<br/>taxonomy, state, enumeration date"]
    C -->|"no"| D2["Skip enrichment, flag low_confidence"]
    D -->|"cache hit or 200 OK"| E["Enrichment cache<br/>SQLite keyed by NPI, TTL 90d"]
    D -->|"API down / rate limited"| F["Offline fallback<br/>last cached NPPES snapshot"]
    E --> G["Feature extraction<br/>age band, geo, taxonomy group, education"]
    F --> G
    D2 --> G
    G --> H["Raking weights w_i<br/>IPF vs NPPES taxonomy x state margins"]
    H -->|"ESS-normalized w-tilde_i"| I["Persona store<br/>personas/*.json + store/leads.csv"]
```

### 2. Nightly update loop

```mermaid
flowchart TD
    A["Cron trigger<br/>update.yml schedule"] --> B["Load persona store<br/>personas/*.json"]
    B --> C["Fetch new confirmations<br/>since last_update_ts"]
    C --> D{"Any new events?"}
    D -->|"no"| Z["Exit clean, no commit"]
    D -->|"yes"| E["Decay counts A,B by lambda^dt<br/>once per batch, priors a0,b0 untouched"]
    E --> F["E-step: responsibilities r_is<br/>log-space posterior predictives"]
    F --> G["M-step: weighted conjugate updates<br/>rho_is = w-tilde_i * r_is"]
    G --> H{"EM converged?"}
    H -->|"no"| F
    H -->|"yes"| I["Entropy check<br/>mean Beta/Dirichlet entropy per persona"]
    I --> J{"split / merge<br/>BIC-gated"}
    J -->|"accepted"| K["Apply split/merge<br/>log lineage"]
    J -->|"rejected"| L["Write personas/*.json<br/>+ store/metrics/history.jsonl"]
    K --> L
    L --> M["git add personas/ && git commit<br/>[skip ci] auto: nightly persona update"]
    M --> N["git push"]
```

### 3. Confirmation feedback loop

```mermaid
flowchart LR
    subgraph Sources
        A["Campaign events<br/>opens, clicks, replies"]
        B["CRM outcomes<br/>meetings, opportunities, closed-won"]
        C["Interview labels<br/>manual research annotations"]
    end
    A --> D["Confirmation bus"]
    B --> D
    C --> D
    D["Confirmation bus<br/>webhook intake + CSV drop zone"] --> E["Schema validation<br/>typed events: binary | categorical"]
    E -->|"malformed"| Q["Dead-letter queue + alert"]
    E --> F["Event store<br/>append-only confirmations log"]
    F --> G["Update engine<br/>r_is, lambda-decay, rho_is-weighted conjugate updates"]
    G --> H["Persona store<br/>personas/*.json"]
    G --> I["Metrics<br/>Brier score per attribute over time"]
    I --> J["Dashboard<br/>score trend + ESS + drift alarms"]
    J -.->|"rising Brier score = concept drift signal"| K["Retune N_max / lambda<br/>ALGORITHM.md sec. 4.3"]
    K -.-> G
```

### 4. CI/CD

```mermaid
flowchart TD
    subgraph ci["ci.yml"]
        A1["Trigger: pull_request, push to main"] --> B1["Checkout + setup Python<br/>pip install -e .[dev]"]
        B1 --> C1["Lint<br/>ruff check"]
        C1 --> D1["pytest<br/>unit: conjugacy, EM monotonicity,<br/>IPF margins, decay bounds"]
        D1 --> E1["Persona store schema validation<br/>JSON schema + invariant checks"]
        E1 --> F1["Status check<br/>required for merge"]
    end
    subgraph update["update.yml"]
        A2["Trigger: cron 0 3 * * *<br/>+ workflow_dispatch"] --> B2["Checkout main"]
        B2 --> C2["Run pipeline<br/>ingest + nightly update cycle"]
        C2 --> D2{"personas/*.json changed?"}
        D2 -->|"no"| E2["Exit clean"]
        D2 -->|"yes"| F2["git commit [skip ci]<br/>auto: nightly persona update DATE"]
        F2 --> G2["git push main"]
        G2 --> H2["Optional later: build dashboard<br/>store/metrics/history.jsonl -> GitHub Pages"]
    end
    F1 -.->|"merge keeps main green<br/>so nightly commits are safe"| A2
```

## Privacy

**No PII is ever persisted.** Persona JSONs and the lead store contain only
sha256-hashed lead IDs/NPIs, calibration-cell assignments (taxonomy, state)
and aggregate counts. Raw names, emails and NPIs are hashed in memory
immediately after use; NPIs leave the machine only as lookups to the public
NPPES registry they came from. Personas below the minimum responsibility mass
`n_min` are folded back into the global prior (k-anonymity guard,
ALGORITHM.md §7.5).

## Docs

- [docs/ALGORITHM.md](docs/ALGORITHM.md) — the math: conjugacy, EM proof,
  raking, drift bounds, split/merge, worked micro-example.
- [docs/WORKFLOWS.md](docs/WORKFLOWS.md) — pipeline and CI/CD design.
- [SPEC.md](SPEC.md) — exact module interfaces.

## License

MIT — see [LICENSE](LICENSE).
