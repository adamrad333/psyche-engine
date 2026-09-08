# psyche-engine — Workflow Design

Version 0.1 — Companion to `ALGORITHM.md`; notation is shared ($r_{is}$ = responsibility, $w_i$/$\tilde w_i$ = raking weights, $\lambda$ = decay, $(\alpha_{js},\beta_{js})$ / $\boldsymbol\alpha_{js}$ = persona hyperparameters, counts $A,B$).

---

## 1. Ingestion & enrichment pipeline

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

**Explanation.** Leads arrive as CSV or a CRM export and first pass schema validation and deduplication (email hash, then NPI as the stronger key); bad rows are quarantined, never silently dropped. Leads with an NPI are enriched from the NPPES public API, with every response cached in SQLite for 90 days so re-runs are cheap and deterministic. If the API is unavailable or rate-limited, the pipeline falls back to the most recent cached NPPES snapshot and marks the enrichment as stale rather than failing. Feature extraction then derives the calibration variables (taxonomy group, state) used by the raking step, which computes design weights $w_i$ against NPPES margins and normalizes them to $\tilde w_i$ with $\sum_i \tilde w_i = \mathrm{ESS}$ (see ALGORITHM.md §3). Everything lands in the persona store: per-lead features in `store/leads.csv`, per-persona hyperparameters and counts in `personas/*.json`.

---

## 2. Nightly update loop

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

**Explanation.** The nightly job loads the persona store and pulls only confirmations newer than the stored `last_update_ts`, exiting without a commit when there is nothing to do so history stays clean. Each batch runs the EM cycle of ALGORITHM.md §2: the stored counts $A,B$ are first decayed once by $\lambda^{\Delta t}$ (priors are immutable), responsibilities $r_{is}$ are then computed in log-space from the post-decay posterior predictives, and the M-step applies increments weighted by $\rho_{is}=\tilde w_i r_{is}$. The order matters: decay is applied exactly once per batch to pre-batch counts, before the E-step, so responsibilities always use post-decay, pre-M-step counts. After convergence, per-persona mean posterior entropy is compared against $\tau_{\text{split}}$ and candidate merges against $\tau_{\text{merge}}$; structural changes are only kept if BIC decreases, and every accepted change writes a `lineage` entry. Finally the updated `personas/*.json` and `store/metrics/history.jsonl` are committed with `[skip ci]` (so the auto-commit does not retrigger CI) and pushed, giving a fully auditable, git-versioned history of every persona.

---

## 3. Confirmation feedback loop

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

**Explanation.** Confirmations arrive from three source families — campaign events, CRM outcomes, and manual interview labels — and converge on a single confirmation bus that accepts webhooks for real-time sources and CSV drops for batch exports. Every event is validated into one of two typed forms (binary confirm/refute or categorical observation), matching the two conjugate families of ALGORITHM.md §1; malformed events go to a dead-letter queue rather than corrupting posteriors. Valid events land in an append-only log, which makes the nightly update idempotent and replayable: the store can always be rebuilt from the log plus priors. The update engine consumes events as in §2–4 of ALGORITHM.md, and — critically — each event is scored with the persona posterior *predictive before* the event is folded in, so the per-attribute Brier score $BS_j = \tfrac{1}{n}\sum_i (q_{js}(x_{ij}) - x_{ij})^2$ measures genuine out-of-sample skill. A sustained rise in Brier score indicates concept drift faster than the current $\lambda$ can track, feeding back into retuning $N_{\max}$ per the bound $\mathrm{bias}\le\delta\lambda/(1-\lambda)$.

---

## 4. CI/CD

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

**Explanation.** Two workflows separate code quality from data motion. `ci.yml` runs on every PR and push to `main`: lint, then pytest with property-style tests pinned to the math in ALGORITHM.md — Beta/Dirichlet update correctness, monotonicity of the EM lower bound, exact margin match after IPF, and the §4.3 bias bound checked by simulation — plus JSON-schema and invariant validation of the persona store (priors immutable, counts non-negative, $\sum_k\alpha_{jsk}$ consistent). `update.yml` runs on a nightly cron (with `workflow_dispatch` for manual runs), executes the ingestion and nightly-update cycle, and commits changed persona JSONs with `[skip ci]` and a timestamped message so the auto-commit never triggers CI and history stays readable. The `[skip ci]` tag plus the required status check on PRs together guarantee that `main` is always green when the nightly job commits on top of it. A GitHub Pages dashboard rendering `store/metrics/history.jsonl` (Brier trends, ESS, persona entropy) is a deliberate later addition — the data contract (`store/metrics/history.jsonl` schema) is established now so the dashboard is pure presentation work.

---

## Cross-document consistency notes

* $\tilde w_i$ (ESS-normalized raking weights) are computed once per ingestion run and frozen until leads change; the nightly loop reuses them.
* The decay factor $\lambda$ and $N_{\max}=1/(1-\lambda)$ live in each persona's JSON (`decay` block), not in workflow config, so drift tuning is per-persona and versioned.
* Every persona mutation path (nightly update, split/merge, manual seed) must pass the same schema + invariant validation used in `ci.yml`.
* The confirmation log is the single source of truth; persona JSONs are a derived, rebuildable artifact.
