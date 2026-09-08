# psyche-engine — Algorithm Design

Version 0.1 — Engineering design document. Companion document: `WORKFLOWS.md` (same notation).

## 0. Data model and notation

A **lead** $i \in \{1,\dots,n\}$ carries static fields (name, email, age, geo, work location, education, NPI, provider taxonomy) and generates **confirmations** over time: campaign responses, CRM outcomes, interview labels. A confirmation is a typed observation of a persona **attribute**.

* Attributes $j = 1,\dots,J$ split into binary attributes $j \in \mathcal{J}_B$ (e.g. "responds to email outreach": $x_{ij}\in\{0,1\}$) and categorical attributes $j \in \mathcal{J}_C$ with $K_j$ levels (e.g. preferred channel, specialty mix: one-hot $x_{ijk}\in\{0,1\}$, $\sum_k x_{ijk}=1$).
* Personas (segments) $s = 1,\dots,S$ with mixing weights $\pi_s$, $\sum_s \pi_s = 1$.
* Persona $s$'s binary attribute $j$ is governed by propensity $\theta_{js}\in[0,1]$; categorical attribute $j$ by $\boldsymbol\phi_{js} \in \Delta^{K_j-1}$.
* All Beta hyperparameters are $(\alpha_{js},\beta_{js})$; Dirichlet hyperparameters $\boldsymbol\alpha_{js}=(\alpha_{js1},\dots,\alpha_{jsK_j})$, with total $\alpha_{js,0}=\sum_k \alpha_{jsk}$.
* Priors are denoted with a $(0)$ superscript: $\alpha^{(0)}_{js}$, etc., with **prior strength** $\kappa$ (Section 5).
* Lead $i$ has **responsibility** $r_{is}$ for persona $s$, $\sum_s r_{is}=1$ (Section 2), and **design weight** $w_i$ from raking (Section 3). The combined update weight is
$$\rho_{is} \;=\; \tilde w_i \, r_{is},$$
where $\tilde w_i$ is the ESS-normalized design weight (Section 3.3).
* $\lambda \in (0,1]$ is the per-period **decay factor**, $\Delta t$ the elapsed periods since last update, $N_{\max}$ the effective sample cap (Section 4).
* $\psi(\cdot)$ is the digamma function; $B(\cdot,\cdot)$ the beta function; $H(\cdot)$ entropy (nats).

The engine stores, per persona, only the **accumulated sufficient statistics** (data counts) $A_{js}, B_{js}$ (binary hits/misses) and $A_{jsk}$ (categorical counts), plus fixed priors:
$$\alpha_{js} = \alpha^{(0)}_{js} + A_{js},\qquad \beta_{js} = \beta^{(0)}_{js} + B_{js},\qquad \alpha_{jsk} = \alpha^{(0)}_{jsk} + A_{jsk}.$$

---

## 1. Attribute posteriors (conjugate updates)

### 1.1 Binary attributes: Beta–Bernoulli

**Definition.** Model $\theta_{js} \sim \mathrm{Beta}(\alpha_{js},\beta_{js})$ and a confirmation $x_{ij}\mid\theta_{js}\sim\mathrm{Bernoulli}(\theta_{js})$.

**Update rule.** Multiplying prior by likelihood,
$$p(\theta_{js}\mid x_{ij}) \;\propto\; \theta_{js}^{\alpha_{js}-1}(1-\theta_{js})^{\beta_{js}-1}\cdot \theta_{js}^{x_{ij}}(1-\theta_{js})^{1-x_{ij}}
= \theta_{js}^{\alpha_{js}+x_{ij}-1}(1-\theta_{js})^{\beta_{js}+1-x_{ij}-1},$$
which is the kernel of a Beta density. Hence
$$\boxed{\;\theta_{js}\mid x_{ij}\sim\mathrm{Beta}(\alpha_{js}+x_{ij},\;\beta_{js}+1-x_{ij})\;}$$
i.e. a confirm increments $\alpha$, a refute increments $\beta$ — one pseudo-count each.

**Estimators.** With $n_0=\alpha+\beta$:
$$\text{mean: }\ \hat\theta = \frac{\alpha}{n_0},\qquad
\text{var: }\ \frac{\alpha\beta}{n_0^2(n_0+1)},\qquad
\text{MAP: }\ \frac{\alpha-1}{n_0-2}\ \ (\alpha,\beta>1,\ \text{else boundary } 0/1).$$

**Posterior predictive.** For the next event, $P(x=1)=\mathbb{E}[\theta]=\alpha/(\alpha+\beta)$.

### 1.2 Categorical attributes: Dirichlet–Categorical

**Definition.** $\boldsymbol\phi_{js}\sim\mathrm{Dirichlet}(\alpha_{js1},\dots,\alpha_{jsK_j})$ and $\mathbf x_{ij}\mid\boldsymbol\phi_{js}\sim\mathrm{Categorical}(\boldsymbol\phi_{js})$.

**Update rule.** $p(\boldsymbol\phi_{js}\mid\mathbf x_{ij})\propto\prod_k \phi_{jsk}^{\alpha_{jsk}-1}\cdot\prod_k\phi_{jsk}^{x_{ijk}}=\prod_k\phi_{jsk}^{\alpha_{jsk}+x_{ijk}-1}$, so
$$\boxed{\;\boldsymbol\phi_{js}\mid \mathbf x_{ij}\sim\mathrm{Dirichlet}(\alpha_{js1}+x_{ij1},\dots,\alpha_{jsK_j}+x_{ijK_j})\;}$$

**Estimators.** With $\alpha_0=\sum_k\alpha_k$:
$$\text{mean: }\ \hat\phi_k=\frac{\alpha_k}{\alpha_0},\qquad
\text{var: }\ \frac{\alpha_k(\alpha_0-\alpha_k)}{\alpha_0^2(\alpha_0+1)},\qquad
\text{MAP: }\ \frac{\alpha_k-1}{\alpha_0-K_j}\ \ (\alpha_k>1).$$

**Posterior predictive.** $P(x_k=1)=\alpha_k/\alpha_0$.

**Justification.** Both updates are exact Bayes by conjugacy: the displayed kernels are unnormalized Beta/Dirichlet densities, and normalization is unique. The posterior depends on data only through counts $(A,B)$ / $(A_k)$ — this is why the persona store holds only sufficient statistics.

---

## 2. Responsibility-weighted soft updates (EM-style)

Leads are not hard-assigned: lead $i$ has soft membership $r_{is}$ in persona $s$.

### 2.1 E-step: responsibilities from posterior predictives

Assume conditional independence of attributes given persona membership (naive-Bayes mixture). Under persona $s$'s *current* posteriors, the posterior predictive of lead $i$'s observed confirmation vector $\mathbf x_i$ is
$$p(\mathbf x_i\mid s)\;=\;\prod_{j\in\mathrm{obs}(i)} q_{js}(x_{ij}),\qquad
q_{js}(1)=\frac{\alpha_{js}}{\alpha_{js}+\beta_{js}},\quad
q_{jsk}=\frac{\alpha_{jsk}}{\alpha_{js,0}},$$
where $\mathrm{obs}(i)$ indexes attributes actually confirmed for lead $i$ (missing attributes contribute no factor — correct under MCAR/MAR since the likelihood factorizes). Then
$$\boxed{\;r_{is}\;=\;\frac{\pi_s\, p(\mathbf x_i\mid s)}{\sum_{s'}\pi_{s'}p(\mathbf x_i\mid s')}\;}$$
Compute in log-space: $\log r_{is} = \log\pi_s + \sum_j \log q_{js}(x_{ij}) - \mathrm{logsumexp}_{s'}(\cdot)$.

### 2.2 M-step: weighted conjugate updates

Each confirmation from lead $i$ updates persona $s$ with weight $\rho_{is}=\tilde w_i r_{is}$:
$$\boxed{\;A_{js}\mathrel{+}= \sum_i \rho_{is}\, x_{ij},\qquad B_{js}\mathrel{+}= \sum_i \rho_{is}(1-x_{ij}),\qquad A_{jsk}\mathrel{+}=\sum_i\rho_{is}\,x_{ijk}\;}$$
Mixing weights: $\pi_s \propto \sum_i \rho_{is}$, normalized (the classical unweighted identity $\pi_s=\tfrac1n\sum_i r_{is}$ of §2.3 is the special case $\tilde w_i\equiv 1$).

### 2.3 Proof: coordinate ascent on a complete-data lower bound

Let $z_i\in\{1,\dots,S\}$ be the latent persona label, $\Theta=\{(\pi_s, q_{js\cdot})\}$ the predictive parameters, and $q_i(\cdot)$ any distribution over $z_i$. By Jensen,
$$\log p(\mathbf x\mid\Theta)=\sum_i\log\sum_s q_i(s)\frac{\pi_s p(\mathbf x_i\mid s)}{q_i(s)}
\;\ge\;\underbrace{\sum_i\sum_s q_i(s)\log\frac{\pi_s p(\mathbf x_i\mid s)}{q_i(s)}}_{\mathcal F(q,\Theta)}.$$

* **E-step** maximizes $\mathcal F$ over $q$ with $\Theta$ fixed: $\mathcal F = \log p(\mathbf x\mid\Theta) - \sum_i \mathrm{KL}\big(q_i \,\|\, p(z_i\mid\mathbf x_i,\Theta)\big)$, maximized uniquely at $q_i(s)=p(z_i\mid\mathbf x_i,\Theta)=r_{is}$, making the bound tight.
* **M-step** maximizes $\mathcal F$ over $\Theta$ with $q$ fixed. The objective separates: $\pi$ maximizes $\sum_i\sum_s r_{is}\log\pi_s$ s.t. $\sum_s\pi_s=1$, giving $\pi_s=\sum_i r_{is}/n$ (Lagrange multiplier). Each binary $q_{js}$ maximizes $\sum_i\rho_{is}[x_{ij}\log q + (1-x_{ij})\log(1-q)]$ plus the Beta-prior pseudo-counts, giving exactly the weighted conjugate update of §2.2 (set derivative to zero: $q_{js}=(\alpha^{(0)}_{js}+\sum_i\rho_{is}x_{ij})/(\alpha^{(0)}_{js}+\beta^{(0)}_{js}+\sum_i\rho_{is})$). Categorical case identical with a simplex constraint. With design weights, $\mathcal F$ is replaced by the weighted bound $\mathcal F_w=\sum_i \tilde w_i\sum_s q_i(s)\log[\pi_s p(\mathbf x_i\mid s)/q_i(s)]$; the E-step is unchanged and both ascent steps are still closed form, so the monotonicity argument below goes through verbatim.

Monotonicity: $\log p(\mathbf x\mid\Theta^{(t+1)}) \ge \mathcal F(q^{(t+1)},\Theta^{(t+1)}) \ge \mathcal F(q^{(t+1)},\Theta^{(t)}) = \log p(\mathbf x\mid\Theta^{(t)})$. Hence the observed-data (penalized) likelihood never decreases and iterates converge to a stationary point. ∎

**Remark (empirical-Bayes character).** We plug posterior-mean predictives $q_{js}$ into the E-step rather than integrating over the Beta/Dirichlet posteriors. The exact integrated predictive is also available in closed form (Beta–Bernoulli / Dirichlet–Categorical marginal likelihood) and SHOULD be substituted once per-attribute counts are large; with $\alpha+\beta \gtrsim 30$ the plug-in error is $O((\alpha+\beta)^{-1})$.

**Identifiability and initialization.** The mixture likelihood is invariant to permuting the persona labels (label switching), and EM converges to a local optimum. Engineering mitigations, all required: (i) fix a canonical persona ordering after each run (sort by $\pi_s$, ties by $q_{1s}$) so the JSON diffs in git are stable; (ii) initialize EM from the *previous* persona store rather than random restarts — the nightly problem is a warm-start tracking problem, not a de novo clustering; (iii) require a minimum responsibility mass $n_s=\sum_i\rho_{is} > n_{\min}$ (default $n_{\min}=5$ effective confirmations) for a persona to survive; under-mass personas are folded into the global prior (Section 5.1) rather than left as near-duplicates of it.

**Interaction with decay.** Responsibilities are recomputed every batch (E-step) but the *counts* persist and are decayed (Section 4). The M-step therefore adds this batch's $\rho_{is}$-weighted increments on top of decayed historical counts — the EM proof above applies within a batch (unit-weight), while across batches the recursion is the bounded-memory estimator analyzed in §4.3. The two analyses compose because decay is a deterministic scalar on the sufficient statistics and does not depend on the E-step output.

---

## 3. Post-stratification / raking against population margins

Confirmations come from self-selected engagers — a biased sample of the healthcare-professional population. We correct with design weights from **iterative proportional fitting** (IPF, raking) against public margins (NPPES counts per provider taxonomy × state).

### 3.1 Setup

Partition leads into cells $g$ defined by the joint (or marginal) levels of calibration variables — here taxonomy and state. Public benchmarks give population cell totals $T_g$ (from NPPES). Sample cell totals are $M_g=\sum_{i\in g} 1$ (or current weight sums); $M_g$ is only an intermediate in IPF — do not confuse with the persona count $S$.

### 3.2 IPF update

Start from $w_i^{(0)}=1$. Cycle over calibration dimensions $d$ (taxonomy, state); for each, rescale every cell margin:
$$\boxed{\;w_i \;\leftarrow\; w_i\cdot\frac{T^{(d)}_{g_d(i)}}{\sum_{i':\, g_d(i')=g_d(i)} w_{i'}}\;}\qquad\text{repeated until margin error } < \varepsilon.$$

**Justification (sketch).** Raking finds weights minimizing the discrimination information $\sum_i w_i\log(w_i/u_i)$ w.r.t. base weights $u_i$ subject to the margin constraints $\sum_{i\in g} w_i = T_g$ (Ireland–Kullback 1968). Each rescaling above is an exact Bregman ($I$-divergence) projection of the weight vector onto one linear constraint set; cyclic Bregman projections onto convex sets converge to the unique projection onto their intersection (Csiszár 1975), provided the margins are mutually consistent (nonempty intersection — guaranteed here because both margins come from the same NPPES frame). ∎

### 3.3 Weighted confirmation updates and the design-effect penalty

Raw raking weights are relative; used directly they would inflate the pseudo-count mass fed to Section 1–2 updates by $\sum_i w_i$, overstating confidence. We therefore:

1. Compute Kish's **effective sample size** and design effect:
$$\boxed{\;\mathrm{ESS}=\frac{\big(\sum_i w_i\big)^2}{\sum_i w_i^2},\qquad \mathrm{deff}=\frac{n}{\mathrm{ESS}}=1+\mathrm{CV}^2(w)\;}$$
2. Renormalize $\boxed{\tilde w_i = w_i\cdot \mathrm{ESS}/\sum_{i'} w_{i'}}$ so that $\sum_i\tilde w_i=\mathrm{ESS}\le n$.

Confirmations then enter the M-step (§2.2) with combined weight $\rho_{is}=\tilde w_i\,r_{is}$, so the total pseudo-count added per nightly batch equals the *effective* number of independent confirmations, not the raw count. **Justification:** for a weighted mean of i.i.d. Bernoulli data, $\mathrm{Var}(\bar x_w)=p(1-p)\sum_i w_i^2/(\sum_i w_i)^2 = p(1-p)/\mathrm{ESS}$; i.e. $\mathrm{ESS}$ is exactly the sample size that makes the conjugate posterior variance match the weighted-estimator variance (Kish 1965). ∎

**Edge cases.** Trim $w_i$ at a quantile (e.g. 99th) before ESS normalization to bound leverage; leads in cells absent from NPPES margins get $w_i$ frozen at the median weight and a `weight_flag` in the persona store.

---

## 4. Recency decay / forgetting under concept drift

To keep personas adaptive, **decay the data counts toward the prior before each update**:
$$\boxed{\;A_{js}\leftarrow \lambda^{\Delta t}\,A_{js},\quad B_{js}\leftarrow \lambda^{\Delta t}\,B_{js},\quad A_{jsk}\leftarrow\lambda^{\Delta t}\,A_{jsk}\;}\quad\text{(priors } \alpha^{(0)},\beta^{(0)} \text{ untouched).}$$

### 4.1 Equivalence to an effective-sample cap

With unit-weight observations, the decayed total count evolves as $T_n=\lambda T_{n-1}+1$, giving $T_n=\sum_{j=0}^{n-1}\lambda^j = (1-\lambda^n)/(1-\lambda)$, deterministic regardless of outcomes. Hence
$$T_n \le T_\infty = \frac{1}{1-\lambda}\equiv N_{\max},\qquad\text{i.e.}\qquad \boxed{\;\lambda = 1-\frac{1}{N_{\max}}\;}$$
Decay is *exactly* a sliding geometric window with capacity $N_{\max}$ pseudo-observations.

**Remark (weighted batches).** This equivalence assumes unit-weight events ($T_n=\lambda T_{n-1}+1$). With raking weights (§3.3) a batch adds pseudo-mass $\mathrm{ESS}$ rather than $n$, so the recursion is $T_n=\lambda T_{n-1}+\mathrm{ESS}_n$: the observed count mass per period is ESS-dependent, and the $N_{\max}$ cap binds in units of *effective* (not raw) confirmations.

### 4.2 Theorem (consistency without decay)

Let $x_1,x_2,\dots\stackrel{\text{iid}}{\sim}\mathrm{Bernoulli}(p)$, no decay ($\lambda=1$), prior $(\alpha^{(0)},\beta^{(0)})$ fixed. Then the posterior mean $\hat\theta_n=(\alpha^{(0)}+S_n)/(\alpha^{(0)}+\beta^{(0)}+n)$, $S_n=\sum_{k\le n}x_k$, satisfies $\hat\theta_n\to p$ a.s., and the posterior concentrates: $\mathrm{Var}(\theta\mid \mathbf x)\to 0$.

**Proof.** $S_n/n\to p$ a.s. by the strong law of large numbers. Then
$$\hat\theta_n=\frac{n}{\alpha^{(0)}+\beta^{(0)}+n}\cdot\frac{S_n}{n}+\frac{\alpha^{(0)}}{\alpha^{(0)}+\beta^{(0)}+n}\;\xrightarrow{\text{a.s.}}\;1\cdot p + 0 = p.$$
$\mathrm{Var}=\alpha_n\beta_n/[\nu_n^2(\nu_n+1)]=O(1/n)\to0$ with $\nu_n=\alpha_n+\beta_n=\alpha^{(0)}+\beta^{(0)}+n$. ∎

### 4.3 Theorem (stationary bias under bounded drift)

Let the true rate drift: $x_k\sim\mathrm{Bernoulli}(p_k)$ independently, $|p_k-p_{k-1}|\le\delta$. With decay $\lambda$ and zero prior strength (for the bound; prior only adds shrinkage toward $\hat\theta_0$), define $\hat\theta_n=A_n/T_n$ with $A_n=\lambda A_{n-1}+x_n$. Then, writing $T_n=(1-\lambda^n)/(1-\lambda)$:

**(a) Bias.**
$$\big|\,\mathbb{E}\hat\theta_n - p_n\,\big| \;\le\; \delta\,\frac{\sum_{j=1}^{n-1} j\,\lambda^j}{T_n}\;\xrightarrow{n\to\infty}\;\boxed{\;\frac{\delta\,\lambda}{1-\lambda}\;=\;O\!\left(\frac{\delta}{1-\lambda}\right)\;}$$

**(b) Variance.**
$$\mathrm{Var}(\hat\theta_n)\;=\;\frac{\sum_{k=1}^n \lambda^{2(n-k)}p_k(1-p_k)}{T_n^2}\;\le\;\frac{1}{4(1-\lambda^2)T_n^2}\;\xrightarrow{n\to\infty}\;\frac{1-\lambda}{4(1+\lambda)}.$$

**(c) Tuning.** The stationary RMSE is $\le\sqrt{\delta^2\lambda^2/(1-\lambda)^2 + (1-\lambda)/[4(1+\lambda)]}$. Writing $\varepsilon=1-\lambda$ and equating the (squared) terms, $\delta^2/\varepsilon^2 \approx \varepsilon/8$ gives
$$\boxed{\;\varepsilon^\star = 1-\lambda^\star = (8\delta^2)^{1/3}=2\,\delta^{2/3},\qquad \mathrm{RMSE}^\star = O\!\left(\delta^{1/3}\right)\;}$$
i.e. with optimal forgetting, error scales as the cube root of the drift rate. This matches (up to constants) the minimax rate $\asymp\delta^{1/3}$ for tracking a Lipschitz-drifting mean — geometric forgetting is rate-optimal for this drift model, so no finer adaptation scheme can improve the exponent.

**Proof of (a).** Unrolling, $A_n=\sum_{k=1}^n\lambda^{n-k}x_k$, so $\mathbb{E}A_n=\sum_k\lambda^{n-k}p_k$ and, since $T_n$ is deterministic, $\mathbb{E}\hat\theta_n=T_n^{-1}\sum_{k}\lambda^{n-k}p_k$. Then, substituting $j=n-k$ and using $|p_k-p_n|\le\delta(n-k)$,
$$\mathbb{E}\hat\theta_n - p_n = T_n^{-1}\sum_{k=1}^n \lambda^{n-k}(p_k-p_n)\;\Rightarrow\;\big|\mathbb{E}\hat\theta_n-p_n\big|\le \delta\,T_n^{-1}\sum_{j=0}^{n-1}j\lambda^j \le \delta\,T_n^{-1}\frac{\lambda}{(1-\lambda)^2},$$
using $\sum_{j\ge0}j\lambda^j=\lambda/(1-\lambda)^2$. Since $T_n\to(1-\lambda)^{-1}$, the limit is $\delta\lambda/(1-\lambda)$. ∎

**Proof of (b).** Independence gives $\mathrm{Var}(A_n)=\sum_k\lambda^{2(n-k)}p_k(1-p_k)\le\tfrac14\sum_{j\ge0}\lambda^{2j}=1/[4(1-\lambda^2)]$; divide by $T_n^2$ and take the limit. ∎

**With a retained prior** $(\alpha^{(0)},\beta^{(0)})$ of strength $\kappa_0$, the mean is a convex combination of the prior predictive and $\hat\theta_n$; the bias bound (a) gains a factor $T_\infty/(\kappa_0+T_\infty)<1$ plus a static shrinkage term $\kappa_0|\hat\theta_0-p_n|/(\kappa_0+T_\infty)$ — strictly smaller for large $n$. **With design weights**, replace (b) by $\mathrm{Var}\approx p(1-p)/\mathrm{ESS}_\lambda$ with discounted ESS $\mathrm{ESS}_\lambda=(\sum_k\lambda^{n-k}w_k)^2/\sum_k\lambda^{2(n-k)}w_k^2$; pick $\lambda$ so $\mathrm{ESS}_\lambda$ matches the drift–variance trade-off above.

**Implementation.** Decay is applied to stored counts, not hyperparameters: `A *= lam**dt` etc., floored at 0, with $\lambda=1-1/N_{\max}$ and a default $N_{\max}=200$ per attribute (tunable per persona in `personas/*.json`).

---

## 5. Initialization and cold start

### 5.1 Priors from population margins

For each binary attribute $j$, take a population base rate $\bar p_j$ estimated from public margins (NPPES taxonomy/state proportions, benchmark response rates) and set, with **prior strength** $\kappa$:
$$\boxed{\;\alpha^{(0)}_{js}=\kappa\,\bar p_j,\qquad \beta^{(0)}_{js}=\kappa\,(1-\bar p_j)\;}$$
For categorical attribute $j$ with population shares $\bar p_{jk}$: $\alpha^{(0)}_{jsk}=\kappa_j\,\bar p_{jk}$. Interpretation: the prior is worth $\kappa$ pseudo-observations; under decay it is the permanent floor every persona reverts toward. Default $\kappa=20$ (small enough that $\sim$50 confirmations dominate, large enough to keep cold personas sane). New personas are seeded from the global prior plus a small random jitter (or from split children, §5.3).

### 5.2 Persona count selection via BIC

For candidate $S$, run the EM of Section 2 to convergence on the current confirmation history and compute
$$\boxed{\;\mathrm{BIC}(S) = -2\,\hat\ell_S + d_S\log n_{\mathrm{eff}},\qquad d_S = S\Big(|\mathcal J_B| + \textstyle\sum_{j\in\mathcal J_C}(K_j-1)\Big) + (S-1),\quad n_{\mathrm{eff}}=\mathrm{ESS}\;}$$
Choose $S^\star=\arg\min_S \mathrm{BIC}(S)$ over $S\in\{1,\dots,S_{\max}\}$ (default $S_{\max}=12$). Justification: BIC is the Laplace approximation to the log marginal likelihood of the mixture, $\log p(\mathbf x\mid S) = \hat\ell_S - \tfrac{d_S}{2}\log n + O(1)$; minimizing it is asymptotically consistent for the true number of well-separated mixture components under standard regularity. Use $n_{\mathrm{eff}}$ rather than $n$ so that raking-weight redundancy does not over-penalize complexity.

### 5.3 Split / merge via posterior entropy

Per persona $s$, compute average attribute uncertainty using the exact Beta entropy
$$H[\mathrm{Beta}(\alpha,\beta)] = \ln B(\alpha,\beta) - (\alpha-1)\psi(\alpha) - (\beta-1)\psi(\beta) + (\alpha+\beta-2)\psi(\alpha+\beta),$$
and the Dirichlet entropy
$$H[\mathrm{Dir}(\boldsymbol\alpha)] = \ln B(\boldsymbol\alpha) + (\alpha_0-K)\psi(\alpha_0) - \textstyle\sum_k(\alpha_k-1)\psi(\alpha_k),$$
averaged over attributes: $\bar H_s = \tfrac{1}{J}\sum_j H_{js}$.

* **Split** persona $s$ if $\bar H_s > \tau_{\text{split}}$ **and** its responsibility mass $n_s=\sum_i r_{is}\tilde w_i > n_{\min}$: cluster the member leads (weighted $k$-means, $k=2$, on their confirmation vectors) and seed two child personas whose counts are the cluster-conditional weighted counts (this preserves total pseudo-mass; children re-tune via EM).
* **Merge** personas $s,t$ if the symmetrized KL between their predictive vectors is below $\tau_{\text{merge}}$ (computed on $q_{js\cdot}$ with a per-attribute average), and their combined responsibility entropy stays below $\tau_{\text{split}}$: counts add ($A_{j,\text{new}}=A_{js}+A_{jt}$, etc.).
* Both operations are accepted only if BIC (§5.2) decreases, and are logged in the persona JSON (`lineage` field) for auditability.

---

## 6. Nightly update cycle — pseudocode

```python
def nightly_update(store, confirmations, margins, cfg):
    # store: persona store (JSON) holding priors a0/b0, counts A/B, pi, meta
    # confirmations: typed events since last run {lead_id, attr, value, ts, source}
    # margins: NPPES taxonomy x state population totals (cached, offline fallback)
    # cfg: lambda/N_max, kappa, tau_split, tau_merge, S_max, eps_rake

    leads   = resolve_leads(confirmations, store.leads)      # dedupe via NPI + email hash
    w       = rake(leads, margins, eps=cfg.eps_rake)         # IPF, Section 3.2
    ess     = w.sum()**2 / (w**2).sum()                      # Kish ESS, Section 3.3
    w_tilde = w * ess / w.sum()                              # sum(w_tilde) == ESS

    # --- out-of-sample scoring BEFORE the batch is consumed (invariant v) ---
    metrics = brier_scores(store.personas, confirmations)    # per-attribute, over time

    dt = periods_since(store.last_update_ts)
    for s in store.personas:                                 # decay counts only, Section 4
        s.A *= cfg.lam ** dt;  s.B *= cfg.lam ** dt          # binary + categorical counts

    # --- EM to convergence on the batch (Sections 2.1-2.2) ---
    for _ in range(cfg.max_em_iter):
        r = e_step(store.personas, confirmations)            # log-space responsibilities
        if converged(r, store.prev_r, cfg.tol): break
        m_step(store.personas, confirmations, r, w_tilde)    # rho_is = w_tilde_i * r_is
        store.prev_r = r
    store.pi = normalize(r.weighted_sum(axis=0, w=w_tilde))

    # --- structural adaptation (Section 5.3), BIC-gated ---
    for s in store.personas:
        if mean_entropy(s) > cfg.tau_split and s.mass > cfg.n_min:
            try_split(store, s, r, w_tilde, cfg)             # accept iff BIC decreases
    for s, t in pairs(store.personas):
        if sym_kl(s.predictive, t.predictive) < cfg.tau_merge:
            try_merge(store, s, t, cfg)                      # accept iff BIC decreases

    persist(store, metrics)                                  # personas/*.json + git commit
    return store
```

**Invariants the implementation must preserve:** (i) priors $\alpha^{(0)},\beta^{(0)}$ are immutable after seeding — decay applies only to counts $A,B$; (ii) every count increment carries weight $\rho_{is}=\tilde w_i r_{is}$, never raw $r_{is}$ alone; (iii) decay is applied exactly once per batch to pre-batch counts, and responsibilities are computed from post-decay, pre-M-step counts; (iv) $\lambda=1-1/N_{\max}$ is stored alongside counts so drift-bias bounds (§4.3) remain auditable; (v) Brier scores are computed on held-out confirmations *before* they are consumed by the update, so the metric tracks genuine predictive skill.

---

## 7. Failure modes and operational guards

The algorithms above are only safe under the following guards; all are enforced in CI (WORKFLOWS.md §4) and at runtime.

1. **Confirmation-selection bias beyond raking.** Raking corrects *observed* marginals (taxonomy, state) but not unobserved engagement propensity. Consequence: posteriors describe the *engager-reweighted* population, and $\tilde w_i$ must never be interpreted as making estimates population-unbiased. Mitigations: keep priors anchored to external base rates (§5.1), report per-persona $\mathrm{ESS}$ next to every predictive, and flag personas where $\mathrm{deff}>5$ (weight concentration) for margin review.
2. **Echo chambers.** Confirmations are generated by campaigns that were themselves targeted using current personas — a feedback loop that can inflate $\alpha$ for already-favored attributes. Guard: exploration floor — every campaign must reserve a fraction $\epsilon_{\text{explore}}$ (default 10%) of sends to uniformly random channels, and confirmations from exploration sends carry a `source=exploration` tag so their $w_i$ is computed against the *sent* sample margins, not the engaged margins.
3. **Numerical safety.** All responsibility computations in log-space with log-sum-exp; counts floored at $0$ after decay; Dirichlet totals capped at $\kappa + N_{\max}$ (§4.1) so digamma evaluations stay in the well-conditioned regime; entropy and KL computed with a $10^{-9}$ probability floor.
4. **Replayability.** The append-only confirmation log plus immutable priors fully determine the persona store: `rebuild(log, priors, cfg)` must reproduce any historical commit bit-for-bit (tested in CI). Manual edits to `personas/*.json` are forbidden outside this path.
5. **Privacy.** NPI, name, and email never enter persona JSONs — only hashed lead IDs, cell assignments, and aggregate counts. Persona stores must pass the $k$-anonymity check ($n_s \ge n_{\min}$, §2) before any dashboard export.

---

## Appendix A — Estimator cheat sheet

| Family | Posterior | Mean | Variance | MAP |
|---|---|---|---|---|
| Binary | $\mathrm{Beta}(\alpha,\beta)$ | $\alpha/(\alpha+\beta)$ | $\alpha\beta/[(\alpha+\beta)^2(\alpha+\beta+1)]$ | $(\alpha-1)/(\alpha+\beta-2)$ |
| Categorical | $\mathrm{Dir}(\alpha_1..\alpha_K)$ | $\alpha_k/\alpha_0$ | $\alpha_k(\alpha_0-\alpha_k)/[\alpha_0^2(\alpha_0+1)]$ | $(\alpha_k-1)/(\alpha_0-K)$ |

## Appendix B — Worked micro-example (sanity check for implementers)

One binary attribute ($j=1$, "responds to email"), $S=2$ personas, prior strength $\kappa=20$, global base rate $\bar p_1=0.30$: prior per persona $\alpha^{(0)}=6,\ \beta^{(0)}=14$, predictive $q=0.30$.

History after several batches, persona 1: counts $A=40, B=60$, so $\alpha=46,\beta=74$, $q_{1}=46/120\approx0.383$; persona 2: $A=5,B=40$, $\alpha=11,\beta=54$, $q_2=11/65\approx0.169$. Equal mixing $\pi_1=\pi_2=0.5$.

A new batch arrives 3 days later ($\Delta t=3$, $\lambda=0.995\Rightarrow\lambda^3\approx0.9851$, i.e. $N_{\max}=200$):

1. **Decay counts:** $A_1=39.40, B_1=59.11$; $A_2=4.93, B_2=39.40$. Predictives (with priors): $q_1=(6+39.40)/(6+14+39.40+59.11)=45.40/118.51\approx0.383$; $q_2=10.93/64.33\approx0.170$.
2. **Event:** lead $i$ with $\tilde w_i=0.8$ (post-ESS normalization) confirms ($x=1$). Likelihoods: $q_1=0.383$, $q_2=0.170$; responsibilities $r_{i1}=0.383/(0.383+0.170)\approx0.693$, $r_{i2}\approx0.307$. Combined weights: $\rho_{i1}=0.554,\ \rho_{i2}=0.246$.
3. **M-step increments:** $A_1\mathrel{+}=0.554,\ B_1\mathrel{+}=0$; $A_2\mathrel{+}=0.246$. New predictives: $q_1=45.96/119.07\approx0.386$, $q_2=11.17/64.58\approx0.173$.
4. **Check:** $\alpha+\beta$ for persona 1 is $119.07 \le \kappa+N_{\max}=220$ — the decay cap holds (§4.1); entropy $H[\mathrm{Beta}(45.96,74.11)]\approx -1.70$ nats (differential entropy; concentrated posteriors are negative — compare personas *relative* to the prior's entropy $H[\mathrm{Beta}(6,14)]\approx -0.90$ nats, not to zero) — no structural action.

If instead the same event arrived after 300 days ($\lambda^{300}\approx0.222$): counts collapse toward the prior, $q_1\to0.353$ before the update (versus $0.383$), and the single confirmation moves the predictive far more — exactly the intended adaptivity under §4.3.

## Appendix C — Symbol index

| Symbol | Meaning | Symbol | Meaning |
|---|---|---|---|
| $\alpha_{js},\beta_{js}$ | Beta hyperparams, persona $s$, attr $j$ | $r_{is}$ | responsibility of lead $i$ for persona $s$ |
| $\alpha_{jsk},\alpha_{js,0}$ | Dirichlet hyperparams / total | $\pi_s$ | persona mixing weight |
| $\alpha^{(0)},\beta^{(0)}$ | fixed prior hyperparams | $w_i,\tilde w_i$ | raking weight / ESS-normalized weight |
| $\kappa$ | prior strength (pseudo-counts) | $\rho_{is}=\tilde w_i r_{is}$ | combined update weight |
| $A,B$ | stored decayed data counts | $\lambda,\Delta t$ | decay factor / elapsed periods |
| $\theta_{js},\boldsymbol\phi_{js}$ | latent propensity / category probs | $N_{\max}=(1-\lambda)^{-1}$ | effective sample cap |
| $x_{ij},x_{ijk}$ | confirmation values (binary / one-hot) | $\delta$ | per-period drift bound (§4.3) |
| $q_{js},q_{jsk}$ | posterior predictive probabilities | $\mathrm{ESS}$, deff | Kish effective sample size / design effect |
| $S$ | number of personas | $\tau_{\text{split}},\tau_{\text{merge}}$ | structural-adaptation thresholds |

## References

* Dempster, Laird, Rubin (1977), *Maximum likelihood from incomplete data via the EM algorithm* — §2.
* Neal & Hinton (1998), *A view of the EM algorithm that justifies incremental, sparse, and other variants* — the $\mathcal F$ lower-bound / coordinate-ascent view in §2.3.
* Ireland & Kullback (1968), *Contingency tables with given marginals*; Csiszár (1975), *$I$-divergence geometry of probability distributions and minimization problems* — IPF convergence, §3.2.
* Kish (1965), *Survey Sampling* — ESS / design effect, §3.3.
* Gelman et al., *Bayesian Data Analysis* (3rd ed.) — Beta/Dirichlet conjugate families, prior-strength interpretation, §1, §5.1.
* Schwarz (1978), *Estimating the dimension of a model* — BIC, §5.2.
