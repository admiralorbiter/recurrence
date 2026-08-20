# Backlog & Priority Queue

## Decision & Demotion Rules

Retire or demote a direction when:
- effects vanish under basic paraphrase/order controls;
- observer conditions match self conditions across well-powered tests;
- latent-state benefits vanish after memory and compute matching;
- intervention effects are fully explained by norm/damage;
- a new paper resolves the question more convincingly than we can;
- required access exceeds available compute with no smaller analogue;
- the experiment cannot discriminate competing explanations;
- the result would be interesting only under an unjustified consciousness interpretation.

---

## Completed / Frozen Foundations (H0–H2 Core)

| Sprint / Phase | Area | Key Delivered Outcome | Status |
|---|---|---|---|
| **H0 (S00–S03)** | Harness & Measurement | Reproducible replay harness, forced-choice KV baseline, Level 0 PAI observer controls. | **FROZEN** |
| **H1 (S04–S09)** | Scaffolded Persistence | Evaluated prompt-level memory vs state across E03–E09. Established narrative primacy and memory dominance; no independent state leverage. | **FROZEN** |
| **S10** | Replay Invariants | RecurrentGemma multi-store adapter; proved exact deterministic replay reconstruction ($S_t = \mathcal{F}_\theta(x_{1:t})$). | **FROZEN** |
| **S11b** | Physical Retention | Branch-specific RG-LRU separation remains resolved at $2W=4096$ tokens across 4 filler regimes; factual cloze decays within window. | **FROZEN** |
| **S12b** | Causal Steering | Surgical RG-LRU transplantation causally steers downstream logits ($P_{\text{RGLRU}} = +74.10$ $[+46.79, +106.72]$). | **FROZEN** |
| **S12c** | Value Specificity | Specificity Microscope: holding template fixed, matching history adds $+38.49$ over same-template wrong values. | **FROZEN** |
| **S13 / S13.3** | Dynamics & Geometry | 24-pair confirmatory + 4-pair strict sensitivity: historical $u_0$ steering decays to zero, state difference reorients ($C_R \to 0.12$), contemporaneous steerability remains positive ($V^{(N)} = +13.95$). | **FROZEN** |
| **S14 / S14.0C** | Metacognition & Provenance | C/D/R/A framework, BOP calibration (100% visible pass rate), strict-C report modulation (`quartz_basalt`), and matched-POST temporal equivalence ($\Delta M_{\text{timing}} \approx 0$, $p_{\text{TOST}} = 0.0048$). *State-conditioned reporting $\neq$ historical-provenance discrimination.* | **FROZEN** |

---

| **Gate B (Q04)** | Garden v0 Baseline | GRU 100% vs Current-MLP 51% vs History-MLP (K=4) 50% vs Sham 100% vs Reset 51-53% across 8 seeds. Zero construct leakage verified. | **FROZEN** |
| **Gate B (Q04b)** | Value Specificity | 800 paired transplants: same-$z$ 100%, opposite-$z$ 0% on recipient world (100% on counterfactual $1-z$ criterion). $h_t$ is causally value-specific. | **FROZEN** |
| **Gate B (Q05/Q05b)** | Reconstructibility & Geometry | Replay achieves exact deterministic parity ($\cos=1.0000, d=0.0000$). Cue restoration achieves behavioral recovery via distinct state ($\cos=0.235$). Intact state displays seed-dependent dynamical heterogeneity ($95.7\% \to 78.1\%$). | **FROZEN** |

---

## P0 — Moonshot Gate C: Learned Controllability & Agency (Active Frontier)

| Item | Scope | Exit Condition / Gate |
|---|---|---|
| **Q07: Behavioral Controllability** | Yoked controllability arena ($W_{\text{ctrl}}$ vs $W_{\text{yoked}}$). Dual loss: forward prediction + experienced return actor-critic. Payoff: $+0.70$ (ctrl) vs $-0.10$ (yoked) vs $0.00$ (abstain). | $\mathbb{E}[R \mid W_{\text{ctrl}}] \ge 0.50$, $P(\text{Abstain} \mid W_{\text{yoked}}) \ge 0.70$, 3-tier observers pass. |
| **Q08: Controllability Vector Decoding** | Probe internal state $h_t$ for linearly separable controllability direction after controlling for marginals. | Decoding accuracy $\ge 0.85$ beating input-only observer. |
| **Q09a: Surgical Controllability Confusion** | Bidirectional causal patching of candidate $\pm c_{\text{ctrl}}$ subspace between matched $W_{\text{ctrl}}$ and $W_{\text{yoked}}$ trials. | Induces selective exploitation in $W_{\text{yoked}}$ and abstention in $W_{\text{ctrl}}$. |

---

## P1 — Moonshot Gate A: Final Frozen-Model Bridge (Scaffold Ready / Live GPU Queued)

| Item | Focus | Key Question / Theoretical Stakes |
|---|---|---|
| **Q01 & Q02: Monitor / Content Coda** | Two-phase live scout on `recurrentgemma-2b-it`. Phase A discovery on 128 items $\to$ Phase B causal $2 \times 2$ ($m_\perp$). | Out-of-sample incremental $p < 0.01$, $|\Delta D| \le 0.15$, $|\Delta M| \ge 0.40$. |
| **Q03: Base vs. IT Metacognition Scout** | Live evaluation of `google/recurrentgemma-2b` (base) on strict-C transport and visible BOP reporting. | Determines whether report modulation is intrinsic to pretraining or alignment-installed. |

---

## Methods Sidecars (Do Not Block Main Frontier)

| Sidecar Topic | Motivation | Handling Rule |
|---|---|---|
| **B1 vs B5 Finite-Precision Bifurcation Mechanism** | Conv1D 4-tap rolling buffer turnover vs BF16 GEMM accumulation. | Maintain as dedicated numerical methods note; do not divert compute from S14 unless S14 estimand is affected. |
| **Regime-Specific Recurrent Carry Dynamics** | Constant expansion ($Q_R \approx 15.3$) vs random contraction ($Q_R \approx 0.94$). | Formulate as interaction hypothesis for future analysis; keep secondary to metacognition. |

---

## Do Not Chase

- Uncontrolled prompt self-narratives without matched-observer controls.
- Generic language benchmark fine-tuning without architectural intervention.
- Speculative claims about consciousness, sentience, or human-like qualia.
- Large-scale compute scaling (>10 hours) before small-scale measurement validity is proven.