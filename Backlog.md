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

## P0 — Moonshot Gate A: Final Frozen-Model Bridge (Active Frontier)

| Item | Scope | Exit Condition / Gate |
|---|---|---|
| **Q01 & Q02: Monitor / Content Coda** | Two-phase scout on `recurrentgemma-2b-it`: Phase A Discovery $\to$ Phase B Causal $2 \times 2$ ($C_\pm \times M_\pm$). Protocol: `docs/S16_Monitor_Content_Dissociation_Protocol.md`. | Pass if double dissociation found ($|\Delta D| \le 0.15$, $|\Delta \text{Report}| \ge 0.40$); if symmetrical coupling, terminate and proceed to Gate B. |
| **Q03: Base vs. IT Metacognition Scout** | Strict-C `quartz_basalt` micro-assay on matched `google/recurrentgemma-2b` (base) vs `google/recurrentgemma-2b-it`. | Determines whether report modulation is intrinsic to pretraining or installed by alignment. |

---

## P1 — Moonshot Gate B: Minimal Developmental Substrate (The Continuity Garden)

| Item | Focus | Key Question / Theoretical Stakes |
|---|---|---|
| **Q04: Temporally Extended POMDP** | Small recurrent organism baseline in minimal symbolic POMDP without language. | Does persistent latent state confer a selective advantage over feed-forward and reset baselines? |
| **Q05: Latent Continuity vs. Restored Memory** | Interruption assay: memory restored vs. latent preserved vs. replay. | Does latent developmental state carry function beyond explicit prompt memory restoration? |
| **Q06: Ground-Truth Construct Separation** | 3-layer architecture: Ground Truth $\to$ Sensor $\to$ Learned State. | Zero target-variable leakage into observations (`uncertainty`, `self_state`, `source`). |

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