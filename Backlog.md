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

---

## P0 — Current Priority: Sprint S14 Measurement Validation

| Item | Scope | Exit Condition / Gate |
|---|---|---|
| **S14.0a Task Feasibility Scout** | 20–40 trials on `google/recurrentgemma-2b`: sham vs secret on-manifold RG-LRU transplant from matched branch trajectory. | Model can distinguish on-manifold internal intervention from sham above chance without generic collapse. |
| **S14.0b Observer Ladder Integration** | Implement input-only, public-history replay, and matched-compute observer baselines with randomized opaque option labels. | Task separates genuine privileged access from public-token reconstruction. Target must beat observer baseline. |
| **S14.0c Source-Control Matrix** | 4-way forced choice: internal state intervention vs external input anomaly vs prefill anomaly vs sham. | Source-localization accuracy exceeds chance and is not driven by simple latency or norm artifacts. |
| **S14.0d Base vs IT Scout** | Compare base (`google/recurrentgemma-2b`) vs instruction-tuned (`google/recurrentgemma-2b-it`) on the validated 4-way task. | Determines whether self-attribution is an intrinsic representation vs a post-training verbal habit. |

---

## P1 — Post-S14 Frontier Candidates

| Item | Focus | Why It Matters |
|---|---|---|
| **S14.1 Confirmatory Metacognition Battery** | Scaled 24-pair confirmatory run on the validated S14.0 design. | Provides definitive causal evidence on whether latent recurrence confers privileged introspective self-access. |
| **S15 Recurrent Adapter Prototype** | Trainable low-rank recurrent state around frozen backbone. | Tests whether dedicated recurrent adapters provide cleaner state continuity than raw frozen weights. |
| **S16 Monitor/Content Dissociation & Level 2 Synthesis** | Attention Schema probe dissociated from first-order factual content. | Final Level 2 Synthesis Memo and Go/No-Go Decision for Horizon 3. |

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