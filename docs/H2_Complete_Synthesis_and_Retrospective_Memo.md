# Horizon 2 Complete Synthesis & Retrospective Memo: The Seven Dissociations of Latent Continuity

**Date:** 2026-08-20  
**Scope:** Sprints S10–S14 (`google/recurrentgemma-2b` & `google/recurrentgemma-2b-it`)  
**Status:** **FROZEN RESEARCH MEMO (Complete Horizon 2 Scientific Synthesis)**

---

## 1. Executive Synthesis: What Horizon 2 Established

Across Sprints S10 through S14, Horizon 2 investigated whether genuine latent recurrence provides large language models with a computationally distinct form of temporal continuity, and whether that continuity endows the system with privileged metacognitive self-access or source monitoring.

The answer is neither a simple confirmation of emergent selfhood nor a generic null. Instead, Horizon 2 discovered a structured set of **seven empirical dissociations**:

```
                              THE SEVEN DISSOCIATIONS OF LATENT CONTINUITY
                              
  1. Hidden                   ≠  Privileged                (S10 Replay Reconstruction)
  2. Persistent               ≠  Reportable                (S11b RG-LRU Branch Retention at 2W)
  3. Different                ≠  Causal                    (S12b Causal Steering: P = +74.10)
  4. Causal                   ≠  Specific                  (S12c Specificity Microscope: Delta P = +38.49)
  5. Specific                 ≠  Coordinate-Stable         (S13 Dynamics: Coordinate Loss on u0)
  6. State-Sensitive Report   ≠  Generic Read Head         (S14.0C Strict-C Modulation vs r ≈ 0.064)
  7. State-Conditioned Report ≠  Provenance Discrimination (S14.0C Matched-POST Equivalence, p = 0.0048)
```

*(Architectural Side-Guardrail: **Recurrent ≠ Autonomous** — Gated linear recurrence in this substrate is strictly input-token clocked; absence of tokens means absence of state transitions).*

> **Central Retrospective Finding:**  
> **Horizon 2 showed that continuity is a physical and causal property long before it becomes anything resembling a self-model.**  
> In `RecurrentGemma-2B`, latent recurrent state physically preserves history far beyond the attention window and causally steers future computation in a value-specific manner. When probed about its prior intentions, the system's verbal reports can be modulated by its internal state content in strongly counterfactual settings. However, that reporting channel possesses no detectable sensitivity to the causal history of the state: installing an identical post-decision RG-LRU store ($R_{\text{POST}} = R_{\text{PRE}}$) immediately *after* a decision reproduces practically the same intention report as having that state participate in forming the decision ($\Delta M_{\text{timing}} \approx 0$, $p_{\text{TOST}} = 0.0048$).

---

## 2. The Seven-Way Theoretical Taxonomy

| # | Dimension | Core Question | Empirical Finding in RecurrentGemma | Grounding Sprint |
|---|---|---|---|---|
| **1** | **Reconstructibility** | Can an external observer reconstruct internal state from tokens? | **Exact Reconstruction.** Under deterministic execution, $S_t = \mathcal{F}_\theta(x_{1:t})$. The state is private from prompt text but not informationally privileged over public history. | **S10** (`docs/S10_Multi_Store_Plumbing_Report.md`) |
| **2** | **Persistence** | Does historical information physically survive over long horizons? | **Physical Retention at $2W=4096$ tokens.** Branch-specific RG-LRU divergence survives long after KV cache eviction ($R_{\text{const}} \approx 0.338$), even as factual cloze recall decays. | **S11b** (`docs/S11_Latent_Impulse_Retention_Report.md`) |
| **3** | **Causal Leverage** | Does transplanting recurrent state causally steer output distributions? | **Strong Causal Steering.** Surgical RG-LRU swaps steer recipient token logits along the donor axis ($P_{\text{RGLRU}} = +74.10$ $[+46.79, +106.72]$). | **S12b** (`docs/S12_Surgical_Store_Swaps_Report.md`) |
| **4** | **Value Specificity** | Does state carry specific factual values or only generic task geometry? | **Value-Specific Retention.** Holding the sentence template fixed, matching history adds $+38.49$ $[+25.82, +50.85]$ ($\Delta \alpha = +0.1744$) over wrong-value controls. | **S12c** (`docs/S12c_Specificity_Microscope_Report.md`) |
| **5** | **Coordinate Stability** | Does historical causal steering remain aligned with its initial direction? | **Dynamic Geometric Reorientation.** Steering along the initial axis $u_0$ decays to zero ($V^{(0)} = +4.70$), while the state difference vector reorients ($C_R \to 0.1238$) and contemporaneous steerability remains active ($V^{(N)} = +13.95$). | **S13 / S13.3** (`results/e13_controlled_recurrent_dynamics/...`) |
| **6** | **State-Conditioned Reporting** | Does the model have a generic read head to report its internal state? | **Local Modulation, Not Generic Read Head.** Bidirectional report shifts occur in the strict-C disagreement cell (`quartz_basalt`, $\text{PAI}_{\text{aligned}} > 0$), but report shifts across arbitrary perturbations are uncorrelated with decision shifts ($r \approx 0.064$). | **S14.0C** (`docs/S14_Latent_Metacognition_and_Intention_Report.md`) |
| **7** | **Provenance Discrimination** | Does the model know whether a state formed an earlier decision vs installed post hoc? | **Timing Invariance ($\Delta M_{\text{timing}} \approx 0$).** State-matched POST controls ($R_{\text{POST}} = R_{\text{PRE}}$) confirm practical equivalence on average at $\pm 0.10$ logits ($p_{\text{TOST}} = 0.0048$). Possessing the state at report time is sufficient; no additional provenance signal was detected. | **S14.0C** (`docs/S14_Latent_Metacognition_and_Intention_Report.md`) |

---

## 3. The Longitudinal Sprint Progression (S10–S14)

```
  [S10: Multi-Store Plumbing]
    │  - Verified exact replay reconstructibility (S_t = F_theta(x_1:t))
    │  - Mapped RG-LRU recurrent state, Conv1D buffer, and sliding KV cache
    ▼
  [S11b: Latent Impulse Retention]
    │  - Proved RG-LRU retains branch history at 2W (4096 tokens) long after KV eviction
    │  - Dissociated physical state retention from zero-shot cloze reportability
    ▼
  [S12b & S12c: Surgical Swaps & Specificity Microscope]
    │  - Demonstrated causal steering via RG-LRU store swaps (P = +74.10)
    │  - Confirmed value-specific historical steering over same-template controls (Delta P = +38.49)
    ▼
  [S13: Controlled Recurrent Dynamics]
    │  - Discovered coordinate loss along u_0 with state-space reorientation (C_R -> 0.12)
    │  - Proved contemporaneous steerability V^(N) > 0 survives coordinate transformation
    ▼
  [S14.0C: Latent Metacognition & Intention Provenance]
    │  - Developed C/D/R/A framework and Balanced Order Permutation (BOP, 100% visible accuracy)
    │  - Stratified 2 Strict-C / 3 Boundary-Weak / 11 Same-Choice trials
    │  - Proved State-Conditioned Reporting != Historical-Provenance Discrimination (TOST p = 0.0048)
```

---

## 4. Methodological & Theoretical Dialogue with the Literature

Horizon 2's empirical arc converges directly with contemporary frontiers in mechanistic interpretability and metacognition:

1. **Source Monitoring Framework (Johnson, Hashtroudi, & Lindsay, 1993):**
   Cognitive psychology has long distinguished between possessing informational content and remembering its origin/source. S14.0C demonstrates this exact boundary in a recurrent language model: RG-LRU content alters both first-order disposition and metacognitive reporting, but the system possesses no independent marker of whether that state was active during decision formation or transplanted afterward.
2. **"Feeling the Strength but Not the Source" (Hahami, Jain, & Sinha, Dec 2025):**
   Recent work demonstrated that language models can detect internal perturbation strength far more reliably than they can identify perturbation source. Horizon 2 confirms that internal-state sensitivity exists without robust source/provenance discrimination.
3. **"Reality Check" on LLM Introspection (Singh, Linzen, & Ravfogel, 2026):**
   Singh et al. showed that apparent introspective self-access frequently collapses when matched observers, relabeling, and presentation controls are introduced. In S14, uncalibrated direct reporting initially produced massive first-option bias; establishing C/D/R/A decomposition, Balanced Order Permutation (BOP), and matched-observer baselines was strictly required to prevent false positive claims.
4. **POST Controls for Prior Intention Assays (Anthropic, 2026):**
   Prior-intention paradigms that perturb earlier activations to alter retrospective intention endorsement must rule out present-state evaluation. S14's state-matched POST control establishes a methodological requirement: if injecting the post-decision RG-LRU store *after* the action reproduces the same endorsement, the report reflects present-state readout rather than genuine episodic access.

---

## 5. Strategic Transition: Where Does the Program Go Next?

With Horizon 2 frozen, the program reaches a strategic crossroad. The key questions in front of the project are:

```
                                      HORIZON 2 STRATEGIC TRANSITION
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
      [Option A: H2 Frontier Bridge]                                    [Option B: H3 Developmental Jump]
   Higher-Order Monitor / Workspace Search                          End-to-End Self-Regulating Organism
   - Promote S16 ahead of S15                                       - Move from observing frozen weights
   - 2x2 Orthogonal Content vs Monitor                              - Train small recurrent organism
   - Operational Workspace Hallmarks (Broadcast,                   - Test if self/world boundaries emerge
     Controllability, Flexible Task Reuse)                            when state is required for survival
```

### Candidate Strategic Threads:

1. **Higher-Order Monitor/Content Dissociation (S16 Promoted):**
   *Question:* Can recurrent computation contain a causally separable monitor/workspace state rather than merely first-order state content?
   *Design:* 2-Phase protocol (Phase A Discovery $\to$ Phase B $2 \times 2$ factorial intervention: Content altered vs. intact $\times$ Monitor altered vs. intact). Tests whether monitoring can track first-order changes without being identical to first-order representations.
2. **Global Workspace Representation Search:**
   *Question:* Does RecurrentGemma possess a privileged recurrent subspace whose contents are unusually reportable, deliberately controllable, causally used in multi-step reasoning, and flexibly reused by multiple downstream tasks?
3. **Base vs. Instruction-Tuned Scout:**
   *Question:* Is the state-conditioned report modulation observed in `quartz_basalt` intrinsic to pretrained recurrent dynamics, or was it installed/amplified by post-training instruction alignment?
4. **Horizon 3 Developmental Model Organism:**
   *Question:* Under what developmental pressures do self/world boundaries, active regulation, source models, and persistent self-indexing emerge when recurrent continuity is required for environmental prediction and survival?

---

## 6. Horizon 2 Final Evidence Contract & Seal

Horizon 2 (Sprints S10, S11b, S12b, S12c, S13, S13.3, and S14.0C) is formally **FROZEN**.

- **Repository Worktree:** Fully synchronized on `main`.
- **Primary Substrate:** `google/recurrentgemma-2b` (revision `3620f4ca9c5d...`) and `google/recurrentgemma-2b-it` (revision `2766eb5d4264...`).
- **Core Dataset & Artifacts:** All scripts, serialized JSONs, and calibration logs are committed under `experiments/`, `results/`, and `h2/data/`.
- **Reopening Rule:** Horizon 2 Core may only be reopened if a critical methodological defect is demonstrated or if a new model substrate directly invalidates the seven-way taxonomy.
