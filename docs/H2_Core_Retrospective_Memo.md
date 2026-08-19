# Horizon 2 Core Retrospective Memo: Causal Latent Continuity & Empirical Dissociations

**Date:** 2026-08-18  
**Scope:** Sprints S10–S12c (`google/recurrentgemma-2b`)  
**Status:** **FROZEN RESEARCH MEMO (Calibrated with S12c.1 Specificity Microscope Results)**

---

## 1. Executive Synthesis: What Horizon 2 Established

Across Sprints S10, S11b, S12b, and S12c, Horizon 2 established **causal latent memory carrying value-specific historical information, not privileged or autonomous continuity**:

> *The recurrent state of `RecurrentGemma-2B` is hidden from prompt text but exactly reconstructible from public token history ($S_t = \mathcal{F}_\theta(x_{1:t})$); it persists physically long after local sliding-window attention has evicted direct access to historical tokens ($L=4096 = 2W$); it directly and causally steers the downstream logit distribution along the donor trajectory ($P_{\text{RGLRU}} = +74.10$); it carries value-specific historical information beyond syntactic sentence templates ($\Delta P_{\text{value\_spec}} = +38.49$, $\Delta \alpha_{\text{value\_spec}} = +0.1744$); but explicit factual retrieval of the original binding has largely disappeared from behavioral output. The state has not yet demonstrated autonomous internal evolution, source ownership, metacognitive access, or informational privilege.*

---

## 2. The Five-Way Theoretical Taxonomy

Horizon 2 replaces the coarse binary distinction ("external prompt memory vs. native recurrence") with a five-way property taxonomy:

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│ THE HORIZON 2 THEORETICAL TAXONOMY                                                                │
├──────────────────────┬───────────────────────────────┬──────────────────────┬─────────────────────┤
│ Dimension            │ Core Question                 │ Empirical Status     │ Empirical Grounding │
├──────────────────────┼───────────────────────────────┼──────────────────────┼─────────────────────┤
│ 1. Reconstructibility│ Can an external observer      │ YES                  │ S10 Replay Invariant│
│                      │ reconstruct state from text?  │ (Privacy != Priv)    │ (S_t = F_theta(x))  │
├──────────────────────┼───────────────────────────────┼──────────────────────┼─────────────────────┤
│ 2. Persistence       │ Does historical information   │ YES                  │ S11b RG-LRU Traces  │
│                      │ physically survive over time? │ (At 2W = 4096 tokens)│ (R_RGLRU ~ 0.34)    │
├──────────────────────┼───────────────────────────────┼──────────────────────┼─────────────────────┤
│ 3. Causal Leverage   │ Does changing state change    │ YES                  │ S12b Surgical Swaps │
│                      │ subsequent model computation? │ (P_RGLRU = +74.10)   │ (CI [46.79, 106.72])│
├──────────────────────┼───────────────────────────────┼──────────────────────┼─────────────────────┤
│ 4. Value Specificity │ Does state carry specific     │ YES                  │ S12c Specificity    │
│                      │ token value vs template info? │ (Delta P = +38.49)   │ (CI [25.82, 50.85]) │
├──────────────────────┼───────────────────────────────┼──────────────────────┼─────────────────────┤
│ 5. Access / Ownership│ Can the model monitor its own │ UNTESTED / UNKNOWN   │ Topic of S14        │
│                      │ state with unique privilege?  │ (Secret Injections)  │ (Base vs IT Models) │
└──────────────────────┴───────────────────────────────┴──────────────────────┴─────────────────────┘
```

### Critical Epistemic Guardrails
- **Hidden $\ne$ Privileged:** An internal state variable not exposed in prompt text can still be 100% determined by public tokens.
- **Recurrent $\ne$ Autonomous:** Gated linear recurrence is input-driven; absence of token input means absence of transition clock.
- **Causal $\ne$ Conscious / Metacognitive:** A physical state can steer logits without the system having introspective access to that state.
- **Representation $\ne$ Reportability:** Latent output dispositions can carry historical structure even when factual cloze recovery fails.

---

## 3. The Three-Way Memory Dissociation at $2W=4096$ Tokens

At twice the local attention window ($2W=4096$), S11b, S12b, and S12c establish a profound three-way dissociation:

$$\text{Physical State Divergence Remains } \implies \text{ Value-Specific Causal Steering Remains } \centernot\implies \text{ Explicit Fact Retrieval}$$

1. **Physical Persistence (S11b):**
   Branch-specific RG-LRU state separation remains clearly above the sham floor across all 4 filler regimes at $2W=4096$ tokens, with retention ranging roughly $0.045$ to $0.338$ ($R_{\text{constant}} \approx 0.338$, $R_{\text{interfering}} \approx 0.080$, $R_{\text{natural}} \approx 0.051$, $R_{\text{random}} \approx 0.045$; all 95% CIs strictly exclude zero).
2. **Value-Specific Causal Output Steering (S12c):**
   Transplanting matching RG-LRU state produces a strongly resolved value-specific advantage over same-template wrong-value states ($\Delta P_{\text{value\_spec}} = +38.49$, 95% CI $[+25.82, +50.85]$; $\Delta \alpha = +0.1744$ $[+0.1001, +0.2536]$).
3. **Behavioral Usability Failure (S11b/S12b):**
   Zero-shot factual cloze margins at $2W$ span zero in all regimes (e.g. natural: $+0.009$ $[-0.071, +0.089]$), and task-specific signed graft effects are unresolved.

**Scientific Interpretation:**  
The model's latent state remains distinct according to what happened in history, and that specific difference causally shifts the output distribution—without necessarily making the fact explicitly reportable in output text.

---

## 4. The Descriptive Contrast Ladder (S12c)

Rather than asserting orthogonal latent sub-components, the S12c results form a clear **descriptive contrast ladder**:

$$\begin{aligned}
P_{\text{noise}} &= +48.23 \quad \text{[Matched-norm Frobenius noise control]} \\
P_{\text{wrong\_val}} &= +83.13 \quad \text{[Same-template wrong-value history: } \Delta P = +34.89 \text{ over noise]} \\
P_{\text{match}} &= +121.62 \quad \text{[Matching historical value: } \Delta P = +38.49 \text{ over wrong-value]} \\
P_{\text{whole}} &= +218.76 \quad \text{[Whole-state positive reference]}
\end{aligned}$$

- **Structured Nonmatching Histories Steer More Than Noise:** Any structured historical state ($+75$ to $+83$) provides a broad recurrent baseline exceeding Gaussian noise ($+48.23$).
- **Matching History Provides a Selective Advantage:** Holding sentence template identical, matching history adds $+38.49$ $[+25.82, +50.85]$.
- **Template Increment Unresolved:** The contrast between same-template wrong-value and cross-template historical states ($\Delta P = +7.38$ [$-8.26, +24.73$]) spans zero. We do not resolve an additional template increment over the cross-template control used, rather than establishing equivalence.
- **Whole-State Reference is Not a Ceiling:** In individual cells, chimeric state transplantation ($P_{\text{RGLRU}} = +288.10$) overshoots the whole-state reference ($P_{\text{whole}} = +255.38$).

---

## 5. Methodological & Cognitive Connections

- **Causal Abstraction (Geiger et al., 2021; 2024):**  
  Our multi-store swap harness implements interchange interventions on neural representations to causally verify value-level historical bindings.
- **Activity-Silent Working Memory (Wolff et al., 2017; Rose et al., 2016; Stokes, 2015):**  
  Item-specific representations that cease to be visible in ongoing zero-shot generation remain latent in recurrent state dynamics and are revealed through causal intervention probes.

---

## 6. Horizon 2 Roadmap from Core Freeze to Completion

```
+---------------------------------------------------------------------------------------------------+
| HORIZON 2 SPRINT PLAN (S10–S16)                                                                   |
+---------------------------------------------------------------------------------------------------+
| S10: Fail-Closed Model Bring-Up & Replay Invariants       | COMPLETE (Replay Invariant Verified)  |
| S11b: Latent Impulse Retention & Temporal Anatomy         | FROZEN (Physical Persistence at 2W)   |
| S12b: Multi-Store Surgical Swaps & Causal Attribution     | FROZEN (Causal Steering P = +74.10)   |
| S12c: Specificity Microscope (Within-Template vs Value)   | FROZEN (Value Spec = +38.49 [25, 50]) |
| S13: Null-Observation / Controlled Recurrent Dynamics     | S13.0 Audit -> S13.1 Controlled Sweep |
| S14: Latent Metacognition, Reality Monitoring & Ownership | Secret Injections, Base vs IT Models  |
| S15: Recurrent Adapter Prototype & Low-Rank Continuity    | Cross-Session Parameterized Memory    |
| S16: Monitor/Content Dissociation & Level 2 Synthesis     | Final H2 Go/No-Go Decision for H3     |
+---------------------------------------------------------------------------------------------------+
```
