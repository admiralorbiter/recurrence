# Horizon 2 Core Retrospective Memo: Causal Latent Continuity & Empirical Dissociations

**Date:** 2026-08-18  
**Scope:** Sprints S10–S12b (`google/recurrentgemma-2b`)  
**Status:** **FROZEN RESEARCH MEMO**

---

## 1. Executive Synthesis: What Horizon 2 Established

Across Sprints S10, S11b, and S12b, Horizon 2 established **causal latent memory, not privileged or autonomous continuity**:

> *The recurrent state of `RecurrentGemma-2B` is hidden from the prompt text but exactly reconstructible from public token history ($S_t = \mathcal{F}_\theta(x_{1:t})$); it persists physically long after local sliding-window attention has evicted direct access to historical tokens ($L=4096 = 2W$); it directly and causally steers the downstream logit distribution along the donor trajectory ($P_{\text{RGLRU}} = +74.10$); matching history has a selective advantage over other structured cross-pair histories ($\Delta P_{\text{spec\_unrel}} = +19.68$); but explicit factual retrieval of the original binding has largely disappeared from behavioral output. The state has not yet demonstrated autonomous internal evolution, source ownership, metacognitive access, or informational privilege.*

---

## 2. The Four-Way Theoretical Taxonomy

Horizon 2 replaces the coarse binary distinction ("external prompt memory vs. native recurrence") with a four-way property taxonomy:

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
│ 4. Access / Ownership│ Can the model monitor its own │ UNTESTED / UNKNOWN   │ Topic of S14        │
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

At twice the local attention window ($2W=4096$), S11b and S12b establish a three-way dissociation:

$$\text{Physical State Divergence Remains } \implies \text{ Causal Output Leverage Remains } \centernot\implies \text{ Explicit Fact Retrieval}$$

1. **Physical Persistence (S11b):**
   Branch-specific RG-LRU state separation remains clearly above the sham floor across all 4 filler regimes ($R_{\text{RGLRU}} \approx 0.34$, 95% CI $[0.26, 0.42]$).
2. **Causal Output Leverage (S12b):**
   Transplanting matching RG-LRU state produces a strongly positive directional logit displacement ($P_{\text{RGLRU}} = +74.10$, 95% CI $[+46.79, +106.72]$).
3. **Behavioral Usability Failure (S11b/S12b):**
   Zero-shot factual cloze margins at $2W$ span zero in all regimes (e.g. natural: $+0.009$ $[-0.071, +0.089]$), and task-specific signed graft effects are unresolved.

**Scientific Interpretation:**  
The historical trajectory actively alters the model's latent output disposition in a history-dependent way long after explicit retrieval of the lexical binding has ceased to be resolved.

---

## 4. Partial Specificity: Matching Enrichment on a Cross-History Manifold

S12b resolves matching-history enrichment above cross-pair donors:
- $\text{Matching RG-LRU: } P_{\text{match}} = +74.10$
- $\text{Unrelated (+1) Donor: } P_{\text{unrel}} = +54.42$
- $\text{Permuted (+7) Donor: } P_{\text{perm}} = +44.46$
- $\text{Frobenius Noise: } P_{\text{noise}} = +17.64$

The matching increments ($\Delta P_{\text{spec\_unrel}} = +19.68$, $\Delta P_{\text{spec\_perm}} = +29.64$) prove a selective matching-history component. However, the substantial baseline displacement of cross-pair donors ($+54.42$) indicates that recurrent states carry shared event-manifold or task-template geometry across items.

---

## 5. Horizon 2 Roadmap from Core Freeze to Completion

```
+---------------------------------------------------------------------------------------------------+
| HORIZON 2 SPRINT PLAN (S10–S16)                                                                   |
+---------------------------------------------------------------------------------------------------+
| S10: Fail-Closed Model Bring-Up & Replay Invariants       | COMPLETE (Replay Invariant Verified)  |
| S11b: Latent Impulse Retention & Temporal Anatomy         | FROZEN (Physical Persistence at 2W)   |
| S12b: Multi-Store Surgical Swaps & Causal Attribution     | FROZEN (Causal Steering P = +74.10)   |
| S12c: Specificity Microscope (Compact Held-Out Panel)     | NEXT (Within-Template vs Value Null)  |
| S13: Null-Observation / Controlled Recurrent Dynamics     | S13.0 Audit -> S13.1 Controlled Sweep |
| S14: Latent Metacognition, Reality Monitoring & Ownership | Secret Injections, Base vs IT Models  |
| S15: Recurrent Adapter Prototype & Low-Rank Continuity    | Cross-Session Parameterized Memory    |
| S16: Monitor/Content Dissociation & Level 2 Synthesis     | Final H2 Go/No-Go Decision for H3     |
+---------------------------------------------------------------------------------------------------+
```
