# Horizon 2 Final Synthesis Map

**Status:** FROZEN CANONICAL BLUEPRINT  
**Target Substrate:** `google/recurrentgemma-2b` & `google/recurrentgemma-2b-it`  
**Core Retrospective Aphorism:**  
> *"Horizon 2 showed that continuity is a physical and causal property long before it becomes anything resembling a self-model."*

---

## 1. The Seven-Way Horizon 2 Empirical Taxonomy (Frozen Core)

| Property / Dimension | Core Scientific Question | Empirical Answer | Concrete Evidence | Critical Boundary / Scope | Next-Level Implication |
|---|---|:---:|---|---|---|
| **1. Reconstructibility** | Can public token history reproduce hidden internal state? | **YES** | **S10 Replay Invariant:** Exact reconstruction $S_t = \mathcal{F}_\theta(x_{1:t})$ across deterministic execution runs. | Exact reconstruction was verified under deterministic execution; stochastic generation was not tested. | **Hidden $\neq$ Privileged.** Internal state is private from prompt text but not informationally privileged over public history. |
| **2. Persistence** | Does historical information physically survive local attention eviction? | **YES** | **S11b RG-LRU Traces:** Branch-specific separation remains resolved at $2W=4096$ tokens across 4 filler regimes ($R_{\text{const}} \approx 0.338$). | Factual cloze retrieval probe simultaneously collapses to chance at 2W. | **Persistence $\neq$ Reportability.** Physical retention in recurrent weights does not guarantee symbolic/factual recall. |
| **3. Causal Leverage** | Does transplanting recurrent state causally steer output distributions? | **YES** | **S12b Surgical Swaps:** Matching RG-LRU store swaps steer downstream token generation ($P_{\text{RGLRU}} = +74.10$ $[+46.79, +106.72]$). | Interventions must respect multi-store boundaries (RG-LRU vs. Conv1D vs. KV). | **Different $\neq$ Causal.** Physical divergence is confirmed to have real causal leverage over future computation. |
| **4. Value Specificity** | Does the surviving state carry specific values or generic task template geometry? | **YES** | **S12c Specificity Microscope:** Matching history adds $+38.49$ $[+25.82, +50.85]$ ($\Delta \alpha = +0.1744$) over same-template wrong values at 2W. | Evaluated on controlled synthetic template families; template increment itself was unresolved (+7.38). | **Causal $\neq$ Specific.** Latent recurrence carries value-specific historical content rather than generic perturbation energy. |
| **5. Dynamical Coordinate Stability** | Does historical memory remain fixed in its original output-space coordinate frame? | **NO** | **S13 Confirmatory Dynamics:** Steering along initial baseline axis $u_0$ decays to zero ($V^{(0)} = +4.70$), state difference reorients ($C_R \to 0.1238$), while contemporaneous steerability remains positive ($V^{(N)} = +13.95$). | Trajectory details are execution-batch sensitive ($B=1$ vs $B=5$), but aggregate state reorientation $C_R$ is robust. | **Specific $\neq$ Coordinate-Stable.** Memory survives not as a static coordinate, but as a dynamically transforming manifold. |
| **6. State-Conditioned Reportability** | Can the model report its private computational state when prompted? | **LOCALLY YES** | **S14.0C Strict-C Assay:** In the strongly counterfactual `quartz_basalt` cell ($\Delta = \pm 1.02$), PRE reports shifted toward private facts in both directions ($\text{PAI}_{\text{aligned}} > 0$). | Only 1/2 directions yielded the correct discrete choice; across 11 same-choice controls, decision and report shifts were uncorrelated ($r \approx 0.064$). | **State Sensitivity $\neq$ Generic Read Head.** Report modulation is a local state-conditioned effect, not a universal introspective channel. |
| **7. Historical Provenance Discrimination** | Does the model's report distinguish whether a state formed a prior decision vs installed post hoc? | **NO SIGNAL DETECTED** | **S14.0C State-Matched POST Control:** Installing the exact post-decision RG-LRU after the forced output yields practically equivalent reports on average ($\Delta M_{\text{timing}} = +0.0348$, $p_{\text{TOST}} = 0.0048$). | Equivalence established at $\pm 0.10$ SESOI with modest cell-level heterogeneity; RG-LRU matched while Conv/KV remained recipient-derived. No additional causal-history/provenance signal was detected beyond current RG-LRU content in the tested assay. | **State Access $\neq$ Provenance Access.** Possessing internal state content is sufficient; the model does not detect its causal history. |

---

## 2. Two Open Strategic Frontiers

| Strategic Frontier | Core Scientific Question | Status & Candidate Next Step | Grounding & Target Architecture |
|---|---|---|---|
| **A. Monitor / Content Dissociation** | Is there a second-order latent variable that monitors first-order computation without duplicating it? | **OPEN (Final H2 Mechanistic Coda).** Pre-registered $2 \times 2$ factorial scout ($C_\pm \times M_\pm$) across candidate subspace directions. | Protocol: `docs/S16_Monitor_Content_Dissociation_Protocol.md` |
| **B. Developmental Self / Source Emergence** | Can source models, self/world boundaries, and active regulation emerge through learning? | **OPEN (Horizon 3 Frontier).** Move from observing frozen weights to training an interactive developmental organism where persistent state is necessary for survival. | Spec: `9. The Continuity Garden.md` |

---

## 3. Epistemic Guardrail Summary

```
                      THE HORIZON 2 METHODOLOGICAL GUARDRAILS
                      
  1. Hidden                   ≠  Privileged
  2. Persistent               ≠  Reportable
  3. Different                ≠  Causal
  4. Causal                   ≠  Specific
  5. Specific                 ≠  Coordinate-Stable
  6. State-Sensitive Report   ≠  Generic Read Head
  7. State-Conditioned Report ≠  Historical-Provenance Discrimination
```

---

## 4. Downstream Document Integration

This synthesis map serves as the shared blueprint for:
1. **The Technical Retrospective:** `docs/H2_Complete_Synthesis_and_Retrospective_Memo.md`
2. **The Interactive Story Exhibit:** `h2/content/core-story.md` (Acts I–X)
3. **The Conceptual Essay:** `notes/the_memory_that_couldnt_remember_essay.md`
4. **The Mechanistic Coda Protocol:** `docs/S16_Monitor_Content_Dissociation_Protocol.md`
5. **The Horizon 3 Specification:** `9. The Continuity Garden.md`
