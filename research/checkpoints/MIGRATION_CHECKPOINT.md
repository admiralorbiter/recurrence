# Post-Hoc Migration Checkpoint

> **This is a post-hoc workflow migration checkpoint, not a preregistration or claim that these constraints were frozen before the preceding work.**

---

## 1. Metadata & Commit Lineage

- **Repository / Project Name**: `recurrence` (Continuity Garden / Structured Continuity Research Program)
- **Checkpoint Date**: `2026-08-21`
- **Pre-Migration Research Base**: `52c2fc48396811a581886074591378515e2eab9e`
  * *Final research experiment execution and 3-lesion battery under the prior ChatGPT review loop.*
- **Migration & Stabilization Commit**: `5d699569b329ad4cff13137976e1074e5ad08520`
  * *Established repository control layer (`AGENTS.md`, contract templates, migration checkpoint), fixed mechanical Rust compilation in `run_gate_e_scouts.rs`, harmonized finite-difference checks in `report_q16a32.md` / `run_q16a32_delayed_role_binding.rs`, and tightened Q16b.1 claim wording from "Epistemic Laundering Solved" to the defensible "Epistemic Laundering Discrimination" in `report_q16b1.md`.*
- **Current Research Stage / Gate**: Gate E — Directional Provenance, Epistemic Laundering, and Causal History
- **Current Active Question**: Q16b.2 (Completed & Stabilized)

---

## 2. What Has Just Been Completed

- **Q16a Closure (Q16a.1, Q16a.2, Q16a.3, Q16a.3.1, Q16a.3.2)**:
  - Validated that prospective/phase-indexed recurrent states preserve learnable relational role structure across matched delay intervals ($\Delta \in \{0, 1, 2, 4\}$), with delay-matched retrospective final-state baselines locked at chance nulls.
  - Verified analytic REINFORCE policy gradients against central finite differences.
- **Q16b Causal Reachability & Multi-Hop Disagreement (Q16b)**:
  - Validated counterfactual intervention engine estimating pairwise causal transmission.
- **Q16b.1 Transitive Ancestry Composition & Unmasked Laundering Corroboration**:
  - Implemented strictly unmasked Bayesian generative sampling for all challenge worlds (evaluating naturally occurring agreement/disagreement without synthetic overwriting).
  - Masked direct $A \to C$ shocks during development, deriving transitive reachability $\hat{A}_{AC}$ via generic path composition ($A \to B \circ B \to C$).
  - Differentiated inherited agreement ($A = C$, $P=92.0\%$, choosing VERIFY at threshold $+1.60$) from independent corroboration ($A = D$, $P=99.25\%$, choosing COMMIT).
- **Q16b.2 Zero-Shot Generalization & 3-Lesion Causal Battery**:
  - Strictly withheld $(A, C)$ from both developmental causal shocks and entity query encoder training.
  - Demonstrated zero-shot conflict resolution ($75.0\%$ Root $A$ choice, $+1.33$ return) and laundering detection ($68.8\%$ VERIFY, $+1.53$ return) on the unseen pair.
  - Executed a 3-lesion double dissociation (Local-Only $\hat{E}$, Upstream Path-Break $\hat{E}_{AB}=0$, Downstream Path-Break $\hat{E}_{BC}=0$, and Transposition $\hat{A}^T$) proving that transitive graph composition is the causal mechanism correcting laundering overconfidence.

---

## 3. Strongest Supported Findings

1. **Phase-Indexed Event Decodability**:
   - Access to recurrent representations at encoding phases ($h_{s1}, h_{s2}$) maintains near-perfect entity decodability and linear role separability, whereas final blended recurrent states ($h_{\text{final}}$) collapse to chance role-assignment accuracy.
2. **Autonomous Local Causal Estimation**:
   - Pairwise counterfactual shocks reliably estimate local directional influence ($\hat{E}_{AB} \approx 69.2\%$, $\hat{E}_{BC} \approx 61.1\%$, $\hat{E}_{AD} \approx 0.0\%$) across $16/16$ seeds.
3. **Algebraic Transitive Path Composition**:
   - Local causal edges compose algebraically ($\hat{A}_{\text{comp}} = \hat{E} + \hat{E}^2$) into a non-zero transitive reachability score ($\hat{A}_{AC} \approx +0.423$) without direct $A \to C$ observation.
4. **Behavioral Laundering Discrimination Under Bayes Policy**:
   - A fixed Bayes decision policy utilizing composed ancestry scores reliably separates inherited redundancy ($A=C \implies$ VERIFY) from independent corroboration ($A=D \implies$ COMMIT) under a high verification payoff ($V=+1.60$).
5. **Zero-Shot Transfer to Withheld Endpoint Pair**:
   - Provenance query mechanisms trained exclusively on 1-hop local and independent pairs successfully transfer composed ancestry to an unpracticed 2-hop endpoint pair $(A, C)$.

---

## 4. Current Claim Limitations & Ceilings Documented by Project

1. **Supplied Composition Grammar (Engineered Transitivity)**:
   - The transitive composition operator ($\hat{A} = \hat{E} + \hat{E}^2$) is supplied algorithmically by the experiment scaffolding rather than discovered endogenously by the neural architecture.
2. **Fixed Bayes Decision Mapping**:
   - The decision policy mapping ancestry scores into actions is an engineered Bayesian decision rule with calibrated thresholds, not an endogenously learned confidence distribution.
3. **Composed Score vs Calibrated Flip Probability**:
   - The product $\hat{E}_{AB} \cdot \hat{E}_{BC} \approx 0.423$ represents a path-reachability score rather than a calibrated conditional counterfactual probability ($P(C \text{ flips} \mid \text{do}(A)) \approx 0.518$).
4. **Scaffolded Sidecar vs Endogenous Recurrent Memory**:
   - Causal graphs reside in an external sidecar matrix rather than directly within endogenous persistent weights or recurrent dynamics.

---

## 5. Important Invariants & Frozen Decisions

- **Seed Standard**: 16 paired random seeds ($101 \dots 116$) evaluated on matched trajectories.
- **Lesion Protocol**: Paired seed-level differences ($\text{Metric}_{\text{intact}} - \text{Metric}_{\text{lesion}}$) with reported standard error of the mean (STE) and explicit seed promotion counts ($k/16$).
- **Generative Sampling**: All challenge episodes generated through genuine Bayesian DAGs with rejection filtering, prohibiting synthetic report forcing/overwriting.
- **Payoff Regimes**: Standard conflict regime ($\text{VERIFY} = +1.00$) and high-threshold corroboration regime ($\text{VERIFY} = +1.60$).

---

## 6. Unresolved Anomalies & Review Findings

- **Seed-Level Attractor Misrouting**: In Q16b.1 independent conditions, $15/16$ seeds succeeded while 1 seed (seed 101) fell into a misrouted query attractor. Seed-level robustness requires explicit monitoring.
- **Zero-Shot Variance**: In Q16b.2 zero-shot transfer, performance on the withheld pair achieves $75.0\%$ ($12/16$ seeds) in conflict and $68.8\%$ ($11/16$ seeds) in laundering agreement, reflecting non-trivial variance across seeds when training is restricted strictly to local pairs.

---

## 7. Known Open Questions

1. Can an organism autonomously learn the generic transitivity composition operator ($\circ$) across variable-depth chains ($A \to B \to C \to D$) without explicit matrix multiplication scaffolding?
2. Can the causal history representation be moved from an external sidecar into endogenous recurrent synaptic plasticity or recurrent memory?
3. Can the mapping from causal ancestry to action confidence be learned autonomously from reward signals without engineered thresholds?

---

## 8. Candidate Next Step

`NOT YET FROZEN` (To be defined under the new research contract workflow).
