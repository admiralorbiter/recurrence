---
contract_id: CONTRACT-E-Q17A
status: FROZEN
base_sha: 6849c64ab5881f10f9c30b7b26ce0f653d0567f4
resource_class: cpu
long_running: false
exclusive_gpu: false
interruptible: true
author: human
creation_date: "2026-08-21"
target_completion_date: "2026-08-22"
---

# Research Contract: CONTRACT-E-Q17A — Endogenous Transitive Composition

> **CRITICAL REPRODUCIBILITY CONTRACT**:
> This contract defines the preregistered hypotheses, experimental design, estimands, acceptance criteria, and claim ceilings for Question 17A. Execution must adhere strictly to these frozen specifications.

---

## 1. Metadata & Lifecycle Status

- **Contract ID**: `CONTRACT-E-Q17A`
- **Contract Status**: `FROZEN`
- **Base Git Commit (`base_sha`)**: `6849c64ab5881f10f9c30b7b26ce0f653d0567f4`
- **Target Experiment / Phase**: Gate E / Frontier Transition (Q17A — Endogenous Transitivity)
- **Date Frozen**: 2026-08-21
- **Protocol Version**: `v0.1`

---

## 2. Research Question & Scientific Rationale

- **Primary Research Question**:
  Can a neural composition kernel autonomously compose multi-hop transitive causal provenance ($\hat{A}_{AC} = \hat{E}_{AB} \circ \hat{E}_{BC}$) and generalize zero-shot to unpracticed endpoint pairs without relying on external matrix multiplication sidecars?
- **Why This Follows from Existing Evidence**:
  In Q16b.2 (commit `52c2fc4`), zero-shot multi-hop conflict resolution ($+75.0\%$) and laundering discrimination ($+68.8\%$) were validated across a 3-lesion battery. However, as documented in `MIGRATION_CHECKPOINT.md` Section 4, the algebraic composition operator ($\hat{A} = \hat{E} + \hat{E}^2$) was supplied algorithmically by the scaffolding rather than computed endogenously. Q17A tests whether the composition operator itself can be learned and executed endogenously.
- **Hypotheses & Competing Explanations**:
  - **$H_1$ (Primary Hypothesis)**: An endogenous relational composition module trained only on local step compositions reliably estimates multi-hop reachability on unseen transitive pairs, preserving the double dissociation observed in Q16b.2.
  - **$H_0$ (Null Hypothesis)**: Without hardcoded matrix algebra sidecars, endogenous relational states fail to bind multi-hop ancestry, collapsing to local association or chance on 2-hop/3-hop pairs.
  - **$H_{\text{alt}}$ (Alternative Mechanism)**: The agent relies on non-directional entity similarity or frequency bias rather than true directional reachability.

---

## 3. Frozen Experimental Design

- **Environment & Generative Model**:
  Provenance Garden DAG over entities $(A, B, C, D)$ with causal chains ($A \to B \to C$) and independent controls ($A \to D$). Challenge episodes generated via exact Bayesian DAG sampling with unmasked likelihood verification.
- **Experimental Population & Seed Standard**:
  16 fixed paired random seeds ($101 \dots 116$) evaluated on matched challenge trajectories.
- **Experimental Conditions & Interventions**:
  1. **Intact Endogenous Composed**: Full endogenous composition network active.
  2. **Local-Only Lesion**: Restrict reachability input strictly to direct 1-hop local estimates ($E_{AC} = 0$).
  3. **Upstream Path-Break Lesion**: Sever $A \to B$ transmission ($\hat{E}_{AB} = 0$).
  4. **Downstream Path-Break Lesion**: Sever $B \to C$ transmission ($\hat{E}_{BC} = 0$).
  5. **Transposition Control**: Invert reachability matrix ($\hat{A}^T$) to verify directional sensitivity.

---

## 4. Estimands & Primary Metrics

- **Primary Estimands**:
  - **Zero-Shot Conflict Choice Accuracy ($M_1$)**: Root originator selection rate on unseen $(A, C)$ multi-hop conflict.
  - **Zero-Shot Laundering Discrimination ($M_2$)**: Rate of choosing `VERIFY` on laundered agreement ($A = C$) under high threshold ($V = +1.60$).
  - **Independent Corroboration Accuracy ($M_3$)**: Rate of choosing `COMMIT` on genuinely independent agreement ($A = D$).
- **Secondary Diagnostic Metrics**:
  - Realized mean return across matched challenge regimes.
  - Paired Lesion Drop: $\Delta\text{Acc}_{\text{lesion}} = \text{Acc}_{\text{intact}} - \text{Acc}_{\text{local\_only}}$.
- **Statistical Aggregation Protocol**:
  - Paired seed-level differences ($N=16$).
  - Standard error of the mean (STE) across seeds.
  - Seed promotion threshold ($k/16$).

---

## 5. Acceptance & Failure Criteria

- **Success / Promotion Criteria**:
  1. **Criterion 1 (Multi-Hop Conflict)**: Zero-shot conflict resolution on $(A, C) \ge 70.0\%$ with seed promotion $\ge 12/16$.
  2. **Criterion 2 (Laundering Discrimination)**: Zero-shot laundering discrimination $\ge 60.0\%$ with seed promotion $\ge 11/16$.
  3. **Criterion 3 (Independent Baseline)**: Independent corroboration ($A = D$) maintains $\ge 90.0\%$ with seed promotion $\ge 14/16$.
  4. **Criterion 4 (Double Dissociation)**: Significant performance drop under both Upstream and Downstream path-breaks ($p < 0.01$).
- **Falsification / Failure Criteria**:
  1. Failure to outperform local-only lesion on withheld pair $(A, C)$.
  2. Performance under transposition control does not drop below chance nulls.

---

## 6. Claim Ceiling

- **Maximum Authorized Claim**:
  Autonomous endogenous neural computation of 2-hop transitive causal provenance without external algebraic sidecar scaffolding.
- **Explicit Exclusions / Prohibited Overclaims**:
  - Does not claim general arbitrary-depth symbolic theorem proving.
  - Does not claim full human-level metacognitive theory of mind.

---

## 7. Operational Boundaries & Governance

- **Permitted Autonomous Implementation Changes**:
  - Bug fixes, numerical stability improvements, test harness updates, and artifact serialization.
- **Prohibited Autonomous Changes**:
  - Modifying hypotheses, altering estimands/metrics, redefining population/seeds, or softening acceptance criteria.
- **Mandatory Escalation Trigger**:
  Any finding requiring contract modification must halt and trigger:
  `ESCALATE — HUMAN EPISTEMIC DECISION REQUIRED`

---

## 8. Required Evidence & Artifacts

- **Executable Binary / Harness**: `crates/continuity_garden_core/src/bin/run_q17a_endogenous_transitivity.rs`
- **Serialized Results**: `results/e27_q17_endogenous_transitivity/q17a_summary.json`
- **Diagnostic Report**: `results/e27_q17_endogenous_transitivity/report_q17a.md`
- **Promotion Document**: `research/promotions/PROMOTION-E-Q17A.md`

---

## 9. Compute & Resources

- **Resource Class**: `cpu`
- **Estimated Compute Budget**: ~15 seconds across 16 seeds on CPU
- **Long-Running Process Flag**: `false`
- **Exclusive GPU**: `false`
- **Interruptible**: `true`

---

## 10. Completion Definition

The contract is fulfilled when:
1. The experiment executes deterministically across all 16 seeds.
2. All required artifacts (`q17a_summary.json`, `report_q17a.md`) are serialized and tracked in git.
3. Promotion document is populated comparing observed metrics against acceptance criteria.
4. Repository test suite (`cargo test`) passes cleanly.
