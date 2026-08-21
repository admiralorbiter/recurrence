---
contract_id: CONTRACT-E-Q17A-R1
status: DRAFT
base_sha: 6849c64ab5881f10f9c30b7b26ce0f653d0567f4
resource_class: cpu
long_running: false
exclusive_gpu: false
interruptible: true
author: human
creation_date: "2026-08-21"
target_completion_date: "2026-08-22"
---

# Research Contract (Draft Revision R1): CONTRACT-E-Q17A-R1 — Learned 2-Hop Transitive Composition

> **DRAFT PROPOSAL — HUMAN EPISTEMIC REVIEW REQUIRED BEFORE FREEZING**:
> This draft revision supersedes the unexecuted proposal `CONTRACT-E-Q17A`. It hardens the definition of the independent control, establishes strict architectural boundaries on endogenous composition, narrows the scope strictly to 2-hop composition, refines lesion and transposition criteria per scenario, and prevents claim ceiling leakage into recurrent memory migration.

---

## 1. Metadata & Lifecycle Status

- **Contract ID**: `CONTRACT-E-Q17A-R1`
- **Contract Status**: `DRAFT` (Requires human review and explicit freeze authorization)
- **Base Git Commit (`base_sha`)**: `6849c64ab5881f10f9c30b7b26ce0f653d0567f4`
- **Supersedes**: `CONTRACT-E-Q17A` (Unexecuted, escalated due to specification ambiguities)
- **Target Experiment / Phase**: Gate E Frontier (Q17A — Learned 2-Hop Composition)
- **Protocol Version**: `v0.1`

---

## 2. Research Question & Scientific Rationale

- **Primary Research Question**:
  Can a parameterized neural composition module learn to bind multi-hop transitive causal ancestry ($\hat{A}_{AC} = f_\theta(\hat{E}_{AB}, \hat{E}_{BC})$) from local step transitions and generalize zero-shot to an unseen endpoint pair $(A, C)$, replacing the fixed algebraic composition operator ($\hat{A} = \hat{E} + \hat{E}^2$) without relying on algorithmic path multiplication sidecars?
- **Scope Restriction**:
  This contract is strictly scoped to **2-hop transitive composition** ($A \to B \to C$). Variable-depth chains ($\ge 3$ hops) and open-ended graph search are explicitly deferred to subsequent research gates.
- **Hypotheses**:
  - **$H_1$ (Primary Hypothesis)**: An endogenous neural composition kernel $f_\theta$ optimized over local training pairs successfully generalizes transitive reachability to the withheld 2-hop pair $(A, C)$, enabling zero-shot conflict resolution and laundering discrimination.
  - **$H_0$ (Null Hypothesis)**: Without hardcoded matrix algebra sidecars, the learned representation fails to bind transitive reachability, collapsing to local association or chance on $(A, C)$.
  - **$H_{\text{alt}}$ (Alternative Mechanism)**: The network relies on non-directional entity similarity, query co-occurrence, or frequency heuristics rather than true directional reachability.

---

## 3. Frozen Experimental Design & Generative Model

- **Environment & DAG Topology**:
  - Entities: $\{A, B, C, D\}$.
  - Causal Chain: Directed transmission $A \to B \to C$.
  - Independent Comparator: Entity $D$ is causally independent of the chain ($E_{AD} \approx 0, E_{BD} \approx 0, E_{CD} \approx 0$). $D$ serves as the independent corroboration baseline ($A = D$) and independent conflict baseline ($A \neq D$).
  - **Clarification**: $(A, D)$ is NOT a causal relation; it is an orthogonal, unlinked control node used to evaluate false-positive causal binding.
- **Population & Seed Standard**:
  - 16 fixed paired random seeds ($101 \dots 116$) evaluated across matched Bayesian challenge episodes.
- **Withholding Standard**:
  - The endpoint pair $(A, C)$ is strictly withheld from both developmental causal shocks and query encoder supervision during training.

---

## 4. Endogenous Composition Architectural Boundaries

To prevent "disguised algorithmic shortcuts," the implementation must strictly respect the following boundaries:

### Permitted Architecture & Information Flow:
- The composition module $f_\theta$ must be a parameterized neural component (e.g., MLP, bilinear layer, or attention head) whose weights $\theta$ are learned.
- Inputs to $f_\theta$ are continuous representations of local pairwise estimates ($\mathbf{e}_{AB}, \mathbf{e}_{BC}$).
- Output of $f_\theta$ is a continuous reachability score or representation $\hat{\mathbf{a}}_{AC}$.
- Supervision may only use local 1-hop and independent training pairs (e.g., $A/B, B/C, A/D, B/D, C/D$).

### Prohibited Shortcuts (MUST FAIL/ESCALATE IF DETECTED):
- **No Hardcoded Matrix Multiplication**: Hardcoded matrix operations like $\mathbf{E} \times \mathbf{E}$ or $\sum \mathbf{E}^k$ outside learned weights are prohibited.
- **No Path Enumeration / Graph Algorithms**: Algorithmic graph traversal (e.g., BFS, DFS, Floyd-Warshall) is prohibited.
- **No Direct Endpoint Supervision**: The pair $(A, C)$ must not receive direct ground-truth supervision during training.
- **No Symbolic Lookups**: Hardcoded lookup tables mapping $(A, C) \mapsto \text{reachability}$ are prohibited.

---

## 5. Estimands, Primary Metrics & Statistical Protocol

- **Primary Estimands**:
  1. **Zero-Shot Conflict Choice Accuracy ($M_1$)**: Selection of Root Originator $A$ on unseen conflict $(A \neq C)$.
  2. **Zero-Shot Laundering Discrimination ($M_2$)**: Frequency of choosing `VERIFY` on laundered agreement ($A = C$) under high threshold ($V = +1.60$).
  3. **Independent Corroboration Accuracy ($M_3$)**: Frequency of choosing `COMMIT` on independent agreement ($A = D$).
  4. **Independent Conflict Baseline ($M_4$)**: Accuracy on independent conflict ($A \neq D$, $V = +1.00$).

- **Statistical Inferential Protocol**:
  - Seed-level paired evaluation across $N=16$ seeds.
  - Paired comparisons: $\Delta_{\text{local}} = \text{Metric}_{\text{intact}} - \text{Metric}_{\text{local\_lesion}}$.
  - **[HUMAN REVIEW REQUIRED]**: Exact inferential test specification (e.g., Paired two-tailed Student's $t$-test or Wilcoxon signed-rank test with $\alpha = 0.01$).

---

## 6. Acceptance & Falsification Criteria

- **Success / Promotion Criteria**:
  1. **Criterion 1 (Zero-Shot Conflict)**: Intact multi-hop conflict choice accuracy $\ge 70.0\%$ with seed promotion $\ge 12/16$.
  2. **Criterion 2 (Zero-Shot Laundering)**: Intact laundering discrimination $\ge 60.0\%$ with seed promotion $\ge 11/16$.
  3. **Criterion 3 (Independent Baselines)**: Independent corroboration ($A = D$) maintains $\ge 95.0\%$ ($k \ge 15/16$) and independent conflict ($A \neq D$) maintains $\ge 95.0\%$ ($k \ge 15/16$).
  4. **Criterion 4 (Double Dissociation on Laundering)**: Local-Only lesion ($\hat{E}_{AC} = 0$) causes a significant drop in laundering correction compared to intact (target $\ge +20.0\%$ drop, $[p < 0.01 \text{ under specified paired test}]$).
  5. **Criterion 5 (Path-Break Sensitivity)**: Severing either upstream ($E_{AB}=0$) or downstream ($E_{BC}=0$) transmission selectively collapses multi-hop reachability $\hat{A}_{AC}$ toward zero.

- **Scenario-Specific Transposition Falsification Criteria**:
  - **Multi-Hop Conflict ($A \neq C$) under Transposition ($\hat{A}^T$)**: Accuracy must collapse to chance or below (target $\le 10.0\%$, return $< 0.00$), confirming that conflict resolution strictly requires correct causal directionality $A \to C$.
  - **Laundering Agreement ($A = C$) under Transposition**: Because transposition reflects a non-zero symmetric reachability matrix ($C \to A$), laundering verification is expected to remain above zero (consistent with Q16b.2 findings of $\approx 68.8\%$). Transposition is therefore evaluated as a directional discriminator for conflict, not an indiscriminate nullifier for agreement.

---

## 7. Claim Ceiling & Epistemic Boundaries

- **Maximum Authorized Claim**:
  Endogenous parameterized neural computation of 2-hop transitive causal reachability from local pairwise estimates, demonstrating zero-shot transfer to an unpracticed endpoint pair without algorithmic matrix multiplication scaffolding.
- **Explicit Exclusions / Prohibited Overclaims**:
  1. **Sidecar vs Recurrent Memory Exclusion**: This experiment does NOT claim that causal history representation has been migrated from external representations into endogenous recurrent synaptic/activation memory. (That is an independent open question for subsequent milestones).
  2. **Arbitrary Graph Traversal Exclusion**: Does NOT claim general $N$-hop graph traversal or symbolic reasoning beyond 2-hop transitive composition.
  3. **Uncalibrated Probability Exclusion**: Composed reachability scores represent relative directional ancestry, not calibrated Bayesian posterior probabilities.

---

## 8. Required Artifacts & Execution Harness

- **Harness Binary**: `crates/continuity_garden_core/src/bin/run_q17a_endogenous_transitivity.rs`
- **Results Summary JSON**: `results/e27_q17_endogenous_transitivity/q17a_summary.json`
- **Formal Markdown Report**: `results/e27_q17_endogenous_transitivity/report_q17a.md`
- **Promotion Document**: `research/promotions/PROMOTION-E-Q17A.md`

---

## 9. Compute & Resource Budget

- **Resource Class**: `cpu`
- **Estimated Runtime**: $< 30$ seconds on 16 threads (CPU)
- **Long-Running Process**: `false`
- **Exclusive GPU**: `false`
- **Interruptible**: `true`

---

## 10. Items Marked for Human Review Before Freezing

- [ ] Confirm exact inferential statistical test (e.g. paired t-test vs Wilcoxon) in Section 5.
- [ ] Confirm the exact neural module parameterization constraints in Section 4.
- [ ] Confirm threshold numbers for Criterion 1 ($70\%$) and Criterion 2 ($60\%$) against prior Q16b.2 empirical distributions.
- [ ] Confirm transition from `DRAFT` to `FROZEN`.
