---
contract_id: CONTRACT-E-Q17A-R1
status: DRAFT
base_sha: 6849c64ab5881f10f9c30b7b26ce0f653d0567f4
resource_class: cpu
long_running: false
exclusive_gpu: false
interruptible: true
proposed_by: antigravity
design_review: APPROVED
reviewed_by: chatgpt-pro
authorized_by: null
creation_date: "2026-08-21"
target_completion_date: "2026-08-22"
---

# Research Contract (Draft Revision R1): CONTRACT-E-Q17A-R1 — Learned 2-Hop Transitive Composition

> **DRAFT PROPOSAL — DESIGN REVIEW APPROVED — PENDING STRATEGIC AUTHORIZATION**:
> This draft revision has completed scientific design review (`chatgpt-pro`) and is approved for strategic human authorization. It establishes the formal training curriculum, evaluation sealing, discrete seed-count gates, McNemar and exact permutation statistical protocols, corrected composition ablation mechanics, and double-dissociation specificity requirements.

---

## 1. Metadata & Lifecycle Status

- **Contract ID**: `CONTRACT-E-Q17A-R1`
- **Contract Status**: `DRAFT` (Pending Human Strategic Authorization to Freeze)
- **Base Git Commit (`base_sha`)**: `6849c64ab5881f10f9c30b7b26ce0f653d0567f4`
- **Supersedes**: `CONTRACT-E-Q17A` (Unexecuted, escalated due to specification defects)
- **Target Experiment / Phase**: Gate E Frontier (Q17A — Learned 2-Hop Composition)
- **Protocol Version**: `v0.1`
- **Governance**:
  - **Proposed By**: `antigravity`
  - **Design Review**: `APPROVED` (Reviewed by `chatgpt-pro`)
  - **Authorized By**: `null` (Pending Human Director Strategic Authorization)

---

## 2. Research Question & Scientific Rationale

- **Primary Research Question**:
  Can a parameterized neural composition kernel $f_\theta$ learn a generic two-hop composition operator from development training worlds ($X \to Y \to Z \implies X \to Z$) and generalize zero-shot to infer reachability on a completely withheld test endpoint pair ($A \to B \to C \implies A \to C$), matching the behavioral floor of the engineered matrix scaffolding ($\hat{A} = \hat{E} + \hat{E}^2$) without hardcoded matrix multiplication?

- **The Moonshot Research Ladder**:
  - **Q16 (Validated Baseline)**: Explicit algebraic matrix composition ($\hat{A} = \hat{E} + \hat{E}^2$) solves multi-hop conflict resolution and laundering discrimination.
  - **Q17A (This Milestone)**: Can a parameterized neural module learn the generic 2-hop composition operator from auxiliary training relations and transfer it zero-shot to an unpracticed endpoint pair?
  - **Q17B (Future Milestone)**: Can the architecture induce composition without explicit 2-hop training targets (endogenous compositional discovery)?
  - **Q17C (Future Milestone)**: Can causal history representation migrate from external representations into endogenous recurrent synaptic/activation memory?

- **Hypotheses**:
  - **$H_1$ (Primary Hypothesis)**: An endogenous composition kernel $f_\theta$ trained on auxiliary 2-hop relations learns a generic directional reachability function that transfers zero-shot to $(A, C)$, matching the Q16b.2 behavioral floor on conflict resolution and laundering discrimination.
  - **$H_0$ (Null Hypothesis)**: Without hardcoded matrix algebra sidecars, learned composition fails to generalize to the withheld pair $(A, C)$, collapsing to local-only association or chance.
  - **$H_{\text{alt}}$ (Alternative Mechanism)**: The network relies on entity similarity or non-directional co-occurrence heuristics rather than true transitive causal reachability.

---

## 3. Training Curriculum & Experimental Design

- **Auxiliary Training Worlds (Composition Supervision)**:
  - Development environments generate synthetic pairwise causal relations over disjoint entity vocabularies (e.g., $X \to Y \to Z$, $P \to Q \to R$).
  - $f_\theta$ is supervised on generic step chaining: inputs $(\mathbf{e}_{1}, \mathbf{e}_{2})$ mapped to multi-hop reachability target $\mathbf{a}_{13}$.
  - **Strict Entity Withholding**: Entities $A, B, C, D$ and the test chain $A \to B \to C$ are strictly excluded from all training worlds and query encoder training.

- **Evaluation Sealing & Model Selection Leakage Protection**:
  - Architecture decisions, optimizer configurations, hyperparameters, learning rate schedules, stopping rules, and model selection criteria may use **only auxiliary training/development worlds**.
  - The final $(A, B, C, D)$ test environment remains sealed until all implementation choices are frozen.
  - Final-test performance may never be used to tune, select, or iterate on the model implementation.

- **Test Evaluation Environment (Zero-Shot Transfer)**:
  - Topology: Directed transmission chain $A \to B \to C$.
  - Independent Comparator: Entity $D$ is causally independent of the chain ($E_{AD} \approx 0, E_{BD} \approx 0, E_{CD} \approx 0$). $D$ provides independent corroboration ($A = D$) and independent conflict ($A \neq D$) baselines.
  - Evaluation Population: 16 fixed paired random seeds ($101 \dots 116$) evaluated across matched Bayesian challenge episodes.

---

## 4. Architectural Boundaries & Prohibited Shortcuts

- **Permitted Architecture**:
  - $f_\theta$ is a parameterized neural layer (MLP, bilinear interaction, or attention kernel) parameterized by learnable weights $\theta$.
  - Inputs are continuous local pairwise representations ($\mathbf{e}_{AB}, \mathbf{e}_{BC}$).
  - Output is a continuous reachability representation or scalar $\hat{a}_{AC} \in [0, 1]$.
- **Prohibited Shortcuts (Falsifies Experiment if Present)**:
  - No hardcoded matrix multiplication ($\mathbf{E} \times \mathbf{E}$ or $\sum \mathbf{E}^k$).
  - No algorithmic path search or traversal (BFS, DFS, Floyd-Warshall).
  - No direct $(A, C)$ data or ground truth in the training curriculum.
  - No symbolic or hardcoded entity-identity lookup tables.

---

## 5. Estimands & Statistical Protocol

### Primary Behavioral Estimands (Binary Decisions per Seed):
1. **$M_1$ Zero-Shot Multi-Hop Conflict Accuracy**: Root originator choice on $(A \neq C, V = +1.00)$.
2. **$M_2$ Zero-Shot Laundering Discrimination**: Frequency of `VERIFY` on laundered agreement $(A = C, V = +1.60)$.
3. **$M_3$ Independent Corroboration**: Frequency of `COMMIT` on independent agreement $(A = D, V = +1.60)$.
4. **$M_4$ Independent Conflict**: Frequency of `VERIFY` on independent conflict $(A \neq D, V = +1.00)$.

### Statistical Analysis Protocol:
- **Behavioral Evaluation**:
  - Report exact seed promotion counts ($k/16$).
  - Report paired difference ($\text{Acc}_{\text{intact}} - \text{Acc}_{\text{ablation}}$).
  - Report complete $2 \times 2$ discordant-pair contingency tables ($n_{10}, n_{01}$).
  - Compute exact binomial McNemar test $p$-values as supporting descriptive evidence (not used as a fragile binary gate at $N=16$).
- **Continuous Mechanistic Evaluation**:
  - Evaluate the continuous composed reachability score $\hat{a}_{AC}$ between intact and path-break conditions.
  - Compute an **exact one-sided paired sign-flip/permutation test** ($2^{16} = 65,536$ exact permutations) on the mean difference $\Delta a = a_{\text{intact}} - a_{\text{pathbreak}}$, testing $\Delta a > 0$ ($p_{\text{exact}} < 0.01$).

---

## 6. Acceptance & Falsification Criteria

- **Success Gates (Matching Q16b.2 Empirical Floors)**:
  1. **Gate 1 (Zero-Shot Conflict Resolution)**: Intact multi-hop conflict choice accuracy $\ge 12/16$ seeds ($75.0\%$).
  2. **Gate 2 (Zero-Shot Laundering Discrimination)**: Intact laundering `VERIFY` rate $\ge 11/16$ seeds ($68.75\%$).
  3. **Gate 3 (Independent Corroboration)**: Independent agreement ($A = D$) maintains $\ge 15/16$ seeds ($93.75\%$).
  4. **Gate 4 (Independent Conflict)**: Independent conflict ($A \neq D$) maintains $\ge 15/16$ seeds ($93.75\%$).
  5. **Gate 5 (Composition Ablation Behavioral Floor)**: Composition Ablation ($\hat{a}_{AC} := 0$) exhibits a discordant behavioral effect floor:
     $$n_{10} - n_{01} \ge 3$$
     where $n_{10}$ is intact-correct / ablation-incorrect, and $n_{01}$ is intact-incorrect / ablation-correct. Report exact binomial McNemar as supporting evidence.
  6. **Gate 6 (Mechanistic Path-Break Specificity)**:
     - Severing $e_{AB} := 0$ or $e_{BC} := 0$ collapses $\hat{a}_{AC}$ under exact one-sided paired permutation test ($\Delta a > 0, p_{\text{exact}} < 0.01$).
     - **Specificity Invariant**: Independent controls ($A/D$) must remain intact ($\ge 15/16$ seeds) during $e_{AB}$ or $e_{BC}$ path-breaks.

- **Scenario-Specific Transposition Controls**:
  - **Multi-Hop Conflict ($A \neq C$) under Transposition ($\hat{A}^T$)**: Accuracy must collapse to $\le 2/16$ seeds (mean return $< 0.00$), confirming that conflict resolution requires correct $A \to C$ directionality.
  - **Laundering Agreement ($A = C$) under Transposition**: Laundering verification is non-directional in downstream Bayes policy and is expected to remain above zero ($\ge 10/16$ seeds), confirming reverse reachability still triggers necessary epistemic caution.

---

## 7. Claim Ceiling & Epistemic Boundaries

- **Authorized Claim (If all gates pass)**:
  Learned parameterized neural composition of 2-hop transitive causal reachability from auxiliary training worlds, demonstrating zero-shot generalization to unpracticed endpoint pairs and matching the behavioral floor of engineered matrix algebra sidecars.
- **Explicit Scientific Exclusions**:
  1. **Recurrent Memory Migration Exclusion**: Does NOT claim causal history is stored in endogenous recurrent synaptic/activation dynamics.
  2. **Self-Supervised Composition Induction Exclusion**: Does NOT claim the organism discovered the need for composition without auxiliary 2-hop training targets (deferred to Q17B).
  3. **Arbitrary-Depth Traversal Exclusion**: Does NOT claim generalized $N$-hop path traversal ($\ge 3$ hops) or symbolic theorem proving.

---

## 8. Required Artifacts & Execution Harness

- **Harness Binary**: `crates/continuity_garden_core/src/bin/run_q17a_endogenous_transitivity.rs`
- **Serialized JSON**: `results/e27_q17_endogenous_transitivity/q17a_summary.json`
- **Diagnostic Report**: `results/e27_q17_endogenous_transitivity/report_q17a.md`
- **Promotion Document**: `research/promotions/PROMOTION-E-Q17A.md`

---

## 9. Compute Budget

- **Resource Class**: `cpu`
- **Estimated Runtime**: $< 30$ seconds across 16 threads (CPU)
- **Long-Running Process**: `false`
- **Exclusive GPU**: `false`
- **Interruptible**: `true`

---

## 10. Human Strategic Authorization Checklist

- [x] Scientific Design Review: `APPROVED` by `chatgpt-pro`.
- [x] Evaluation sealing and model-selection leakage protection enforced.
- [x] Statistical protocols established: $n_{10} - n_{01} \ge 3$ discordant-pair floor + exact one-sided permutation test.
- [x] Thresholds locked to discrete Q16b.2 empirical floor ($12/16, 11/16, 15/16, 15/16$).
- [x] Double dissociation specificity enforced (path-breaks damage $A \to C$ while preserving $A/D$).
- [ ] **Human Director Strategic Authorization**: Awaiting strategic go/no-go from Human Director to freeze (`status: FROZEN`, `authorized_by: human`).

