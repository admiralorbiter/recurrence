---
contract_id: CONTRACT-<GATE>-<QUESTION>
status: DRAFT # DRAFT | FROZEN | SUPERSEDED
base_sha: <git-sha-where-contract-is-frozen>
resource_class: cpu # cpu | gpu | hybrid
long_running: false # contract/operator declaration for scheduling
exclusive_gpu: false
interruptible: true
author: <author-or-agent-id>
creation_date: "YYYY-MM-DD"
target_completion_date: "YYYY-MM-DD"
---

# Research Contract: [Contract Title]

> **WARNING**: An experiment must NOT be represented as preregistered or frozen unless this contract file was committed in `FROZEN` status BEFORE experimental execution began. Retrospective editing of frozen criteria is strictly prohibited.

---

## 2. Research Question & Rationale

- **Primary Research Question**:
  * [State the precise scientific question being answered]
- **Why This Follows from Existing Evidence**:
  * [Reference specific prior checkpoint, findings, or frozen capabilities that necessitate this step]
- **Hypotheses & Competing Explanations**:
  * **H1 (Primary Hypothesis)**: [Specific mechanistic claim]
  * **H0 (Null Hypothesis)**: [Failure mode or baseline explanation]
  * **H_alt (Alternative Mechanism)**: [Confound or competing architectural explanation]

---

## 3. Frozen Experimental Design

- **Environment & Generative Model**:
  * [Specify environment dynamics, transition probabilities, and observation channels]
- **Experimental Population & Seed Standard**:
  * [e.g., 16 fixed paired random seeds (101..116)]
- **Experimental Conditions & Interventions**:
  * Condition 1: [Control / Baseline]
  * Condition 2: [Proposed Mechanism / Intact Condition]
  * Condition 3: [Ablation / Lesion 1]
  * Condition 4: [Ablation / Lesion 2]
- **Oracle / Reference Definition**:
  * [Exact theoretical benchmark, Bayesian upper bound, or empirical teacher]

---

## 4. Estimands & Primary Metrics

- **Primary Estimand(s)**:
  * [Exact mathematical definition of metric, e.g., Paired $\Delta\text{Accuracy} = \text{Acc}_{\text{intact}} - \text{Acc}_{\text{lesion}}$]
- **Secondary Diagnostic Metrics**:
  * [Realized return, error distributions, representation R2, weight norms]
- **Statistical Aggregation Protocol**:
  * [Paired seed-level differences, mean $\pm$ STE across $N$ seeds, promotion threshold $k/N$]

---

## 5. Acceptance & Failure Criteria

- **Success / Promotion Criteria**:
  1. [Criterion 1: e.g., $\text{Paired } \Delta\text{Acc} \ge +50.0\%$ with $p < 0.01$]
  2. [Criterion 2: e.g., Seed promotion count $\ge 14/16$]
  3. [Criterion 3: e.g., Oracle gap $< 10\%$]
- **Falsification / Failure Criteria**:
  1. [Criterion 1: e.g., Lesion fails to produce statistically significant drop]
  2. [Criterion 2: e.g., Performance does not exceed chance baseline]

---

## 6. Claim Ceiling

- **Maximum Authorized Claim**:
  * [Explicit statement of what this experiment CAN and CANNOT claim if successful]
- **Explicit Exclusions / Prohibited Overclaims**:
  * [Document what remains unproven or engineered, preventing premature milestone declarations]

---

## 7. Operational Boundaries & Governance

- **Permitted Autonomous Implementation Changes**:
  * Bug fixes, numerical stability improvements, test harness updates, artifact serialization.
- **Prohibited Autonomous Changes**:
  * Changing hypotheses, modifying estimands/metrics, redefining population/seeds, altering payoff values, softening acceptance thresholds.
- **Mandatory Escalation Triggers**:
  * Any violation requiring contract modification must halt and trigger:
    `ESCALATE — HUMAN EPISTEMIC DECISION REQUIRED`

---

## 8. Required Evidence & Artifacts

- **Executable Binary / Harness**: `crates/.../src/bin/<runner_name>.rs`
- **Serialized Results**: `results/<exp_dir>/<experiment_id>_summary.json`
- **Diagnostic Report**: `results/<exp_dir>/report_<experiment_id>.md`
- **Promotion Document**: `research/promotions/PROMOTION_<EXPERIMENT_ID>.md`

---

## 9. Compute & Resources

- **Estimated Compute Budget**: [e.g., ~30 seconds on 16 threads]
- **Long-Running Task Flag**: `true` | `false`

---

## 10. Completion Definition

The contract is fulfilled when:
1. The experiment executes deterministically across all specified seeds.
2. All required artifacts and serialized summaries are committed.
3. Promotion document is created comparing observed results against acceptance criteria.
4. Repository tests (`cargo test`, `pytest`) pass cleanly.
