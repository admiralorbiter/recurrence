---
contract_id: CONTRACT-<GATE>-<QUESTION>
promotion_id: PROMOTION-<GATE>-<QUESTION>
status: CANDIDATE # CANDIDATE | PROMOTED | REJECTED | REVISED_CONTRACT_REQUIRED | ESCALATED
base_sha: <base-sha>
candidate_sha: <candidate-sha>
audit_date: "YYYY-MM-DD"
author: <author-or-agent-id>
---

# Research Promotion & Diagnostic Audit: [Promotion Title]

> **NOTE**: A promotion audit is initially generated with status `CANDIDATE`. Promotion into repository baselines is NOT considered final until human epistemic review occurs.

---

## 1. Executive Summary & Verdict

- **Question Tested**:
  * [Brief summary of the primary scientific question]
- **Implementation Summary**:
  * [Concise overview of architectural, algorithmic, or experimental changes implemented]
- **Overall Verdict**: `CANDIDATE` | `PROMOTED` | `REJECTED` | `REVISED_CONTRACT_REQUIRED` | `ESCALATED`

---

## 2. Workflow & Process Efficiency Metrics

*Evaluating progress per unit of human attention ("fast exploration, slow promotion").*

| Process Metric | Observed Value | Declared Contract Budget | Assessment |
| :--- | :--- | :--- | :--- |
| **Autonomous Review / Fix Rounds** | [Count] | [Declared budget or N/A] | [Autonomous loop iterations] |
| **Human Interventions Required** | [Count] | 0 (except formal escalation) | [Context switch count] |
| **Total Wall-Clock Time** | [e.g. 2m 15s] | [Declared time budget or N/A] | [Execution throughput] |
| **Compute Footprint** | [e.g. 16 CPU threads, 0 GPU] | [Declared resource_class] | [Resource class compliance] |
| **Attention Context Switches** | [e.g. 1 (final review)] | 1 | [Preserved human attention] |

---

## 3. Experiments & Executions

- **Executable Harnesses Run**:
  * [List exact binary targets and scripts executed]
- **Population & Seeds Evaluated**:
  * [e.g., 16 seeds: 101..116]
- **Compute & Wall-Clock Resources**:
  * [Runtime, thread count, memory profile]

---

## 4. Quantitative Results & Estimands

| Condition / Scenario | Primary Metric (Intact) | Lesion / Control Metric | Paired Effect ($\Delta \pm \text{STE}$) | Seed Promotion ($k/N$) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Condition 1** | | | | | |
| **Condition 2** | | | | | |
| **Condition 3** | | | | | |

---

## 5. Acceptance & Falsification Criteria Audit

- **Criterion 1**: [Criterion description]
  * **Observed**: [Observed value]
  * **Verdict**: `MET` | `UNMET` | `INCONCLUSIVE`
- **Criterion 2**: [Criterion description]
  * **Observed**: [Observed value]
  * **Verdict**: `MET` | `UNMET` | `INCONCLUSIVE`

---

## 6. Verification & Test Suite Status

- **Rust Workspace Unit & Integration Tests**: `PASSED` (`cargo test --manifest-path crates/continuity_garden_core/Cargo.toml`)
- **Python Invariant & Repository Tests**: `PASSED` (`python -m pytest tests/test_repo_state.py`)
- **Deterministic Parity / Finite-Difference Checks**: `VERIFIED`

---

## 7. Hardening & Corrections Made

- **Implementation Fixes**:
  * [List any code, type, or runtime fixes made during hardening]
- **Measurement & Calibration Corrections**:
  * [List any synchronization, assertion, or formatting corrections]

---

## 8. Epistemic Assessment & Human Escalation

- **Potential Epistemic Escalation**: `YES` | `NO`
  * If `YES`, state reason: `ESCALATE — HUMAN EPISTEMIC DECISION REQUIRED: <details>`
- **Unexpected Findings or Anomalies**:
  * [Document any unexpected empirical observations or seed-level variance]
- **Claim & Interpretation Changes**:
  * [Document any refinement in claim ceilings or boundary conditions]
- **Updated Beliefs / Scientific Stance**:
  * [How this outcome updates our mechanistic understanding]
- **Unresolved Issues / Known Gaps**:
  * [Explicitly list what remains untested or scaffolded]

---

## 9. Roadmap & Candidate Next Steps

- **Downstream Capabilities Unlocked**:
  * [Capabilities now established as frozen baselines]
- **Candidate Next Questions**:
  * [Potential future contracts suggested by evidence; state `NOT YET FROZEN` until contracted]
