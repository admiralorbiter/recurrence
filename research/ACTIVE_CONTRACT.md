---
state: ESCALATED
active_contract_id: CONTRACT-E-Q17A
contract_path: research/contracts/CONTRACT-E-Q17A.md
execution_base_sha: add4234f2f00f798e7d4470db2f95d084c4bd936
last_checkpoint: research/checkpoints/MIGRATION_CHECKPOINT.md
last_promotion: null
---

# Active Research Contract Pointer

**Current Operational State**: `ESCALATED`

## Escalation Reason: Epistemic Review Required (Human Decision Gate)

Autonomous experimental execution of `CONTRACT-E-Q17A` is **HALTED**.

A deep epistemic review of `CONTRACT-E-Q17A` identified foundational specification defects prior to execution:
1. **Control Definition Contradiction**: The contract incorrectly referenced $(A \to D)$ as an independent control rather than preserving $D$ as the causally independent comparator ($E_{AD} \approx 0$).
2. **Underspecified Endogenous Composition Boundary**: Insufficient architectural constraints to prevent disguised hardcoded matrix multiplication/path enumeration shortcuts.
3. **Incomplete Falsification & Statistical Criteria**: Unspecified statistical test for path-breaks and inappropriate blanket transposition nulls across asymmetric scenarios.
4. **Scope & Claim Boundary Overreach**: Broadened scope without explicit exclusion of causal-history recurrent memory migration.

A revised proposal has been drafted for human review at [`research/contracts/CONTRACT-E-Q17A-R1.md`](contracts/CONTRACT-E-Q17A-R1.md) (`status: DRAFT`).
