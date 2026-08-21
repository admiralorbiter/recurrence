---
active_contract_id: null
state: IDLE # IDLE | READY | RUNNING | AUDITING | ESCALATED
contract_path: null
execution_base_sha: null
last_checkpoint: "research/checkpoints/MIGRATION_CHECKPOINT.md"
last_promotion: null
---

# Active Research Contract Pointer

This document provides the machine-discoverable operational entry point for autonomous research agents and human auditors.

## Operational Lifecycle States (`state`)
- `IDLE`: No contract is currently active. Agents MUST NOT begin experimental runs or modify hypotheses.
- `READY`: A contract has been frozen and committed at `execution_base_sha`. Ready for execution.
- `RUNNING`: Experimental harnesses are actively executing against frozen contract specifications.
- `AUDITING`: Experimental results are being evaluated against acceptance criteria in a candidate promotion audit.
- `ESCALATED`: Execution halted due to invariant failure, anomaly, or hypothesis/metric boundary violation requiring human epistemic decision.

## Protocol for Agents
1. When starting work, parse the YAML frontmatter above.
2. If `state: IDLE`, halt and wait for human instruction or a frozen contract.
3. If `state: READY` or `state: RUNNING`, inspect the target contract at `contract_path` and optimize implementation strictly against its frozen constraints.
