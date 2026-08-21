---
state: READY
active_contract_id: CONTRACT-E-Q17A
contract_path: research/contracts/CONTRACT-E-Q17A.md
execution_base_sha: add4234f2f00f798e7d4470db2f95d084c4bd936
last_checkpoint: research/checkpoints/MIGRATION_CHECKPOINT.md
last_promotion: null
---

# Active Research Contract Pointer

**Current Operational State**: `READY`

This document provides the machine-discoverable operational entry point for autonomous research agents and Mother Base.

## Active Contract Details
- **Active Contract ID**: `CONTRACT-E-Q17A`
- **Contract Path**: [`research/contracts/CONTRACT-E-Q17A.md`](contracts/CONTRACT-E-Q17A.md)
- **Execution Base Commit (`execution_base_sha`)**: `add4234f2f00f798e7d4470db2f95d084c4bd936`
- **Last Checkpoint**: [`research/checkpoints/MIGRATION_CHECKPOINT.md`](checkpoints/MIGRATION_CHECKPOINT.md)

## Protocol for Agents
1. When starting work, parse the YAML frontmatter above.
2. If `state: READY` or `state: RUNNING`, inspect the target contract at `contract_path` and optimize implementation strictly against its frozen constraints.
3. Prohibited from modifying hypotheses, estimands, metrics, or claim ceilings after freezing.
