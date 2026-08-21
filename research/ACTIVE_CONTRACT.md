---
state: ESCALATED
active_contract_id: CONTRACT-E-Q17A-R1
contract_path: research/contracts/CONTRACT-E-Q17A-R1.md
execution_base_sha: d50f6c36a097ad6305c2ee07f4d1a1119473cfc6
last_checkpoint: research/checkpoints/MIGRATION_CHECKPOINT.md
last_promotion: null
---

# Active Research Contract Pointer

**Current Operational State**: `READY`

This document provides the machine-discoverable operational entry point for autonomous research agents and Mother Base.

## Active Contract Details
- **Active Contract ID**: `CONTRACT-E-Q17A-R1`
- **Contract Path**: [`research/contracts/CONTRACT-E-Q17A-R1.md`](contracts/CONTRACT-E-Q17A-R1.md)
- **Execution Base Commit (`execution_base_sha`)**: `d50f6c36a097ad6305c2ee07f4d1a1119473cfc6`
- **Last Checkpoint**: [`research/checkpoints/MIGRATION_CHECKPOINT.md`](checkpoints/MIGRATION_CHECKPOINT.md)
- **Governance**: Design Review `APPROVED`, Authorized by `human`.

## Protocol for Agents
1. When starting work, parse the YAML frontmatter above.
2. If `state: ESCALATED` or `state: ESCALATED`, inspect the target contract at `contract_path` and optimize implementation strictly against its frozen constraints.
3. Prohibited from modifying hypotheses, estimands, metrics, or claim ceilings after freezing.
