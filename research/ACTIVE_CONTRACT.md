# Active Research Contract Pointer

This file provides a standardized, machine-discoverable pointer to the currently active research contract.

```yaml
active_contract_id: null
status: IDLE
contract_path: null
base_sha: "5d699569b329ad4cff13137976e1074e5ad08520"
last_promoted_contract: null
last_checkpoint: "research/checkpoints/MIGRATION_CHECKPOINT.md"
```

---

## Instructions for Automated Tools & Agents

1. **Active Contract Location**: When an experiment is authorized and active, this file MUST point to the canonical contract path in `research/contracts/CONTRACT-<GATE>-<QUESTION>.md`.
2. **State Machine**:
   - `IDLE`: No contract is currently active. Agents MUST NOT begin experiments until a contract is frozen and registered here.
   - `FROZEN`: A contract is actively being executed against its frozen specifications.
   - `AUDITING`: Experimental results are being evaluated against acceptance criteria in a candidate promotion audit.
3. **Preregistration Invariant**: The contract file referenced in `contract_path` must be committed with `status: FROZEN` at `base_sha` before execution begins.
