# Research Contracts Directory

This directory stores all frozen, active, and historical research contracts for the repository.

## Naming Convention
Contracts MUST follow the pattern:
```text
CONTRACT-<GATE>-<QUESTION_OR_TOPIC>.md
```
*Example:* `CONTRACT-E-Q17a_endogenous_transitivity.md`

## Lifecycle (`status`)
1. **DRAFT**: Authoring the hypothesis, population, metrics, oracle, and acceptance criteria.
2. **FROZEN**: Committed and locked before experimental execution begins. Referenced in [`research/ACTIVE_CONTRACT.md`](../ACTIVE_CONTRACT.md) during execution. A frozen contract remains frozen as permanent historical evidence.
3. **SUPERSEDED**: Explicitly replaced or updated via a formal subsequent contract. Promotion/rejection outcomes are recorded separately in promotion documents.
