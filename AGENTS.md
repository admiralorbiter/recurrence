# Agent Guidelines & Repository Research Control Layer

This document defines the operational boundaries, verification protocols, and escalation rules for autonomous agents working in this repository.

---

## 1. Core Operating Principle

> **Agents optimize implementation against the frozen research contract. They do not optimize the research contract until the implementation passes.**

If satisfying an experimental contract or passing an invariant test requires changing the contract, hypothesis, or estimand, agents MUST stop and escalate immediately.

---

## 2. Autonomous Action Boundaries

### Agents MAY Autonomously Repair:
- Implementation bugs and code errors.
- Test failures, missing edge cases, and flaky assertions that do not alter the underlying scientific estimand.
- Artifact tracking, persistence, and filesystem serialization bugs.
- Documentation, report, and result synchronization discrepancies.
- Reproducibility failures (e.g., seeding, deterministic RNG initialization, numerical precision).
- Invariant and property-test failures enforcing already-frozen requirements.
- Malformed schemas, manifests, identifiers, and Cargo/Python configuration errors.
- Violations of already-frozen scientific and architectural requirements.

### Agents MUST ESCALATE (Human Epistemic Decision Required):
- Changes to the underlying research question or hypothesis.
- Modifications to the estimand, primary metric, or payoff structure.
- Redefinition of benchmark populations, control tasks, or reference conditions.
- Altering oracle definitions or loosening checks to artificially pass tests.
- Reinterpreting causal mechanisms or rewriting scientific conclusions.
- Raising or shifting the claim ceiling beyond documented repository evidence.
- Retracting, replacing, or modifying prior frozen conclusions.
- Modifying the research roadmap or initiating new gates, rounds, or experiments.
- Any repair or refactoring that causes an experiment to answer a materially different question.

When an escalation condition is met, agents must halt the affected portion of the work and clearly label the issue:
```text
ESCALATE — HUMAN EPISTEMIC DECISION REQUIRED
```

---

## 3. Repository Verification Commands

All agents must execute and pass the appropriate verification commands before declaring work complete:

### Rust Core Test Suite & Binaries
- Run all unit and integration tests across the Rust workspace:
  ```powershell
  cargo test --manifest-path crates/continuity_garden_core/Cargo.toml
  ```
- Build and run specific experimental binaries in release mode:
  ```powershell
  cargo run --release --manifest-path crates/continuity_garden_core/Cargo.toml --bin <bin_name>
  ```

### Python Test Suite & Repository Invariants
- Run repo state and invariant tests:
  ```powershell
  python -m pytest tests/test_repo_state.py
  ```
- Run full Python test suite:
  ```powershell
  python -m pytest
  ```
