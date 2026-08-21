---
promotion_id: PROMOTION-CONTRACT-E-Q17B
contract_id: CONTRACT-E-Q17B
status: CANDIDATE
candidate_sha: 13bb5593b7ee2111ed45fbd0d296afd4c53af578
generated_at: "2026-08-21 22:42:00Z"
repair_rounds: 0
reviewed_by: codex
authorized_by: null
---

# Candidate Promotion Record: PROMOTION-CONTRACT-E-Q17B

**Lifecycle Status**: `CANDIDATE` (Awaiting Human Director & ChatGPT Pro Promotion Review)

## 1. Execution & Audit Summary
- **Target Contract**: `CONTRACT-E-Q17B`
- **Phase / Milestone**: Gate E Frontier (Q17B — Self-Supervised Endogenous Composition)
- **Candidate Branch**: `mb/CONTRACT-E-Q17B`
- **Contract Acceptance Verifier**: `PASS` (`verify_contract_q17b` executed and cleanly passed)
- **Auditor Verdict**: `PASS` (All 6 Success Gates and Controls satisfied)
- **Repair Iterations**: 0 rounds (Clean first-pass verification)

## 2. Experimental Verification & Gate Audit

| Gate / Estimand | Pre-registered Floor | Observed Empirical Result | Status |
| :--- | :--- | :--- | :--- |
| **Gate 1 (Zero-Shot Multi-Hop Conflict)** | $\ge 10/16$ seeds ($62.5\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 2 (Laundering Discrimination)** | $\ge 10/16$ seeds ($62.5\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 3 (Temporal Shuffle Control Superiority)** | $n_{10} - n_{01} \ge 3, p < 0.05$ | **$n_{10}=16, n_{01}=0, \Delta=16$** ($p = 1.5259 \times 10^{-5}$) | **PASS** |
| **Gate 4 (Directional Transposition Falsification)** | $\le 2/16$ seeds, return $< 0.00$ | **0/16 seeds passed, mean return = -1.000** | **PASS** |
| **Gate 5 (Transposition Laundering Invariant)** | $\ge 10/16$ seeds | **16/16 seeds** | **PASS** |
| **Gate 6 (Mechanistic Path-Break Specificity)** | $p < 0.01$ | **$p = 4.8828 \times 10^{-4}$** | **PASS** |
| **Supervised Reference (Q17A)** | Upper benchmark reference | **16/16 parity maintained** | **INFORMATIONAL** |

## 3. Strict Epistemic Boundaries & Narrow Claim Ceiling
- **Claim**: Self-supervised multi-step trajectory prediction induces a composition-capable neural operator on local transition representations without explicit two-hop reachability labels.
- **Exclusions**: Does NOT claim that the architecture autonomously discovered that composition exists (the kernel is architecturally handed adjacent transition pairs). Does NOT claim arbitrary $N$-hop path planning or lifetime memory consolidation (deferred to Q17C).
