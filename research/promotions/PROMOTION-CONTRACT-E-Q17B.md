---
promotion_id: PROMOTION-CONTRACT-E-Q17B
contract_id: CONTRACT-E-Q17B
status: CANDIDATE
candidate_sha: 9c9a3d95b060b1a9f49bb4ec318e867e1b05bfe5
generated_at: "2026-08-21 22:50:00Z"
repair_rounds: 1
reviewed_by: codex
authorized_by: null
---

# Candidate Promotion Record: PROMOTION-CONTRACT-E-Q17B (Revision 1 — Matched Control Hardening)

**Lifecycle Status**: `CANDIDATE` (Awaiting Human Director & ChatGPT Pro Promotion Review)

## 1. Execution & Audit Summary
- **Target Contract**: `CONTRACT-E-Q17B`
- **Phase / Milestone**: Gate E Frontier (Q17B — Self-Supervised Endogenous Composition)
- **Candidate Branch**: `mb/CONTRACT-E-Q17B`
- **Candidate Commit SHA**: `9c9a3d95b060b1a9f49bb4ec318e867e1b05bfe5`
- **Contract Acceptance Verifier**: `PASS` (`verify_contract_q17b` executed and cleanly passed)
- **Auditor Verdict**: `PASS` (All 6 Success Gates, Dataset Matched Permutation Controls, and Transposition Falsification verified)
- **Repair Iterations**: 1 round (Replaced continuous simulator parameters with empirical observed trajectory frequencies; implemented strictly matched permutation negative control with identical input marginals and identical target sum $N=10,072$).

## 2. Experimental Verification & Gate Audit

| Gate / Estimand | Pre-registered Floor | Observed Empirical Result | Status |
| :--- | :--- | :--- | :--- |
| **Dataset Matched Control Integrity** | $N_{\text{samples}}=2500$, Identical Target Sum | **$N=2500$, Target Sum: 10,072 Intact vs 10,072 Shuffled (EXACT MATCH)** | **PASS** |
| **Gate 1 (Zero-Shot Multi-Hop Conflict)** | $\ge 10/16$ seeds ($62.5\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 2 (Laundering Discrimination)** | $\ge 10/16$ seeds ($62.5\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 3 (Temporal Shuffle Control Superiority)** | $n_{10} - n_{01} \ge 3, p < 0.05$ | **$n_{10}=13, n_{01}=0, \Delta=13$** ($p = 1.2207 \times 10^{-4}$) | **PASS** |
| **Gate 4 (Directional Transposition Falsification)** | $\le 2/16$ seeds, return $< 0.00$ | **0/16 seeds passed, mean return = -1.000** | **PASS** |
| **Gate 5 (Transposition Laundering Invariant)** | $\ge 10/16$ seeds | **16/16 seeds** | **PASS** |
| **Gate 6 (Mechanistic Path-Break Specificity)** | $p < 0.01$ | **$p = 1.2207 \times 10^{-4}$** | **PASS** |
| **Supervised Reference (Q17A)** | Upper benchmark reference | **16/16 parity maintained** | **INFORMATIONAL** |

## 3. Strict Epistemic Boundaries & Narrow Claim Ceiling
- **Claim**: Self-supervised multi-step trajectory prediction induces a composition-capable neural operator on local empirical transition representations without explicit two-hop reachability labels.
- **Exclusions**: Does NOT claim that the architecture autonomously discovered that composition exists (the kernel is architecturally handed adjacent transition pairs). Does NOT claim arbitrary $N$-hop path planning or lifetime memory consolidation (deferred to Q17C).
