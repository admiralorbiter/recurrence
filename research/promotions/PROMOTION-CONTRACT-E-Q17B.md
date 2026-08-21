---
promotion_id: PROMOTION-CONTRACT-E-Q17B
contract_id: CONTRACT-E-Q17B
status: CANDIDATE
candidate_sha: da925179bbe769d9da544239c6db9604fcbad243
generated_at: "2026-08-21 22:58:00Z"
repair_rounds: 2
reviewed_by: codex
authorized_by: null
---

# Candidate Promotion Record: PROMOTION-CONTRACT-E-Q17B (Revision 2 — Continuous Permutation & Transposed Arm)

**Lifecycle Status**: `CANDIDATE` (Awaiting Human Director & ChatGPT Pro Promotion Review)

## 1. Execution & Audit Summary
- **Target Contract**: `CONTRACT-E-Q17B`
- **Phase / Milestone**: Gate E Frontier (Q17B — Self-Supervised Endogenous Composition)
- **Candidate Branch**: `mb/CONTRACT-E-Q17B`
- **Candidate Commit SHA**: `da925179bbe769d9da544239c6db9604fcbad243`
- **Contract Acceptance Verifier**: `PASS` (`verify_contract_q17b` executed and cleanly passed with continuous delta permutation test and transposed laundering arm)
- **Auditor Verdict**: `PASS` (All 6 Success Gates, Dataset Matched Permutation Controls, Continuous Lesion Permutations, and Transposition Falsification verified)
- **Repair Iterations**: 2 rounds (Implemented genuine transposed laundering arm; evaluated continuous $\Delta a_i$ sign-flip permutation test; independently aggregated per-seed target sums).

## 2. Experimental Verification & Gate Audit

| Gate / Estimand | Pre-registered Floor | Observed Empirical Result | Status |
| :--- | :--- | :--- | :--- |
| **Dataset Matched Control Integrity** | $N_{\text{samples}}=2500$, Independently Aggregated | **$N=2500$, Target Sum: 10,072 Intact vs 10,072 Shuffled (EXACT MATCH)** | **PASS** |
| **Gate 1 (Zero-Shot Multi-Hop Conflict)** | $\ge 10/16$ seeds ($62.5\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 2 (Laundering Discrimination)** | $\ge 10/16$ seeds ($62.5\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 3 (Temporal Shuffle Control Superiority)** | $n_{10} - n_{01} \ge 3, p < 0.05$ | **$n_{10}=13, n_{01}=0, \Delta=13$** ($p = 1.2207 \times 10^{-4}$) | **PASS** |
| **Gate 4 (Directional Transposition Falsification)** | $\le 2/16$ seeds, return $< 0.00$ | **0/16 seeds passed, mean return = -1.000** | **PASS** |
| **Gate 5 (Transposition Laundering Arm Invariant)** | $\ge 10/16$ seeds under $A^T$ | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 6 (Mechanistic Continuous Delta Permutation)** | $p < 0.01$ on continuous $\Delta a_i$ | **$p = 1.5259 \times 10^{-5}$** | **PASS** |
| **Supervised Reference (Q17A)** | Upper benchmark reference | **16/16 parity maintained** | **INFORMATIONAL** |

## 3. Strict Epistemic Boundaries & Narrow Claim Ceiling
- **Claim**: Self-supervised multi-step trajectory prediction induces a composition-capable neural operator on local empirical transition representations without explicit two-hop reachability labels.
- **Exclusions**: Does NOT claim that the architecture autonomously discovered that composition exists (the kernel is architecturally handed adjacent transition pairs). Does NOT claim arbitrary $N$-hop path planning or lifetime memory consolidation (deferred to Q17C).
