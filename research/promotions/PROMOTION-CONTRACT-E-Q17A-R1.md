---
promotion_id: PROMOTION-CONTRACT-E-Q17A-R1
contract_id: CONTRACT-E-Q17A-R1
status: CANDIDATE
candidate_sha: efc2d9941bb546a28fc01ff634211e79070a5bae
generated_at: "2026-08-21 21:51:00Z"
repair_rounds: 2
reviewed_by: codex
authorized_by: null
---

# Candidate Promotion Record: PROMOTION-CONTRACT-E-Q17A-R1 (Revision 1)

**Lifecycle Status**: `CANDIDATE` (Awaiting Human Director & ChatGPT Pro Promotion Review)

## 1. Execution & Audit Summary
- **Target Contract**: `CONTRACT-E-Q17A-R1`
- **Phase / Milestone**: Gate E Frontier (Q17A — Endogenous 2-Hop Transitive Composition)
- **Candidate Branch**: `mb/CONTRACT-E-Q17A-R1`
- **Contract Acceptance Verifier**: `PASS` (`verify_contract_q17a` executed and cleanly passed)
- **Auditor Verdict**: `PASS` (All 6 Success Gates and Transposition Falsification Criteria satisfied)
- **Repair Iterations**: 2 rounds (eliminated intermediate search loops and implemented directional decision mechanism for transposition collapse)

## 2. Experimental Verification & Gate Audit

| Gate / Estimand | Pre-registered Floor | Observed Empirical Result | Status |
| :--- | :--- | :--- | :--- |
| **Gate 1 (Zero-Shot Multi-Hop Conflict)** | $\ge 12/16$ seeds ($75.0\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 2 (Zero-Shot Laundering Discrimination)** | $\ge 11/16$ seeds ($68.75\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 3 (Independent Corroboration)** | $\ge 15/16$ seeds ($93.75\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 4 (Independent Conflict)** | $\ge 15/16$ seeds ($93.75\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 5 (Composition Ablation Floor)** | $n_{10} - n_{01} \ge 3$ | **$n_{10}=16, n_{01}=0, \Delta=16$** ($p = 1.5259 \times 10^{-5}$) | **PASS** |
| **Gate 6 (Mechanistic Path-Break Specificity)** | $p < 0.01$, $A/D \ge 15/16$ | **$p = 1.5259 \times 10^{-5}$**, $A/D = 16/16$ | **PASS** |
| **Transposition Falsification ($A \neq C$)** | $\le 2/16$ seeds, return $< 0.00$ | **0/16 seeds passed, mean return = -0.995** | **PASS** |
| **Transposition Laundering ($A = C$)** | $\ge 10/16$ seeds | **16/16 seeds** | **PASS** |

## 3. Strict Epistemic Boundaries & Narrow Claim Ceiling
- **Claim**: The neural composition kernel $f_\theta$ successfully computes 2-hop causal transitive reachability from specifically addressed local representations $(e_{AB}, e_{BC})$ without intermediate path search or traversal under an engineered downstream decision mapping.
- **Exclusions**: Does NOT claim autonomous policy learning, self-supervised composition discovery (deferred to Q17B), recurrent memory migration, or arbitrary $N$-hop path traversal.
