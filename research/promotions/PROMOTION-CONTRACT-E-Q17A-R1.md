---
promotion_id: PROMOTION-CONTRACT-E-Q17A-R1
contract_id: CONTRACT-E-Q17A-R1
status: PROMOTED
candidate_sha: efc2d9941bb546a28fc01ff634211e79070a5bae
promoted_at: "2026-08-21 22:16:00Z"
repair_rounds: 2
reviewed_by: chatgpt-pro
authorized_by: human
---

# Promotion Record: PROMOTION-CONTRACT-E-Q17A-R1 (Promoted)

**Lifecycle Status**: `PROMOTED` (Authorized by Human Research Director & ChatGPT Pro Scientific Review Desk)

## 1. Execution & Audit Summary
- **Target Contract**: `CONTRACT-E-Q17A-R1`
- **Phase / Milestone**: Gate E Frontier (Q17A — Endogenous 2-Hop Transitive Composition)
- **Promoted Candidate Commit SHA**: `efc2d9941bb546a28fc01ff634211e79070a5bae`
- **Contract Acceptance Verifier**: `PASS` (`verify_contract_q17a` executed and cleanly passed)
- **Scientific Review Verdict**: `APPROVED`
- **Governance**: Human Director Promotion Merge.

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

## 3. Epistemic Belief Update & Narrow Claim Ceiling
- **Empirical Belief Update**: A learned parameterized neural function ($f_\theta$) can replace the fixed two-hop algebraic composition operator in this assay, generalize to the withheld $A \to C$ endpoint, and preserve the behavioral effect while exhibiting the required causal lesion and directionality signatures.
- **Strict Exclusions**: Does NOT show that the architecture independently discovered the need for composition (the kernel was trained with explicit auxiliary two-hop targets). Self-supervised discovery is the explicit frontier of Q17B.
