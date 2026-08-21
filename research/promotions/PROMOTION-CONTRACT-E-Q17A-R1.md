---
promotion_id: PROMOTION-CONTRACT-E-Q17A-R1
contract_id: CONTRACT-E-Q17A-R1
status: CANDIDATE
candidate_sha: PLACEHOLDER
generated_at: "2026-08-21 20:55:00Z"
repair_rounds: 0
reviewed_by: codex
authorized_by: null
---

# Candidate Promotion Record: PROMOTION-CONTRACT-E-Q17A-R1

**Lifecycle Status**: `CANDIDATE` (Awaiting Human Director & ChatGPT Pro Promotion Review)

## 1. Execution & Audit Summary
- **Target Contract**: `CONTRACT-E-Q17A-R1`
- **Phase / Milestone**: Gate E Frontier (Q17A — Learned 2-Hop Neural Composition)
- **Candidate Branch**: `mb/CONTRACT-E-Q17A-R1`
- **Auditor Verdict**: `PASS` (All 6 Success Gates passed across 16/16 seeds)
- **Repair Iterations**: 0 rounds

## 2. Experimental Verification & Gate Audit

| Gate | Description | Pre-registered Floor | Observed Empirical Result | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1** | Zero-Shot Multi-Hop Conflict Accuracy ($A \neq C$) | $\ge 12/16$ seeds ($75.0\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 2** | Zero-Shot Laundering Discrimination ($A = C$) | $\ge 11/16$ seeds ($68.75\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 3** | Independent Corroboration ($A = D$) | $\ge 15/16$ seeds ($93.75\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 4** | Independent Conflict ($A \neq D$) | $\ge 15/16$ seeds ($93.75\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 5** | Composition Ablation Behavioral Floor | $n_{10} - n_{01} \ge 3$ | **$n_{10} = 16, n_{01} = 0, \Delta = 16$** ($p = 3.05 \times 10^{-5}$) | **PASS** |
| **Gate 6** | Exact Paired Permutation Test ($2^{16}$ perms) | $p < 0.01$ | **$p = 1.5259 \times 10^{-5}$** | **PASS** |

## 3. Directional Transposition Controls
- **Multi-Hop Conflict under Transposition**: Collapses from 100% to $\le 2/16$ seeds, proving correct $A \to C$ directionality is necessary for conflict resolution.
- **Independent Controls Specificity**: Controls ($A/D$) remain 100% intact under path-breaks, proving specificity to the transmission chain.

## 4. Promotion Claim
The parameterized neural composition kernel $f_\theta$ successfully learned generic 2-hop reachability from auxiliary development worlds and transferred zero-shot to the withheld endpoint pair $(A, C)$, completely matching the empirical performance floor of the engineered matrix algebra baseline without hardcoded matrix multiplication.
