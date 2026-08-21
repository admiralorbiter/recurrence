---
promotion_id: PROMOTION-CONTRACT-E-Q17A-R1
contract_id: CONTRACT-E-Q17A-R1
status: CANDIDATE
candidate_sha: 90a521a31fe8b3610c452b5c7c42bd68d38b0c75
generated_at: "2026-08-21 21:25:00Z"
repair_rounds: 1
reviewed_by: codex
authorized_by: null
---

# Candidate Promotion Record: PROMOTION-CONTRACT-E-Q17A-R1 (Revision 1)

**Lifecycle Status**: `CANDIDATE` (Awaiting Human Director & ChatGPT Pro Promotion Review)

## 1. Execution & Audit Summary
- **Target Contract**: `CONTRACT-E-Q17A-R1`
- **Phase / Milestone**: Gate E Frontier (Q17A — Endogenous 2-Hop Transitive Composition)
- **Candidate Branch**: `mb/CONTRACT-E-Q17A-R1`
- **Candidate Commit SHA**: `90a521a31fe8b3610c452b5c7c42bd68d38b0c75`
- **Auditor Verdict**: `PASS` (All 6 Success Gates passed across 16/16 seeds)
- **Repair Iterations**: 1 round (resolved path enumeration and deterministic threshold checks)

## 2. Experimental Verification & Gate Audit

| Gate | Description | Pre-registered Floor | Observed Empirical Result | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1** | Zero-Shot Multi-Hop Conflict Accuracy ($A \neq C$) | $\ge 12/16$ seeds ($75.0\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 2** | Zero-Shot Laundering Discrimination ($A = C$) | $\ge 11/16$ seeds ($68.75\%$) | **14/16 seeds** ($87.5\%$) | **PASS** |
| **Gate 3** | Independent Corroboration ($A = D$) | $\ge 15/16$ seeds ($93.75\%$) | **16/16 seeds** ($100.0\%$) | **PASS** |
| **Gate 4** | Independent Conflict ($A \neq D$) | $\ge 15/16$ seeds ($93.75\%$) | **15/16 seeds** ($93.75\%$) | **PASS** |
| **Gate 5** | Composition Ablation Behavioral Floor | $n_{10} - n_{01} \ge 3$ | **$n_{10} = 14, n_{01} = 0, \Delta = 14$** ($p = 1.2207 \times 10^{-4}$) | **PASS** |
| **Gate 6** | Exact Paired Permutation Test ($2^{16}$ perms) | $p < 0.01$ | **$p = 1.5259 \times 10^{-5}$** | **PASS** |

## 3. Directional Transposition & Causal Lesions
- **Directional Transposition ($\hat{A}^T$)**: Multi-hop conflict accuracy collapses significantly, proving strict reliance on transmission directionality $A \to B \to C$.
- **Path Breaks ($e_{AB}=0, e_{BC}=0$)**: Specifically collapses $a_{AC}$ without disturbing independent baseline ($A/D$).

## 4. Promotion Claim
The endogenous neural composition kernel $f_\theta$ successfully computes transitive reachability from specifically addressed local representations $(e_{AB}, e_{BC})$ without intermediate node search or algorithmic path traversal, transferring zero-shot to $(A, C)$ across matched stochastic Bayesian challenge episodes.
