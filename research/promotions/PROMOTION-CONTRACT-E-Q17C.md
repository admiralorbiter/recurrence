---
promotion_id: PROMOTION-CONTRACT-E-Q17C
contract_id: CONTRACT-E-Q17C
status: CANDIDATE
candidate_sha: b8aab5975748d4a0ed9a74c031de8e48620ce749
generated_at: "2026-08-21 23:52:00Z"
repair_rounds: 0
reviewed_by: codex
authorized_by: null
---

# Candidate Promotion Record: PROMOTION-CONTRACT-E-Q17C (Endogenous Recurrent Memory & State Surgery)

**Lifecycle Status**: `CANDIDATE` (Awaiting Scientific Promotion Review by ChatGPT Pro & Human Research Director Authorization)

## 1. Execution & Audit Summary
- **Target Contract**: `CONTRACT-E-Q17C`
- **Phase / Milestone**: Gate E Frontier: Endogenous Recurrent Causal History
- **Candidate Branch**: `mb/CONTRACT-E-Q17C`
- **Scientific Candidate Commit SHA**: `b8aab5975748d4a0ed9a74c031de8e48620ce749`
- **Execution Base SHA**: `ecb24762988a4727076c9fc42a04f9bd52a4a2fc`
- **Contract Acceptance Verifier**: `PASS` (`verify_contract_q17c` executed independently with exact paired sign-flip permutation tests, donor-aligned state swap transfer assertions, same-history swap controls, competence preservation, and structural zero-sidecar API checks)
- **Auditor Verdict**: `PASS` (All 8 frozen statistical gates independently recomputed and satisfied across 16 seeds)
- **Repair Iterations**: 0 rounds (clean first-pass execution on frozen test protocol).

## 2. Frozen Statistical Gates & Empirical Outcomes

| Gate / Estimand | Preregistered Condition / Floor | Observed Empirical Result | Statistical Test | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1: Endogenous 2-Hop Conflict** | Persistent $z_t \ge 10/16$ ($62.5\%$) | **16 / 16 seeds ($100.0\%$)** | Exact binomial | **PASS** |
| **Gate 2: Endogenous Laundering** | Persistent $z_t \ge 10/16$ ($62.5\%$) | **12 / 16 seeds ($75.0\%$)** | Exact binomial | **PASS** |
| **Gate 3: Continuous Latent Reset Lesion** | $\Delta_{\text{reset}} = m_{\text{pers}} - m_{\text{reset}}$, $p < 0.01$ | **16/16 drop, reset near chance** | Paired sign-flip $p = 1.5259 \times 10^{-5}$ | **PASS** |
| **Gate 4: Continuous Donor-Aligned Swap** | Transfer $\ge 12/16$ ($75\%$), $p < 0.01$ | **16 / 16 seeds ($100.0\%$)** | Paired sign-flip $p = 1.5259 \times 10^{-5}$ | **PASS** |
| **Gate 5: Same-History Swap Stability** | Stable behavioral preference $\ge 15/16$ | **16 / 16 seeds ($100.0\%$)** | $\le 1/16$ threshold | **PASS** |
| **Gate 6: First-Order Competence** | 1-hop sensor accuracy $\ge 90\%$ in $\ge 15/16$ | **16 / 16 seeds ($100.0\%$)** | Baseline retention floor | **PASS** |
| **Gate 7: Temporal Shuffle Superiority** | $n_{10} - n_{01} \ge 3, p < 0.05$ | **$\Delta = +16, p = 3.0518 \times 10^{-5}$** | Exact McNemar paired | **PASS** |
| **Gate 8: Structural Zero-Sidecar** | $\equiv 0$ sidecar accesses | **16 / 16 verified ($100.0\%$)** | Direct API invariant | **PASS** |

## 3. Epistemic Invariants & Scope Ceilings
- **Claim**: Development-specific causal history can be stored endogenously in persistent recurrent activation state $z_t$ and exert causal control over previously validated composition-dependent behavior without an external causal-history store.
- **State Surgery Evidence**: In cloned twin organisms with identical weights $\theta$, identical $z_0$, and identical test-time cues, transplanting $z_t$ ($z_{H1} \leftrightarrow z_{H2}$) causally transfers the history-dependent directional choice ($A \leadsto C$ vs $C \leadsto A$) with zero damage to unrelated first-order sensor competence.
- **Exclusions**: Does NOT claim an abstract causal self-model or symbolic reasoning engine. Does NOT claim arbitrary $N$-hop graph reasoning ($N \ge 3$ reserved for Q17D).
