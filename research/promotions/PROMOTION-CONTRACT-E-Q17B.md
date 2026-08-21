---
promotion_id: PROMOTION-CONTRACT-E-Q17B
contract_id: CONTRACT-E-Q17B
status: PROMOTED
candidate_sha: da925179bbe769d9da544239c6db9604fcbad243
generated_at: "2026-08-21 23:12:00Z"
repair_rounds: 2
reviewed_by: chatgpt-pro
authorized_by: human
---

# Promotion Record: PROMOTION-CONTRACT-E-Q17B (Self-Supervised Endogenous Composition)

**Lifecycle Status**: `PROMOTED` (Authorized by Human Research Director following Scientific Review)

## 1. Execution & Audit Summary
- **Target Contract**: `CONTRACT-E-Q17B`
- **Phase / Milestone**: Gate E Frontier (Q17B — Self-Supervised Endogenous Composition)
- **Candidate Branch**: `mb/CONTRACT-E-Q17B`
- **Candidate Commit SHA**: `da925179bbe769d9da544239c6db9604fcbad243`
- **Contract Acceptance Verifier**: `PASS` (`verify_contract_q17b` executed and cleanly passed with continuous delta permutation test and transposed laundering arm)
- **Scientific Review Desk Verdict**: `APPROVED`
- **Human Director Verdict**: `PROMOTED`
- **Audit Findings**:
  - Training dataset $D$ ($N = 2500$) matched 1-to-1 against $D_{\text{shuffled}}$ with identical input marginal distributions, sample sizes, and identical target sums ($N_{\text{target}}=10,072$).
  - Separate transposed laundering arm confirmed circular consistency preservation under $A^T$.
  - Exact sign-flip permutation test evaluated directly on continuous path-break deltas $\Delta a_i$ confirmed mechanistic specificity ($p = 1.5259 \times 10^{-5} < 0.01$).
  - 13–0 superiority observed over the matched temporally shuffled negative control ($p = 1.2207 \times 10^{-4} < 0.05$).

## 2. Empirical Verification & Gate Audit

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
- **Claim**: Temporally aligned self-supervised trajectory experience produces a reliable improvement in composition-capable behavior over an otherwise matched temporally shuffled learning condition without explicit two-hop reachability labels.
- **Scientific Nuance**: Even without temporal alignment, the shuffled control learner achieves substantial baseline conflict accuracy ($\sim 0.90\text{--}1.0$), demonstrating that marginal transition statistics provide significant structural signal. However, temporally ordered trajectory pairing yields a statistically significant and mechanistic improvement ($\Delta = +13, p = 1.2207 \times 10^{-4}$).
- **Exclusions**: Does NOT claim that temporal alignment is strictly required for composition. Does NOT claim that the architecture autonomously discovered that composition exists (the kernel is architecturally handed adjacent transition pairs). Does NOT claim arbitrary $N$-hop path planning or lifetime memory consolidation.
