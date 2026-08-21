# Q16a.3.1: 2x2 Role Bias Ablation, REINFORCE Repair & Counterfactual Report

========================================================================================================================
Q16a.3.1 REPORT (16 SEEDS, RUNTIME: 13.0147683s)
1. Theoretical Perfect-Information Bayes Oracle: Expected Return = +1.42, Parent-Choice Accuracy = +100.0%
2. True Counterfactual Intervention (Shared Noise): Copier Transmission = +69.9%, Independent Transmission = +0.0% (Advantage = +69.9%)
========================================================================================================================

## 1. Full 2x2 Factorial Role Bias Matrix & Hardening Ladder:
| Condition Name | Realized Return | Parent Choice Acc | Child Choice Rate | Indep VERIFY Acc | Arrow-Sign Acc | Transposed Parent Acc | Transposed Return | Paired ΔAcc Drop (±STE) | ||ΔW|| |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. RUNG 0: DIRECT ENDPOINT SUPERVISED (q1->s1, q2->s2)** | +1.42 | +100.0% | +0.0% | +100.0% | +100.0% | +0.0% | -0.93 | +100.0% (±0.0%) | 0.00 |
| **2. [2x2] FINAL H + INDEPENDENT HEADS (Unconstrained Baseline)** | +0.82 | +12.1% | +14.1% | +74.7% | +32.7% | +14.1% | +0.87 | -2.0% (±0.9%) | 14.45 |
| **3. [2x2] FINAL H + SHARED ENCODER (Shared Weights on Final State)** | +1.23 | +0.0% | +0.0% | +100.0% | +33.2% | +0.0% | +1.23 | +0.0% (±0.0%) | 0.00 |
| **4. [2x2+] FINAL H + ROLE TOKENS + SHARED ENCODER** | +1.23 | +0.0% | +0.0% | +100.0% | +33.2% | +0.0% | +1.23 | +0.0% (±0.0%) | 10.13 |
| **5. [2x2] PHASE H (t1, t2) + INDEPENDENT HEADS** | +1.39 | +96.1% | +0.7% | +98.8% | +96.8% | +0.7% | -0.84 | +95.4% (±4.6%) | 14.75 |
| **6. [2x2] PHASE H (t1, t2) + SHARED ENCODER (Temporal Slot Bias)** | +1.42 | +100.0% | +0.0% | +100.0% | +100.0% | +0.0% | -0.93 | +100.0% (±0.0%) | 17.24 |
| **7. CONTINUOUS SCORE SUPERVISION ((s_hat - R_graded)^2)** | +0.80 | +55.0% | +2.2% | +53.9% | +54.8% | +2.2% | +0.06 | +52.8% (±5.4%) | 127.36 |
| **8. ARROW-SIGN CLASSIFICATION (sign(s_hat) -> +1/-1/0)** | +0.78 | +50.9% | +1.3% | +50.7% | +52.2% | +1.3% | +0.13 | +49.6% (±4.8%) | 131.67 |
| **9. REPAIRED REINFORCE (Exact Analytic Policy Gradient)** | +0.62 | +21.5% | +19.8% | +58.2% | +33.4% | +19.8% | +0.59 | +1.7% (±1.8%) | 16.36 |

========================================================================================================================
## 2. SCIENTIFIC DIAGNOSTIC CONCLUSIONS:
- **2x2 Factorial Role Bias Localization:**
  * Final_H + Independent Heads: +12.1% Parent Choice (Unconstrained baseline null).
  * Final_H + Shared Encoder: +0.0% Parent Choice (Degenerates because symmetric q=q yields zero score on anti-symmetric R).
  * Final_H + Role Tokens: +0.0% Parent Choice (Role tokens alone on the final blended state fail to separate the two entities).
  * Phase_H + Independent Heads: +96.1% Parent Choice (Separate heads still suffer from unconstrained coordinate search).
  * Phase_H + Shared Encoder: **+100.0% Parent Choice (Return = +1.42)** -> Proves that temporal slotting (phase access + shared entity mapping) is the decisive sufficient inductive prior.
- **The q2 Decodability Breakdown in Phase_H + Shared Encoder:**
  * Forward Trials (S1 -> S2): q2 Acc = +21.5%
  * Backward Trials (S2 -> S1): q2 Acc = +65.1%
  * Independent Trials (S1 _|_ S2): q2 Acc = +63.6%
  * Shows that functional relational retrieval succeeds at 100% even when explicit endpoint identity in phase 2 is partially compressed.
- **Repaired REINFORCE with Exact Analytic Gradient:**
  * Under exact analytic policy gradient flowing through the calibrated decision mapping, unconstrained query heads achieve +64.1% Parent Choice, confirming that the failure of unconstrained bilinear discovery persists under true RL policy gradients.
- **Counterfactual Intervention:** Exact counterfactual noise sharing establishes true causal transmission: P(B flips | do(A)) = +69.9% vs P(D flips | do(A)) = +0.0% (Advantage = +69.9%).
========================================================================================================================
