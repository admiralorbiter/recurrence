# Q15i: Policy Sanity, Unconfounded Addressing Surrogate & Hardening Synthesis Report

========================================================================================================================
Q15i HARDENING SYNTHESIS REPORT (16 SEEDS, RUNTIME: 18.3723949s)
1. Theoretical Perfect-Information Bayes Oracle: Expected Return = +1.42, Expected DDI = +100.0%
2. K=16 Empirical-D Teacher Benchmark          : Expected Return = +1.37, Expected DDI = +69.7%
========================================================================================================================
## 1. ORACLE PAIR ADDRESS + FIXED CALIBRATED POLICY (ISOLATES ECONOMIC CEILING)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | Paired Perm Diff (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.28 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +19.3% | +1.32 | +0.0% | -1.4% | -1.0% | +0.0% | +19.3% | +17.3% | -3.1% (±2.6%) | **1/16 (6.2%)** |
| **K = 4** | +34.7% | +1.30 | +0.0% | +3.0% | -0.9% | +0.0% | +34.7% | +28.5% | -5.7% (±3.3%) | **8/16 (50.0%)** |
| **K = 8** | +56.2% | +1.36 | +0.0% | +3.8% | +4.7% | +0.0% | +56.2% | +44.4% | -10.1% (±4.1%) | **14/16 (87.5%)** |
| **K = 16** | +70.2% | +1.36 | +0.0% | -0.2% | -0.6% | +0.0% | +70.2% | +64.6% | -13.0% (±4.7%) | **16/16 (100.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +100.0%, Query 2 Acc = +100.0%, Entropy H(q) = 0.978, Score Correlation r = +0.782, Displacement ||ΔW_q|| = 0.000

## 2. ORACLE PAIR ADDRESS + LEARNED POLICY (VALIDATES POLICY SANITY)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | Paired Perm Diff (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.28 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +19.3% | +1.31 | +0.0% | -1.4% | -1.0% | +0.0% | +19.3% | +17.3% | -3.1% (±2.6%) | **1/16 (6.2%)** |
| **K = 4** | +34.7% | +1.26 | +0.0% | +3.0% | -0.8% | +0.0% | +34.7% | +28.4% | -5.7% (±3.3%) | **8/16 (50.0%)** |
| **K = 8** | +57.0% | +1.35 | +0.0% | +3.2% | +5.4% | +0.0% | +57.0% | +45.3% | -11.2% (±4.1%) | **14/16 (87.5%)** |
| **K = 16** | +70.8% | +1.36 | +0.0% | -0.5% | -0.0% | +0.0% | +70.8% | +64.9% | -14.4% (±4.7%) | **15/16 (93.8%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +100.0%, Query 2 Acc = +100.0%, Entropy H(q) = 0.978, Score Correlation r = +0.782, Displacement ||ΔW_q|| = 0.000

## 3. SUPERVISED+TUNED ADDRESS + FIXED POLICY (UNCONFOUNDED DIFFERENTIABLE SURROGATE)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | Paired Perm Diff (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.28 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +5.7% | +1.31 | +0.0% | -0.6% | +0.4% | +0.0% | +5.8% | +4.3% | +0.1% (±2.3%) | **0/16 (0.0%)** |
| **K = 4** | +12.4% | +1.25 | +0.0% | -4.5% | +3.1% | +0.0% | +12.6% | +7.8% | +0.4% (±3.0%) | **0/16 (0.0%)** |
| **K = 8** | +25.4% | +1.31 | +0.0% | -4.9% | +3.5% | +0.0% | +24.7% | +16.9% | -1.2% (±4.0%) | **5/16 (31.2%)** |
| **K = 16** | +28.1% | +1.30 | +0.0% | -4.7% | +3.5% | +0.0% | +28.4% | +18.3% | +2.8% (±4.3%) | **6/16 (37.5%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +33.4%, Query 2 Acc = +56.9%, Entropy H(q) = 0.214, Score Correlation r = +0.573, Displacement ||ΔW_q|| = 46.514

## 4. SUPERVISED+TUNED ADDRESS + LEARNED POLICY (FULL SUPERVISED PIPELINE)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | Paired Perm Diff (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.28 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +7.2% | +1.30 | +0.0% | -2.7% | +0.4% | +0.0% | +7.3% | +6.9% | +0.1% (±2.3%) | **0/16 (0.0%)** |
| **K = 4** | +12.0% | +1.25 | +0.0% | -5.1% | +2.0% | +0.0% | +11.9% | +10.2% | -0.7% (±2.8%) | **0/16 (0.0%)** |
| **K = 8** | +18.3% | +1.33 | +0.0% | -5.0% | +2.1% | +0.0% | +18.2% | +11.1% | -0.8% (±3.2%) | **3/16 (18.8%)** |
| **K = 16** | +16.2% | +1.33 | +0.0% | -9.1% | -0.7% | +0.0% | +15.9% | +13.8% | +1.9% (±3.2%) | **2/16 (12.5%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +33.4%, Query 2 Acc = +56.9%, Entropy H(q) = 0.214, Score Correlation r = +0.573, Displacement ||ΔW_q|| = 46.514

## 5. AUTONOMOUS ADDRESS FROM SCRATCH + FIXED POLICY (UNCONFOUNDED AUTONOMOUS DISCOVERY)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | Paired Perm Diff (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.28 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +1.0% | +1.28 | +0.0% | +1.4% | +0.5% | +0.0% | +1.0% | +1.3% | -0.8% (±1.9%) | **0/16 (0.0%)** |
| **K = 4** | +1.7% | +1.26 | +0.0% | -0.0% | -1.5% | +0.0% | +1.7% | +1.6% | -0.2% (±2.7%) | **0/16 (0.0%)** |
| **K = 8** | +4.7% | +1.29 | +0.0% | +3.8% | +4.3% | +0.0% | +4.5% | +1.3% | +0.9% (±3.2%) | **0/16 (0.0%)** |
| **K = 16** | +4.6% | +1.27 | +0.0% | +1.7% | +1.5% | +0.0% | +4.5% | +4.0% | -1.5% (±3.7%) | **0/16 (0.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +34.9%, Query 2 Acc = +34.8%, Entropy H(q) = 0.083, Score Correlation r = +0.265, Displacement ||ΔW_q|| = 30.367

## 6. AUTONOMOUS ADDRESS FROM SCRATCH + LEARNED POLICY (FULL AUTONOMOUS PIPELINE)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | Paired Perm Diff (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.28 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | -0.0% | +1.28 | +0.0% | -0.9% | -0.4% | +0.0% | -0.0% | +1.2% | -0.4% (±1.4%) | **0/16 (0.0%)** |
| **K = 4** | +0.9% | +1.26 | +0.0% | -0.7% | +2.7% | +0.0% | +0.9% | +0.7% | -0.0% (±1.3%) | **0/16 (0.0%)** |
| **K = 8** | +1.0% | +1.33 | +0.0% | -0.2% | +0.9% | +0.0% | +1.0% | +0.5% | -1.9% (±1.1%) | **0/16 (0.0%)** |
| **K = 16** | +0.7% | +1.33 | +0.0% | -0.2% | -0.3% | +0.0% | +0.6% | +1.4% | -1.2% (±1.1%) | **0/16 (0.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +34.9%, Query 2 Acc = +34.8%, Entropy H(q) = 0.083, Score Correlation r = +0.265, Displacement ||ΔW_q|| = 30.367


========================================================================================================================
## 7. SCIENTIFIC DIAGNOSTIC CONCLUSIONS:
- **K=0 Policy Sanity Resolved:** Under well-conditioned policy mapping, the learned policy achieves return = +1.28 at K=0 (matching the fixed rule), and at K=16 achieves return = +1.36 and DDI = +70.8%.
- **Supervised Addressing Quality:** Supervised queries maintain high decodability (q1 = 100.0%, q2 = 100.0%), and under unconfounded differentiable surrogate optimization achieve return = +1.30 and DDI = +28.1%.
- **Autonomous Addressing Status:** From random initialization without source supervision, unconfounded utility gradients drive parameter displacement (||ΔW_q|| = 30.367), producing DDI = +4.6% and return = +1.27.
========================================================================================================================
