# Q15i: Policy Sanity, Unconfounded Addressing Surrogate & Hardening Synthesis Report

========================================================================================================================
Q15i HARDENING SYNTHESIS REPORT (16 SEEDS, RUNTIME: 18.8277672s)
1. Theoretical Perfect-Information Bayes Oracle: Expected Return = +1.42, Expected DDI = +100.0%
2. K=16 Empirical-D Teacher Benchmark          : Expected Return = +1.37, Expected DDI = +69.7%
========================================================================================================================
## 1. ORACLE PAIR ADDRESS + FIXED CALIBRATED POLICY (ISOLATES ECONOMIC CEILING)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | Paired ΔDDI (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.28 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +19.3% | +1.32 | +0.0% | -1.4% | -1.0% | +0.0% | +19.3% | +17.3% | +20.7% (±2.3%) | **1/16 (6.2%)** |
| **K = 4** | +34.7% | +1.30 | +0.0% | +3.0% | -0.9% | +0.0% | +34.7% | +28.5% | +31.7% (±2.3%) | **8/16 (50.0%)** |
| **K = 8** | +56.2% | +1.36 | +0.0% | +3.8% | +4.7% | +0.0% | +56.2% | +44.4% | +52.4% (±3.2%) | **14/16 (87.5%)** |
| **K = 16** | +70.2% | +1.36 | +0.0% | -0.2% | -0.6% | +0.0% | +70.2% | +64.6% | +70.4% (±3.2%) | **16/16 (100.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +100.0%, Query 2 Acc = +100.0%, Entropy H(q) = 0.978, Score Correlation r = +0.782, Displacement ||ΔW_q|| = 0.000

## 2. ORACLE PAIR ADDRESS + LEARNED POLICY (VALIDATES POLICY SANITY)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | Paired ΔDDI (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.28 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +19.3% | +1.31 | +0.0% | -1.4% | -1.0% | +0.0% | +19.3% | +17.3% | +20.7% (±2.3%) | **1/16 (6.2%)** |
| **K = 4** | +34.7% | +1.26 | +0.0% | +3.0% | -0.8% | +0.0% | +34.7% | +28.4% | +31.7% (±2.3%) | **9/16 (56.2%)** |
| **K = 8** | +57.2% | +1.31 | +0.0% | +3.4% | +5.5% | +0.0% | +57.2% | +45.4% | +53.8% (±2.9%) | **13/16 (81.2%)** |
| **K = 16** | +70.3% | +1.35 | +0.0% | +0.6% | -0.2% | +0.0% | +70.3% | +63.7% | +69.7% (±3.7%) | **15/16 (93.8%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +100.0%, Query 2 Acc = +100.0%, Entropy H(q) = 0.978, Score Correlation r = +0.782, Displacement ||ΔW_q|| = 0.000

## 3. SUPERVISED+TUNED ADDRESS + FIXED POLICY (UNCONFOUNDED DIFFERENTIABLE SURROGATE)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | Paired ΔDDI (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.28 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +5.7% | +1.31 | +0.0% | -0.6% | +0.4% | +0.0% | +5.8% | +4.3% | +6.3% (±2.4%) | **0/16 (0.0%)** |
| **K = 4** | +12.4% | +1.25 | +0.0% | -4.5% | +3.1% | +0.0% | +12.6% | +7.8% | +16.8% (±3.5%) | **0/16 (0.0%)** |
| **K = 8** | +25.4% | +1.31 | +0.0% | -4.9% | +3.5% | +0.0% | +24.7% | +16.9% | +30.3% (±7.8%) | **5/16 (31.2%)** |
| **K = 16** | +28.1% | +1.30 | +0.0% | -4.7% | +3.5% | +0.0% | +28.4% | +18.3% | +32.8% (±7.0%) | **6/16 (37.5%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +33.4%, Query 2 Acc = +56.9%, Entropy H(q) = 0.214, Score Correlation r = +0.573, Displacement ||ΔW_q|| = 46.514

## 4. SUPERVISED+TUNED ADDRESS + LEARNED POLICY (FULL SUPERVISED PIPELINE)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | Paired ΔDDI (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.27 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +7.1% | +1.28 | +0.0% | -2.8% | +0.4% | +0.0% | +7.1% | +6.8% | +9.8% (±3.7%) | **0/16 (0.0%)** |
| **K = 4** | +12.3% | +1.23 | +0.0% | -4.9% | +1.5% | +0.0% | +12.0% | +10.2% | +17.2% (±4.5%) | **0/16 (0.0%)** |
| **K = 8** | +18.2% | +1.30 | +0.0% | -4.8% | +2.7% | +0.0% | +18.0% | +11.0% | +23.1% (±7.3%) | **2/16 (12.5%)** |
| **K = 16** | +16.9% | +1.31 | +0.0% | -8.8% | -0.6% | +0.0% | +16.7% | +14.5% | +25.7% (±7.5%) | **2/16 (12.5%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +33.4%, Query 2 Acc = +56.9%, Entropy H(q) = 0.214, Score Correlation r = +0.573, Displacement ||ΔW_q|| = 46.514

## 5. AUTONOMOUS ADDRESS FROM SCRATCH + FIXED POLICY (UNCONFOUNDED AUTONOMOUS DISCOVERY)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | Paired ΔDDI (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.28 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +1.0% | +1.28 | +0.0% | +1.4% | +0.5% | +0.0% | +1.0% | +1.3% | -0.4% (±1.5%) | **0/16 (0.0%)** |
| **K = 4** | +1.7% | +1.26 | +0.0% | -0.0% | -1.5% | +0.0% | +1.7% | +1.6% | +1.7% (±1.9%) | **0/16 (0.0%)** |
| **K = 8** | +4.7% | +1.29 | +0.0% | +3.8% | +4.3% | +0.0% | +4.5% | +1.3% | +0.8% (±2.7%) | **0/16 (0.0%)** |
| **K = 16** | +4.6% | +1.27 | +0.0% | +1.7% | +1.5% | +0.0% | +4.5% | +4.0% | +2.9% (±4.4%) | **0/16 (0.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +34.9%, Query 2 Acc = +34.8%, Entropy H(q) = 0.083, Score Correlation r = +0.265, Displacement ||ΔW_q|| = 30.367

## 6. AUTONOMOUS ADDRESS + LEARNED POLICY (SCAFFOLDED REPORT DECODERS)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | Paired ΔDDI (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.25 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +0.1% | +1.25 | +0.0% | -1.1% | -0.5% | +0.0% | +0.1% | +1.5% | +1.2% (±1.2%) | **0/16 (0.0%)** |
| **K = 4** | +1.3% | +1.24 | +0.0% | -1.1% | +2.6% | +0.0% | +1.2% | +1.0% | +2.4% (±1.2%) | **0/16 (0.0%)** |
| **K = 8** | +1.6% | +1.32 | +0.0% | -0.2% | +0.5% | +0.0% | +1.5% | +1.4% | +1.8% (±2.6%) | **0/16 (0.0%)** |
| **K = 16** | +0.4% | +1.31 | +0.0% | +0.1% | +0.2% | +0.0% | +0.4% | +1.3% | +0.3% (±2.4%) | **0/16 (0.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +34.9%, Query 2 Acc = +34.8%, Entropy H(q) = 0.083, Score Correlation r = +0.265, Displacement ||ΔW_q|| = 30.367


========================================================================================================================
## 7. SCIENTIFIC DIAGNOSTIC CONCLUSIONS:
- **K=0 Policy Sanity Resolved:** Under well-conditioned continuous feature policy mapping, the learned policy achieves return = +1.28 at K=0 (matching the fixed rule), and at K=16 under oracle addressing achieves return = +1.35 and DDI = +70.3%.
- **Supervised Addressing Dynamics:** After surrogate utility tuning, query heads reorganize into a functional relational code (Query 1 Acc = +33.4%, Query 2 Acc = +56.9%, Score Correlation r = +0.573), achieving return = +1.30 and DDI = +28.1%.
- **Autonomous Addressing Status:** From random initialization without source supervision, unconfounded utility gradients drive parameter displacement (||ΔW_q|| = 30.367), producing DDI = +4.6% and return = +1.27.
========================================================================================================================
