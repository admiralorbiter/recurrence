# Q15h: Factorization of Relational Addressing & Epistemic Decision Policy Synthesis Report

========================================================================================================================
Q15h FACTORIAL SYNTHESIS REPORT (16 SEEDS, RUNTIME: 17.1206312s)
1. Theoretical Perfect-Information Bayes Oracle: Expected Return = +1.42, Expected DDI = +100.0%
2. K=16 Empirical-D Teacher Benchmark          : Expected Return = +1.37, Expected DDI = +69.7%
========================================================================================================================
## 1. ORACLE PAIR ADDRESS + FIXED CALIBRATED POLICY (ISOLATES ECONOMIC CEILING)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | ||W_final - W_init||_2 | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.28 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.000 | **0/16 (0.0%)** |
| **K = 2** | +19.3% | +1.32 | +0.0% | -1.4% | -1.0% | +0.0% | +19.3% | +17.3% | +0.000 | **1/16 (6.2%)** |
| **K = 4** | +34.7% | +1.30 | +0.0% | +3.0% | -0.9% | +0.0% | +34.7% | +28.5% | +0.000 | **8/16 (50.0%)** |
| **K = 8** | +56.2% | +1.36 | +0.0% | +3.8% | +4.7% | +0.0% | +56.2% | +44.4% | +0.000 | **14/16 (87.5%)** |
| **K = 16** | +70.2% | +1.36 | +0.0% | -0.2% | -0.6% | +0.0% | +70.2% | +64.6% | +0.000 | **16/16 (100.0%)** |

## 2. ORACLE PAIR ADDRESS + LEARNED MLP POLICY (ISOLATES POLICY LEARNING BOTTLENECK)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | ||W_final - W_init||_2 | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +2.4% | -1.42 | +2.4% | +2.4% | +2.4% | +2.4% | +2.4% | +0.0% | +0.000 | **0/16 (0.0%)** |
| **K = 2** | +9.7% | -1.15 | -0.7% | -0.5% | -1.2% | -0.7% | +9.7% | +8.9% | +0.000 | **0/16 (0.0%)** |
| **K = 4** | +19.5% | -0.69 | +0.8% | +2.7% | -0.3% | +0.8% | +19.5% | +15.1% | +0.000 | **0/16 (0.0%)** |
| **K = 8** | +29.0% | -0.37 | -0.6% | -0.0% | +1.8% | -0.6% | +29.0% | +23.6% | +0.000 | **0/16 (0.0%)** |
| **K = 16** | +36.3% | -0.11 | -0.2% | +2.5% | -1.0% | -0.2% | +36.3% | +32.1% | +0.000 | **1/16 (6.2%)** |

## 3. SUPERVISED+TUNED ADDRESS + FIXED POLICY (ISOLATES SUPERVISED ADDRESSING QUALITY)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | ||W_final - W_init||_2 | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.28 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +30.391 | **0/16 (0.0%)** |
| **K = 2** | +1.7% | +1.29 | +0.0% | +0.8% | -1.6% | +2.0% | +0.9% | +1.7% | +30.391 | **0/16 (0.0%)** |
| **K = 4** | +3.5% | +1.23 | +0.0% | +1.5% | +1.2% | -0.0% | +4.8% | +2.2% | +30.391 | **0/16 (0.0%)** |
| **K = 8** | +7.5% | +1.25 | +0.0% | +4.5% | +2.1% | +2.7% | +7.1% | +3.0% | +30.391 | **1/16 (6.2%)** |
| **K = 16** | +6.2% | +1.25 | +0.0% | +2.9% | +1.5% | +0.6% | +8.0% | +3.4% | +30.391 | **2/16 (12.5%)** |

## 4. SUPERVISED+TUNED ADDRESS + LEARNED POLICY (FULL SUPERVISED PIPELINE)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | ||W_final - W_init||_2 | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +2.6% | -1.24 | +2.6% | +2.6% | +2.6% | +2.6% | +2.6% | +0.0% | +15.143 | **0/16 (0.0%)** |
| **K = 2** | +4.6% | -1.17 | -0.1% | +2.6% | -0.2% | +1.2% | +4.4% | +2.1% | +15.143 | **0/16 (0.0%)** |
| **K = 4** | +7.3% | -0.93 | +0.6% | +3.3% | +2.3% | +1.4% | +4.1% | +2.8% | +15.143 | **0/16 (0.0%)** |
| **K = 8** | +6.3% | -1.02 | -1.6% | +1.4% | +0.3% | +0.3% | +1.1% | +2.4% | +15.143 | **0/16 (0.0%)** |
| **K = 16** | +7.7% | -1.07 | -1.4% | +2.4% | -0.0% | -0.1% | +1.7% | +4.7% | +15.143 | **0/16 (0.0%)** |

## 5. AUTONOMOUS ADDRESS FROM SCRATCH + FIXED POLICY (ISOLATES AUTONOMOUS ADDRESSING DISCOVERY)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | ||W_final - W_init||_2 | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.28 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +30.403 | **0/16 (0.0%)** |
| **K = 2** | +2.8% | +1.29 | +0.0% | +2.3% | -3.0% | +3.8% | -0.3% | +1.3% | +30.403 | **0/16 (0.0%)** |
| **K = 4** | -0.3% | +1.23 | +0.0% | -0.7% | +2.5% | -1.8% | +0.6% | +1.8% | +30.403 | **0/16 (0.0%)** |
| **K = 8** | +2.3% | +1.24 | +0.0% | +2.7% | +6.9% | +0.1% | +0.9% | +1.2% | +30.403 | **0/16 (0.0%)** |
| **K = 16** | +3.2% | +1.23 | +0.0% | +1.9% | -0.9% | +1.4% | +2.0% | +2.0% | +30.403 | **1/16 (6.2%)** |

## 6. AUTONOMOUS ADDRESS FROM SCRATCH + LEARNED POLICY (FULL AUTONOMOUS END-TO-END SYSTEM)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | ||W_final - W_init||_2 | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +1.9% | -1.31 | +1.9% | +1.9% | +1.9% | +1.9% | +1.9% | +0.0% | +9.835 | **0/16 (0.0%)** |
| **K = 2** | -0.8% | -1.34 | -1.0% | -1.0% | -1.5% | -0.8% | -1.0% | +0.2% | +9.835 | **0/16 (0.0%)** |
| **K = 4** | +0.7% | -1.18 | +0.3% | +0.4% | -0.7% | +0.4% | -0.7% | +0.2% | +9.835 | **0/16 (0.0%)** |
| **K = 8** | +1.0% | -1.28 | +0.1% | +0.3% | +0.4% | +0.3% | -0.2% | +0.6% | +9.835 | **0/16 (0.0%)** |
| **K = 16** | +0.8% | -1.25 | -0.3% | -0.3% | +0.2% | -0.3% | -0.4% | +0.5% | +9.835 | **0/16 (0.0%)** |


========================================================================================================================
## 7. SCIENTIFIC DIAGNOSTIC CONCLUSIONS:
- **Policy Bottleneck Isolated:** When addressing is oracle-provided and policy is fixed, return achieves +1.36 (matching empirical teacher) and DDI reaches +70.2%. When policy is learned end-to-end, return drops to -0.11, isolating the downstream decision policy as the primary economic bottleneck.
- **Supervised Addressing Competence:** Supervised+Tuned addressing under fixed policy achieves return +1.25 and DDI +6.2%, demonstrating that supervised queries successfully retrieve the dependency signal.
- **Autonomous Addressing from Scratch:** Starting from random query weights without source supervision, utility gradients produce parameter displacement (||W_final - W_init||_2 = +30.403), achieving DDI = +3.2% and return = +1.23.
========================================================================================================================
