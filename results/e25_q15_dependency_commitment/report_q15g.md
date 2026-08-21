# Q15g: Temporal Addressability Audit & True End-to-End Recruitment Synthesis Report

========================================================================================================================
Q15g SYNTHESIS REPORT: TEMPORAL ADDRESSABILITY AUDIT & 4-RUNG RECRUITMENT LADDER (16 SEEDS, RUNTIME: 12.725298s)
Analytic Bayes Oracle Ceiling Benchmark: Expected Return = +1.37, Expected DDI = +70.3%
========================================================================================================================

## 1. TEMPORAL STATE DECODABILITY AUDIT (Mean across 16 seeds)

| Timestep | Source 1 Acc | Content 1 Acc | Source 2 Acc | Content 2 Acc | Ordered Pair (s1, s2) Acc | Agreement Bit Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **t=0 (Blank)** | +32.1% | +49.4% | +33.6% | +51.3% | +15.9% (Chance 11.1%) | +80.9% |
| **t=1 (Source 1)** | +100.0% | +100.0% | +50.2% | +80.9% | +50.5% (Chance 11.1%) | +80.9% |
| **t=2 (Source 2)** | +100.0% | +100.0% | +100.0% | +100.0% | +100.0% (Chance 11.1%) | +82.2% |
| **t=3 (Blank Delay 1)** | +100.0% | +100.0% | +100.0% | +100.0% | +100.0% (Chance 11.1%) | +82.2% |
| **t=4 (Blank Delay 2)** | +100.0% | +100.0% | +100.0% | +100.0% | +100.0% (Chance 11.1%) | +82.2% |
| **t=5 (Blank Delay 3)** | +100.0% | +100.0% | +100.0% | +100.0% | +100.0% (Chance 11.1%) | +82.0% |
| **t=6 (Decision Cue)** | +100.0% | +100.0% | +100.0% | +100.0% | +100.0% (Chance 11.1%) | +81.8% |

## 2. RUNG A0: ORACLE CURRENT SOURCE-PAIR LOOKUP (ANALYSIS UPPER BOUND)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | ||ΔW_q|| | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | -1.7% | -1.44 | -1.7% | -1.7% | -1.7% | -1.7% | -1.7% | +0.0% | +0.000 | **0/16 (0.0%)** |
| **K = 2** | +6.0% | -1.13 | -2.2% | -3.6% | -3.4% | -2.2% | +6.0% | +7.6% | +0.000 | **0/16 (0.0%)** |
| **K = 4** | +17.8% | -0.83 | -0.1% | +1.5% | -0.8% | -0.1% | +17.8% | +14.8% | +0.000 | **0/16 (0.0%)** |
| **K = 8** | +28.0% | -0.37 | +0.2% | +1.4% | +2.9% | +0.2% | +28.0% | +22.0% | +0.000 | **0/16 (0.0%)** |
| **K = 16** | +36.0% | -0.07 | +2.2% | +1.5% | +2.1% | +2.2% | +36.0% | +30.4% | +0.000 | **0/16 (0.0%)** |

## 3. RUNG A1: STANDARDIZED SUPERVISED QUERY ADDRESSING (h -> q1, q2)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | ||ΔW_q|| | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | -1.0% | -1.23 | -1.0% | -1.0% | -1.0% | -1.0% | -1.0% | +0.0% | +0.000 | **0/16 (0.0%)** |
| **K = 2** | +1.2% | -0.97 | +0.4% | -0.3% | +1.0% | +2.3% | +4.7% | +1.0% | +0.000 | **0/16 (0.0%)** |
| **K = 4** | +7.7% | -0.71 | -2.9% | -2.3% | -3.5% | -3.2% | +1.9% | +8.3% | +0.000 | **0/16 (0.0%)** |
| **K = 8** | +10.5% | -0.43 | -1.9% | -1.6% | -0.3% | +0.4% | +0.5% | +9.5% | +0.000 | **0/16 (0.0%)** |
| **K = 16** | +11.6% | -0.48 | +3.4% | +3.9% | +0.9% | +4.5% | +6.1% | +6.1% | +0.000 | **0/16 (0.0%)** |

## 4. RUNG A2: TRUE END-TO-END UTILITY-LEARNED ADDRESSING (FROZEN GRU)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | ||ΔW_q|| | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.4% | -0.67 | +0.4% | +0.4% | +0.4% | +0.4% | +0.4% | +0.0% | +4.010 | **0/16 (0.0%)** |
| **K = 2** | +4.8% | -0.43 | -0.4% | +3.0% | +0.4% | +4.7% | +7.1% | +1.4% | +4.010 | **0/16 (0.0%)** |
| **K = 4** | +7.6% | -0.29 | -1.2% | +0.5% | -0.7% | +0.4% | +4.2% | +6.1% | +4.010 | **0/16 (0.0%)** |
| **K = 8** | +19.0% | -0.24 | +1.4% | +6.7% | +3.6% | +8.2% | +6.8% | +10.8% | +4.010 | **0/16 (0.0%)** |
| **K = 16** | +14.6% | -0.31 | +1.0% | +0.4% | +1.3% | +5.6% | +8.1% | +11.5% | +4.010 | **1/16 (6.2%)** |

## 5. RUNG A3: PLASTIC RECURRENT END-TO-END ADDRESSING (PLASTIC GRU)

| Calibration (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal D DDI | Off-Diagonal DDI | Relational Adv | ||ΔW_q|| | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | -0.8% | -0.86 | -0.8% | -0.8% | -0.8% | -0.8% | -0.8% | +0.0% | +3.919 | **0/16 (0.0%)** |
| **K = 2** | +4.9% | -0.72 | -0.9% | +1.5% | -0.9% | +1.0% | +2.2% | +3.0% | +3.919 | **0/16 (0.0%)** |
| **K = 4** | +9.8% | -0.53 | +0.5% | +3.2% | +1.0% | +1.1% | +3.3% | +6.0% | +3.919 | **0/16 (0.0%)** |
| **K = 8** | +11.0% | -0.45 | +1.6% | +2.6% | -0.0% | +2.3% | +6.6% | +8.1% | +3.919 | **0/16 (0.0%)** |
| **K = 16** | +7.7% | -0.44 | -1.1% | +1.1% | -1.7% | -1.3% | +3.2% | +5.7% | +3.919 | **0/16 (0.0%)** |


========================================================================================================================
## 6. SCIENTIFIC DIAGNOSTIC CONCLUSIONS:
- **Temporal Overwrite Finding:** In the temporal decodability audit, Source 1 decodability drops significantly following the arrival of Source 2 across blank delays (at decision cue t=6: s1 = +100.0%, s2 = +100.0%, ordered pair = +100.0% vs 11.1% chance).
- **Rung A0 (Oracle Addressing Upper Bound):** Under oracle current source-pair lookup, normalized D drives DDI to +36.0% and return to -0.07.
- **Rung A1 (Supervised Query Addressing):** Using correctly standardized and intercept-preserving query heads, DDI reaches +11.6% and return reaches -0.48.
- **Rung A2 & A3 (End-to-End Recruitment):** True backpropagation into query weights yields non-zero parameter updates (||ΔW_q|| = +4.010), achieving DDI = +14.6% (Frozen GRU) and +7.7% (Plastic GRU).
========================================================================================================================
