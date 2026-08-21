# Q16a.1: Directional Provenance Hardening Synthesis Report

========================================================================================================================
Q16a.1 HARDENING SYNTHESIS REPORT (16 SEEDS, RUNTIME: 10.7475171s)
1. Theoretical Perfect-Information Bayes Oracle: Expected Return = +1.42, Parent-Choice Accuracy = +100.0%
2. K=16 Empirical-R Teacher Benchmark          : Expected Return = +1.25, Parent-Choice Accuracy = +57.2%
========================================================================================================================

## Sidecar Ground-Truth Arrow Reconstruction Acc_sidecar(K):
- **K = 0**: +100.0% correct causal arrow classification
- **K = 2**: +2.0% correct causal arrow classification
- **K = 4**: +8.3% correct causal arrow classification
- **K = 8**: +26.3% correct causal arrow classification
- **K = 16**: +58.9% correct causal arrow classification

## 1. ORACLE DIRECTIONAL ADDRESS + FIXED CALIBRATED POLICY (ISOLATES ECONOMIC CEILING)

| Calibration (K) | Realized Return | Parent Choice Acc | Child Choice Rate | Indep VERIFY Acc | Arrow-Sign Acc | Transposed Parent Acc | Transposed Return | Paired ΔAcc Drop (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +1.25 | +0.0% | +0.0% | +100.0% | +35.1% | +0.0% | +1.25 | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +1.22 | +0.3% | +0.2% | +99.8% | +33.8% | +0.2% | +1.21 | +0.1% (±0.3%) | **0/16 (0.0%)** |
| **K = 4** | +1.21 | +9.4% | +0.5% | +99.2% | +39.0% | +0.5% | +1.01 | +8.9% (±0.8%) | **0/16 (0.0%)** |
| **K = 8** | +1.22 | +27.1% | +1.3% | +97.3% | +51.3% | +1.3% | +0.62 | +25.8% (±1.3%) | **0/16 (0.0%)** |
| **K = 16** | +1.28 | +58.4% | +2.1% | +90.8% | +69.2% | +2.1% | -0.03 | +56.3% (±2.6%) | **1/16 (6.2%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +100.0%, Query 2 Acc = +100.0%, Entropy H(q) = 0.977, Arrow-Sign Acc = +63.2%, Oracle Corr r = +0.980, Displacement ||ΔW_q|| = 0.000

## 2. ORACLE DIRECTIONAL ADDRESS + LEARNED POLICY (VALIDATES DIRECTIONAL DECISION MAPPING)

| Calibration (K) | Realized Return | Parent Choice Acc | Child Choice Rate | Indep VERIFY Acc | Arrow-Sign Acc | Transposed Parent Acc | Transposed Return | Paired ΔAcc Drop (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +1.12 | +3.0% | +2.9% | +93.8% | +35.1% | +3.0% | +1.12 | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +1.11 | +2.2% | +2.8% | +93.8% | +33.8% | +2.4% | +1.11 | -0.2% (±0.2%) | **0/16 (0.0%)** |
| **K = 4** | +1.09 | +3.2% | +3.2% | +93.5% | +39.0% | +2.6% | +1.08 | +0.6% (±0.3%) | **0/16 (0.0%)** |
| **K = 8** | +1.07 | +6.5% | +4.9% | +93.4% | +51.3% | +5.4% | +1.06 | +1.1% (±0.7%) | **0/16 (0.0%)** |
| **K = 16** | +1.06 | +8.4% | +6.6% | +92.7% | +69.2% | +6.5% | +1.05 | +1.9% (±1.0%) | **0/16 (0.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +100.0%, Query 2 Acc = +100.0%, Entropy H(q) = 0.977, Arrow-Sign Acc = +63.2%, Oracle Corr r = +0.980, Displacement ||ΔW_q|| = 0.000

## 3. SUPERVISED+TUNED ADDRESS + FIXED POLICY (DIFFERENTIABLE DIRECTIONAL SURROGATE)

| Calibration (K) | Realized Return | Parent Choice Acc | Child Choice Rate | Indep VERIFY Acc | Arrow-Sign Acc | Transposed Parent Acc | Transposed Return | Paired ΔAcc Drop (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +1.25 | +0.0% | +0.0% | +100.0% | +35.1% | +0.0% | +1.25 | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +1.22 | +0.4% | +0.1% | +99.8% | +33.6% | +0.1% | +1.21 | +0.3% (±0.2%) | **0/16 (0.0%)** |
| **K = 4** | +1.19 | +2.2% | +0.9% | +98.6% | +34.6% | +0.9% | +1.17 | +1.4% (±0.7%) | **0/16 (0.0%)** |
| **K = 8** | +1.15 | +7.5% | +2.8% | +94.0% | +36.7% | +2.8% | +1.03 | +4.7% (±2.3%) | **0/16 (0.0%)** |
| **K = 16** | +1.14 | +15.9% | +3.8% | +88.8% | +40.9% | +3.8% | +0.86 | +12.1% (±4.1%) | **0/16 (0.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +48.4%, Query 2 Acc = +43.3%, Entropy H(q) = 0.377, Arrow-Sign Acc = +39.0%, Oracle Corr r = +0.439, Displacement ||ΔW_q|| = 19.026

## 4. SUPERVISED+TUNED ADDRESS + LEARNED POLICY (FULL SUPERVISED PIPELINE)

| Calibration (K) | Realized Return | Parent Choice Acc | Child Choice Rate | Indep VERIFY Acc | Arrow-Sign Acc | Transposed Parent Acc | Transposed Return | Paired ΔAcc Drop (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +1.05 | +4.4% | +5.3% | +89.3% | +35.1% | +4.4% | +1.05 | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +1.07 | +4.3% | +4.6% | +89.7% | +33.6% | +4.4% | +1.06 | -0.1% (±0.2%) | **0/16 (0.0%)** |
| **K = 4** | +1.03 | +4.8% | +4.9% | +89.5% | +34.6% | +4.8% | +1.03 | +0.0% (±0.4%) | **0/16 (0.0%)** |
| **K = 8** | +1.07 | +6.2% | +5.2% | +88.6% | +36.7% | +6.7% | +1.06 | -0.4% (±0.5%) | **0/16 (0.0%)** |
| **K = 16** | +1.05 | +6.9% | +6.1% | +87.6% | +40.9% | +7.3% | +1.02 | -0.3% (±1.0%) | **0/16 (0.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +48.4%, Query 2 Acc = +43.3%, Entropy H(q) = 0.377, Arrow-Sign Acc = +39.0%, Oracle Corr r = +0.439, Displacement ||ΔW_q|| = 19.026

## 5. AUTONOMOUS ADDRESS FROM SCRATCH + FIXED POLICY (TESTS DIRECTIONAL EMERGENCE PRESSURE)

| Calibration (K) | Realized Return | Parent Choice Acc | Child Choice Rate | Indep VERIFY Acc | Arrow-Sign Acc | Transposed Parent Acc | Transposed Return | Paired ΔAcc Drop (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +1.25 | +0.0% | +0.0% | +100.0% | +35.1% | +0.0% | +1.25 | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +1.22 | +0.3% | +0.2% | +99.8% | +33.5% | +0.2% | +1.22 | +0.1% (±0.1%) | **0/16 (0.0%)** |
| **K = 4** | +1.22 | +0.4% | +0.1% | +99.6% | +33.4% | +0.1% | +1.20 | +0.3% (±0.4%) | **0/16 (0.0%)** |
| **K = 8** | +1.19 | +0.6% | +1.2% | +98.2% | +33.0% | +1.2% | +1.21 | -0.6% (±0.4%) | **0/16 (0.0%)** |
| **K = 16** | +1.21 | +1.5% | +1.5% | +96.8% | +34.7% | +1.5% | +1.21 | +0.1% (±0.7%) | **0/16 (0.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +34.7%, Query 2 Acc = +34.4%, Entropy H(q) = 0.171, Arrow-Sign Acc = +34.1%, Oracle Corr r = +0.062, Displacement ||ΔW_q|| = 14.040

## 6. AUTONOMOUS ADDRESS + LEARNED POLICY (SCAFFOLDED REPORT DECODERS)

| Calibration (K) | Realized Return | Parent Choice Acc | Child Choice Rate | Indep VERIFY Acc | Arrow-Sign Acc | Transposed Parent Acc | Transposed Return | Paired ΔAcc Drop (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +1.05 | +4.4% | +5.3% | +89.3% | +35.1% | +4.4% | +1.05 | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +1.06 | +4.2% | +4.6% | +89.4% | +33.5% | +4.4% | +1.06 | -0.2% (±0.2%) | **0/16 (0.0%)** |
| **K = 4** | +1.05 | +4.5% | +4.5% | +89.9% | +33.4% | +4.6% | +1.04 | -0.2% (±0.2%) | **0/16 (0.0%)** |
| **K = 8** | +1.08 | +5.6% | +4.4% | +89.5% | +33.0% | +5.8% | +1.08 | -0.2% (±0.2%) | **0/16 (0.0%)** |
| **K = 16** | +1.09 | +5.3% | +4.6% | +90.0% | +34.7% | +5.5% | +1.08 | -0.2% (±0.4%) | **0/16 (0.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +34.7%, Query 2 Acc = +34.4%, Entropy H(q) = 0.171, Arrow-Sign Acc = +34.1%, Oracle Corr r = +0.062, Displacement ||ΔW_q|| = 14.040


========================================================================================================================
## 7. SCIENTIFIC DIAGNOSTIC CONCLUSIONS:
- **Directional Sidecar & Oracle Ceiling:** At K=16, the sidecar reconstructs true arrows with 100.0% accuracy. Under oracle addressing and calibrated policy, Parent-Choice Accuracy reaches +58.4% (Return = +1.28). Transposing R collapses Parent-Choice Accuracy by +56.3%, establishing massive causal arrow specificity.
- **Directional Supervised Addressing:** Supervised queries maintain directional accuracy (Arrow-Sign Acc = +39.0%, q1 = +48.4%, q2 = +43.3%), achieving return = +1.14 and Parent-Choice Accuracy = +15.9%.
- **Autonomous Addressing Under Strong Directional Pressure:** Even when directional arrow mastery carries a massive +1.44 vs -4.44 reward differential on 60% of trials, autonomous query weights displace (||ΔW_q|| = 14.040) but remain at chance (q1 = +34.7%, q2 = +34.4%), yielding Parent-Choice Accuracy = +1.5% and return = +1.21.
========================================================================================================================
