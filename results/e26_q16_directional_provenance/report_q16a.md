# Q16a: Minimal Directional Provenance & Anti-Symmetric Addressing Synthesis Report

========================================================================================================================
Q16a DIRECTIONAL SYNTHESIS REPORT (16 SEEDS, RUNTIME: 13.1574137s)
1. Theoretical Perfect-Information Bayes Oracle: Expected Return = +1.40, Expected DDI = +100.0%
2. K=16 Empirical-R Teacher Benchmark          : Expected Return = +1.40, Expected DDI = +40.1%
========================================================================================================================
## 1. ORACLE DIRECTIONAL ADDRESS + FIXED CALIBRATED POLICY (ISOLATES ECONOMIC CEILING)

| Calibration (K) | Intact DDI % | Realized Return | Forward Acc | Backward Acc | Indep Acc | Transposed R DDI | Permuted R DDI | Paired ΔTrans (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.46 | +100.0% | +100.0% | +100.0% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +0.7% | +1.41 | +100.0% | +100.0% | +100.0% | +0.7% | -0.3% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 4** | +5.4% | +1.44 | +100.0% | +100.0% | +100.0% | +5.4% | +1.9% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 8** | +16.7% | +1.40 | +100.0% | +100.0% | +100.0% | +16.7% | -1.2% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 16** | +37.7% | +1.41 | +100.0% | +100.0% | +100.0% | +37.7% | -1.6% | +0.0% (±0.0%) | **0/16 (0.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +100.0%, Query 2 Acc = +100.0%, Entropy H(q) = 0.978, Score Correlation r = +0.982, Displacement ||ΔW_q|| = 0.000

## 2. ORACLE DIRECTIONAL ADDRESS + LEARNED POLICY (VALIDATES DIRECTIONAL DECISION MAPPING)

| Calibration (K) | Intact DDI % | Realized Return | Forward Acc | Backward Acc | Indep Acc | Transposed R DDI | Permuted R DDI | Paired ΔTrans (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.45 | +100.0% | +99.7% | +99.3% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +0.1% | +1.39 | +97.9% | +99.1% | +98.6% | +0.3% | +0.1% | -0.3% (±0.3%) | **0/16 (0.0%)** |
| **K = 4** | +0.6% | +1.43 | +93.9% | +93.0% | +97.6% | +0.5% | +0.3% | +0.1% (±0.3%) | **0/16 (0.0%)** |
| **K = 8** | +2.1% | +1.37 | +80.5% | +81.4% | +95.4% | +0.8% | +0.0% | +1.3% (±0.5%) | **0/16 (0.0%)** |
| **K = 16** | +2.5% | +1.47 | +53.3% | +57.9% | +89.4% | +3.6% | +0.1% | -1.1% (±0.9%) | **0/16 (0.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +100.0%, Query 2 Acc = +100.0%, Entropy H(q) = 0.978, Score Correlation r = +0.982, Displacement ||ΔW_q|| = 0.000

## 3. SUPERVISED+TUNED ADDRESS + FIXED POLICY (DIFFERENTIABLE DIRECTIONAL SURROGATE)

| Calibration (K) | Intact DDI % | Realized Return | Forward Acc | Backward Acc | Indep Acc | Transposed R DDI | Permuted R DDI | Paired ΔTrans (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.46 | +100.0% | +100.0% | +100.0% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | -0.1% | +1.40 | +98.7% | +99.3% | +98.9% | -0.1% | +0.2% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 4** | +1.4% | +1.44 | +95.4% | +94.6% | +98.1% | +1.4% | +0.5% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 8** | +2.6% | +1.35 | +86.2% | +84.1% | +91.1% | +2.6% | +0.8% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 16** | +4.5% | +1.39 | +73.0% | +72.9% | +81.6% | +4.5% | +1.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +36.2%, Query 2 Acc = +39.7%, Entropy H(q) = 0.242, Score Correlation r = +0.187, Displacement ||ΔW_q|| = 27.862

## 4. SUPERVISED+TUNED ADDRESS + LEARNED POLICY (FULL SUPERVISED PIPELINE)

| Calibration (K) | Intact DDI % | Realized Return | Forward Acc | Backward Acc | Indep Acc | Transposed R DDI | Permuted R DDI | Paired ΔTrans (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.44 | +99.6% | +99.4% | +98.6% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +0.1% | +1.39 | +97.2% | +98.3% | +98.1% | +0.1% | +0.1% | +0.0% (±0.1%) | **0/16 (0.0%)** |
| **K = 4** | -0.0% | +1.44 | +93.1% | +91.9% | +97.1% | -0.2% | -0.2% | +0.2% (±0.2%) | **0/16 (0.0%)** |
| **K = 8** | -0.7% | +1.38 | +79.3% | +78.6% | +94.3% | +0.1% | -0.6% | -0.8% (±0.6%) | **0/16 (0.0%)** |
| **K = 16** | +0.2% | +1.45 | +52.3% | +54.1% | +88.2% | +0.5% | -0.0% | -0.3% (±0.7%) | **0/16 (0.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +36.2%, Query 2 Acc = +39.7%, Entropy H(q) = 0.242, Score Correlation r = +0.187, Displacement ||ΔW_q|| = 27.862

## 5. AUTONOMOUS ADDRESS FROM SCRATCH + FIXED POLICY (TESTS DIRECTIONAL EMERGENCE PRESSURE)

| Calibration (K) | Intact DDI % | Realized Return | Forward Acc | Backward Acc | Indep Acc | Transposed R DDI | Permuted R DDI | Paired ΔTrans (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.46 | +100.0% | +100.0% | +100.0% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | -0.7% | +1.41 | +98.1% | +99.1% | +98.9% | -0.7% | +0.2% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 4** | +0.0% | +1.44 | +93.8% | +93.3% | +98.1% | +0.0% | +0.3% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 8** | -1.6% | +1.38 | +82.4% | +83.0% | +90.6% | -1.6% | -1.4% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 16** | -0.8% | +1.37 | +66.6% | +66.5% | +80.3% | -0.8% | -1.3% | +0.0% (±0.0%) | **0/16 (0.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +33.3%, Query 2 Acc = +33.4%, Entropy H(q) = 0.182, Score Correlation r = -0.048, Displacement ||ΔW_q|| = 21.452

## 6. AUTONOMOUS ADDRESS + LEARNED POLICY (SCAFFOLDED REPORT DECODERS)

| Calibration (K) | Intact DDI % | Realized Return | Forward Acc | Backward Acc | Indep Acc | Transposed R DDI | Permuted R DDI | Paired ΔTrans (±STE) | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | +0.0% | +1.45 | +99.6% | +99.7% | +99.2% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 2** | +0.0% | +1.39 | +97.6% | +98.5% | +97.6% | +0.0% | +0.0% | +0.0% (±0.0%) | **0/16 (0.0%)** |
| **K = 4** | -0.1% | +1.43 | +93.2% | +92.0% | +97.0% | +0.1% | +0.0% | -0.2% (±0.1%) | **0/16 (0.0%)** |
| **K = 8** | -0.8% | +1.37 | +79.2% | +78.0% | +93.5% | +0.2% | -0.9% | -1.0% (±0.9%) | **0/16 (0.0%)** |
| **K = 16** | +0.1% | +1.46 | +52.1% | +52.1% | +87.2% | -0.6% | -0.3% | +0.8% (±1.1%) | **0/16 (0.0%)** |

**Addressing Quality Metrics (K=16):** Query 1 Acc = +33.3%, Query 2 Acc = +33.4%, Entropy H(q) = 0.182, Score Correlation r = -0.048, Displacement ||ΔW_q|| = 21.452


========================================================================================================================
## 7. SCIENTIFIC DIAGNOSTIC CONCLUSIONS:
- **Directional Relational Ceiling:** Under anti-symmetric relational matrix R (where R^T = -R), oracle fixed addressing achieves return = +1.41 and DDI = +37.7%, and oracle learned policy achieves return = +1.47 and DDI = +2.5%.
- **Directional Supervised Addressing:** Supervised queries maintain directional correlation (r = +0.187, q1 = +36.2%, q2 = +39.7%), achieving return = +1.39 and DDI = +4.5%.
- **Autonomous Directional Recruitment Status:** Under asymmetric environmental pressure where arrow directionality is consequential, autonomous query weights displace (||ΔW_q|| = 21.452), yielding q1 = +33.3%, q2 = +33.4%, DDI = -0.8%, and return = +1.37.
========================================================================================================================
