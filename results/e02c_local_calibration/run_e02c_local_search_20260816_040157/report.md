# Experiment E02c: Horizon 0 v2.4 Local Calibration & Validation Report

**Run ID:** `run_e02c_local_search_20260816_040157`  
**Target Model:** `llama3.2:3b` (`a80c4f17acd5...`)  
**Date:** 2026-08-16T04:07:23.851923+00:00  
**Mode:** `local_search`  
**Response Format:** Matched 2-Alternative Forced Choice (Direct-Value Candidate Strings)  
**Total Trials:** 120  

---

## 1. Executive Summary & Calibration Gate Status

| Coordinate `(H, D)` | Trials | Correct | Accuracy | 95% Wilson CI | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Gate Status | Meta-$d'$ Status | AUROC2 | Brier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `H1_D16` | 24 | 18 | **75.0%** | [55.1%, 88.0%] | +1.77 [+1.15, +2.38] | -0.88 [-1.19, -0.58] | FAIL | `estimable` | 0.583 | 0.208 |
| `H1_D32` | 24 | 20 | **83.3%** | [64.1%, 93.3%] | +1.81 [+1.01, +2.97] | -0.29 [-0.79, +0.29] | FAIL | `estimable` | 0.600 | 0.167 |
| `H2_D4` | 24 | 15 | **62.5%** | [42.7%, 78.8%] | +0.80 [-0.12, +1.77] | -0.80 [-1.32, -0.40] | FAIL | `confidence_degenerate` | 0.500 | 0.375 |
| `H2_D8` | 24 | 16 | **66.7%** | [46.7%, 82.0%] | +1.37 [+0.57, +1.96] | -1.08 [-1.48, -0.79] | FAIL | `confidence_degenerate` | 0.500 | 0.333 |
| `H2_D16` | 24 | 14 | **58.3%** | [38.8%, 75.5%] | +0.47 [-0.47, +1.57] | -0.63 [-1.20, -0.19] | FAIL | `estimable` | 0.464 | 0.458 |

---

## 2. Detailed Diagnostics by Coordinate Cell

### Cell: `H1_D16` (H=1, D=16)
- **Trials:** 24 | **Accuracy:** 75.0% (18/24)
- **Type-1 Sensitivity ($d'$):** +1.77 (95% CI: [+1.15, +2.38])
- **Type-1 Criterion ($c$):** -0.88 (95% CI: [-1.19, -0.58])
- **Option 1 Primacy Selection Rate:** 75.0%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): FAIL
  - Response Bias Gate ($|c| \le 0.50$): FAIL
  - Accuracy Gate ($60\% \le Acc \le 80\%$): PASS
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `estimable`
  - Mean Confidence: 95.8%
  - Confidence Separation (Correct - Incorrect): +16.7%
  - AUROC2: 0.5833333333333334
  - Brier Score: 0.20833333333333334

### Cell: `H1_D32` (H=1, D=32)
- **Trials:** 24 | **Accuracy:** 83.3% (20/24)
- **Type-1 Sensitivity ($d'$):** +1.81 (95% CI: [+1.01, +2.97])
- **Type-1 Criterion ($c$):** -0.29 (95% CI: [-0.79, +0.29])
- **Option 1 Primacy Selection Rate:** 58.3%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): FAIL
  - Response Bias Gate ($|c| \le 0.50$): PASS
  - Accuracy Gate ($60\% \le Acc \le 80\%$): FAIL
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `estimable`
  - Mean Confidence: 91.7%
  - Confidence Separation (Correct - Incorrect): +20.0%
  - AUROC2: 0.6
  - Brier Score: 0.16666666666666666

### Cell: `H2_D4` (H=2, D=4)
- **Trials:** 24 | **Accuracy:** 62.5% (15/24)
- **Type-1 Sensitivity ($d'$):** +0.80 (95% CI: [-0.12, +1.77])
- **Type-1 Criterion ($c$):** -0.80 (95% CI: [-1.32, -0.40])
- **Option 1 Primacy Selection Rate:** 79.2%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): FAIL
  - Response Bias Gate ($|c| \le 0.50$): FAIL
  - Accuracy Gate ($60\% \le Acc \le 80\%$): PASS
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `confidence_degenerate`
  - Mean Confidence: 100.0%
  - Confidence Separation: N/A
  - AUROC2: 0.5
  - Brier Score: 0.375

### Cell: `H2_D8` (H=2, D=8)
- **Trials:** 24 | **Accuracy:** 66.7% (16/24)
- **Type-1 Sensitivity ($d'$):** +1.37 (95% CI: [+0.57, +1.96])
- **Type-1 Criterion ($c$):** -1.08 (95% CI: [-1.48, -0.79])
- **Option 1 Primacy Selection Rate:** 83.3%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): PASS
  - Response Bias Gate ($|c| \le 0.50$): FAIL
  - Accuracy Gate ($60\% \le Acc \le 80\%$): PASS
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `confidence_degenerate`
  - Mean Confidence: 100.0%
  - Confidence Separation: N/A
  - AUROC2: 0.5
  - Brier Score: 0.3333333333333333

### Cell: `H2_D16` (H=2, D=16)
- **Trials:** 24 | **Accuracy:** 58.3% (14/24)
- **Type-1 Sensitivity ($d'$):** +0.47 (95% CI: [-0.47, +1.57])
- **Type-1 Criterion ($c$):** -0.63 (95% CI: [-1.20, -0.19])
- **Option 1 Primacy Selection Rate:** 75.0%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): FAIL
  - Response Bias Gate ($|c| \le 0.50$): FAIL
  - Accuracy Gate ($60\% \le Acc \le 80\%$): FAIL
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `estimable`
  - Mean Confidence: 95.8%
  - Confidence Separation (Correct - Incorrect): -7.1%
  - AUROC2: 0.4642857142857143
  - Brier Score: 0.4583333333333333
