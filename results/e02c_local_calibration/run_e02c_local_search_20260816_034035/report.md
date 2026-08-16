# Experiment E02c: Horizon 0 v2.4 Local Calibration & Validation Report

**Run ID:** `run_e02c_local_search_20260816_034035`  
**Target Model:** `qwen2.5:14b` (`7cdf5a0187d5...`)  
**Date:** 2026-08-16T04:01:47.192040+00:00  
**Mode:** `local_search`  
**Response Format:** Matched 2-Alternative Forced Choice (Direct-Value Candidate Strings)  
**Total Trials:** 120  

---

## 1. Executive Summary & Calibration Gate Status

| Coordinate `(H, D)` | Trials | Correct | Accuracy | 95% Wilson CI | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Gate Status | Meta-$d'$ Status | AUROC2 | Brier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `H2_D32` | 24 | 22 | **91.7%** | [74.2%, 97.7%] | +2.40 [+1.59, +3.54] | -0.00 [-0.58, +0.58] | FAIL | `confidence_degenerate` | 0.500 | 0.083 |
| `H2_D64` | 24 | 18 | **75.0%** | [55.1%, 88.0%] | +1.39 [+0.47, +2.38] | -0.50 [-0.98, -0.05] | FAIL | `confidence_degenerate` | 0.500 | 0.250 |
| `H3_D4` | 24 | 13 | **54.2%** | [35.1%, 72.1%] | +0.19 [-0.74, +1.23] | -0.10 [-0.63, +0.40] | FAIL | `confidence_degenerate` | 0.500 | 0.458 |
| `H3_D8` | 24 | 15 | **62.5%** | [42.7%, 78.8%] | +0.62 [-0.25, +1.81] | -0.31 [-0.91, +0.11] | FAIL | `confidence_degenerate` | 0.500 | 0.375 |
| `H3_D16` | 24 | 19 | **79.2%** | [59.5%, 90.8%] | +1.48 [+0.59, +2.97] | -0.13 [-0.69, +0.44] | FAIL | `confidence_degenerate` | 0.500 | 0.208 |

---

## 2. Detailed Diagnostics by Coordinate Cell

### Cell: `H2_D32` (H=2, D=32)
- **Trials:** 24 | **Accuracy:** 91.7% (22/24)
- **Type-1 Sensitivity ($d'$):** +2.40 (95% CI: [+1.59, +3.54])
- **Type-1 Criterion ($c$):** -0.00 (95% CI: [-0.58, +0.58])
- **Option 1 Primacy Selection Rate:** 50.0%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): FAIL
  - Response Bias Gate ($|c| \le 0.50$): PASS
  - Accuracy Gate ($60\% \le Acc \le 80\%$): FAIL
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `confidence_degenerate`
  - Mean Confidence: 100.0%
  - Confidence Separation: N/A
  - AUROC2: 0.5
  - Brier Score: 0.08333333333333333

### Cell: `H2_D64` (H=2, D=64)
- **Trials:** 24 | **Accuracy:** 75.0% (18/24)
- **Type-1 Sensitivity ($d'$):** +1.39 (95% CI: [+0.47, +2.38])
- **Type-1 Criterion ($c$):** -0.50 (95% CI: [-0.98, -0.05])
- **Option 1 Primacy Selection Rate:** 66.7%
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
  - Brier Score: 0.25

### Cell: `H3_D4` (H=3, D=4)
- **Trials:** 24 | **Accuracy:** 54.2% (13/24)
- **Type-1 Sensitivity ($d'$):** +0.19 (95% CI: [-0.74, +1.23])
- **Type-1 Criterion ($c$):** -0.10 (95% CI: [-0.63, +0.40])
- **Option 1 Primacy Selection Rate:** 54.2%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): FAIL
  - Response Bias Gate ($|c| \le 0.50$): PASS
  - Accuracy Gate ($60\% \le Acc \le 80\%$): FAIL
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `confidence_degenerate`
  - Mean Confidence: 100.0%
  - Confidence Separation: N/A
  - AUROC2: 0.5
  - Brier Score: 0.4583333333333333

### Cell: `H3_D8` (H=3, D=8)
- **Trials:** 24 | **Accuracy:** 62.5% (15/24)
- **Type-1 Sensitivity ($d'$):** +0.62 (95% CI: [-0.25, +1.81])
- **Type-1 Criterion ($c$):** -0.31 (95% CI: [-0.91, +0.11])
- **Option 1 Primacy Selection Rate:** 62.5%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): FAIL
  - Response Bias Gate ($|c| \le 0.50$): PASS
  - Accuracy Gate ($60\% \le Acc \le 80\%$): PASS
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `confidence_degenerate`
  - Mean Confidence: 100.0%
  - Confidence Separation: N/A
  - AUROC2: 0.5
  - Brier Score: 0.375

### Cell: `H3_D16` (H=3, D=16)
- **Trials:** 24 | **Accuracy:** 79.2% (19/24)
- **Type-1 Sensitivity ($d'$):** +1.48 (95% CI: [+0.59, +2.97])
- **Type-1 Criterion ($c$):** -0.13 (95% CI: [-0.69, +0.44])
- **Option 1 Primacy Selection Rate:** 54.2%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): FAIL
  - Response Bias Gate ($|c| \le 0.50$): PASS
  - Accuracy Gate ($60\% \le Acc \le 80\%$): PASS
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `confidence_degenerate`
  - Mean Confidence: 100.0%
  - Confidence Separation: N/A
  - AUROC2: 0.5
  - Brier Score: 0.20833333333333334
