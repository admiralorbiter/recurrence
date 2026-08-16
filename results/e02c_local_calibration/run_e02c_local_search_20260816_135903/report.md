# Experiment E02c: Horizon 0 v2.4 Local Calibration & Validation Report

**Run ID:** `run_e02c_local_search_20260816_135903`  
**Target Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Date:** 2026-08-16T14:02:14.775610+00:00  
**Mode:** `local_search`  
**Response Format:** Matched 2-Alternative Forced Choice (Direct-Value Candidate Strings)  
**Total Trials:** 72  

---

## 1. Executive Summary & Calibration Gate Status

| Coordinate `(H, D)` | Trials | Correct | Accuracy | 95% Wilson CI | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Gate Status | Meta-$d'$ Status | AUROC2 | Brier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `H1_D8` | 24 | 16 | **66.7%** | [46.7%, 82.0%] | +0.81 [-0.20, +1.81] | -0.21 [-0.80, +0.29] | FAIL | `eligible_for_fit` | 0.445 | 0.379 |
| `H1_D12` | 24 | 14 | **58.3%** | [38.8%, 75.5%] | +0.47 [-0.29, +1.57] | -0.63 [-1.32, -0.20] | FAIL | `eligible_for_fit` | 0.239 | 0.538 |
| `H1_D16` | 24 | 17 | **70.8%** | [50.8%, 85.1%] | +1.20 [+0.22, +2.38] | -0.60 [-1.08, -0.11] | FAIL | `eligible_for_fit` | 0.450 | 0.366 |

---

## 2. Detailed Diagnostics by Coordinate Cell

### Cell: `H1_D8` (H=1, D=8)
- **Trials:** 24 | **Accuracy:** 66.7% (16/24)
- **Type-1 Sensitivity ($d'$):** +0.81 (95% CI: [-0.20, +1.81])
- **Type-1 Criterion ($c$):** -0.21 (95% CI: [-0.80, +0.29])
- **Option 1 Primacy Selection Rate:** 58.3%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): FAIL
  - Response Bias Gate ($|c| \le 0.50$): PASS
  - Accuracy Gate ($60\% \le Acc \le 80\%$): PASS
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `eligible_for_fit`
  - Mean Confidence: 68.0%
  - Confidence Separation (Correct - Incorrect): -3.0%
  - AUROC2: 0.4453125
  - Brier Score: 0.378925

### Cell: `H1_D12` (H=1, D=12)
- **Trials:** 24 | **Accuracy:** 58.3% (14/24)
- **Type-1 Sensitivity ($d'$):** +0.47 (95% CI: [-0.29, +1.57])
- **Type-1 Criterion ($c$):** -0.63 (95% CI: [-1.32, -0.20])
- **Option 1 Primacy Selection Rate:** 75.0%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): FAIL
  - Response Bias Gate ($|c| \le 0.50$): FAIL
  - Accuracy Gate ($60\% \le Acc \le 80\%$): FAIL
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `eligible_for_fit`
  - Mean Confidence: 81.3%
  - Confidence Separation (Correct - Incorrect): -29.4%
  - AUROC2: 0.2392857142857143
  - Brier Score: 0.5378

### Cell: `H1_D16` (H=1, D=16)
- **Trials:** 24 | **Accuracy:** 70.8% (17/24)
- **Type-1 Sensitivity ($d'$):** +1.20 (95% CI: [+0.22, +2.38])
- **Type-1 Criterion ($c$):** -0.60 (95% CI: [-1.08, -0.11])
- **Option 1 Primacy Selection Rate:** 70.8%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): PASS
  - Response Bias Gate ($|c| \le 0.50$): FAIL
  - Accuracy Gate ($60\% \le Acc \le 80\%$): PASS
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `eligible_for_fit`
  - Mean Confidence: 63.8%
  - Confidence Separation (Correct - Incorrect): -4.7%
  - AUROC2: 0.4495798319327731
  - Brier Score: 0.36625833333333335
