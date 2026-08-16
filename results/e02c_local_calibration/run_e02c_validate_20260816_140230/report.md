# Experiment E02c: Horizon 0 v2.4 Local Calibration & Validation Report

**Run ID:** `run_e02c_validate_20260816_140230`  
**Target Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Date:** 2026-08-16T14:05:21.334482+00:00  
**Mode:** `validate`  
**Response Format:** Matched 2-Alternative Forced Choice (Direct-Value Candidate Strings)  
**Total Trials:** 64  

---

## 1. Executive Summary & Calibration Gate Status

| Coordinate `(H, D)` | Trials | Correct | Accuracy | 95% Wilson CI | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Gate Status | Meta-$d'$ Status | AUROC2 | Brier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `H1_D8` | 64 | 43 | **67.2%** | [55.0%, 77.4%] | +0.88 [+0.25, +1.56] | -0.21 [-0.53, +0.09] | FAIL | `eligible_for_fit` | 0.445 | 0.365 |

---

## 2. Detailed Diagnostics by Coordinate Cell

### Cell: `H1_D8` (H=1, D=8)
- **Trials:** 64 | **Accuracy:** 67.2% (43/64)
- **Type-1 Sensitivity ($d'$):** +0.88 (95% CI: [+0.25, +1.56])
- **Type-1 Criterion ($c$):** -0.21 (95% CI: [-0.53, +0.09])
- **Option 1 Primacy Selection Rate:** 57.8%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): FAIL
  - Response Bias Gate ($|c| \le 0.50$): PASS
  - Accuracy Gate ($60\% \le Acc \le 80\%$): PASS
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `eligible_for_fit`
  - Mean Confidence: 79.7%
  - Confidence Separation (Correct - Incorrect): -7.6%
  - AUROC2: 0.44462901439645625
  - Brier Score: 0.364859375
