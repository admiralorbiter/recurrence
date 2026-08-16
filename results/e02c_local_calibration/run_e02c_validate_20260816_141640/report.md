# Experiment E02c: Horizon 0 v2.4 Local Calibration & Validation Report

**Run ID:** `run_e02c_validate_20260816_141640`  
**Target Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Date:** 2026-08-16T14:19:35.362317+00:00  
**Mode:** `validate`  
**Response Format:** Matched 2-Alternative Forced Choice (Direct-Value Candidate Strings)  
**Total Trials:** 64  

---

## 1. Executive Summary & Calibration Gate Status

| Coordinate `(H, D)` | Trials | Correct | Accuracy | 95% Wilson CI | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Gate Status | Meta-$d'$ Status | AUROC2 | Brier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `H1_D4` | 64 | 45 | **70.3%** | [58.2%, 80.1%] | +1.25 [+0.65, +2.05] | -0.62 [-1.05, -0.32] | FAIL | `eligible_for_fit` | 0.477 | 0.377 |

---

## 2. Detailed Diagnostics by Coordinate Cell

### Cell: `H1_D4` (H=1, D=4)
- **Trials:** 64 | **Accuracy:** 70.3% (45/64)
- **Type-1 Sensitivity ($d'$):** +1.25 (95% CI: [+0.65, +2.05])
- **Type-1 Criterion ($c$):** -0.62 (95% CI: [-1.05, -0.32])
- **Option 1 Primacy Selection Rate:** 70.3%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): PASS
  - Response Bias Gate ($|c| \le 0.50$): FAIL
  - Accuracy Gate ($60\% \le Acc \le 80\%$): PASS
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `eligible_for_fit`
  - Mean Confidence: 54.5%
  - Confidence Separation (Correct - Incorrect): -2.0%
  - AUROC2: 0.47719298245614034
  - Brier Score: 0.37684375000000003
