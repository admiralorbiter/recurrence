# Experiment E02c: Horizon 0 v2.4 Local Calibration & Validation Report

**Run ID:** `run_e02c_validate_20260816_033733`  
**Target Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Date:** 2026-08-16T03:40:26.680752+00:00  
**Mode:** `validate`  
**Response Format:** Matched 2-Alternative Forced Choice (Direct-Value Candidate Strings)  
**Total Trials:** 64  

---

## 1. Executive Summary & Calibration Gate Status

| Coordinate `(H, D)` | Trials | Correct | Accuracy | 95% Wilson CI | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Gate Status | Meta-$d'$ Status | AUROC2 | Brier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `H1_D16` | 64 | 41 | **64.1%** | [51.8%, 74.7%] | +0.75 [+0.08, +1.41] | -0.37 [-0.72, -0.08] | FAIL | `estimable` | 0.366 | 0.442 |

---

## 2. Detailed Diagnostics by Coordinate Cell

### Cell: `H1_D16` (H=1, D=16)
- **Trials:** 64 | **Accuracy:** 64.1% (41/64)
- **Type-1 Sensitivity ($d'$):** +0.75 (95% CI: [+0.08, +1.41])
- **Type-1 Criterion ($c$):** -0.37 (95% CI: [-0.72, -0.08])
- **Option 1 Primacy Selection Rate:** 64.1%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): FAIL
  - Response Bias Gate ($|c| \le 0.50$): PASS
  - Accuracy Gate ($60\% \le Acc \le 80\%$): PASS
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `estimable`
  - Mean Confidence: 69.2%
  - Confidence Separation (Correct - Incorrect): -17.6%
  - AUROC2: 0.36585365853658536
  - Brier Score: 0.4416890625
