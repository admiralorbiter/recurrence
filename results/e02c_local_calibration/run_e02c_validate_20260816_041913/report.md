# Experiment E02c: Horizon 0 v2.4 Local Calibration & Validation Report

**Run ID:** `run_e02c_validate_20260816_041913`  
**Target Model:** `qwen2.5:14b` (`7cdf5a0187d5...`)  
**Date:** 2026-08-16T04:28:48.419993+00:00  
**Mode:** `validate`  
**Response Format:** Matched 2-Alternative Forced Choice (Direct-Value Candidate Strings)  
**Total Trials:** 64  

---

## 1. Executive Summary & Calibration Gate Status

| Coordinate `(H, D)` | Trials | Correct | Accuracy | 95% Wilson CI | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Gate Status | Meta-$d'$ Status | AUROC2 | Brier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `H3_D16` | 64 | 45 | **70.3%** | [58.2%, 80.1%] | +1.03 [+0.47, +1.70] | -0.04 [-0.36, +0.27] | **PASS** | `confidence_degenerate` | 0.500 | 0.297 |

---

## 2. Detailed Diagnostics by Coordinate Cell

### Cell: `H3_D16` (H=3, D=16)
- **Trials:** 64 | **Accuracy:** 70.3% (45/64)
- **Type-1 Sensitivity ($d'$):** +1.03 (95% CI: [+0.47, +1.70])
- **Type-1 Criterion ($c$):** -0.04 (95% CI: [-0.36, +0.27])
- **Option 1 Primacy Selection Rate:** 51.6%
- **Calibration Gate Evaluation:**
  - Sensitivity Gate ($d' \in [0.9, 1.4]$): PASS
  - Response Bias Gate ($|c| \le 0.50$): PASS
  - Accuracy Gate ($60\% \le Acc \le 80\%$): PASS
  - Schema Compliance Gate ($\ge 95\%$): PASS
- **Type-2 Metacognitive Metrics:**
  - Status: `confidence_degenerate`
  - Mean Confidence: 100.0%
  - Confidence Separation: N/A
  - AUROC2: 0.5
  - Brier Score: 0.296875
