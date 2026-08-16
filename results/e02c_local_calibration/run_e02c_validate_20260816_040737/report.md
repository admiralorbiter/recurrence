# Experiment E02c: Horizon 0 v2.4 Local Calibration & Validation Report

**Run ID:** `run_e02c_validate_20260816_040737`  
**Target Model:** `qwen2.5:14b` (`7cdf5a0187d5...`)  
**Date:** 2026-08-16T04:18:47.071709+00:00  
**Mode:** `validate`  
**Response Format:** Matched 2-Alternative Forced Choice (Direct-Value Candidate Strings)  
**Total Trials:** 64  

---

## 1. Executive Summary & Calibration Gate Status

| Coordinate `(H, D)` | Trials | Correct | Accuracy | 95% Wilson CI | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Gate Status | Meta-$d'$ Status | AUROC2 | Brier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `H2_D64` | 64 | 57 | **89.1%** | [79.1%, 94.6%] | +2.34 [+1.60, +3.40] | -0.08 [-0.44, +0.30] | FAIL | `confidence_degenerate` | 0.500 | 0.109 |

---

## 2. Detailed Diagnostics by Coordinate Cell

### Cell: `H2_D64` (H=2, D=64)
- **Trials:** 64 | **Accuracy:** 89.1% (57/64)
- **Type-1 Sensitivity ($d'$):** +2.34 (95% CI: [+1.60, +3.40])
- **Type-1 Criterion ($c$):** -0.08 (95% CI: [-0.44, +0.30])
- **Option 1 Primacy Selection Rate:** 51.6%
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
  - Brier Score: 0.109375
