# Experiment E02b: Horizon 0 v2 Difficulty-Grid Mapping Report

**Run ID:** `run_e02b_diff_20260815_212030`  
**Target Model:** `qwen2.5:14b` (`7cdf5a0187d5...`)  
**Date:** 2026-08-15T21:52:04.974791+00:00  
**Response Format:** Matched 2-Alternative Forced Choice (2AFC, response_mode=`direct_value`)  
**Total Trials:** 192  

---

## 1. Executive Summary & Calibration Gate Status

| Sweep / Task Family | Monotonicity (Spearman $\rho$) | Kendall $\tau$ | Min Acc | Max Acc | Acc Span | Readiness Verdict | Calibration Gate Pass |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **Multi Hop** | -0.847 | -0.781 | 25.0% | 95.8% | 70.8% | **Staircase Ready** | No (None) |

---

## 2. Psychometric Curves by Task Family

### Sweep: Multi Hop

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Gate Pass | Mean Conf | Conf Sep | Brier | Prompt Tok |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | 24 | 23 | **95.8%** | [79.8%, 99.3%] | 54.2% | +2.97 [+2.38, +3.54] | -0.29 [-0.58, +0.00] | **FAIL** | 100.0% | +0.0% | 0.042 | 419 |
| `2` | 24 | 21 | **87.5%** | [69.0%, 95.7%] | 54.2% | +2.07 [+1.23, +3.54] | -0.16 [-0.69, +0.45] | **FAIL** | 100.0% | +0.0% | 0.125 | 448 |
| `3` | 24 | 15 | **62.5%** | [42.7%, 78.8%] | 45.8% | +0.59 [-0.39, +1.74] | +0.10 [-0.40, +0.63] | **FAIL** | 100.0% | +0.0% | 0.375 | 478 |
| `4` | 24 | 13 | **54.2%** | [35.1%, 72.1%] | 54.2% | +0.19 [-0.68, +1.27] | -0.10 [-0.62, +0.40] | **FAIL** | 100.0% | +0.0% | 0.458 | 509 |
| `5` | 24 | 11 | **45.8%** | [27.9%, 64.9%] | 79.2% | -0.25 [-1.57, +0.80] | -0.74 [-1.48, -0.31] | **FAIL** | 100.0% | +0.0% | 0.542 | 535 |
| `6` | 24 | 6 | **25.0%** | [12.0%, 44.9%] | 66.7% | -1.39 [-2.64, -0.47] | -0.50 [-0.98, +0.00] | **FAIL** | 100.0% | +0.0% | 0.750 | 566 |
| `7` | 24 | 13 | **54.2%** | [35.1%, 72.1%] | 70.8% | +0.22 [-0.80, +1.39] | -0.51 [-1.08, -0.10] | **FAIL** | 100.0% | +0.0% | 0.458 | 595 |

---

## 3. Elicitation Policy Reactivity Analysis

- **Paired Trials Evaluated:** 24
- **Answer-Only Accuracy:** 95.8%
- **Answer+Confidence Accuracy:** 95.8%
- **Delta Accuracy (Confidence - Only):** +0.0%
- **Exact Option Concordance Rate:** 100.0%
- **McNemar Chi2 Statistic:** 0.000 (Asymptotic p = 1.0000)
- **Exact Binomial McNemar Test:** p = 1.0000
- **Reactivity Verdict:** `negligible_reactivity`
- **Policy Reactivity Note:** Exact p < 0.05 or Concordance < 85% demonstrates causal perturbation of the first-order choice policy by confidence elicitation.

---

## 4. Scientific Takeaways for Adaptive Calibration

1. **Monotonicity Assessment:** A difficulty axis is staircase-ready if it exhibits a consistent negative rank correlation (rho <= -0.70, negative_step_ratio >= 0.70) and spans the operational target window (~55-90%).
2. **Response Bias (Criterion $c$):** Positive $c$ indicates an Option-B selection bias under Signal=A conventions. Extreme criterion shifts ($|c| > 1.0$) indicate response collapse rather than sensitivity loss.
3. **Within-Item Transitions:** High rebound rates ($0 \to 1$) indicate order/placement sensitivity or item noise, whereas high degradation ($1 \to 0$) confirms true capacity degradation.