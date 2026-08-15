# Experiment E02b: Horizon 0 v2 Difficulty-Grid Mapping Report

**Run ID:** `run_e02b_diff_20260815_211153`  
**Target Model:** `llama3.2:3b` (`a80c4f17acd5...`)  
**Date:** 2026-08-15T21:20:22.755202+00:00  
**Response Format:** Matched 2-Alternative Forced Choice (2AFC, response_mode=`direct_value`)  
**Total Trials:** 192  

---

## 1. Executive Summary & Calibration Gate Status

| Sweep / Task Family | Monotonicity (Spearman $\rho$) | Kendall $\tau$ | Min Acc | Max Acc | Acc Span | Readiness Verdict | Calibration Gate Pass |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **Multi Hop** | -0.727 | -0.651 | 33.3% | 75.0% | 41.7% | **Staircase Ready** | No (None) |

---

## 2. Psychometric Curves by Task Family

### Sweep: Multi Hop

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Gate Pass | Mean Conf | Conf Sep | Brier | Prompt Tok |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | 24 | 18 | **75.0%** | [55.1%, 88.0%] | 75.0% | +1.77 [+1.15, +2.38] | -0.88 [-1.19, -0.58] | **FAIL** | 89.6% | -13.9% | 0.344 | 410 |
| `2` | 24 | 14 | **58.3%** | [38.8%, 75.5%] | 75.0% | +0.47 [-0.57, +1.57] | -0.63 [-1.20, -0.19] | **FAIL** | 100.0% | +0.0% | 0.417 | 438 |
| `3` | 24 | 12 | **50.0%** | [31.4%, 68.6%] | 75.0% | +0.00 [-1.06, +1.01] | -0.62 [-1.32, -0.19] | **FAIL** | 100.0% | +0.0% | 0.500 | 469 |
| `4` | 24 | 11 | **45.8%** | [27.9%, 64.9%] | 79.2% | -0.25 [-1.37, +0.90] | -0.74 [-1.32, -0.29] | **FAIL** | 100.0% | +0.0% | 0.542 | 500 |
| `5` | 24 | 14 | **58.3%** | [38.8%, 75.5%] | 66.7% | +0.42 [-0.48, +1.58] | -0.40 [-0.98, +0.00] | **FAIL** | 95.8% | +10.0% | 0.375 | 526 |
| `6` | 24 | 12 | **50.0%** | [31.4%, 68.6%] | 75.0% | +0.00 [-1.20, +1.20] | -0.62 [-1.32, -0.10] | **FAIL** | 94.8% | -2.1% | 0.492 | 557 |
| `7` | 24 | 8 | **33.3%** | [18.0%, 53.3%] | 58.3% | -0.81 [-1.96, +0.19] | -0.21 [-0.80, +0.29] | **FAIL** | 95.8% | -12.5% | 0.708 | 586 |

---

## 3. Elicitation Policy Reactivity Analysis

- **Paired Trials Evaluated:** 24
- **Answer-Only Accuracy:** 62.5%
- **Answer+Confidence Accuracy:** 75.0%
- **Delta Accuracy (Confidence - Only):** +12.5%
- **Exact Option Concordance Rate:** 87.5%
- **McNemar Chi2 Statistic:** 1.333 (Asymptotic p = 0.2482)
- **Exact Binomial McNemar Test:** p = 0.2500
- **Reactivity Verdict:** `severe_reactivity`
- **Policy Reactivity Note:** Exact p < 0.05 or Concordance < 85% demonstrates causal perturbation of the first-order choice policy by confidence elicitation.

---

## 4. Scientific Takeaways for Adaptive Calibration

1. **Monotonicity Assessment:** A difficulty axis is staircase-ready if it exhibits a consistent negative rank correlation (rho <= -0.70, negative_step_ratio >= 0.70) and spans the operational target window (~55-90%).
2. **Response Bias (Criterion $c$):** Positive $c$ indicates an Option-B selection bias under Signal=A conventions. Extreme criterion shifts ($|c| > 1.0$) indicate response collapse rather than sensitivity loss.
3. **Within-Item Transitions:** High rebound rates ($0 \to 1$) indicate order/placement sensitivity or item noise, whereas high degradation ($1 \to 0$) confirms true capacity degradation.