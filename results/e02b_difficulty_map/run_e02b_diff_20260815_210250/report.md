# Experiment E02b: Horizon 0 v2 Difficulty-Grid Mapping Report

**Run ID:** `run_e02b_diff_20260815_210250`  
**Target Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Date:** 2026-08-15T21:11:34.083473+00:00  
**Response Format:** Matched 2-Alternative Forced Choice (2AFC, response_mode=`direct_value`)  
**Total Trials:** 192  

---

## 1. Executive Summary & Calibration Gate Status

| Sweep / Task Family | Monotonicity (Spearman $\rho$) | Kendall $\tau$ | Min Acc | Max Acc | Acc Span | Readiness Verdict | Calibration Gate Pass |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **Multi Hop** | +0.000 | +0.048 | 37.5% | 70.8% | 33.3% | **Non Monotonic** | Yes (`1`) |

---

## 2. Psychometric Curves by Task Family

### Sweep: Multi Hop

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Gate Pass | Mean Conf | Conf Sep | Brier | Prompt Tok |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | 24 | 17 | **70.8%** | [50.8%, 85.1%] | 62.5% | +1.06 [+0.19, +2.16] | -0.34 [-0.91, +0.11] | **PASS** | 69.0% | -1.5% | 0.342 | 419 |
| `2` | 24 | 9 | **37.5%** | [21.2%, 57.3%] | 70.8% | -0.68 [-1.81, +0.33] | -0.53 [-1.19, -0.10] | **FAIL** | 69.7% | -10.3% | 0.497 | 448 |
| `3` | 24 | 14 | **58.3%** | [38.8%, 75.5%] | 83.3% | +0.58 [-0.57, +1.57] | -0.91 [-1.48, -0.43] | **FAIL** | 66.3% | +28.6% | 0.224 | 478 |
| `4` | 24 | 11 | **45.8%** | [27.9%, 64.9%] | 79.2% | -0.25 [-1.37, +0.87] | -0.74 [-1.32, -0.29] | **FAIL** | 68.2% | +2.5% | 0.368 | 509 |
| `5` | 24 | 10 | **41.7%** | [24.5%, 61.2%] | 83.3% | -0.58 [-1.57, +0.33] | -0.91 [-1.48, -0.40] | **FAIL** | 59.0% | -25.6% | 0.555 | 535 |
| `6` | 24 | 12 | **50.0%** | [31.4%, 68.6%] | 75.0% | +0.00 [-1.15, +1.15] | -0.62 [-1.20, -0.19] | **FAIL** | 70.9% | +13.1% | 0.289 | 566 |
| `7` | 24 | 16 | **66.7%** | [46.7%, 82.0%] | 83.3% | +1.37 [+0.57, +1.96] | -1.08 [-1.48, -0.79] | **FAIL** | 57.1% | +6.9% | 0.327 | 595 |

---

## 3. Elicitation Policy Reactivity Analysis

- **Paired Trials Evaluated:** 24
- **Answer-Only Accuracy:** 70.8%
- **Answer+Confidence Accuracy:** 70.8%
- **Delta Accuracy (Confidence - Only):** +0.0%
- **Exact Option Concordance Rate:** 83.3%
- **McNemar Chi2 Statistic:** 0.250 (Asymptotic p = 0.6171)
- **Exact Binomial McNemar Test:** p = 1.0000
- **Reactivity Verdict:** `moderate_reactivity`
- **Policy Reactivity Note:** Exact p < 0.05 or Concordance < 85% demonstrates causal perturbation of the first-order choice policy by confidence elicitation.

---

## 4. Scientific Takeaways for Adaptive Calibration

1. **Monotonicity Assessment:** A difficulty axis is staircase-ready if it exhibits a consistent negative rank correlation (rho <= -0.70, negative_step_ratio >= 0.70) and spans the operational target window (~55-90%).
2. **Response Bias (Criterion $c$):** Positive $c$ indicates an Option-B selection bias under Signal=A conventions. Extreme criterion shifts ($|c| > 1.0$) indicate response collapse rather than sensitivity loss.
3. **Within-Item Transitions:** High rebound rates ($0 \to 1$) indicate order/placement sensitivity or item noise, whereas high degradation ($1 \to 0$) confirms true capacity degradation.