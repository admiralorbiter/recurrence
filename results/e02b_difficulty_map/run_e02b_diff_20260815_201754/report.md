# Experiment E02b: Horizon 0 v2 Difficulty-Grid Mapping Report

**Run ID:** `run_e02b_diff_20260815_201754`  
**Target Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Date:** 2026-08-15T20:29:57.966568+00:00  
**Response Format:** Matched 2-Alternative Forced Choice (2AFC) with 50/50 Counterbalancing  
**Total Trials:** 256  

---

## 1. Executive Summary & Staircase Readiness

| Sweep / Task Family | Monotonicity (Spearman $\rho$) | Kendall $\tau$ | Min Acc | Max Acc | Acc Span | Readiness Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Distractor Load** | -0.396 | -0.293 | 46.9% | 81.2% | 34.4% | **Non Monotonic** |

---

## 2. Psychometric Curves by Task Family

### Sweep: Distractor Load

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4` | 32 | 21 | **65.6%** | [48.3%, 79.6%] | 34.4% | +0.82 [+0.00, +1.81] | +0.41 [-0.00, +0.94] | 75.0% | -9.6% | 0.348 | 197 | 100.0% |
| `8` | 32 | 26 | **81.2%** | [64.7%, 91.1%] | 43.8% | +1.68 [+0.93, +2.94] | +0.21 [-0.27, +0.72] | 53.7% | +12.2% | 0.288 | 251 | 100.0% |
| `16` | 32 | 22 | **68.8%** | [51.4%, 82.0%] | 31.2% | +1.05 [+0.19, +2.19] | +0.52 [+0.09, +1.09] | 55.6% | -4.3% | 0.336 | 360 | 100.0% |
| `32` | 32 | 19 | **59.4%** | [42.3%, 74.5%] | 28.1% | +0.52 [-0.30, +1.51] | +0.56 [+0.15, +1.09] | 60.4% | +10.1% | 0.278 | 577 | 100.0% |
| `64` | 32 | 21 | **65.6%** | [48.3%, 79.6%] | 40.6% | +0.78 [+0.00, +1.68] | +0.24 [-0.16, +0.75] | 66.2% | -13.8% | 0.367 | 1013 | 100.0% |
| `128` | 32 | 24 | **75.0%** | [57.9%, 86.7%] | 37.5% | +1.35 [+0.60, +2.52] | +0.37 [-0.00, +0.94] | 73.0% | -4.8% | 0.255 | 1884 | 100.0% |
| `256` | 32 | 15 | **46.9%** | [30.9%, 63.6%] | 59.4% | -0.15 [-0.93, +0.75] | -0.22 [-0.64, +0.17] | 46.5% | -15.2% | 0.425 | 3626 | 100.0% |

#### Within-Item Paired Transitions (Adjacent Levels):

| Transition ($D_k \to D_{k+1}$) | Retained ($1 \to 1$) | Degraded ($1 \to 0$) | Persisted Wrong ($0 \to 0$) | Rebounded ($0 \to 1$) | Degradation Rate | Rebound Rate | Net $\Delta$ Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4 -> 8` | 16 | 5 | 1 | 10 | 23.8% | 90.9% | +15.6% |
| `8 -> 16` | 19 | 7 | 3 | 3 | 26.9% | 50.0% | -12.5% |
| `16 -> 32` | 14 | 8 | 5 | 5 | 36.4% | 50.0% | -9.4% |
| `32 -> 64` | 16 | 3 | 8 | 5 | 15.8% | 38.5% | +6.2% |
| `64 -> 128` | 15 | 6 | 2 | 9 | 28.6% | 81.8% | +9.4% |
| `128 -> 256` | 10 | 14 | 3 | 5 | 58.3% | 62.5% | -28.1% |

---

## 3. Elicitation Reactivity Control (Answer-Only vs. Answer+Confidence)

- **Paired Trials Evaluated:** 32
- **Answer-Only Accuracy:** 93.8%
- **Answer+Confidence Accuracy:** 65.6%
- **Delta Accuracy (Confidence - Only):** -28.1%
- **Exact Option Concordance Rate:** 65.6%
- **McNemar Chi2 Statistic:** 5.818 (p = 0.0159)
- **Reactivity Verdict:** `severe_reactivity`
- **Policy Reactivity Note:** Concordance below 85% reflects item-level decision changes even if net accuracy difference is small.

---

## 4. Scientific Takeaways for Adaptive Calibration

1. **Monotonicity Assessment:** A difficulty axis is staircase-ready if it exhibits a consistent negative rank correlation (rho <= -0.70, negative_step_ratio >= 0.70) and spans the operational target window (~55-90%).
2. **Response Bias (Criterion $c$):** Positive $c$ indicates an Option-B selection bias under Signal=A conventions. Extreme criterion shifts ($|c| > 1.0$) indicate response collapse rather than sensitivity loss.
3. **Within-Item Transitions:** High rebound rates ($0 \to 1$) indicate order/placement sensitivity or item noise, whereas high degradation ($1 \to 0$) confirms true capacity degradation.