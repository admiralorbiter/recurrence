# Experiment E02b: Horizon 0 v2 Difficulty-Grid Mapping Report

**Run ID:** `run_e02b_diff_20260815_203008`  
**Target Model:** `llama3.2:3b` (`a80c4f17acd5...`)  
**Date:** 2026-08-15T20:42:03.455852+00:00  
**Response Format:** Matched 2-Alternative Forced Choice (2AFC) with 50/50 Counterbalancing  
**Total Trials:** 256  

---

## 1. Executive Summary & Staircase Readiness

| Sweep / Task Family | Monotonicity (Spearman $\rho$) | Kendall $\tau$ | Min Acc | Max Acc | Acc Span | Readiness Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Distractor Load** | -0.955 | -0.878 | 53.1% | 84.4% | 31.2% | **Staircase Ready** |

---

## 2. Psychometric Curves by Task Family

### Sweep: Distractor Load

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4` | 32 | 27 | **84.4%** | [68.2%, 93.1%] | 34.4% | +2.35 [+1.74, +2.94] | +0.72 [+0.42, +1.02] | 96.9% | +20.0% | 0.125 | 188 | 100.0% |
| `8` | 32 | 24 | **75.0%** | [57.9%, 86.7%] | 25.0% | +1.89 [+1.26, +2.52] | +0.94 [+0.63, +1.26] | 96.9% | +12.5% | 0.219 | 242 | 100.0% |
| `16` | 32 | 22 | **68.8%** | [51.4%, 82.0%] | 18.8% | +1.59 [+1.07, +2.19] | +1.09 [+0.80, +1.36] | 93.8% | +20.0% | 0.250 | 351 | 100.0% |
| `32` | 32 | 22 | **68.8%** | [51.4%, 82.0%] | 18.8% | +1.59 [+1.07, +2.19] | +1.09 [+0.80, +1.36] | 86.7% | +28.0% | 0.236 | 568 | 100.0% |
| `64` | 32 | 18 | **56.2%** | [39.3%, 71.8%] | 6.2% | +0.84 [+0.00, +1.43] | +1.47 [+1.17, +1.89] | 81.2% | +30.2% | 0.312 | 1004 | 100.0% |
| `128` | 32 | 20 | **62.5%** | [45.3%, 77.1%] | 12.5% | +1.26 [+0.54, +1.89] | +1.26 [+0.94, +1.62] | 88.3% | +4.6% | 0.377 | 1875 | 100.0% |
| `256` | 32 | 17 | **53.1%** | [36.4%, 69.1%] | 3.1% | +0.54 [+0.00, +1.07] | +1.62 [+1.36, +1.89] | 92.2% | +10.4% | 0.414 | 3617 | 100.0% |

#### Within-Item Paired Transitions (Adjacent Levels):

| Transition ($D_k \to D_{k+1}$) | Retained ($1 \to 1$) | Degraded ($1 \to 0$) | Persisted Wrong ($0 \to 0$) | Rebounded ($0 \to 1$) | Degradation Rate | Rebound Rate | Net $\Delta$ Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4 -> 8` | 23 | 4 | 4 | 1 | 14.8% | 20.0% | -9.4% |
| `8 -> 16` | 20 | 4 | 6 | 2 | 16.7% | 25.0% | -6.2% |
| `16 -> 32` | 20 | 2 | 8 | 2 | 9.1% | 20.0% | +0.0% |
| `32 -> 64` | 18 | 4 | 10 | 0 | 18.2% | 0.0% | -12.5% |
| `64 -> 128` | 17 | 1 | 11 | 3 | 5.6% | 21.4% | +6.2% |
| `128 -> 256` | 16 | 4 | 11 | 1 | 20.0% | 8.3% | -9.4% |

---

## 3. Elicitation Reactivity Control (Answer-Only vs. Answer+Confidence)

- **Paired Trials Evaluated:** 32
- **Answer-Only Accuracy:** 90.6%
- **Answer+Confidence Accuracy:** 84.4%
- **Delta Accuracy (Confidence - Only):** -6.2%
- **Exact Option Concordance Rate:** 87.5%
- **McNemar Chi2 Statistic:** 0.250 (p = 0.6171)
- **Reactivity Verdict:** `moderate_reactivity`
- **Policy Reactivity Note:** Concordance below 85% reflects item-level decision changes even if net accuracy difference is small.

---

## 4. Scientific Takeaways for Adaptive Calibration

1. **Monotonicity Assessment:** A difficulty axis is staircase-ready if it exhibits a consistent negative rank correlation (rho <= -0.70, negative_step_ratio >= 0.70) and spans the operational target window (~55-90%).
2. **Response Bias (Criterion $c$):** Positive $c$ indicates an Option-B selection bias under Signal=A conventions. Extreme criterion shifts ($|c| > 1.0$) indicate response collapse rather than sensitivity loss.
3. **Within-Item Transitions:** High rebound rates ($0 \to 1$) indicate order/placement sensitivity or item noise, whereas high degradation ($1 \to 0$) confirms true capacity degradation.