# Experiment E02b: Horizon 0 v2 Difficulty-Grid Mapping Report

**Run ID:** `run_e02b_diff_20260816_202207_dryrun`  
**Target Model:** `mock-qwen2.5:3b` (`mock_2afc_di...`)  
**Date:** 2026-08-16T20:22:07.159566+00:00  
**Response Format:** Matched 2-Alternative Forced Choice (2AFC, response_mode=`direct_value`)  
**Total Trials:** 40  

---

## 1. Executive Summary & Calibration Gate Status

| Sweep / Task Family | Monotonicity (Spearman $\rho$) | Kendall $\tau$ | Min Acc | Max Acc | Acc Span | Readiness Verdict | Calibration Gate Pass |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **Distractor Load** | +0.000 | +0.000 | 50.0% | 50.0% | 0.0% | **Flat Or Ceiling** | No (None) |
| **Multi Hop** | +0.000 | +0.000 | 50.0% | 50.0% | 0.0% | **Flat Or Ceiling** | No (None) |
| **Overwrite Load** | +0.000 | +0.000 | 50.0% | 50.0% | 0.0% | **Flat Or Ceiling** | No (None) |

---

## 2. Psychometric Curves by Task Family

### Sweep: Distractor Load

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Gate Pass | Mean Conf | Conf Sep | Brier | Prompt Tok |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 160 |
| `8` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 200 |
| `16` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 282 |
| `32` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 440 |
| `64` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 758 |
| `128` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 1393 |
| `256` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 2670 |

#### Within-Item Paired Transitions (Adjacent Levels):

| Transition ($D_k \to D_{k+1}$) | Retained ($1 \to 1$) | Degraded ($1 \to 0$) | Persisted Wrong ($0 \to 0$) | Rebounded ($0 \to 1$) | Degradation Rate | Rebound Rate | Net $\Delta$ Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4 -> 8` | 1 | 0 | 1 | 0 | 0.0% | 0.0% | +0.0% |
| `8 -> 16` | 1 | 0 | 1 | 0 | 0.0% | 0.0% | +0.0% |
| `16 -> 32` | 1 | 0 | 1 | 0 | 0.0% | 0.0% | +0.0% |
| `32 -> 64` | 1 | 0 | 1 | 0 | 0.0% | 0.0% | +0.0% |
| `64 -> 128` | 1 | 0 | 1 | 0 | 0.0% | 0.0% | +0.0% |
| `128 -> 256` | 1 | 0 | 1 | 0 | 0.0% | 0.0% | +0.0% |

### Sweep: Multi Hop

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Gate Pass | Mean Conf | Conf Sep | Brier | Prompt Tok |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 332 |
| `2` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 356 |
| `3` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 378 |
| `4` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 401 |
| `5` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 424 |
| `6` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 450 |
| `7` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 476 |

### Sweep: Overwrite Load

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Gate Pass | Mean Conf | Conf Sep | Brier | Prompt Tok |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 339 |
| `1` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 356 |
| `2` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 368 |
| `3` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 381 |
| `4` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | +0.00 | -0.67 | **FAIL** | 80.0% | +0.0% | 0.340 | 394 |

---

## 3. Elicitation Policy Reactivity Analysis

- **Paired Trials Evaluated:** 2
- **Answer-Only Accuracy:** 50.0%
- **Answer+Confidence Accuracy:** 50.0%
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