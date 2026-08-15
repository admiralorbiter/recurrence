# Experiment E02b: Horizon 0 v2 Difficulty-Grid Mapping Report

**Run ID:** `run_e02b_diff_20260815_181036`  
**Target Model:** `qwen2.5:14b` (`7cdf5a0187d5...`)  
**Date:** 2026-08-15T18:48:16.750621+00:00  
**Response Format:** Matched 2-Alternative Forced Choice (2AFC) with 50/50 Counterbalancing  
**Total Trials:** 272  

---

## 1. Executive Summary & Staircase Readiness

| Sweep / Task Family | Monotonicity (Spearman $\rho$) | Kendall $\tau$ | Min Acc | Max Acc | Acc Span | Readiness Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Distractor Load** | +0.000 | +0.000 | 100.0% | 100.0% | 0.0% | **Flat Or Ceiling** |
| **Multi Hop** | -0.707 | -0.632 | 93.8% | 100.0% | 6.2% | **Flat Or Ceiling** |
| **Overwrite Load** | +0.000 | +0.000 | 100.0% | 100.0% | 0.0% | **Flat Or Ceiling** |

---

## 2. Psychometric Curves by Task Family

### Sweep: Distractor Load

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 131 | 100.0% |
| `8` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 170 | 100.0% |
| `16` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 250 | 100.0% |
| `32` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 409 | 100.0% |
| `64` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 728 | 100.0% |
| `128` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 1367 | 100.0% |
| `256` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 2641 | 100.0% |

### Sweep: Multi Hop

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 291 | 100.0% |
| `2` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 301 | 100.0% |
| `3` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 315 | 100.0% |
| `4` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 326 | 100.0% |
| `5` | 16 | 15 | **93.8%** | [71.7%, 98.9%] | 43.8% | 100.0% | +0.0% | 0.062 | 338 | 100.0% |

### Sweep: Overwrite Load

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 313 | 100.0% |
| `1` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 326 | 100.0% |
| `2` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 340 | 100.0% |
| `3` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 353 | 100.0% |
| `4` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 366 | 100.0% |

---

## 3. Elicitation Reactivity Control (Answer-Only vs. Answer+Confidence)

- **Paired Trials Evaluated:** 16
- **Answer-Only Accuracy:** 100.0%
- **Answer+Confidence Accuracy:** 100.0%
- **Delta Accuracy (Confidence - Only):** +0.0%
- **Exact Option Concordance Rate:** 100.0%
- **McNemar Chi2 Statistic:** 0.000 (p = 1.0000)
- **Reactivity Verdict:** `negligible_reactivity`

---

## 4. Scientific Takeaways for Adaptive Calibration

1. **Monotonicity Assessment:** A difficulty axis is staircase-ready if it exhibits a consistent negative rank correlation (rho <= -0.70) and spans the operational target window (~55-90%).
2. **Context vs. Structure:** If distractor count saturates at ceiling for larger models, multi-hop pointer depth (H) provides a structural composition bottleneck.
3. **Reactivity Stability:** Confidence elicitation must maintain high concordance (>= 85%) with pure answer-only behavior to avoid perturbing first-order policy during metacognitive readout.