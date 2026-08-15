# Experiment E02b: Horizon 0 v2 Difficulty-Grid Mapping Report

**Run ID:** `run_e02b_diff_20260815_175717`  
**Target Model:** `llama3.2:3b` (`a80c4f17acd5...`)  
**Date:** 2026-08-15T18:10:28.520863+00:00  
**Response Format:** Matched 2-Alternative Forced Choice (2AFC) with 50/50 Counterbalancing  
**Total Trials:** 272  

---

## 1. Executive Summary & Staircase Readiness

| Sweep / Task Family | Monotonicity (Spearman $\rho$) | Kendall $\tau$ | Min Acc | Max Acc | Acc Span | Readiness Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Distractor Load** | -0.847 | -0.683 | 56.2% | 100.0% | 43.8% | **Staircase Ready** |
| **Multi Hop** | -0.264 | -0.224 | 50.0% | 68.8% | 18.8% | **Non Monotonic** |
| **Overwrite Load** | -0.707 | -0.632 | 50.0% | 56.2% | 6.2% | **Flat Or Ceiling** |

---

## 2. Psychometric Curves by Task Family

### Sweep: Distractor Load

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4` | 16 | 15 | **93.8%** | [71.7%, 98.9%] | 43.8% | 100.0% | +0.0% | 0.062 | 131 | 100.0% |
| `8` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 100.0% | N/A | 0.000 | 170 | 100.0% |
| `16` | 16 | 14 | **87.5%** | [64.0%, 96.5%] | 37.5% | 87.5% | +100.0% | 0.000 | 250 | 100.0% |
| `32` | 16 | 14 | **87.5%** | [64.0%, 96.5%] | 37.5% | 100.0% | +0.0% | 0.125 | 409 | 100.0% |
| `64` | 16 | 9 | **56.2%** | [33.2%, 76.9%] | 6.2% | 87.5% | +28.6% | 0.312 | 728 | 100.0% |
| `128` | 16 | 12 | **75.0%** | [50.5%, 89.8%] | 25.0% | 98.4% | -2.1% | 0.254 | 1367 | 100.0% |
| `256` | 16 | 11 | **68.8%** | [44.4%, 85.8%] | 18.8% | 93.8% | +20.0% | 0.250 | 2641 | 100.0% |

### Sweep: Multi Hop

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | 16 | 9 | **56.2%** | [33.2%, 76.9%] | 6.2% | 98.4% | +3.6% | 0.410 | 291 | 100.0% |
| `2` | 16 | 9 | **56.2%** | [33.2%, 76.9%] | 6.2% | 100.0% | +0.0% | 0.438 | 301 | 100.0% |
| `3` | 16 | 8 | **50.0%** | [28.0%, 72.0%] | 0.0% | 98.4% | +3.1% | 0.473 | 315 | 100.0% |
| `4` | 16 | 11 | **68.8%** | [44.4%, 85.8%] | 18.8% | 100.0% | +0.0% | 0.312 | 326 | 100.0% |
| `5` | 16 | 8 | **50.0%** | [28.0%, 72.0%] | 0.0% | 93.8% | +12.5% | 0.438 | 338 | 100.0% |

### Sweep: Overwrite Load

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | 16 | 9 | **56.2%** | [33.2%, 76.9%] | 6.2% | 100.0% | +0.0% | 0.438 | 313 | 100.0% |
| `1` | 16 | 8 | **50.0%** | [28.0%, 72.0%] | 0.0% | 98.8% | +2.5% | 0.478 | 326 | 100.0% |
| `2` | 16 | 8 | **50.0%** | [28.0%, 72.0%] | 0.0% | 93.4% | +10.6% | 0.394 | 340 | 100.0% |
| `3` | 16 | 8 | **50.0%** | [28.0%, 72.0%] | 0.0% | 96.2% | -0.1% | 0.469 | 353 | 100.0% |
| `4` | 16 | 8 | **50.0%** | [28.0%, 72.0%] | 0.0% | 93.8% | +5.0% | 0.424 | 366 | 100.0% |

---

## 3. Elicitation Reactivity Control (Answer-Only vs. Answer+Confidence)

- **Paired Trials Evaluated:** 16
- **Answer-Only Accuracy:** 100.0%
- **Answer+Confidence Accuracy:** 93.8%
- **Delta Accuracy (Confidence - Only):** -6.2%
- **Exact Option Concordance Rate:** 93.8%
- **McNemar Chi2 Statistic:** 0.000 (p = 1.0000)
- **Reactivity Verdict:** `moderate_reactivity`

---

## 4. Scientific Takeaways for Adaptive Calibration

1. **Monotonicity Assessment:** A difficulty axis is staircase-ready if it exhibits a consistent negative rank correlation (rho <= -0.70) and spans the operational target window (~55-90%).
2. **Context vs. Structure:** If distractor count saturates at ceiling for larger models, multi-hop pointer depth (H) provides a structural composition bottleneck.
3. **Reactivity Stability:** Confidence elicitation must maintain high concordance (>= 85%) with pure answer-only behavior to avoid perturbing first-order policy during metacognitive readout.