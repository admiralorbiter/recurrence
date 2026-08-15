# Experiment E02b: Horizon 0 v2 Difficulty-Grid Mapping Report

**Run ID:** `run_e02b_diff_20260815_174343`  
**Target Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Date:** 2026-08-15T17:57:00.971712+00:00  
**Response Format:** Matched 2-Alternative Forced Choice (2AFC) with 50/50 Counterbalancing  
**Total Trials:** 272  

---

## 1. Executive Summary & Staircase Readiness

| Sweep / Task Family | Monotonicity (Spearman $\rho$) | Kendall $\tau$ | Min Acc | Max Acc | Acc Span | Readiness Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Distractor Load** | -0.893 | -0.714 | 43.8% | 100.0% | 56.2% | **Staircase Ready** |
| **Multi Hop** | -0.616 | -0.527 | 56.2% | 81.2% | 25.0% | **Partially Monotonic** |
| **Overwrite Load** | -0.718 | -0.527 | 37.5% | 75.0% | 37.5% | **Staircase Ready** |

---

## 2. Psychometric Curves by Task Family

### Sweep: Distractor Load

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4` | 16 | 16 | **100.0%** | [80.6%, 100.0%] | 50.0% | 75.3% | N/A | 0.128 | 131 | 100.0% |
| `8` | 16 | 14 | **87.5%** | [64.0%, 96.5%] | 62.5% | 57.5% | +8.6% | 0.247 | 170 | 100.0% |
| `16` | 16 | 15 | **93.8%** | [71.7%, 98.9%] | 56.2% | 67.9% | -34.3% | 0.264 | 250 | 100.0% |
| `32` | 16 | 11 | **68.8%** | [44.4%, 85.8%] | 56.2% | 46.4% | -22.6% | 0.457 | 409 | 100.0% |
| `64` | 16 | 12 | **75.0%** | [50.5%, 89.8%] | 50.0% | 66.6% | -4.6% | 0.308 | 728 | 100.0% |
| `128` | 16 | 7 | **43.8%** | [23.1%, 66.8%] | 56.2% | 61.9% | +27.9% | 0.220 | 1367 | 100.0% |
| `256` | 16 | 8 | **50.0%** | [28.0%, 72.0%] | 50.0% | 45.1% | +4.9% | 0.307 | 2641 | 100.0% |

### Sweep: Multi Hop

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | 16 | 13 | **81.2%** | [57.0%, 93.4%] | 56.2% | 58.8% | +34.6% | 0.232 | 291 | 100.0% |
| `2` | 16 | 11 | **68.8%** | [44.4%, 85.8%] | 31.2% | 66.2% | -5.2% | 0.376 | 301 | 100.0% |
| `3` | 16 | 13 | **81.2%** | [57.0%, 93.4%] | 31.2% | 79.2% | -9.1% | 0.242 | 315 | 100.0% |
| `4` | 16 | 12 | **75.0%** | [50.5%, 89.8%] | 37.5% | 72.1% | +4.4% | 0.245 | 326 | 100.0% |
| `5` | 16 | 9 | **56.2%** | [33.2%, 76.9%] | 31.2% | 75.9% | +10.4% | 0.301 | 338 | 100.0% |

### Sweep: Overwrite Load

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | 16 | 12 | **75.0%** | [50.5%, 89.8%] | 37.5% | 66.1% | +23.1% | 0.219 | 313 | 100.0% |
| `1` | 16 | 11 | **68.8%** | [44.4%, 85.8%] | 56.2% | 77.6% | +10.2% | 0.237 | 326 | 100.0% |
| `2` | 16 | 12 | **75.0%** | [50.5%, 89.8%] | 50.0% | 52.2% | +8.0% | 0.302 | 340 | 100.0% |
| `3` | 16 | 6 | **37.5%** | [18.5%, 61.4%] | 50.0% | 69.3% | -18.9% | 0.516 | 353 | 100.0% |
| `4` | 16 | 10 | **62.5%** | [38.6%, 81.5%] | 37.5% | 58.4% | -1.2% | 0.360 | 366 | 100.0% |

---

## 3. Elicitation Reactivity Control (Answer-Only vs. Answer+Confidence)

- **Paired Trials Evaluated:** 16
- **Answer-Only Accuracy:** 93.8%
- **Answer+Confidence Accuracy:** 100.0%
- **Delta Accuracy (Confidence - Only):** +6.2%
- **Exact Option Concordance Rate:** 93.8%
- **McNemar Chi2 Statistic:** 0.000 (p = 1.0000)
- **Reactivity Verdict:** `moderate_reactivity`

---

## 4. Scientific Takeaways for Adaptive Calibration

1. **Monotonicity Assessment:** A difficulty axis is staircase-ready if it exhibits a consistent negative rank correlation (rho <= -0.70) and spans the operational target window (~55-90%).
2. **Context vs. Structure:** If distractor count saturates at ceiling for larger models, multi-hop pointer depth (H) provides a structural composition bottleneck.
3. **Reactivity Stability:** Confidence elicitation must maintain high concordance (>= 85%) with pure answer-only behavior to avoid perturbing first-order policy during metacognitive readout.