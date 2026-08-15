# Experiment E02b: Horizon 0 v2 Difficulty-Grid Mapping Report

**Run ID:** `run_e02b_diff_20260815_173855_dryrun`  
**Target Model:** `mock-qwen2.5:3b` (`mock_2afc_di...`)  
**Date:** 2026-08-15T17:38:55.262619+00:00  
**Response Format:** Matched 2-Alternative Forced Choice (2AFC) with 50/50 Counterbalancing  
**Total Trials:** 34  

---

## 1. Executive Summary & Staircase Readiness

| Sweep / Task Family | Monotonicity (Spearman $\rho$) | Kendall $\tau$ | Min Acc | Max Acc | Acc Span | Readiness Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Distractor Load** | +0.000 | +0.000 | 50.0% | 50.0% | 0.0% | **Flat Or Ceiling** |
| **Multi Hop** | +0.000 | +0.000 | 50.0% | 50.0% | 0.0% | **Flat Or Ceiling** |
| **Overwrite Load** | +0.000 | +0.000 | 50.0% | 50.0% | 0.0% | **Flat Or Ceiling** |

---

## 2. Psychometric Curves by Task Family

### Sweep: Distractor Load

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 132 | 100.0% |
| `8` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 171 | 100.0% |
| `16` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 249 | 100.0% |
| `32` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 409 | 100.0% |
| `64` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 724 | 100.0% |
| `128` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 1368 | 100.0% |
| `256` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 2643 | 100.0% |

### Sweep: Multi Hop

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 293 | 100.0% |
| `2` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 303 | 100.0% |
| `3` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 314 | 100.0% |
| `4` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 327 | 100.0% |
| `5` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 338 | 100.0% |

### Sweep: Overwrite Load

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 312 | 100.0% |
| `1` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 327 | 100.0% |
| `2` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 340 | 100.0% |
| `3` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 355 | 100.0% |
| `4` | 2 | 1 | **50.0%** | [9.5%, 90.5%] | 100.0% | 80.0% | +0.0% | 0.340 | 364 | 100.0% |

---

## 3. Elicitation Reactivity Control (Answer-Only vs. Answer+Confidence)

- **Paired Trials Evaluated:** 2
- **Answer-Only Accuracy:** 50.0%
- **Answer+Confidence Accuracy:** 50.0%
- **Delta Accuracy (Confidence - Only):** +0.0%
- **Exact Option Concordance Rate:** 100.0%
- **McNemar Chi2 Statistic:** 0.000 (p = 1.0000)
- **Reactivity Verdict:** `negligible_reactivity`

---

## 4. Scientific Takeaways for Adaptive Calibration

1. **Monotonicity Assessment:** A difficulty axis is staircase-ready if it exhibits a consistent negative rank correlation (rho <= -0.70) and spans the operational target window (~55-90%).
2. **Context vs. Structure:** If distractor count saturates at ceiling for larger models, multi-hop pointer depth (H) provides a structural composition bottleneck.
3. **Reactivity Stability:** Confidence elicitation must maintain high concordance (>= 85%) with pure answer-only behavior to avoid perturbing first-order policy during metacognitive readout.