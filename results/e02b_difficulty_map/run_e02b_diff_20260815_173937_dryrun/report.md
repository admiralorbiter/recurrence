# Experiment E02b: Horizon 0 v2 Difficulty-Grid Mapping Report

**Run ID:** `run_e02b_diff_20260815_173937_dryrun`  
**Target Model:** `qwen2.5:3b` (`mock_2afc_di...`)  
**Date:** 2026-08-15T17:39:37.103152+00:00  
**Response Format:** Matched 2-Alternative Forced Choice (2AFC) with 50/50 Counterbalancing  
**Total Trials:** 68  

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
| `4` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 131 | 100.0% |
| `8` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 170 | 100.0% |
| `16` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 250 | 100.0% |
| `32` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 411 | 100.0% |
| `64` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 725 | 100.0% |
| `128` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 1368 | 100.0% |
| `256` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 2644 | 100.0% |

### Sweep: Multi Hop

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 292 | 100.0% |
| `2` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 302 | 100.0% |
| `3` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 312 | 100.0% |
| `4` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 327 | 100.0% |
| `5` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 337 | 100.0% |

### Sweep: Overwrite Load

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 312 | 100.0% |
| `1` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 326 | 100.0% |
| `2` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 340 | 100.0% |
| `3` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 355 | 100.0% |
| `4` | 4 | 2 | **50.0%** | [15.0%, 85.0%] | 100.0% | 80.0% | +0.0% | 0.340 | 364 | 100.0% |

---

## 3. Elicitation Reactivity Control (Answer-Only vs. Answer+Confidence)

- **Paired Trials Evaluated:** 4
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