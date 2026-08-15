# Experiment E02b: Horizon 0 v2 Difficulty-Grid Mapping Report

**Run ID:** `run_e02b_diff_20260815_192236`  
**Target Model:** `llama3.2:3b` (`a80c4f17acd5...`)  
**Date:** 2026-08-15T19:34:22.343031+00:00  
**Response Format:** Matched 2-Alternative Forced Choice (2AFC) with 50/50 Counterbalancing  
**Total Trials:** 256  

---

## 1. Executive Summary & Staircase Readiness

| Sweep / Task Family | Monotonicity (Spearman $\rho$) | Kendall $\tau$ | Min Acc | Max Acc | Acc Span | Readiness Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Distractor Load** | -0.927 | -0.851 | 59.4% | 96.9% | 37.5% | **Staircase Ready** |

---

## 2. Psychometric Curves by Task Family

### Sweep: Distractor Load

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | SDT $d'$ | SDT $c$ | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4` | 32 | 31 | **96.9%** | [84.3%, 99.4%] | 46.9% | +3.24 | +0.27 | 100.0% | +0.0% | 0.031 | 188 | 100.0% |
| `8` | 32 | 28 | **87.5%** | [71.9%, 95.0%] | 37.5% | +2.52 | +0.63 | 100.0% | +0.0% | 0.125 | 243 | 100.0% |
| `16` | 32 | 25 | **78.1%** | [61.2%, 89.0%] | 28.1% | +2.04 | +0.87 | 100.0% | +0.0% | 0.219 | 351 | 100.0% |
| `32` | 32 | 25 | **78.1%** | [61.2%, 89.0%] | 28.1% | +2.04 | +0.87 | 93.8% | +28.6% | 0.156 | 568 | 100.0% |
| `64` | 32 | 19 | **59.4%** | [42.3%, 74.5%] | 9.4% | +1.07 | +1.36 | 93.0% | +14.1% | 0.346 | 1003 | 100.0% |
| `128` | 32 | 21 | **65.6%** | [48.3%, 79.6%] | 15.6% | +1.43 | +1.17 | 83.6% | +16.6% | 0.314 | 1874 | 100.0% |
| `256` | 32 | 19 | **59.4%** | [42.3%, 74.5%] | 9.4% | +1.07 | +1.36 | 92.2% | +16.0% | 0.332 | 3617 | 100.0% |

---

## 3. Elicitation Reactivity Control (Answer-Only vs. Answer+Confidence)

- **Paired Trials Evaluated:** 32
- **Answer-Only Accuracy:** 100.0%
- **Answer+Confidence Accuracy:** 96.9%
- **Delta Accuracy (Confidence - Only):** -3.1%
- **Exact Option Concordance Rate:** 96.9%
- **McNemar Chi2 Statistic:** 0.000 (p = 1.0000)
- **Reactivity Verdict:** `negligible_reactivity`

---

## 4. Scientific Takeaways for Adaptive Calibration

1. **Monotonicity Assessment:** A difficulty axis is staircase-ready if it exhibits a consistent negative rank correlation (rho <= -0.70) and spans the operational target window (~55-90%).
2. **Context vs. Structure:** If distractor count saturates at ceiling for larger models, multi-hop pointer depth (H) provides a structural composition bottleneck.
3. **Reactivity Stability:** Confidence elicitation must maintain high concordance (>= 85%) with pure answer-only behavior to avoid perturbing first-order policy during metacognitive readout.