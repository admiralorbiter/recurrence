# Experiment E02b: Horizon 0 v2 Difficulty-Grid Mapping Report

**Run ID:** `run_e02b_diff_20260815_191031`  
**Target Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Date:** 2026-08-15T19:22:27.192423+00:00  
**Response Format:** Matched 2-Alternative Forced Choice (2AFC) with 50/50 Counterbalancing  
**Total Trials:** 256  

---

## 1. Executive Summary & Staircase Readiness

| Sweep / Task Family | Monotonicity (Spearman $\rho$) | Kendall $\tau$ | Min Acc | Max Acc | Acc Span | Readiness Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Distractor Load** | -0.821 | -0.619 | 40.6% | 90.6% | 50.0% | **Promising Monotonic Trend** |

---

## 2. Psychometric Curves by Task Family

### Sweep: Distractor Load

| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | SDT $d'$ | SDT $c$ | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4` | 32 | 25 | **78.1%** | [61.2%, 89.0%] | 40.6% | +1.51 | +0.30 | 72.4% | +2.2% | 0.235 | 197 | 100.0% |
| `8` | 32 | 29 | **90.6%** | [75.8%, 96.8%] | 46.9% | +2.40 | +0.15 | 56.9% | -12.6% | 0.293 | 252 | 100.0% |
| `16` | 32 | 21 | **65.6%** | [48.3%, 79.6%] | 53.1% | +0.76 | -0.08 | 55.6% | -7.5% | 0.333 | 360 | 100.0% |
| `32` | 32 | 22 | **68.8%** | [51.4%, 82.0%] | 43.8% | +0.93 | +0.16 | 65.1% | -1.3% | 0.279 | 577 | 100.0% |
| `64` | 32 | 13 | **40.6%** | [25.5%, 57.7%] | 46.9% | -0.45 | +0.08 | 65.0% | -1.3% | 0.349 | 1012 | 100.0% |
| `128` | 32 | 20 | **62.5%** | [45.3%, 77.1%] | 43.8% | +0.61 | +0.15 | 69.3% | +4.5% | 0.284 | 1883 | 100.0% |
| `256` | 32 | 16 | **50.0%** | [33.6%, 66.4%] | 43.8% | +0.00 | +0.15 | 39.6% | +9.4% | 0.321 | 3626 | 100.0% |

---

## 3. Elicitation Reactivity Control (Answer-Only vs. Answer+Confidence)

- **Paired Trials Evaluated:** 32
- **Answer-Only Accuracy:** 81.2%
- **Answer+Confidence Accuracy:** 78.1%
- **Delta Accuracy (Confidence - Only):** -3.1%
- **Exact Option Concordance Rate:** 71.9%
- **McNemar Chi2 Statistic:** 0.000 (p = 1.0000)
- **Reactivity Verdict:** `moderate_reactivity`

---

## 4. Scientific Takeaways for Adaptive Calibration

1. **Monotonicity Assessment:** A difficulty axis is staircase-ready if it exhibits a consistent negative rank correlation (rho <= -0.70) and spans the operational target window (~55-90%).
2. **Context vs. Structure:** If distractor count saturates at ceiling for larger models, multi-hop pointer depth (H) provides a structural composition bottleneck.
3. **Reactivity Stability:** Confidence elicitation must maintain high concordance (>= 85%) with pure answer-only behavior to avoid perturbing first-order policy during metacognitive readout.