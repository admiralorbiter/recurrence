# Experiment E09c: Fixed-Target Metacognitive Interaction Screen Report (Sprint S09d)

**Run ID:** `run_e09c_meta_20260817_153208_exploratory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-17T20:32:08.629542+00:00  
**Scope:** 4 Episodes | 20 Fixed Target Decisions | 80 Total Evaluator Probes  
**Primary Question:** *Under strictly frozen, identical first-order target decisions across all conditions, does scaffolded persistence alter the self-observer metacognitive calibration gap?*

---

## 1. 2x2 Fixed-Target Metacognitive Factorial Matrix

| Evaluator | Memory Format | Trials | Target Accuracy | Mean Confidence | Brier Score (Lower is Better) | AUROC Error Prediction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Agent (Self / alpha)** | `Transcript-Only` | 20 | 25.0% | 67.5% | **0.6846** | **0.333** |
| **Auditing Observer (gamma)** | `Transcript-Only` | 20 | 25.0% | 58.5% | **0.5136** | **0.493** |
| **Primary Agent (Self / alpha)** | `Scaffolded Persistence` | 20 | 25.0% | 45.0% | **0.5921** | **0.307** |
| **Auditing Observer (gamma)** | `Scaffolded Persistence` | 20 | 25.0% | 49.1% | **0.5132** | **0.413** |

---

## 2. Primary Estimands: Brier & AUROC Difference-in-Differences

| Estimand | Point Estimate | 95% Clustered CI | Permutation $p$ (Method) | Scientific Inference |
| :--- | :---: | :---: | :---: | :--- |
| **`Brier_Diff_in_Diff_Interaction`** | **-0.0921** | [-0.4570, +0.2728] | 0.8824 (`exact_sign_flip_2^4`) | **Invariant Calibration Gap** |
| **`AUROC_Metacognitive_Interaction`** | **+0.053** | [-0.147, +0.253] | 0.7500 (`exact_format_block_swap_65k`) | **Invariant Resolution** |
| **`Delta_Brier_Transcript`** | **+0.1710** | [+0.0500, +0.2838] | N/A (`cluster_bootstrap_ci_only`) | Observer calibrated better |
| **`Delta_Brier_Scaffolded`** | **+0.0789** | [-0.1942, +0.3697] | N/A (`cluster_bootstrap_ci_only`) | Observer calibrated better |

---

## 3. Scientific Conclusion

- **First-Order Choice Invariance:** Primary agent choice distribution held fixed at **25.0% accuracy** across all evaluators and formats.
- **Brier Calibration Diff-in-Diff:** $\text{Interaction}_{\text{Brier}} = \mathbf{-0.0921}$ ($p = 0.8824$).
- **AUROC Resolution Diff-in-Diff:** $\text{Interaction}_{\text{AUROC}} = \mathbf{+0.053}$ ($p = 0.7500$).