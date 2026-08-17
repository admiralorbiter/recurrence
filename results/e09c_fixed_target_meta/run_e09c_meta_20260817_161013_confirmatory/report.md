# Experiment E09c: Fixed-Target Metacognitive Interaction Screen Report (Sprint S09d)

**Run ID:** `run_e09c_meta_20260817_161013_confirmatory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `CONFIRMATORY` (Seed: `1337`)  
**Date:** 2026-08-17T21:10:13.446505+00:00  
**Scope:** 16 Episodes | 80 Fixed Target Decisions | 320 Total Evaluator Probes  
**Primary Question:** *Under strictly frozen, identical first-order target decisions across all conditions, does scaffolded persistence alter the self-observer metacognitive calibration gap?*

---

## 1. 2x2 Fixed-Target Metacognitive Factorial Matrix

| Evaluator | Memory Format | Trials | Target Accuracy | Mean Confidence | Brier Score (Lower is Better) | AUROC Error Prediction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Agent (Self / alpha)** | `Transcript-Only` | 80 | 47.5% | 44.8% | **0.4312** | **0.534** |
| **Auditing Observer (gamma)** | `Transcript-Only` | 80 | 47.5% | 60.6% | **0.4596** | **0.520** |
| **Primary Agent (Self / alpha)** | `Scaffolded Persistence` | 80 | 47.5% | 47.0% | **0.5517** | **0.419** |
| **Auditing Observer (gamma)** | `Scaffolded Persistence` | 80 | 47.5% | 71.4% | **0.3921** | **0.613** |

---

## 2. Primary Estimands: Brier & AUROC Difference-in-Differences

| Estimand | Point Estimate | 95% Clustered CI | Permutation $p$ (Method) | Scientific Inference |
| :--- | :---: | :---: | :---: | :--- |
| **`Brier_Diff_in_Diff_Interaction`** | **+0.1880** | [-0.0232, +0.4242] | 0.1501 (`exact_sign_flip_2^16`) | **Invariant Calibration Gap** |
| **`AUROC_Metacognitive_Interaction`** | **-0.209** | [-0.409, -0.009] | 0.1406 (`exact_format_block_swap_65k`) | **Invariant Resolution** |
| **`Delta_Brier_Transcript`** | **-0.0284** | [-0.1722, +0.1367] | N/A (`cluster_bootstrap_ci_only`) | Self calibrated better |
| **`Delta_Brier_Scaffolded`** | **+0.1596** | [-0.0097, +0.3384] | N/A (`cluster_bootstrap_ci_only`) | Observer calibrated better |

---

## 3. Scientific Conclusion

- **First-Order Choice Invariance:** Primary agent choice distribution held fixed at **47.5% accuracy** across all evaluators and formats.
- **Brier Calibration Diff-in-Diff:** $\text{Interaction}_{\text{Brier}} = \mathbf{+0.1880}$ ($p = 0.1501$).
- **AUROC Resolution Diff-in-Diff:** $\text{Interaction}_{\text{AUROC}} = \mathbf{-0.209}$ ($p = 0.1406$).