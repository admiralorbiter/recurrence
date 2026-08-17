# Experiment E09: Metacognitive Continuity & Item-Paired Error Prediction Report (Sprint S09b)

**Run ID:** `run_e09_meta_20260817_183133_confirmatory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `CONFIRMATORY` (Seed: `1337`)  
**Date:** 2026-08-17T19:05:01.530108+00:00  
**Scope:** 16 Multi-Source Episodes | 320 Total Metacognitive Probes  
**Primary Question:** *Under matched visible public information, does self-referential framing provide a post-choice error-prediction advantage over an auditing observer predicting the exact same target decisions?*  

---

## 1. Metacognitive Calibration Breakdown (Brier Score & AUROC Resolution)

| Evaluator | Memory Format | Trials | Accuracy | Mean Confidence | Brier Score (Lower is Better) | AUROC Error Prediction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Auditing Observer (gamma)** | `Scaffolded Persistence` | 80 | 32.5% | 61.2% | **0.4507** | **0.594** |
| **Auditing Observer (gamma)** | `Transcript-Only` | 80 | 37.5% | 67.4% | **0.4643** | **0.560** |
| **Primary Agent (Self / alpha)** | `Scaffolded Persistence` | 80 | 32.5% | 53.1% | **0.5440** | **0.440** |
| **Primary Agent (Self / alpha)** | `Transcript-Only` | 80 | 37.5% | 59.2% | **0.3674** | **0.641** |

---

## 2. Item-Paired Metacognitive Estimands (Predicting Identical Target Decisions)

| Item-Paired Estimand | Point Estimate | 95% Clustered CI | Permutation $p$ (Method) | Scientific Inference |
| :--- | :---: | :---: | :---: | :--- |
| **`Delta_AUROC_Transcript`** | **+0.081** | [-0.106, +0.250] | 0.3778 (`exact_confidence_swap_65k`) | **Null / Invariant** |
| **`Delta_AUROC_Scaffolded`** | **-0.154** | [-0.308, -0.029] | 0.0615 (`exact_confidence_swap_65k`) | **Null / Invariant** |
| **`Delta_Brier_Transcript`** | **+0.0969** | [-0.0710, +0.2517] | 0.2658 (`exact_exhaustive`) | **Null / Invariant** |
| **`Delta_Brier_Scaffolded`** | **-0.0934** | [-0.2115, +0.0233] | 0.1525 (`exact_exhaustive`) | **Null / Invariant** |
| **`Scaffolding_Metacognitive_Interaction`** | **-0.235** | [-0.423, -0.052] | 0.0286 (`exact_format_block_swap_65k`) | **Scaffolded Persistence Alters Self-Observer Calibration** |

---

## 3. Scientific Gate Synthesis for Horizon 1 Closeout

1. **Pre-Feedback Correctness Prediction:** Evaluates whether subjective confidence discriminates impending errors prior to external feedback.
2. **Item-Paired Framing Control:** Both Self and Observer evaluate the exact same first-order decisions made by `agent_alpha` under matched public evidence.
3. **Persistence Scaffolding Interaction:** Measures whether explicit Level-1 state shifts the metacognitive gap between internal self-framing and external observer evaluation.