# Experiment E09: Metacognitive Continuity & Item-Paired Error Prediction Report (Sprint S09b)

**Run ID:** `run_e09_meta_20260817_171718_exploratory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-17T17:22:26.579444+00:00  
**Scope:** 4 Multi-Source Episodes | 80 Total Metacognitive Probes  
**Primary Question:** *Under matched visible public information, does self-referential framing provide a post-choice error-prediction advantage over an auditing observer predicting the exact same target decisions?*  

---

## 1. Metacognitive Calibration Breakdown (Brier Score & AUROC Resolution)

| Evaluator | Memory Format | Trials | Accuracy | Mean Confidence | Brier Score (Lower is Better) | AUROC Error Prediction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Auditing Observer (gamma)** | `Scaffolded Persistence` | 20 | 25.0% | 75.0% | **0.5730** | **0.580** |
| **Auditing Observer (gamma)** | `Transcript-Only` | 20 | 15.0% | 67.6% | **0.5510** | **0.471** |
| **Primary Agent (Self / alpha)** | `Scaffolded Persistence` | 20 | 25.0% | 55.9% | **0.5009** | **0.507** |
| **Primary Agent (Self / alpha)** | `Transcript-Only` | 20 | 15.0% | 55.5% | **0.5465** | **0.353** |

---

## 2. Item-Paired Metacognitive Estimands (Predicting Identical Target Decisions)

| Item-Paired Estimand | Point Estimate | 95% Clustered CI | Permutation $p$ (Method) | Scientific Inference |
| :--- | :---: | :---: | :---: | :--- |
| **`Delta_AUROC_Transcript`** | **-0.083** | [-0.438, +0.188] | 1.0000 (`exact_exhaustive`) | **Null / Invariant** |
| **`Delta_AUROC_Scaffolded`** | **-0.062** | [-0.375, +0.188] | 1.0000 (`exact_exhaustive`) | **Null / Invariant** |
| **`Delta_Brier_Transcript`** | **+0.0045** | [-0.3673, +0.3763] | 1.0000 (`exact_exhaustive`) | **Null / Invariant** |
| **`Delta_Brier_Scaffolded`** | **+0.0721** | [-0.1080, +0.1807] | 0.5000 (`exact_exhaustive`) | **Null / Invariant** |
| **`Scaffolding_Metacognitive_Interaction`** | **+0.021** | [-0.167, +0.188] | 1.0000 (`exact_exhaustive`) | **Scaffolding-Invariant Metacognition** |

---

## 3. Scientific Gate Synthesis for Horizon 1 Closeout

1. **Pre-Feedback Correctness Prediction:** Evaluates whether subjective confidence discriminates impending errors prior to external feedback.
2. **Item-Paired Framing Control:** Both Self and Observer evaluate the exact same first-order decisions made by `agent_alpha` under matched public evidence.
3. **Persistence Scaffolding Interaction:** Measures whether explicit Level-1 state shifts the metacognitive gap between internal self-framing and external observer evaluation.