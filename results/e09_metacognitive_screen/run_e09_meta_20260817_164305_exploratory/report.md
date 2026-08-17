# Experiment E09: Metacognitive Continuity & Future-Failure Screen Report (Sprint S09b)

**Run ID:** `run_e09_meta_20260817_164305_exploratory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-17T16:50:02.523590+00:00  
**Scope:** 4 Multi-Source Episodes | 80 Total Metacognitive Probes  
**Primary Question:** *Does scaffolded persistence improve an agent's metacognitive calibration and future-failure resolution over an external observer?*  

---

## 1. Metacognitive Calibration Breakdown (Brier Score & AUROC Resolution)

| Evaluator | Memory Format | Trials | Accuracy | Mean Confidence | Brier Score (Lower is Better) | AUROC Error Prediction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Auditing Observer (gamma)** | `Scaffolded Persistence` | 20 | 80.0% | 88.3% | **0.2937** | **0.344** |
| **Auditing Observer (gamma)** | `Transcript-Only` | 20 | 80.0% | 81.8% | **0.2309** | **0.617** |
| **Primary Agent (Self / alpha)** | `Scaffolded Persistence` | 20 | 65.0% | 85.3% | **0.3014** | **0.604** |
| **Primary Agent (Self / alpha)** | `Transcript-Only` | 20 | 85.0% | 73.5% | **0.2317** | **0.471** |

---

## 2. Self vs Observer Metacognitive Advantage ($\Delta_{\text{meta}}$)

- **Self vs Observer Metacognitive Advantage (Transcript-Only):** **-0.147 AUROC**
- **Self vs Observer Metacognitive Advantage (Scaffolded State):** **+0.261 AUROC**

---

## 3. Scientific Gate Synthesis for Horizon 1 Closeout

1. **Metacognitive Calibration:** Does the agent accurately calibrate confidence against its actual empirical error distribution?
2. **Future-Failure Resolution:** Can subjective confidence discriminate impending attribution errors prior to feedback?
3. **Privileged Self-Access:** Does internal self-evaluation provide an error-predictive advantage over an external observer inspecting identical scaffolded representations?