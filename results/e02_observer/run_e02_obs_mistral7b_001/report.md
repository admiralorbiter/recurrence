# Experiment E02 (Sprint S03.4): Level-0 Privileged Access & Observer Ladder Report

## 1. Executive Summary

Empirical measurement summary for **Experiment E02 (`run_e02_obs_mistral7b_001`)** on `mistral:latest` across 40 counterbalanced 4-way Forced Choice KV retrieval trials (First-order accuracy: 100.0%).

All measurements standardize on $P(\text{Target Correct}) \in [0.0, 1.0]$. Every contrast row below is computed strictly over its exact shared item intersection subset ($N$).

### Strict Item-Paired Intersection Results Table

| Evaluator Condition | Information Vantage & Compute | Shared Items ($N$) | Self AUROC2 | Observer AUROC2 | $\Delta\text{AUROC2}$ (Self - Obs) | Stratified 95% Bootstrap CI | Self Brier | Observer Brier | Forecast Classification Acc |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Counterfactual Reconstruction** | Independent 4-Way Distribution Lookup | 37 | N/A (no errors) | N/A (no errors) | N/A | N/A | 0.000 | 0.211 | 78.4% |
| **Visible: Answer-Only** | Prompt + Answer (Conf stripped) | 40 | N/A (no errors) | N/A (no errors) | N/A | N/A | 0.000 | 0.050 | 95.0% |
| **Visible: Full-Transcript** | Prompt + Answer + Target Conf | 40 | N/A (no errors) | N/A (no errors) | N/A | N/A | 0.000 | 0.000 | 100.0% |
| **Equal-Compute Self-Review** | 2nd Invocation: Self framing | 40 | N/A (no errors) | N/A (no errors) | N/A | N/A | 0.000 | 0.025 | 97.5% |
| **Equal-Compute Other-Review** | 2nd Invocation: Other framing | 40 | N/A (no errors) | N/A (no errors) | N/A | N/A | 0.000 | 0.025 | 97.5% |
| **Ablated: Input-Only** | Prompt Only (Difficulty prior) | 40 | N/A (no errors) | N/A (no errors) | N/A | N/A | 0.000 | 0.001 | 100.0% |
| **Ablated: Output-Full-Response** | Full Output (Fluency prior) | 40 | N/A (no errors) | N/A (no errors) | N/A | N/A | 0.000 | 0.000 | 100.0% |

### Epistemic Status & Interpretation:
> [!NOTE]
> **Ceiling Performance Regime (1st-Order Accuracy: 100.0%):**
> With 0 error trials ($N_{\text{incorrect}} = 0$), Type-2 Signal Detection discrimination ($\text{AUROC2}$) and the Privileged Access Index ($\text{PAI}$) are mathematically non-identifiable / undefined.
> 
> **Metacognitive Sensitivity:** $\text{AUROC2} = \text{N/A}$ (Cannot discriminate correct from incorrect when all items are correct).
> **Privileged Access Index:** $\text{PAI} = \text{N/A}$
> 
> **Informative Calibration & Confidence Dynamics:** Continuous Brier scores and review confidence shifts remain valid for analyzing belief updates under review framing.

---

## 2. Direct Pre-Specified Pairwise Contrasts

### A. Review Framing Test ($N = 40$ shared items)
* **Self-Review AUROC2:** N/A (Brier: 0.025, Forecast Acc: 97.5%)
* **Other-Review AUROC2:** N/A (Brier: 0.025, Forecast Acc: 97.5%)
* **$\Delta\text{AUROC2} (\text{Self} - \text{Other}):$** $\mathbf{N/A} \quad (95\%\text{ CI: } N/A)$
* **Status:** Type-2 AUROC2 undefined at 100% accuracy ceiling. Compare continuous Brier scores and mean confidence shifts instead.

### B. Public Channel Effect Test ($N = 40$ shared items)
* **Visible Answer-Only AUROC2:** N/A (Forecast Acc: 95.0%)
* **Visible Full-Transcript AUROC2:** N/A (Forecast Acc: 100.0%)
* **$\Delta\text{AUROC2} (\text{Answer} - \text{Transcript}):$** $\mathbf{N/A} \quad (95\%\text{ CI: } N/A)$
* **Status:** Type-2 AUROC2 undefined at 100% accuracy ceiling.

---

## 3. Compliance & Gate Verification

* **Primary Condition Compliance:** {
  "self_immediate": 1.0,
  "self_review_equal_compute": 1.0,
  "observer_review_other": 1.0,
  "observer_visible_answer_only": 1.0,
  "observer_visible_full_transcript": 1.0,
  "observer_reconstruction": 0.925
}
* **Minimum Primary Compliance:** 92.5%
* **Compliance Hard Gate (Min $\ge 90\%$):** PASSED
