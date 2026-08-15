# Experiment E02 (Sprint S03.4): Level-0 Privileged Access & Observer Ladder Report

## 1. Executive Summary

Empirical measurement summary for **Experiment E02 (`run_e02_obs_qwen1_5b_001`)** on `qwen2.5:1.5b` across 40 counterbalanced 4-way Forced Choice KV retrieval trials (First-order accuracy: 30.0%).

All measurements standardize on $P(\text{Target Correct}) \in [0.0, 1.0]$. Every contrast row below is computed strictly over its exact shared item intersection subset ($N$).

### Strict Item-Paired Intersection Results Table

| Evaluator Condition | Information Vantage & Compute | Shared Items ($N$) | Self AUROC2 | Observer AUROC2 | $\Delta\text{AUROC2}$ (Self - Obs) | Stratified 95% Bootstrap CI | Self Brier | Observer Brier | Observer Accuracy |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Counterfactual Reconstruction** | Independent 4-Way Distribution Lookup | 29 | 0.474 | 0.382 | +0.092 | [-0.134, 0.324] | 0.371 | 0.368 | 55.2% |
| **Visible: Answer-Only** | Prompt + Answer (Conf stripped) | 40 | 0.527 | 0.518 | +0.009 | [-0.177, 0.183] | 0.319 | 0.323 | 57.5% |
| **Visible: Full-Transcript** | Prompt + Answer + Target Conf | 40 | 0.527 | 0.555 | -0.028 | [-0.165, 0.100] | 0.319 | 0.252 | 70.0% |
| **Equal-Compute Self-Review** | 2nd Invocation: Self framing | 40 | 0.527 | 0.531 | -0.004 | [-0.219, 0.205] | 0.319 | 0.288 | 62.5% |
| **Equal-Compute Other-Review** | 2nd Invocation: Other framing | 40 | 0.527 | 0.680 | -0.153 | [-0.351, 0.022] | 0.319 | 0.266 | 65.0% |
| **Ablated: Input-Only** | Prompt Only (Difficulty prior) | 40 | 0.527 | 0.528 | -0.001 | [-0.159, 0.155] | 0.319 | 0.355 | 52.5% |
| **Ablated: Output-Full-Response** | Full Output (Fluency prior) | 40 | 0.527 | 0.503 | +0.024 | [-0.135, 0.225] | 0.319 | 0.301 | 70.0% |

### Epistemic Status & Interpretation:
> [!WARNING]
> **Measurement Validity Gate Failed (Minimum Primary Compliance: 72.5% < 90.0%).**
> Inferential Privileged Access Index (PAI) and observer contrasts are reported for diagnostic/scouting purposes only and do NOT support a Level-0 privileged-access conclusion.
>
> **Diagnostic PAI (Unpromoted, Shared $N=29$):**
> $\text{Diagnostic PAI} = +0.032 \quad (95\%\text{ CI: } [-0.187, 0.158])$

---

## 2. Direct Pre-Specified Pairwise Contrasts

### A. Review Framing Test ($N = 40$ shared items)
* **Self-Review AUROC2:** 0.531 (Brier: 0.288, Acc: 62.5%)
* **Other-Review AUROC2:** 0.680 (Brier: 0.266, Acc: 65.0%)
* **$\Delta\text{AUROC2} (\text{Self} - \text{Other}):$** $\mathbf{-0.149} \quad (95\%\text{ CI: } [-0.315, 0.039])$
* **Status:** Diagnostic comparison only (measurement gate failed).

### B. Public Channel Effect Test ($N = 40$ shared items)
* **Visible Answer-Only AUROC2:** 0.518 (Acc: 57.5%)
* **Visible Full-Transcript AUROC2:** 0.555 (Acc: 70.0%)
* **$\Delta\text{AUROC2} (\text{Answer} - \text{Transcript}):$** $\mathbf{-0.037} \quad (95\%\text{ CI: } [-0.219, 0.131])$
* **Status:** Diagnostic comparison only (measurement gate failed).

---

## 3. Compliance & Gate Verification

* **Primary Condition Compliance:** {
  "self_immediate": 1.0,
  "self_review_equal_compute": 1.0,
  "observer_review_other": 1.0,
  "observer_visible_answer_only": 1.0,
  "observer_visible_full_transcript": 1.0,
  "observer_reconstruction": 0.725
}
* **Minimum Primary Compliance:** 72.5%
* **Compliance Hard Gate (Min $\ge 90\%$):** FAILED
