# Experiment E02 (Sprint S03.3): Level-0 Privileged Access & Observer Ladder Report

## 1. Executive Summary

Empirical measurement summary for **Experiment E02 (`run_e02_obs_005`)** on `qwen2.5:3b` across 40 counterbalanced 4-way Forced Choice KV retrieval trials (First-order accuracy: 57.5%).

All measurements standardize on $P(\text{Target Correct}) \in [0.0, 1.0]$. Every contrast row below is computed strictly over its exact shared item intersection subset ($N$).

### Strict Item-Paired Intersection Results Table

| Evaluator Condition | Information Vantage & Compute | Shared Items ($N$) | Self AUROC2 | Observer AUROC2 | $\Delta\text{AUROC2}$ (Self - Obs) | Stratified 95% Bootstrap CI | Self Brier | Observer Brier | Observer Accuracy |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Counterfactual Reconstruction** | Independent 4-Way Distribution Lookup | 40 | 0.517 | 0.573 | -0.056 | [-0.272, 0.174] | 0.396 | 0.369 | 47.5% |
| **Visible: Answer-Only** | Prompt + Answer (Conf stripped) | 40 | 0.517 | 0.678 | -0.161 | [-0.428, 0.104] | 0.396 | 0.290 | 62.5% |
| **Visible: Full-Transcript** | Prompt + Answer + Target Conf | 40 | 0.517 | 0.574 | -0.058 | [-0.275, 0.160] | 0.396 | 0.348 | 60.0% |
| **Equal-Compute Self-Review** | 2nd Invocation: Self framing | 40 | 0.517 | 0.428 | +0.088 | [-0.168, 0.367] | 0.396 | 0.483 | 37.5% |
| **Equal-Compute Other-Review** | 2nd Invocation: Other framing | 40 | 0.517 | 0.496 | +0.020 | [-0.192, 0.221] | 0.396 | 0.474 | 47.5% |
| **Ablated: Input-Only** | Prompt Only (Difficulty prior) | 40 | 0.517 | 0.527 | -0.010 | [-0.265, 0.239] | 0.396 | 0.323 | 57.5% |
| **Ablated: Output-Full-Response** | Full Output (Fluency prior) | 40 | 0.517 | 0.684 | -0.168 | [-0.403, 0.109] | 0.396 | 0.276 | 65.0% |

### Epistemic Status & Interpretation:
**Scientific Interpretation (Measurement Gate Passed):**
Measurements satisfied the pre-specified validity gate (min primary compliance 100.0% $\ge 90\%$).
$$\text{Point PAI} = \text{AUROC2}_{\text{Self}} - \max(\text{AUROC2}_{\text{VisAns}}, \text{AUROC2}_{\text{Recon}}) = 0.517 - 0.678 = \mathbf{-0.161}$$
$$\mathbf{\text{Stratified 95\% Bootstrap CI: } [-0.428, 0.070]} \quad (\text{SESOI margin } \pm 0.1)$$
No positive Level-0 privileged-access effect was statistically resolved at the present sample size ($N=40$). The stratified 95% bootstrap confidence interval spans zero and remains compatible with both modest observer advantage and positive self advantage.

---

## 2. Direct Pre-Specified Pairwise Contrasts

### A. Review Framing Test ($N = 40$ shared items)
* **Self-Review AUROC2:** 0.428 (Brier: 0.483, Acc: 37.5%)
* **Other-Review AUROC2:** 0.496 (Brier: 0.474, Acc: 47.5%)
* **$\Delta\text{AUROC2} (\text{Self} - \text{Other}):$** $\mathbf{-0.068} \quad (95\%\text{ CI: } [-0.318, 0.198])$
* **Status:** No framing effect was statistically resolved in this sample.

### B. Public Channel Effect Test ($N = 40$ shared items)
* **Visible Answer-Only AUROC2:** 0.678 (Acc: 62.5%)
* **Visible Full-Transcript AUROC2:** 0.574 (Acc: 60.0%)
* **$\Delta\text{AUROC2} (\text{Answer} - \text{Transcript}):$** $\mathbf{+0.104} \quad (95\%\text{ CI: } [-0.145, 0.354])$
* **Status:** No beneficial transcript-confidence effect was resolved.

---

## 3. Compliance & Gate Verification

* **Primary Condition Compliance:** {
  "self_immediate": 1.0,
  "self_review_equal_compute": 1.0,
  "observer_review_other": 1.0,
  "observer_visible_answer_only": 1.0,
  "observer_visible_full_transcript": 1.0,
  "observer_reconstruction": 1.0
}
* **Minimum Primary Compliance:** 100.0%
* **Compliance Hard Gate (Min $\ge 90\%$):** PASSED
