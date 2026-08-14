# Experiment E02 (Sprint S03.3): Level-0 Privileged Access & Observer Ladder Report

## 1. Executive Summary

Empirical measurement summary for **Experiment E02 (`run_e02_obs_004`)** on `qwen2.5:3b` across 40 counterbalanced 4-way Forced Choice KV retrieval trials (First-order accuracy: 45.0%).

All measurements standardize on $P(\text{Target Correct}) \in [0.0, 1.0]$. Every contrast row below is computed strictly over its exact shared item intersection subset ($N$).

### Strict Item-Paired Intersection Results Table

| Evaluator Condition | Information Vantage & Compute | Shared Items ($N$) | Self AUROC2 | Observer AUROC2 | $\Delta\text{AUROC2}$ (Self - Obs) | Stratified 95% Bootstrap CI | Self Brier | Observer Brier | Observer Accuracy |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Counterfactual Reconstruction** | Independent 4-Way Distribution Lookup | 9 | 0.583 | 0.722 | -0.139 | [-0.611, 0.250] | 0.289 | 0.389 | 44.4% |
| **Visible: Answer-Only** | Prompt + Answer (Conf stripped) | 19 | 0.500 | 0.489 | +0.011 | [-0.339, 0.350] | 0.403 | 0.454 | 47.4% |
| **Visible: Full-Transcript** | Prompt + Answer + Target Conf | 26 | 0.506 | 0.634 | -0.128 | [-0.390, 0.140] | 0.397 | 0.305 | 61.5% |
| **Equal-Compute Self-Review** | 2nd Invocation: Self framing | 19 | 0.583 | 0.644 | -0.061 | [-0.428, 0.333] | 0.318 | 0.431 | 47.4% |
| **Equal-Compute Other-Review** | 2nd Invocation: Other framing | 21 | 0.569 | 0.662 | -0.093 | [-0.389, 0.255] | 0.339 | 0.295 | 61.9% |
| **Ablated: Input-Only** | Prompt Only (Difficulty prior) | 19 | 0.381 | 0.483 | -0.102 | [-0.335, 0.131] | 0.430 | 0.308 | 52.6% |
| **Ablated: Output-Full-Response** | Full Output (Fluency prior) | 18 | 0.375 | 0.590 | -0.215 | [-0.667, 0.292] | 0.440 | 0.374 | 66.7% |

### Joint Privileged Access Index Summary ($N=4$ shared items):
$$\text{Point PAI} = \text{AUROC2}_{\text{Self}} - \max(\text{AUROC2}_{\text{VisAns}}, \text{AUROC2}_{\text{Recon}}) = 0.625 - 1.000 = \mathbf{-0.375}$$
$$\mathbf{\text{Stratified 95\% Bootstrap CI: } [-0.750, 0.000]} \quad (\text{SESOI margin } \pm 0.1)$$

**Scientific Interpretation:**
No positive Level-0 privileged-access effect was statistically resolved at the present sample size ($N=40$). The stratified 95% bootstrap confidence interval spans zero and remains compatible with both modest observer advantage and positive self advantage.

---

## 2. Direct Pre-Specified Pairwise Contrasts

### A. Review Framing Test ($N = 24$ shared items)
* **Self-Review AUROC2:** 0.481 (Brier: 0.424, Acc: 45.8%)
* **Other-Review AUROC2:** 0.559 (Brier: 0.340, Acc: 54.2%)
* **$\Delta\text{AUROC2} (\text{Self} - \text{Other}):$** $\mathbf{-0.078} \quad (95\%\text{ CI: } [-0.400, 0.226])$
* **Conclusion:** No framing effect was statistically resolved in this sample.

### B. Public Channel Effect Test ($N = 29$ shared items)
* **Visible Answer-Only AUROC2:** 0.424 (Acc: 37.9%)
* **Visible Full-Transcript AUROC2:** 0.644 (Acc: 58.6%)
* **$\Delta\text{AUROC2} (\text{Answer} - \text{Transcript}):$** $\mathbf{-0.220} \quad (95\%\text{ CI: } [-0.460, 0.020])$
* **Conclusion:** No beneficial transcript-confidence effect was resolved.

---

## 3. Compliance & Gate Verification

* **Primary Condition Compliance:** {
  "self_immediate": 0.7,
  "self_review_equal_compute": 0.725,
  "observer_review_other": 0.825,
  "observer_visible_answer_only": 0.75,
  "observer_visible_full_transcript": 0.95,
  "observer_reconstruction": 0.375
}
* **Minimum Primary Compliance:** 37.5%
* **Compliance Hard Gate (Min $\ge 90\%$):** FAILED
