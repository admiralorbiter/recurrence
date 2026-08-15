# Experiment E02 (Sprint S03.4): Level-0 Privileged Access & Observer Ladder Report

## 1. Executive Summary

Empirical measurement summary for **Experiment E02 (`run_e02_obs_qwen7b_001`)** on `qwen2.5:7b` across 40 counterbalanced 4-way Forced Choice KV retrieval trials (First-order accuracy: 30.0%).

All measurements standardize on $P(\text{Target Correct}) \in [0.0, 1.0]$. Every contrast row below is computed strictly over its exact shared item intersection subset ($N$).

### Strict Item-Paired Intersection Results Table

| Evaluator Condition | Information Vantage & Compute | Shared Items ($N$) | Self AUROC2 | Observer AUROC2 | $\Delta\text{AUROC2}$ (Self - Obs) | Stratified 95% Bootstrap CI | Self Brier | Observer Brier | Observer Accuracy |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Counterfactual Reconstruction** | Independent 4-Way Distribution Lookup | 27 | 0.600 | 0.594 | +0.006 | [-0.297, 0.312] | 0.341 | 0.316 | 63.0% |
| **Visible: Answer-Only** | Prompt + Answer (Conf stripped) | 40 | 0.522 | 0.537 | -0.015 | [-0.263, 0.246] | 0.293 | 0.325 | 62.5% |
| **Visible: Full-Transcript** | Prompt + Answer + Target Conf | 40 | 0.522 | 0.667 | -0.144 | [-0.400, 0.107] | 0.293 | 0.228 | 72.5% |
| **Equal-Compute Self-Review** | 2nd Invocation: Self framing | 40 | 0.522 | 0.443 | +0.079 | [-0.186, 0.327] | 0.293 | 0.310 | 62.5% |
| **Equal-Compute Other-Review** | 2nd Invocation: Other framing | 40 | 0.522 | 0.673 | -0.150 | [-0.406, 0.130] | 0.293 | 0.235 | 72.5% |
| **Ablated: Input-Only** | Prompt Only (Difficulty prior) | 40 | 0.522 | 0.463 | +0.060 | [-0.229, 0.332] | 0.293 | 0.288 | 67.5% |
| **Ablated: Output-Full-Response** | Full Output (Fluency prior) | 40 | 0.522 | 0.406 | +0.116 | [-0.118, 0.333] | 0.293 | 0.375 | 52.5% |

### Epistemic Status & Interpretation:
> [!WARNING]
> **Measurement Validity Gate Failed (Minimum Primary Compliance: 67.5% < 90.0%).**
> Inferential Privileged Access Index (PAI) and observer contrasts are reported for diagnostic/scouting purposes only and do NOT support a Level-0 privileged-access conclusion.
>
> **Diagnostic PAI (Unpromoted, Shared $N=27$):**
> $\text{Diagnostic PAI} = +0.006 \quad (95\%\text{ CI: } [-0.300, 0.218])$

---

## 2. Direct Pre-Specified Pairwise Contrasts

### A. Review Framing Test ($N = 40$ shared items)
* **Self-Review AUROC2:** 0.443 (Brier: 0.310, Acc: 62.5%)
* **Other-Review AUROC2:** 0.673 (Brier: 0.235, Acc: 72.5%)
* **$\Delta\text{AUROC2} (\text{Self} - \text{Other}):$** $\mathbf{-0.229} \quad (95\%\text{ CI: } [-0.518, 0.080])$
* **Status:** Diagnostic comparison only (measurement gate failed).

### B. Public Channel Effect Test ($N = 40$ shared items)
* **Visible Answer-Only AUROC2:** 0.537 (Acc: 62.5%)
* **Visible Full-Transcript AUROC2:** 0.667 (Acc: 72.5%)
* **$\Delta\text{AUROC2} (\text{Answer} - \text{Transcript}):$** $\mathbf{-0.129} \quad (95\%\text{ CI: } [-0.375, 0.115])$
* **Status:** Diagnostic comparison only (measurement gate failed).

---

## 3. Compliance & Gate Verification

* **Primary Condition Compliance:** {
  "self_immediate": 1.0,
  "self_review_equal_compute": 1.0,
  "observer_review_other": 1.0,
  "observer_visible_answer_only": 1.0,
  "observer_visible_full_transcript": 1.0,
  "observer_reconstruction": 0.675
}
* **Minimum Primary Compliance:** 67.5%
* **Compliance Hard Gate (Min $\ge 90\%$):** FAILED
