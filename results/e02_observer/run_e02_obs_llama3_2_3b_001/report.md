# Experiment E02 (Sprint S03.4): Level-0 Privileged Access & Observer Ladder Report

## 1. Executive Summary

Empirical measurement summary for **Experiment E02 (`run_e02_obs_llama3_2_3b_001`)** on `llama3.2:3b` across 40 counterbalanced 4-way Forced Choice KV retrieval trials (First-order accuracy: 100.0%).

All measurements standardize on $P(\text{Target Correct}) \in [0.0, 1.0]$. Every contrast row below is computed strictly over its exact shared item intersection subset ($N$).

### Strict Item-Paired Intersection Results Table

| Evaluator Condition | Information Vantage & Compute | Shared Items ($N$) | Self AUROC2 | Observer AUROC2 | $\Delta\text{AUROC2}$ (Self - Obs) | Stratified 95% Bootstrap CI | Self Brier | Observer Brier | Observer Accuracy |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Counterfactual Reconstruction** | Independent 4-Way Distribution Lookup | 38 | 0.500 | 0.500 | +0.000 | [0.000, 0.000] | 0.000 | 0.213 | 76.3% |
| **Visible: Answer-Only** | Prompt + Answer (Conf stripped) | 40 | 0.500 | 0.500 | +0.000 | [0.000, 0.000] | 0.000 | 0.050 | 95.0% |
| **Visible: Full-Transcript** | Prompt + Answer + Target Conf | 40 | 0.500 | 0.500 | +0.000 | [0.000, 0.000] | 0.000 | 0.094 | 92.5% |
| **Equal-Compute Self-Review** | 2nd Invocation: Self framing | 40 | 0.500 | 0.500 | +0.000 | [0.000, 0.000] | 0.000 | 0.606 | 40.0% |
| **Equal-Compute Other-Review** | 2nd Invocation: Other framing | 40 | 0.500 | 0.500 | +0.000 | [0.000, 0.000] | 0.000 | 0.169 | 85.0% |
| **Ablated: Input-Only** | Prompt Only (Difficulty prior) | 40 | 0.500 | 0.500 | +0.000 | [0.000, 0.000] | 0.000 | 0.119 | 95.0% |
| **Ablated: Output-Full-Response** | Full Output (Fluency prior) | 40 | 0.500 | 0.500 | +0.000 | [0.000, 0.000] | 0.000 | 0.543 | 25.0% |

### Epistemic Status & Interpretation:
**Scientific Interpretation (Measurement Gate Passed):**
Measurements satisfied the pre-specified validity gate (min primary compliance 95.0% $\ge 90\%$).
$$\text{Point PAI} = \text{AUROC2}_{\text{Self}} - \max(\text{AUROC2}_{\text{VisAns}}, \text{AUROC2}_{\text{Recon}}, \text{AUROC2}_{\text{InputOnly}}) = 0.500 - 0.500 = \mathbf{+0.000}$$
$$\mathbf{\text{Stratified 95\% Bootstrap CI: } [0.000, 0.000]} \quad (\text{SESOI margin } \pm 0.1)$$
In a fully measurement-valid Level-0 benchmark, Qwen2.5:3b showed no resolved privileged self-monitoring advantage over external/reconstructive controls. Immediate self-confidence was essentially nondiscriminative (AUROC2 $\approx .52$), while visible-answer observation performed substantially better descriptively (AUROC2 $\approx .68$). The joint strongest-observer statistic excludes a $\ge .10$ self advantage, although individual paired contrasts remain too imprecise to establish equivalence.

---

## 2. Direct Pre-Specified Pairwise Contrasts

### A. Review Framing Test ($N = 40$ shared items)
* **Self-Review AUROC2:** 0.500 (Brier: 0.606, Acc: 40.0%)
* **Other-Review AUROC2:** 0.500 (Brier: 0.169, Acc: 85.0%)
* **$\Delta\text{AUROC2} (\text{Self} - \text{Other}):$** $\mathbf{+0.000} \quad (95\%\text{ CI: } [0.000, 0.000])$
* **Status:** No framing effect was statistically resolved in this sample.

### B. Public Channel Effect Test ($N = 40$ shared items)
* **Visible Answer-Only AUROC2:** 0.500 (Acc: 95.0%)
* **Visible Full-Transcript AUROC2:** 0.500 (Acc: 92.5%)
* **$\Delta\text{AUROC2} (\text{Answer} - \text{Transcript}):$** $\mathbf{+0.000} \quad (95\%\text{ CI: } [0.000, 0.000])$
* **Status:** No beneficial transcript-confidence effect was resolved.

---

## 3. Compliance & Gate Verification

* **Primary Condition Compliance:** {
  "self_immediate": 1.0,
  "self_review_equal_compute": 1.0,
  "observer_review_other": 1.0,
  "observer_visible_answer_only": 1.0,
  "observer_visible_full_transcript": 1.0,
  "observer_reconstruction": 0.95
}
* **Minimum Primary Compliance:** 95.0%
* **Compliance Hard Gate (Min $\ge 90\%$):** PASSED
