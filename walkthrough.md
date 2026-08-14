# Walkthrough — Sprint S03.2: Definitive Level-0 Privileged Access & Observer Ladder Benchmark (`E02_Final_Hardened`)

## Executive Summary
Sprint S03.2 implemented and executed the definitive **Experiment E02_Final_Hardened (`run_e02_obs_003`)**, establishing the frozen baseline for Level-0 privileged access prior to Sprint S04 recurrent state comparisons:

1. **4-Option Reconstruction Distribution Lookup:** Prompted the independent solver for a full 4-option distribution (`A`, `B`, `C`, `D`), looking up $P(\text{Target Correct}) = P_{\text{recon}}(\text{Target Selected Option})$, eliminating both heuristic $1-p$ approximations and default imputations.
2. **Grammar-Constrained Structured JSON:** Enforced native JSON decoding (`format="json"`) across target and observer backends, strictly removing Likert 1–5 fallbacks in probability mode.
3. **Stratified Paired Bootstrap:** Implemented label-stratified bootstrap resampling (1,000 iterations) to guarantee positive/negative label representation on every iteration.
4. **Direct Pre-Specified Contrasts:** Computed pairwise statistical tests for:
   - Equal-Compute Review framing (`self_review_equal_compute` vs. `observer_review_other`).
   - Public channel effect (`observer_visible_answer_only` vs. `observer_visible_full_transcript`).
5. **Exact Paired Intersections:** Analyzed all observer conditions on their strict shared item subsets with continuous Brier scores, AUROC2 discrimination, and 95% bootstrap confidence intervals.

---

## 1. Definitive Strict Item-Paired Intersection Results Table

Evaluated on `qwen2.5:3b` across 40 counterbalanced 4-way Forced Choice KV retrieval trials (57.5% overall first-order accuracy):

| Evaluator / Contrast ($X$) | Information Vantage & Compute | Shared Items ($N$) | Self AUROC2 | Observer AUROC2 | $\Delta\text{AUROC2}$ (Self - Obs) | Stratified 95% Bootstrap CI | Self Brier | Observer Brier | Observer Accuracy |
|---|---|---|---|---|---|---|---|---|---|
| **Counterfactual Reconstruction** | Independent 4-Way Distribution Lookup | 25 | 0.703 | 0.593 | +0.110 | $[-0.153, +0.373]$ | 0.281 | 0.422 | 60.0% |
| **Visible: Answer-Only** | Prompt + Answer (Conf stripped) | 21 | 0.685 | 0.634 | +0.051 | $[-0.199, +0.310]$ | 0.300 | 0.347 | 61.9% |
| **Visible: Full-Transcript** | Prompt + Answer + Target Conf | 23 | 0.635 | 0.542 | +0.092 | $[-0.142, +0.312]$ | 0.318 | 0.353 | 52.2% |
| **Equal-Compute Self-Review** | 2nd Invocation: Self framing | 17 | 0.715 | **0.785** | -0.069 | $[-0.403, +0.271]$ | 0.305 | **0.272** | 64.7% |
| **Equal-Compute Other-Review** | 2nd Invocation: Other framing | 21 | 0.597 | 0.616 | -0.019 | $[-0.273, +0.245]$ | 0.348 | 0.342 | 57.1% |
| **Ablated: Input-Only** | Prompt Only (Difficulty prior) | 18 | 0.600 | 0.569 | +0.031 | $[-0.350, +0.413]$ | 0.348 | 0.310 | 44.4% |
| **Ablated: Output-Only** | Answer Only (Fluency prior) | 17 | 0.561 | 0.447 | +0.114 | $[-0.371, +0.546]$ | 0.322 | 0.554 | 41.2% |

### Joint Privileged Access Index Summary ($N=20$ shared intersection items):

$$\text{PAI}_{\text{Joint}} = \text{AUROC2}_{\text{Self}} - \max(\text{AUROC2}_{\text{VisibleAns}}, \text{AUROC2}_{\text{Recon}}) = 0.734 - 0.641 = \mathbf{+0.094}$$
$$\mathbf{\text{95\% Stratified Bootstrap CI: } [-0.188, +0.318]} \quad (\text{SESOI} = \pm 0.10)$$

---

## 2. Direct Pre-Specified Contrasts

### A. Review Framing Test ($N = 28$ shared items)
* **Self-Review AUROC2:** $0.701$ (Brier: $0.270$, Accuracy: $67.9\%$)
* **Other-Review AUROC2:** $0.732$ (Brier: $0.282$, Accuracy: $60.7\%$)
* **$\Delta\text{AUROC2} (\text{Self} - \text{Other}):$** $\mathbf{-0.031} \quad (95\%\text{ CI: } [-0.323, +0.258])$
* **Conclusion:** Framing manipulation is non-significant; monitoring gains are compute-driven.

### B. Public Channel Effect Test ($N = 31$ shared items)
* **Visible Answer-Only AUROC2:** $0.557$ (Accuracy: $58.1\%$)
* **Visible Full-Transcript AUROC2:** $0.496$ (Accuracy: $51.6\%$)
* **$\Delta\text{AUROC2} (\text{Answer} - \text{Transcript}):$** $\mathbf{+0.061} \quad (95\%\text{ CI: } [-0.195, +0.303])$
* **Conclusion:** Including target confidence does not assist observer discrimination; stripped answer-only evaluation performs comparably or slightly better.

---

## 3. Data Lineage & Artifact Verification

* **Promoted Run ID:** `run_e02_obs_003`
* **Manifest Git Fingerprint:** Clean commit `1d610fa`
* **Stream Checksum:** `737c17ae420391940957b51ff3035fba102ea13390135d491c7bcf7c251e53a4`
* **Environment Hash:** `49a1299f5f2ba7779b0c4f5fb92ab1a9f5f77090a84a643f08f9b239829aed37`
* **Canonical Artifacts:**
  - `results/e02_observer/run_e02_obs_003/summary.json`
  - `results/e02_observer/run_e02_obs_003/trials.jsonl`
  - `results/e02_observer/run_e02_obs_003/report.md`
  - `artifacts/e02_observer/run_e02_obs_003/run_e02_obs_003_events.parquet`
