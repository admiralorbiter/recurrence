# Walkthrough — Sprint S03.4: Level-0 Reference Baseline (`run_e02_obs_005`)

## Executive Summary

Sprint S03.4 hardened the measurement contract, resolved JSON schema non-compliance, and established the frozen **Level-0 Privileged Access Reference Baseline (`run_e02_obs_005`)** on `qwen2.5:3b`:

1. **Structured JSON Schemas with Bounded Enums (`src/recurrence/core/schemas.py`)**:
   - Passed strict JSON schemas with integer ranges $[0, 100]$ directly to Ollama `/api/chat` via GBNF-constrained decoding.
   - Enforced target forced-choice format: `{"answer": "A"|"B"|"C"|"D", "probability": 0..100}`.
   - Enforced reconstruction format: `{"A": 0..100, "B": 0..100, "C": 0..100, "D": 0..100}`.
2. **Decoupled Answer Parsing vs. Forecast/Schema Compliance**:
   - Separated `answer_parse_valid`, `probability_parse_valid`, and `schema_valid`.
   - Guaranteed accurate first-order task evaluation without conflating format quirks with task failure.
3. **100% Measurement Compliance Across All Primary Conditions**:
   - Every evaluator condition achieved **100.0% compliance (40/40 valid measurements)**.
   - Compliance Hard Gate ($\min \ge 90\%$) passed cleanly (**100.0% vs. 90.0% threshold**).
4. **Substantive Level-0 Findings (No Positive Advantage Resolved)**:
   - Immediate self-confidence was essentially nondiscriminative ($\text{AUROC2} \approx 0.517$, Brier $\approx 0.396$).
   - Visible Answer-Only observer performed substantially better descriptively ($\text{AUROC2} \approx 0.678$, Brier $\approx 0.290$, Binary Accuracy: 62.5%).
   - Joint PAI: $\mathbf{-0.161}$ (Stratified 95% Bootstrap CI: $[-0.428, +0.055]$).
   - The joint strongest-observer statistic excludes a $\ge 0.10$ self advantage, though individual paired contrasts remain imprecise.

---

## 1. Strict Item-Paired Intersection Results Table ($N=40$ for all conditions)

Evaluated on `qwen2.5:3b` across 40 counterbalanced 4-way Forced Choice KV retrieval trials (57.5% overall first-order accuracy; Semantic: 75.0%, Opaque: 40.0%):

| Evaluator / Contrast | Information Vantage & Compute | Shared Items ($N$) | Self AUROC2 | Observer AUROC2 | $\Delta\text{AUROC2}$ (Self - Obs) | Stratified 95% Bootstrap CI | Self Brier | Observer Brier | Observer Accuracy |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Counterfactual Reconstruction** | Independent 4-Way Distribution Lookup | 40 | 0.517 | 0.573 | -0.056 | $[-0.272, +0.174]$ | 0.396 | 0.369 | 47.5% |
| **Visible: Answer-Only** | Prompt + Answer (Conf stripped) | 40 | 0.517 | 0.678 | -0.161 | $[-0.428, +0.104]$ | 0.396 | 0.290 | 62.5% |
| **Visible: Full-Transcript** | Prompt + Answer + Target Conf | 40 | 0.517 | 0.574 | -0.058 | $[-0.275, +0.160]$ | 0.396 | 0.348 | 60.0% |
| **Equal-Compute Self-Review** | 2nd Invocation: Self framing | 40 | 0.517 | 0.428 | +0.088 | $[-0.168, +0.367]$ | 0.396 | 0.483 | 37.5% |
| **Equal-Compute Other-Review** | 2nd Invocation: Other framing | 40 | 0.517 | 0.496 | +0.020 | $[-0.192, +0.221]$ | 0.396 | 0.474 | 47.5% |
| **Ablated: Input-Only** | Prompt Only (Difficulty prior) | 40 | 0.517 | 0.527 | -0.010 | $[-0.265, +0.239]$ | 0.396 | 0.323 | 57.5% |
| **Ablated: Output-Full-Response** | Full Output (Fluency prior) | 40 | 0.517 | 0.684 | -0.168 | $[-0.403, +0.109]$ | 0.396 | 0.276 | 65.0% |

### Joint Privileged Access Index Summary ($N=40$ shared items):

$$\text{Point PAI} = \text{AUROC2}_{\text{Self}} - \max(\text{AUROC2}_{\text{VisAns}}, \text{AUROC2}_{\text{Recon}}, \text{AUROC2}_{\text{InputOnly}}) = 0.517 - 0.678 = \mathbf{-0.161}$$
$$\mathbf{\text{Stratified 95\% Bootstrap CI: } [-0.428, +0.055]} \quad (\text{SESOI margin } \pm 0.10)$$

---

## 2. Direct Pre-Specified Contrasts

### A. Review Framing Test ($N = 40$ shared items)
* **Self-Review AUROC2:** $0.428$ (Brier: $0.483$, Accuracy: $37.5\%$)
* **Other-Review AUROC2:** $0.496$ (Brier: $0.474$, Accuracy: $47.5\%$)
* **$\Delta\text{AUROC2} (\text{Self} - \text{Other}):$** $\mathbf{-0.068} \quad (95\%\text{ CI: } [-0.318, +0.198])$
* **Conclusion:** No framing effect resolved.

### B. Public Channel Effect Test ($N = 40$ shared items)
* **Visible Answer-Only AUROC2:** $0.678$ (Brier: $0.290$, Accuracy: $62.5\%$)
* **Visible Full-Transcript AUROC2:** $0.574$ (Brier: $0.348$, Accuracy: $60.0\%$)
* **$\Delta\text{AUROC2} (\text{Answer} - \text{Transcript}):$** $\mathbf{+0.104} \quad (95\%\text{ CI: } [-0.145, +0.354])$
* **Conclusion:** Presenting the target's explicit confidence report does not improve observer AUROC2; stripped answer-only monitoring is slightly more discriminative.

---

## 3. Provenance & Artifact Verification

* **Frozen Code Commit:** [`f4ace47`](file:///c:/Users/admir/Github/recurrence) (`feat(s03.4): enforce structured JSON schemas, decouple answer validity, and gate inferential claims`)
* **Promoted Run Pointer:** [`results/e02_observer/promoted.json`](file:///c:/Users/admir/Github/recurrence/results/e02_observer/promoted.json) $\to$ `run_e02_obs_005`
* **Latest Attempt Pointer:** [`results/e02_observer/latest_attempt.json`](file:///c:/Users/admir/Github/recurrence/results/e02_observer/latest_attempt.json) $\to$ `run_e02_obs_005`
* **Artifact Hashes:**
  - Stream Checksum: `f9aa1553ef8d69011e0ecbed36e39e2e15417f0782e9398e2e4282418b64eb0f`
  - Environment Hash: `ee73e35d02efcd5248ec0ccc21b4ecff5e532ba92bbfb893fa6580f7b734e11b`
