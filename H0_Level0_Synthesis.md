# Horizon 0: Level-0 Privileged Access & Behavioral Baseline Synthesis

**Program Phase:** Horizon 0 / Level 0 (Stateless / Feedforward Base Evaluation)  
**Reference Benchmark:** `run_e02_obs_005` (Sprint S03.4)  
**Target Model:** `Qwen2.5:3B-Instruct` (Deterministic Greedy Decoding, `temp=0.0`, `seed=42`)  
**Epistemic Status:** Measurement-valid Level-0 reference baseline established; no positive privileged-access advantage resolved.

---

## 1. Executive Summary & Scientific Findings

Horizon 0 (Level 0) of the Recurrence research program was designed to establish a measurement-valid reference baseline for self-monitoring in feedforward models prior to introducing persistent state or latent recurrence:

> **Level-0 Null Expectation:** In a standard autoregressive feedforward transformer, a model exhibits **no privileged access** to its own internal decision accuracy beyond what can be inferred by external observers inspecting its inputs, outputs, or counterfactual reconstructions.

Following four iterative sprints of hardening (S01 through S03.4) that systematically eliminated construct contamination, prompt recency biases, metric direction inversions, parser fragility, and output ungrammaticality, the definitive benchmark `run_e02_obs_005` achieved **100% measurement compliance (40/40 trials across all 6 primary conditions)**, cleanly passing the pre-specified $\ge 90\%$ validity gate.

### Headline Measurements ($N=40$ Exact Item-Paired Intersections):
1. **Contemporaneous Self-Confidence Scarcely Discriminates Accuracy ($\text{AUROC2} \approx 0.517$, Brier $\approx 0.396$):**
   When the model generates an answer and self-reports confidence in the same forward pass, its reported confidence is virtually identical whether it is correct ($P_{\text{mean}} = 74.5\%$) or incorrect ($P_{\text{mean}} = 73.0\%$).
2. **External Observers Exploit First-Order Behavioral Cues ($\text{AUROC2} \approx 0.678$, Brier $\approx 0.290$):**
   An external observer viewing only the task prompt and the model's selected option letter achieves substantially higher discrimination ($P_{\text{mean}|\text{correct}} = 49.1\%$, $P_{\text{mean}|\text{incorrect}} = 21.2\%$) and binary classification accuracy (62.5%).
3. **Privileged Access Index is Non-Positive ($\text{PAI} = -0.161$, 95% Stratified Bootstrap CI: $[-0.428, +0.055]$):**
   The joint strongest-observer statistic ($\max(\text{VisibleAns}, \text{Recon}, \text{InputOnly})$) excludes a meaningful self-advantage ($\ge +0.10$ SESOI margin).
4. **Epistemic Conclusion:**
   **Measurement-valid Level-0 reference baseline established; no positive privileged-access advantage resolved.** While the joint bound rules out a positive self-advantage $\ge +0.10$, individual paired contrasts remain wide enough that strict statistical equivalence ($\text{PAI} \approx 0$) cannot be asserted.

---

## 2. Strict Item-Paired Intersection Results Table ($N=40$)

Evaluated on `Qwen2.5:3b` across 40 counterbalanced 4-way Forced Choice KV retrieval trials (57.5% overall first-order accuracy; Semantic: 65.0%, Opaque: 50.0%):

| Evaluator / Vantage | Information Available | Shared $N$ | Self AUROC2 | Observer AUROC2 | $\Delta\text{AUROC2}$ (Self - Obs) | Stratified 95% Bootstrap CI | Self Brier | Observer Brier | Observer Accuracy |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Counterfactual Reconstruction** | Independent 4-Way Distribution Lookup | 40 | 0.517 | 0.573 | -0.056 | $[-0.272, +0.174]$ | 0.396 | 0.369 | 47.5% |
| **Visible: Answer-Only** | Prompt + Answer (Conf stripped) | 40 | 0.517 | **0.678** | **-0.161** | $[-0.428, +0.104]$ | 0.396 | **0.290** | **62.5%** |
| **Visible: Full-Transcript** | Prompt + Answer + Target Conf | 40 | 0.517 | 0.574 | -0.058 | $[-0.275, +0.160]$ | 0.396 | 0.348 | 60.0% |
| **Equal-Compute Self-Review** | 2nd Invocation: Self framing | 40 | 0.517 | 0.428 | +0.088 | $[-0.168, +0.367]$ | 0.396 | 0.483 | 37.5% |
| **Equal-Compute Other-Review** | 2nd Invocation: Other framing | 40 | 0.517 | 0.496 | +0.020 | $[-0.192, +0.221]$ | 0.396 | 0.474 | 47.5% |
| **Ablated: Input-Only** | Prompt Only (Difficulty prior) | 40 | 0.517 | 0.527 | -0.010 | $[-0.265, +0.239]$ | 0.396 | 0.323 | 57.5% |
| **Ablated: Output-Full-Response** | Full Output (Fluency prior) | 40 | 0.517 | 0.684 | -0.168 | $[-0.403, +0.109]$ | 0.396 | 0.276 | 65.0% |

$$\text{Point PAI} = \text{AUROC2}_{\text{Self}} - \max(\text{AUROC2}_{\text{VisAns}}, \text{AUROC2}_{\text{Recon}}, \text{AUROC2}_{\text{InputOnly}}) = 0.517 - 0.678 = \mathbf{-0.161}$$
$$\mathbf{\text{Stratified 95\% Bootstrap CI: } [-0.428, +0.055]} \quad (\text{SESOI margin } \pm 0.10)$$

---

## 3. Evolution of Findings & Measurement Failures Resolved

The journey from initial exploratory scouting to the frozen Level-0 baseline demonstrates how subtle measurement bugs can mimic or distort metacognitive signals:

```
[Sprint S01: Scout] ──► Tokenization Artifacts & Recency Bias
       │
[Sprint S02: Task Matrix] ──► 4-Way Forced Choice Standardization (57.5% Acc)
       │
[Sprint S03.0: Observers] ──► Probability Direction Inversion (Scored as P(Eval) not P(Target Correct))
       │
[Sprint S03.1: Paired Analysis] ──► Reconstruction Heuristic Bug (1 - p invalid for 4-way choice)
       │
[Sprint S03.2: 4-Way Distribution] ──► Parser Non-Compliance (Ollama format="json" unconstrained)
       │
[Sprint S03.3: Rejection Bounds] ──► Fallback Ground-Truth Leakage & Output Entanglement
       │
[Sprint S03.4: JSON Schemas] ──► 100% Valid Bounded Integer Schemas & Decoupled Validity (run_005)
```

### Key Measurement Lessons:
- **Lesson 1 (Tokenization & Recency in S01):** Free generation in key-value retrieval conflated binding retrieval with BPE token boundary fragmentation and prompt-position recency. Moving to counterbalanced 4-way forced choice (DR-0012) stabilized base accuracy at $57.5\%$ (in the psychophysically optimal $50\text{--}65\%$ regime) with zero option-position bias.
- **Lesson 2 (Probability Direction in S03.0):** Observers evaluating "Evaluation: INCORRECT, Confidence: 5" were initially scored as high probability of target correctness. Standardizing all evaluators strictly on the semantic quantity $P(\text{Target Correct}) \in [0.0, 1.0]$ resolved this fundamental sign error.
- **Lesson 3 (4-Way Reconstruction in S03.1):** Assuming $P(\text{Target Correct}) = 1 - P(\text{Recon Choice})$ when reconstruction disagreed was mathematically invalid for a 4-choice task. Replacing it with an explicit 4-way probability distribution lookup $P_{\text{recon}}(\text{Target Option})$ created a mathematically valid counterfactual baseline.
- **Lesson 4 (Constrained Decoding in S03.4):** Relying on raw text prompts with `format="json"` produced malformed keys, trailing text, and negative numbers. Moving to native JSON Schema dictionaries with `type: "integer"`, `minimum: 0, maximum: 100`, and `enum: ["A","B","C","D"]` elevated primary compliance from $37.5\%$ to $100.0\%$.

---

## 4. Substantive Scientific Insights from Level-0

### A. Contemporaneous Self-Confidence Is Flat
In single-turn feedforward generation, the model's explicit self-confidence scarcely discriminated correct from incorrect responses ($\text{AUROC2} \approx 0.52$). The model exhibits a ceiling-anchored confidence bias ($\sim 74\%$) regardless of whether its retrieval succeeded or failed.

### B. External Visibility Superiority
An independent observer reading the proposed answer achieves $\text{AUROC2} = 0.678$. A working hypothesis is that evaluating a static proposition unburdens the model from the generative process that selected the candidate option, though the underlying mechanism remains to be established.

### C. Direct Pairwise Contrasts Remain Unresolved
- **Public Channel Effect ($N=40$):** Presenting the target's explicit confidence report did not improve observer discrimination (Answer-Only AUROC2 $0.678$ vs. Full-Transcript AUROC2 $0.574$, $\Delta = +0.104$, $95\%$ CI: $[-0.145, +0.354]$). While the point estimate descriptively favors stripped answer-only monitoring, the direct pairwise difference remains statistically unresolved.
- **Review Framing Effect ($N=40$):** Contrasting second-invocation self-review against other-review resolved no framing effect (Self-Review AUROC2 $0.428$ vs. Other-Review AUROC2 $0.496$, $\Delta = -0.068$, $95\%$ CI: $[-0.318, +0.198]$).

---

## 5. What Horizon 0 / Level 0 Does Not Establish

To ensure rigorous epistemic boundaries, Horizon 0 explicitly does **not** establish the following:

1. **No Generalization Across Models:** Tested exclusively on `Qwen2.5:3B-Instruct` in one local quantized engine configuration (`temp=0.0`, `seed=42`).
2. **No Generalization Across Task Families:** Benchmarked exclusively on counterbalanced 4-way forced-choice key-value retrieval.
3. **Behavioral Outputs Only (No Mechanistic Probing):** Measures input/output token behavior only; no internal activation probes, residual stream analyses, or attention head interventions were conducted.
4. **Zero Recurrent / Multi-Turn Dynamics:** Evaluates strictly single-turn feedforward generation; does not test persistent state accumulation or recurrent hidden states.
5. **No Consciousness or Phenomenal Claims:** Governed strictly by DR-0001; operational metacognitive and calibration metrics do not establish subjective experience or phenomenality.

---

## 6. What Questions Level-1 (Horizon 1) Is Meant to Answer

With Level-0 rigorously established as a non-privileged baseline ($\text{PAI} \le 0$), we now have an unambiguous reference point for evaluating persistent state and memory.

### The Research Question for Level 1:
> **Core Question for Level 1:** Does maintaining persistent explicit memory (conversation transcripts, structured state dictionaries, model summaries) across multi-turn trajectories create **positive privileged access** ($\text{PAI} > 0$) or meaningfully enhance metacognitive calibration compared to external observers who inspect only current-turn outputs?

### Hypotheses for Level 1:
1. **$H_1^{\text{accuracy}}$ (State Retention):** Which explicit memory representation (full transcript, structured JSON state, narrative summary) best resists multi-turn context decay under sequential overwrites?
2. **$H_1^{\text{meta}}$ (Metacognitive Calibration):** Does persistent state history allow the model to know *when it has lost track* of an updated variable (producing calibrated low confidence on corrupted state)?
3. **$H_1^{\text{efficiency}}$ (Compute Frontier):** What is the trade-off between token footprint ($T_{\text{prompt}}$) and retrieval accuracy across explicit memory architectures before introducing latent recurrent representations?

---

## 7. Frozen Provenance Summary

| Artifact | Pointer / Value |
|---|---|
| **Frozen Code Commit** | [`f4ace47`](file:///c:/Users/admir/Github/recurrence) |
| **Promoted Results Commit** | [`c6c90e6`](file:///c:/Users/admir/Github/recurrence) |
| **Promoted Run ID** | `run_e02_obs_005` |
| **Dataset Stream Checksum** | `f9aa1553ef8d69011e0ecbed36e39e2e15417f0782e9398e2e4282418b64eb0f` |
| **Environment Fingerprint** | `ee73e35d02efcd5248ec0ccc21b4ecff5e532ba92bbfb893fa6580f7b734e11b` |
| **Decision Record** | [DR-0013](file:///c:/Users/admir/Github/recurrence/Decision%20Log.md) |
