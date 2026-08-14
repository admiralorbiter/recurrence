# H0: Level-0 Privileged Access & Behavioral Baseline Synthesis

**Program Phase:** Level 0 (Stateless / Feedforward Base Evaluation)  
**Reference Benchmark:** `run_e02_obs_005` (Sprint S03.4)  
**Target Model:** `Qwen2.5:3B-Instruct` (Deterministic Greedy Decoding, `temp=0.0`, `seed=42`)  
**Status:** Frozen Reference Baseline Established; $H_0$ Confirmed (No Positive Privileged Access Resolved)

---

## 1. Executive Summary & Core Scientific Conclusion

Level 0 of the Recurrence research program was designed to test the null hypothesis of self-monitoring:

> **Hypothesis $H_0$:** In a standard autoregressive feedforward transformer, a model has **no privileged access** to its own internal decision accuracy beyond what can be inferred by an external observer from its inputs, generated outputs, or counterfactual reconstructions.

Following four iterative sprints of hardening (S01 through S03.4) that systematically eliminated construct contamination, prompt recency biases, metric direction inversions, parser fragility, and output ungrammaticality, the definitive benchmark `run_e02_obs_005` achieved **100% measurement compliance (40/40 trials across all 6 primary conditions)**.

### Headline Findings:
1. **Contemporaneous Self-Confidence is Nondiscriminative ($\text{AUROC2} \approx 0.517$, Brier $\approx 0.396$):**
   When the model generates an answer and self-reports confidence on the same forward pass, its reported confidence is virtually identical whether it is correct ($P_{\text{mean}} = 74.5\%$) or incorrect ($P_{\text{mean}} = 73.0\%$).
2. **External Observers Exploit First-Order Behavioral Cues ($\text{AUROC2} \approx 0.678$, Brier $\approx 0.290$):**
   An external observer viewing only the task prompt and the model's selected option letter achieves substantially higher discrimination ($P_{\text{mean}|\text{correct}} = 49.1\%$, $P_{\text{mean}|\text{incorrect}} = 21.2\%$) and binary classification accuracy (62.5%).
3. **Privileged Access Index is Non-Positive ($\text{PAI} = -0.161$, 95% Bootstrap CI: $[-0.428, +0.055]$):**
   The joint strongest-observer statistic ($\max(\text{VisibleAns}, \text{Recon}, \text{InputOnly})$) definitively rules out a meaningful self-advantage ($\ge +0.10$ SESOI).
4. **Epistemic Classification:**
   **Measurement-valid Level-0 reference baseline established; no positive privileged-access advantage resolved.** While the joint bound rules out a positive self-advantage $\ge +0.10$, individual paired contrasts remain wide enough that strict statistical equivalence ($\text{PAI} \approx 0$) cannot be asserted.

---

## 2. Evolution of Findings & What Failed Along the Way

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
- **Lesson 1 (Tokenization / Recency in S01):** Free generation in key-value retrieval conflated binding retrieval with BPE token boundary fragmentation and prompt-position recency. Moving to counterbalanced 4-way forced choice (DR-0012) stabilized base accuracy at $57.5\%$ (in the psychophysically optimal $50\text{--}65\%$ regime) with zero option-position bias.
- **Lesson 2 (Probability Direction in S03.0):** Observers evaluating "Evaluation: INCORRECT, Confidence: 5" were initially scored as high probability of target correctness. Standardizing all evaluators strictly on the semantic quantity $P(\text{Target Correct}) \in [0.0, 1.0]$ resolved this fundamental sign error.
- **Lesson 3 (4-Way Reconstruction in S03.1):** Assuming $P(\text{Target Correct}) = 1 - P(\text{Recon Choice})$ when reconstruction disagreed was mathematically invalid for a 4-choice task. Replacing it with an explicit 4-way probability distribution lookup $P_{\text{recon}}(\text{Target Option})$ created a proper counterfactual baseline.
- **Lesson 4 (Constrained Decoding in S03.4):** Relying on raw text prompts with `format="json"` produced malformed keys, trailing text, and negative numbers. Moving to native JSON Schema dictionaries with `type: "integer"`, `minimum: 0, maximum: 100`, and `enum: ["A","B","C","D"]` elevated primary compliance from $37.5\%$ to $100.0\%$.

---

## 3. Substantive Scientific Insights from Level-0

### A. The "Disillusioned Self" Phenomenon
In single-turn feedforward generation, the model's internal activation state at token generation time is **not** accurately reflected in its explicit self-confidence token emission. The model exhibits a ceiling-anchored confidence bias ($\sim 74\%$) regardless of whether its retrieval succeeded or failed.

### B. External Visibility Superiority
An independent observer (an identical instance of the same model invoked on a fresh prompt) reading the proposed answer achieves $\text{AUROC2} = 0.678$. The external observer evaluates the plausibility of the key-value pairing with greater objectivity because it does not suffer from the generative commitment bias that produced the first-order token.

### C. The Ineffectiveness of Public Transcript Confidence
Presenting the target's stated confidence to an external observer did **not** improve observer performance:
- Visible Answer-Only AUROC2: $0.678$
- Visible Full-Transcript AUROC2: $0.574$
- Contrast: $\Delta = +0.104$ ($95\%$ CI: $[-0.145, +0.354]$)

The observer is slightly worse off (or at best unimproved) when exposed to the target's inflated confidence numbers, indicating that explicit confidence tokens act as unhelpful distractors rather than informative signals.

### D. Compute vs. Framing
Contrasting second-invocation self-review against other-review revealed no meaningful framing effect:
- Self-Review AUROC2: $0.428$
- Other-Review AUROC2: $0.496$
- Contrast: $\Delta = -0.068$ ($95\%$ CI: $[-0.318, +0.198]$)

Reviewing one's own answer does not confer privileged access; both 2nd-invocation reviews perform near or below chance when re-evaluating forced-choice options without independent solving.

---

## 4. What Questions Level-1 (H1) Must Answer

With Level-0 rigorously established as a non-privileged baseline ($\text{PAI} \le 0$), we now have an unambiguous reference point for evaluating persistent state and memory.

### The Research Question for Level 1:
> **Core Question for H1:** Does maintaining persistent explicit memory (conversation transcripts, structured state dictionaries, model summaries) across multi-turn trajectories create **positive privileged access** ($\text{PAI} > 0$) or enhance metacognitive discrimination ($\text{AUROC2} > 0.52$) compared to external observers who inspect only current-turn outputs?

### Key Hypotheses to Test in Level 1:
1. **$H_1^{\text{accuracy}}$ (State Retention):** Which explicit memory representation (full transcript, structured JSON state, narrative summary) best resists multi-turn context decay under sequential overwrites?
2. **$H_1^{\text{meta}}$ (Metacognitive Calibration):** Does persistent state history allow the model to know *when it has lost track* of an updated variable (producing calibrated low confidence on corrupted state)?
3. **$H_1^{\text{efficiency}}$ (Compute Frontier):** What is the trade-off between token footprint ($T_{\text{prompt}}$) and retrieval accuracy across explicit memory architectures before introducing latent recurrent representations?

---

## 5. Frozen Provenance Summary

| Artifact | Pointer / Value |
|---|---|
| **Frozen Code Commit** | [`f4ace47`](file:///c:/Users/admir/Github/recurrence) |
| **Promoted Results Commit** | [`c6c90e6`](file:///c:/Users/admir/Github/recurrence) |
| **Promoted Run ID** | `run_e02_obs_005` |
| **Dataset Stream Checksum** | `f9aa1553ef8d69011e0ecbed36e39e2e15417f0782e9398e2e4282418b64eb0f` |
| **Environment Fingerprint** | `ee73e35d02efcd5248ec0ccc21b4ecff5e532ba92bbfb893fa6580f7b734e11b` |
| **Decision Record** | [DR-0013](file:///c:/Users/admir/Github/recurrence/Decision%20Log.md) |
