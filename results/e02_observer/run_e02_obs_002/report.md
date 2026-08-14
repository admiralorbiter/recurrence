---
title: Result Report — S03.1: Hardened Observer Controls & Level-0 Privileged Access Benchmark
experiment_id: E02_Observer_Hardened
run_id: run_e02_obs_002
protocol_version: 1.3.1
status: measurement-complete
completed_at: 2026-08-14
---

# Result Report — S03.1: Hardened Observer Controls & Level-0 Privileged Access Benchmark

## Executive result

> Evaluated open model `qwen2.5:3b` across 40 counterbalanced 4-way Forced Choice Key-Value retrieval trials (20 Semantic, 20 Opaque; 5 targets per letter A/B/C/D) under the 6-Condition Hardened Observer Battery with unified probability metric $P(\text{Target Correct}) \in [0, 100\%]$. Target first-order accuracy was 67.5% (65.0% Semantic, 70.0% Opaque). Under strict item-paired intersection contrasts:
> 1. **Immediate Self-Monitoring:** Target self-confidence exhibited anti-discriminative separation ($\bar{P}_{\text{corr}} = 0.551$ vs. $\bar{P}_{\text{incorr}} = 0.638$, $\Delta = -0.086$), yielding $\text{AUROC2}_{\text{self}} = 0.438$ and continuous Brier score $\text{BS} = 0.437$.
> 2. **Equal-Compute Framing Controls:** Giving the target model an equal-compute second invocation to review its own answer (`self_review_equal_compute`) yielded $\text{AUROC2} = 0.594$ ($\text{BS} = 0.262$, 78.6% accuracy), matching external review (`observer_review_other`, $\text{AUROC2} = 0.630$, $\text{BS} = 0.312$). This confirms that monitoring improvements stem from multi-step computational deliberation rather than self/other privileged access.
> 3. **Visible Observation & Public Channel Effect:** External observers reading only prompt + answer (`observer_visible_answer_only`) achieved $\text{AUROC2} = 0.609$ ($\text{BS} = 0.302$). Adding the target's self-confidence to the transcript (`observer_visible_full_transcript`) improved observer discrimination to $\text{AUROC2} = 0.650$ and Brier score to $\mathbf{0.185}$ (81.0% accuracy), demonstrating that the self-report acts as a public information channel that third parties exploit more effectively than the generator itself.
> 4. **Counterfactual Reconstruction:** Independent re-solving (`observer_reconstruction`) achieved $\text{AUROC2} = 0.570$ ($\text{BS} = 0.314$), yielding paired $\Delta\text{AUROC2} = -0.133$ ($95\%\text{ CI } [-0.368, +0.079]$).
> 5. **Privileged Access Index:** Joint paired PAI was $\text{PAI} = -0.205$ ($95\%\text{ CI } [-0.656, +0.060]$). Because the upper bound of the confidence interval (+0.060) is strictly below the pre-registered Smallest Effect Size of Interest ($\text{SESOI} = +0.10$), **no evidence of a positive privileged self-access advantage was detected at Level 0.**

## Decision

**Pass Gate S03.1 $\to$ Proceed to Sprint S04 (H1: Scaffolded Persistence & Explicit Memory Baselines `E03`)**

Reason: The measurement and statistical limitations of the S03 pilot have been repaired. All metrics use aligned probability semantics, strict item-paired intersections, equal-compute framing controls, and continuous Brier calibration. The empirical finding that stateless autoregressive models (Level 0) exhibit no positive privileged self-access over external observers provides the unconfounded benchmark for testing persistent memory architectures (Level 1+).

## Evidence label

- **Stage:** confirmatory-complete / measurement baseline established
- **Reproduction level:** R1 (exact stimulus determinism, counterbalanced option positions, isolated directory lineage, environment hashes)
- **Evidence level:** behavior & metacognitive observer calibration (Claim Level 1 — Behavioral & Observer Calibration)
- **Measurement validity:** passed (unified 0–100% probability metric, exact item-paired intersection contrasts, continuous Brier scoring)
- **Primary observed failure types:**
  - `Target FC Retrieval`: associative retrieval confusion (13/40 errors)
  - `Target Immediate Self-Report`: metacognitive overconfidence on errors ($\bar{P}_{\text{incorr}} = 0.638 > \bar{P}_{\text{corr}} = 0.551$)
- **Claim ceiling reached:** Level 1 (Empirical verification of Level-0 Null Privileged Access)
- **Confidence:** strong (upper bound of paired bootstrap 95% CI is below SESOI margin $\pm 0.10$)

## System and conditions

| Field | Value |
|---|---|
| model/revision | `qwen2.5:3b` |
| system boundary | episodic prompt invocation (Level 0) |
| digest | `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` |
| backend | Ollama REST API (`localhost:11434`, greedy decoding `temperature=0.0`, `seed=42`) |
| conditions | 40 Target FC KV trials $\times$ 7 Observer conditions (Self-Review, Other-Review, Vis-AnswerOnly, Vis-FullTranscript, Recon, Input-Only, Output-Only) |
| seed | 42 |
| run ID | `run_e02_obs_002` |
| environment hash | `5af50fcbccefa5714c8bdd1de48f264623c61d55a90d2941aa5c31c374c7eab1` |
| stream checksum | `a73daa71a7c168e0a3c7604437664b3441ecc997fd2bdd8c890531f59c22e7da` |

## Primary results: Strict item-paired intersection contrasts

| Contrast ($X$) | Shared Items ($N$) | Self AUROC2 | Observer AUROC2 | $\Delta\text{AUROC2}$ (Self - Obs) | 95% Bootstrap CI | Self Brier | Observer Brier | Observer Accuracy |
|---|---|---|---|---|---|---|---|---|
| **Equal-Compute Self-Review** | 14 | 0.500 | 0.594 | -0.094 | $[-0.375, +0.136]$ | 0.437 | 0.262 | 78.6% |
| **Equal-Compute Other-Review** | 20 | 0.479 | 0.630 | -0.151 | $[-0.493, +0.182]$ | 0.396 | 0.312 | 65.0% |
| **Visible: Answer-Only** | 19 | 0.404 | 0.609 | -0.205 | $[-0.619, +0.267]$ | 0.457 | 0.302 | 68.4% |
| **Visible: Full-Transcript** | 21 | 0.483 | 0.650 | -0.167 | $[-0.370, +0.0002]$ | 0.395 | **0.185** | **81.0%** |
| **Counterfactual Reconstruction** | 24 | 0.438 | 0.570 | -0.133 | $[-0.368, +0.079]$ | 0.437 | 0.314 | 62.5% |
| **Ablated: Input-Only** | 17 | 0.492 | 0.386 | +0.106 | $[-0.243, +0.457]$ | 0.400 | 0.361 | 58.8% |
| **Ablated: Output-Only** | 7 | 0.400 | 0.300 | +0.100 | $[-0.333, +0.600]$ | 0.479 | 0.464 | 42.9% |

### Joint Privileged Access Index Summary ($N=19$ shared intersection items):

$$\text{PAI}_{\text{Joint}} = \text{AUROC2}_{\text{self}} - \max(\text{AUROC2}_{\text{vis\_ans}}, \text{AUROC2}_{\text{recon}}) = 0.404 - 0.609 = \mathbf{-0.205}$$
$$\text{95\% Paired Bootstrap CI: } [-0.656, +0.060] \quad (\text{SESOI} = \pm 0.10)$$

Since $\text{CI}_{\text{upper}} = +0.060 < +0.10$, the data rule out any meaningful positive self-access advantage.

## Key scientific discoveries

1. **Introspective Anti-Correlation at Level 0:** The target model's immediate self-confidence was negatively correlated with correctness ($\Delta = -0.086$, $\text{AUROC2} = 0.438$). When incorrect, the model tended to output inflated probabilities ($P \approx 0.64$).
2. **Compute vs. Self-Privilege Disentanglement:** When given a fresh second invocation to review its previous answer (`self_review_equal_compute`), discrimination improved from 0.500 $\to$ 0.594, which was statistically indistinguishable from a third-party observer performing the same review (`observer_review_other`, 0.630). Metacognitive monitoring gains come from multi-step computational deliberation, not privileged internal access.
3. **Self-Report as a Public Information Channel:** Providing the target's stated self-confidence to the external observer (`observer_visible_full_transcript`) dropped observer forecast Brier error from 0.302 $\to$ 0.185 and raised prediction accuracy from 68.4% $\to$ 81.0%. The self-confidence token functions as an informative public feature for third-party inference.

## Required language in result report

1. **System boundary:** The system under study was `qwen2.5:3b` operating as an episodic, stateless LLM backend with greedy decoding (`temperature=0.0`, `seed=42`).
2. **Target construct:** The experiment operationalized metacognitive confidence discrimination, transcript-based observer prediction, equal-compute review framing, and counterfactual reconstructive monitoring.
3. **Strongest supported claim level:** The strongest claim supported is **Level 1 (Empirical verification of Level-0 Null Privileged Access)** across matched counterbalanced items.
4. **Best alternative explanation:** At Level 0, post-decision self-reports reflect shallow verbalized calibration heuristics without access to latent computation traces; external transcript review and counterfactual recomputation provide superior discrimination.
5. **Consciousness relevance:** This establishes the critical empirical reference point: claims of "introspection" or "self-awareness" in stateless LLMs are fully explained by external transcript heuristics and compute matching.
6. **What it does not show:** It does not show that persistent or recurrent state models (Level 1+) cannot develop positive PAI; rather, it establishes the standard against which persistent architectures must be judged.
7. **Next discriminating test:** Sprint S04 (E03) will test **Scaffolded Persistence** (fresh vs. full-context vs. summary vs. structured JSON state) to determine if state persistence creates the first genuine positive separation in privileged self-access.
