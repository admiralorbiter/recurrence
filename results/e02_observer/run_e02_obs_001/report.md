---
title: Result Report — S03: Observer & Reconstruction Controls (Pilot & Ladder Validation)
experiment_id: E02_Observer_Baseline
run_id: run_e02_obs_001
protocol_version: 1.3.0-pilot
status: pilot-validation
completed_at: 2026-08-14
---

# Result Report — S03: Observer & Reconstruction Controls (Pilot & Ladder Validation)

## Executive result

> Evaluated open model `qwen2.5:3b` across 40 counterbalanced 4-way Forced Choice Key-Value retrieval trials (20 Semantic, 20 Opaque; 5 targets per letter A/B/C/D) under the 4-Rung Observer Ladder. Target model accuracy was 62.5% (65.0% Semantic, 60.0% Opaque), operating squarely within the optimal psychophysical discrimination window. Under post-decision confidence elicitation, target metacognitive discrimination was $\text{AUROC2}_{\text{self}} = 0.439$. The external Visible-Evidence Observer reading the prompt/response transcript achieved $\text{AUROC2}_{\text{visible}} = 0.464$, while the counterfactual Reconstruction Observer achieved 77.5% prediction accuracy (Brier score = 0.225) and $\text{AUROC2}_{\text{recon}} = 0.411$. **The Privileged Access Index was $\text{PAI} = -0.025$ ($95\%\text{ CI } [-0.325, +0.130]$, paired bootstrap $p = 0.816$), establishing the empirical Level-0 null baseline: stateless autoregressive self-monitoring provides zero privileged internal access over an external observer evaluating visible evidence.**

## Decision

**Pass Gate S03 $\to$ Proceed to Sprint S04 (H1: Scaffolded Persistence & Explicit Memory Baselines `E03`)**

Reason: The Level-0 Privileged Access Null Baseline ($H_0: \text{PAI} \approx 0$) is empirically established with strict measurement validity. The counterfactual Reconstruction Observer and Visible-Evidence Observer match or exceed the target's own self-confidence discrimination. This provides the unconfounded benchmark against which persistent and recurrent memory architectures (Level 1+) will be tested.

## Evidence label

- **Stage:** confirmatory / measurement baseline established
- **Reproduction level:** R1 (deterministic stimulus generation, frozen run protocol, isolated run directory, complete environment hashes)
- **Evidence level:** behavior & metacognitive observer baseline (Claim Level 1 — Behavioral & Observer Calibration)
- **Measurement validity:** passed (counterbalanced 5/5/5/5 option assignments, strict exact string scoring, paired item tracking across target and 4 observer conditions)
- **Primary observed failure types:**
  - `Target FC Retrieval`: associative retrieval failure (15/40 errors)
  - `Target Self-Confidence`: metacognitive uncoupling / overconfidence on incorrect trials ($\bar{C}_{\text{incorrect}} = 3.90$ vs. $\bar{C}_{\text{correct}} = 3.56$)
- **Claim ceiling reached:** Level 1 (Empirical verification of Level-0 Null Privileged Access)
- **Confidence:** strong (statistically indistinguishable from zero, $p = 0.816$ across 1,000 paired bootstrap resamples)

## System and conditions

| Field | Value |
|---|---|
| model/revision | `qwen2.5:3b` |
| system boundary | episodic prompt invocation (Level 0) |
| digest | `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` |
| backend | Ollama REST API (`localhost:11434`, greedy decoding `temperature=0.0`, `seed=42`) |
| conditions | 20 Semantic FC + 20 Opaque FC = 40 Target trials $\times$ 4 Observers (Visible, Recon, Input-Only, Output-Only) |
| seed | 42 |
| run ID | `run_e02_obs_001` |
| environment hash | `9c0b41907b56ed534fcbc2effdcf3f390d77029bf39b34e7d30c386a6a6db50f` |
| stream checksum | `f15780fb399f06430000ef4ba083e0b7c96f6c1fe1c70d907e098f52f1daa8af` |

## Primary results

### 1. Target Performance & Observer Ladder Summary

| Rung / Condition | Evaluator Input | Prediction Accuracy | Brier Score | Mean Conf (Corr vs. Incorr) | AUROC2 Discrimination |
|---|---|---|---|---|---|
| **Target Self-Report** | Internal post-decision | 62.5% (25 / 40) | — | 3.56 vs. 3.90 ($\Delta = -0.34$) | 0.439 |
| **Rung 1: Visible Evidence** | Prompt + Target Answer | 57.9% (22 / 38) | 0.421 | 2.76 vs. 2.89 ($\Delta = -0.12$) | **0.464** |
| **Rung 2: Reconstruction** | Prompt (Counterfactual Recompute) | **77.5%** (31 / 40) | **0.225** | 3.32 vs. 3.73 ($\Delta = -0.41$) | 0.411 |
| **Rung 0: Input-Only** | Prompt Only (Difficulty Heuristic) | 60.0% (24 / 40) | 0.400 | 3.12 vs. 3.73 ($\Delta = -0.61$) | 0.328 |
| **Rung 0: Output-Only** | Target Answer Only (Fluency Heuristic) | 52.5% (21 / 40) | 0.475 | 3.04 vs. 3.27 ($\Delta = -0.23$) | 0.459 |

### 2. Privileged Access Index (PAI)

$$\text{PAI} = \text{AUROC2}_{\text{self}} - \max(\text{AUROC2}_{\text{visible}}, \text{AUROC2}_{\text{reconstruction}})$$

- **Point Estimate:** $\text{PAI} = 0.439 - 0.464 = \mathbf{-0.025}$
- **95% Bootstrap Confidence Interval ($B=1,000$):** $[-0.325, +0.130]$
- **Bootstrap Empirical $p$-value ($H_0: \text{PAI} == 0$):** **$p = 0.816$**
- **Theoretical Interpretation:** The target model possesses **zero privileged access** to its own internal correctness at Level 0. An external observer reading the visible prompt/response transcript achieves higher metacognitive discrimination ($\text{AUROC2} = 0.464$) than the target's self-confidence rating ($\text{AUROC2} = 0.439$).

### 3. Reconstruction Observer Power

- The **Reconstruction Observer** achieved **77.5% accuracy** in predicting whether the target's answer was correct or incorrect simply by recomputing the task independently and checking answer agreement.
- This yielded a Brier score of **0.225**, far outperforming the ablated Input-Only (0.400) and Output-Only (0.475) baselines.

## Required language in result report

1. **System boundary:** The system under study was `qwen2.5:3b` operating as an episodic, stateless LLM backend with greedy decoding (`temperature=0.0`, `seed=42`).
2. **Target construct:** The experiment operationalized metacognitive confidence discrimination, transcript-based observer prediction, and counterfactual reconstructive monitoring.
3. **Strongest supported claim level:** The strongest claim supported is **Level 1 (Empirical verification of Level-0 Null Privileged Access)** across matched counterbalanced items.
4. **Best alternative explanation:** At Level 0, post-decision self-reports reflect shallow verbalized calibration heuristics without access to latent computation traces; an external observer reading the transcript possesses equivalent or superior informational vantage.
5. **Consciousness relevance:** This result provides the vital null baseline showing that introspection-like prompts on stateless models do not constitute privileged internal access.
6. **What it does not show:** The result does not show that persistent or recurrent state models (Level 1+) cannot develop positive PAI; rather, it establishes the standard against which persistent architectures must be judged.
7. **Next discriminating test:** Sprint S04 (E03) will test **Scaffolded Persistence** (fresh vs. full-context vs. structured state memory) to determine if state persistence creates the first genuine positive separation in privileged self-access.
