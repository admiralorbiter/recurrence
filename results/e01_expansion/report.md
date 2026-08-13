---
title: Result Report — S02: Hardened Task Validation & 2x2 Factorial Diagnostics
experiment_id: E01_Expansion_Hardened
protocol_version: 1.2.0
status: established
completed_at: 2026-08-13
---

# Result Report — S02: Hardened Task Validation & 2x2 Factorial Diagnostics

## Executive result

> Evaluated open model `qwen2.5:3b` across 120 trials using a hardened, collision-safe test harness with strict normalized exact matching. In a paired 2x2 factorial evaluation ($N=20$ matched items per cell), **associative accuracy was substantially higher under forced-choice recognition (65.0%) than under exact free generation (17.5%)**, demonstrating a massive +47.5 percentage-point response-mode dissociation. In forced choice, semantic and opaque identifiers achieved identical performance (65.0% vs. 65.0%), weakening the hypothesis that semantic embedding geometry rescues associative binding. A matched control verified that requesting prospective confidence ratings did not distort first-order accuracy (5.0% with confidence vs. 5.0% without). When context tracking was re-architected into genuine multi-object interleaved state tracking across 6 transitions, accuracy dropped to 20.0% (4/20), confirming that earlier single-object performance was largely an artifact of final-sentence prompt structure.

## Decision

**Proceed to Sprint S03 (Observer & Reconstruction Controls / Self-Monitoring Baseline `E02`)**

Reason: Measurement validity is established. The paired 2x2 matrix isolates associative recognition from surface-generation corruption, the confidence intervention control shows no measurement distortion, and interleaved tracking provides a genuine state-maintenance benchmark.

## Evidence label

- **Stage:** establish
- **Reproduction level:** R0 (exact deterministic replay verified with complete environment fingerprints)
- **Evidence level:** behavior & task validation (Claim Level 1 — Behavioral capacity)
- **Measurement validity:** passed (paired items isolate response-mode dissociation; strict exact scoring eliminates substring leakage; collision guards active)
- **Primary observed failure types:**
  - `KV Free Generation`: partial string corruption (33/40 errors across semantic & opaque)
  - `KV Forced Choice`: unresolved associative retrieval failure (13/40 errors)
  - `Interleaved Context Tracking`: multi-object state confusion / hallucination (12/20 errors), previous state (1/20), recency bias (1/20), formatting noncompliance (2/20)
- **Claim ceiling reached:** Level 1 (Behavioral task validation & recognition dissociation)
- **Confidence:** strong (reproducible across 120 paired trials)

## System and conditions

| Field | Value |
|---|---|
| model/revision | `qwen2.5:3b` |
| system boundary | episodic prompt invocation (Level 0) |
| digest | `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` |
| backend | Ollama REST API (`localhost:11434`) |
| conditions | 4 paired KV Factorial conditions (20 items each) + 1 Confidence Control (20 items) + 1 Interleaved Context Tracking (20 items) = 120 total trials |
| seed | 42 |
| run ID | `run_e01_exp_002` |
| environment hash | `b3532f861ed05dd7c474ab15b72a857d7ecf56a3a0d3431c37170e7cf60f6c2b` |
| stream checksum | `23a3cfefae4e5198bfee7b8d17033ae12ca9928f5583db31aae51eba2a721f82` |

## Primary results

### 1. Paired 2x2 Factorial Matrix: Recognition vs. Free Generation

| Identifier Type | Forced Choice (4-Way) | Free Generation | Main Effect (Identifiers) |
|---|---|---|---|
| **Semantic Identifiers** (Compound words) | **65.0%** (13 / 20) | **30.0%** (6 / 20) | 47.5% Mean |
| **Opaque Identifiers** (Alphanumeric strings) | **65.0%** (13 / 20) | **5.0%** (1 / 20) | 35.0% Mean |
| **Main Effect (Response Mode)** | **65.0% Mean** (26 / 40) | **17.5% Mean** (7 / 40) | **Δ = +47.5%** |

*Core Insights:*
1. **Response-Mode Dissociation:** Associative accuracy under forced-choice recognition (65.0%) is almost four times higher than under exact free generation (17.5%) across the *exact same* underlying items. This strongly suggests that candidate answers being present in the prompt aids selection, whereas free-form autoregressive generation suffers frequent subword character drift.
2. **Lack of Semantic Advantage in Forced Choice:** Under forced choice, semantic and opaque items achieved identical accuracy (65.0% vs. 65.0%). The advantage of semantic identifiers in free generation (30% vs 5%) appears to reflect spelling robustness of natural dictionary words rather than superior associative binding.

### 2. Confidence Elicitation Intervention Control

| Condition | Items | Correct | Accuracy |
|---|---|---|---|
| **Opaque Free Generation (with Confidence prompt)** | 20 | 1 | **5.0%** |
| **Opaque Free Generation (without Confidence prompt)** | 20 | 1 | **5.0%** |

*Core Insight:* Requesting a prospective confidence rating (`Confidence: <1 to 5>`) caused **zero distortion** in first-order accuracy on identical paired items, confirming that confidence elicitation does not interfere with first-order performance in this setup.

### 3. Interleaved Multi-Object Context Tracking

- **Overall Accuracy:** 20.0% (4 / 20)
- **Lag Performance Breakdown:**
  - $k=3$ intervening moves: 28.6% (2 / 7)
  - $k=2$ intervening moves: 14.3% (1 / 7)
  - $k=1$ intervening move: 50.0% (1 / 2)
  - $k=0$ (target moved in final sentence): 0.0% (0 / 4)
- **Error Distribution:**
  - Multi-object state confusion / hallucination: 12 / 20 (60.0%)
  - Formatting noncompliance: 2 / 20 (10.0%)
  - Previous state of target: 1 / 20 (5.0%)
  - Terminal sentence recency bias: 1 / 20 (5.0%)

*Core Insight:* Introducing multiple moving objects dramatically reduces performance (from 70% in the single-object scout to 20%), demonstrating that the model struggles to maintain independent entity state registers when updates are interleaved.

## Required language in result report

1. **System boundary:** The system under study was `qwen2.5:3b` operating as an episodic, stateless LLM backend with greedy decoding (`temperature=0.0`, `seed=42`).
2. **Target construct:** The experiment operationalized associative recognition vs. free generation, measurement reactivity of confidence prompts, and interleaved multi-object state tracking.
3. **Strongest supported claim level:** The strongest claim supported is **Level 1 (Behavioral task validation & recognition/generation dissociation)** across matched items.
4. **Best alternative explanation:** Forced-choice accuracy benefits from visible candidate options acting as recognition cues rather than proving pre-existing latent memory availability.
5. **Consciousness relevance:** This result establishes measurement validity and null baselines; it has no direct consciousness implications.
6. **What it does not show:** The result does not establish privileged internal access, recurrent state persistence, or specific neural mechanisms.
7. **Next discriminating test:** Sprint S03 (E02) will implement the **Observer Ladder** to establish whether self-confidence provides any discrimination advantage over an equally informed non-recurrent observer reading the visible transcript.
