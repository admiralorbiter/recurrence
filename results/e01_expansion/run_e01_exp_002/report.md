---
title: Result Report — S02: Hardened Task Validation & 2x2 Factorial Diagnostics
experiment_id: E01_Expansion_Hardened
run_id: run_e01_exp_002
protocol_version: 1.2.0
status: validation-complete
completed_at: 2026-08-13
---

# Result Report — S02: Hardened Task Validation & 2x2 Factorial Diagnostics

## Executive result

> Evaluated open model `qwen2.5:3b` across 120 trials using an isolated, collision-safe test harness with strict normalized exact matching, cross-process deterministic generation, and exact option-position counterbalancing (5 targets each at A, B, C, D). In the paired 2x2 factorial evaluation ($N=20$ matched items per cell), **associative accuracy was substantially higher under forced-choice recognition (62.5% mean) than under exact free generation (32.5% mean)**, establishing a +30.0 percentage-point recognition vs. free-generation dissociation. A paired discordant analysis across all 40 matched items revealed that 19 items were solved exclusively under forced choice versus only 7 exclusively under free generation (**exact McNemar $p = 0.0290$**). A paired 2x2 contingency analysis on the confidence-prompt intervention check revealed no significant trial-level or aggregate distortion (25.0% with confidence vs. 20.0% without; 2 both correct, 3 confidence-only, 2 no-confidence-only, 13 both wrong; exact McNemar $p=1.00$). When context tracking was tested across 3 moving objects and 6 transitions, accuracy was 15.0% (3/20), confirming that earlier single-object performance (70–80%) was largely an artifact of the target move appearing in the final prompt sentence.

## Decision

**Proceed to Sprint S03 (Observer & Reconstruction Controls / Self-Monitoring Baseline `E02`)**

Reason: Task measurement validity is established. The paired 2x2 matrix isolates associative recognition from free-generation drift under exact option counterbalancing, the confidence intervention control exhibits no significant reactivity, and item-level results are automatically tracked in committed structured artifacts.

## Evidence label

- **Stage:** scout / measurement validation
- **Reproduction level:** R1 (stimulus-generation replay verified across fresh Python processes with complete environment fingerprints)
- **Evidence level:** behavior & task validation (Claim Level 1 — Behavioral capacity)
- **Measurement validity:** passed (paired items isolate response-mode dissociation; exact A/B/C/D counterbalancing eliminates position bias; strict exact scoring active)
- **Primary observed failure types:**
  - `KV Free Generation`: partial string corruption (27/40 errors across semantic & opaque)
  - `KV Forced Choice`: unresolved associative retrieval failure (15/40 errors)
  - `Interleaved Context Tracking`: multi-object state confusion / hallucination (11/20), previous state (3/20), format noncompliance (3/20)
- **Claim ceiling reached:** Level 1 (Behavioral task validation & recognition dissociation)
- **Confidence:** strong (reproducible across 120 paired counterbalanced trials)

## System and conditions

| Field | Value |
|---|---|
| model/revision | `qwen2.5:3b` |
| system boundary | episodic prompt invocation (Level 0) |
| digest | `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` |
| backend | Ollama REST API (`localhost:11434`, timeout=120s with retry) |
| conditions | 4 paired KV Factorial conditions (20 items each) + 1 Confidence Control (20 items) + 1 Interleaved Context Tracking (20 items) = 120 total trials |
| seed | 42 |
| run ID | `run_e01_exp_002` |
| environment hash | `4707bbdaba950622cc88a0f40e42830c98c52ce727d214f904c2084cae46d2c1` |
| stream checksum | `444cfd3d54ee4d65238c0a53c40ecf3e7f7a4d838e129e9e7993a47b3791e88b` |

## Primary results

### 1. Paired 2x2 Factorial Matrix (Counterbalanced Targets: 5 A, 5 B, 5 C, 5 D)

| Identifier Type | Forced Choice (4-Way) | Free Generation | Main Effect (Identifiers) |
|---|---|---|---|
| **Semantic Identifiers** (Compound words) | **65.0%** (13 / 20) | **40.0%** (8 / 20) | 52.5% Mean |
| **Opaque Identifiers** (Alphanumeric strings) | **60.0%** (12 / 20) | **25.0%** (5 / 20) | 42.5% Mean |
| **Main Effect (Response Mode)** | **62.5% Mean** (25 / 40) | **32.5% Mean** (13 / 40) | **Δ = +30.0%** |

### 2. Paired Response-Mode Contingency (Forced Choice vs. Free Generation)

| Dataset Strata | Both Correct | FC Only Correct | FG Only Correct | Both Wrong | Discordant (FC : FG) | Exact McNemar $p$ |
|---|---|---|---|---|---|---|
| **Semantic Identifiers** ($N=20$) | 5 (25.0%) | 8 (40.0%) | 3 (15.0%) | 4 (20.0%) | 8 : 3 (2.67x) | $p = 0.227$ |
| **Opaque Identifiers** ($N=20$) | 1 (5.0%) | 11 (55.0%) | 4 (20.0%) | 4 (20.0%) | 11 : 4 (2.75x) | $p = 0.118$ |
| **Pooled Items** ($N=40$) | 6 (15.0%) | 19 (47.5%) | 7 (17.5%) | 8 (20.0%) | 19 : 7 (2.71x) | **$p = 0.0290$** |

*Core Insights:*
1. **Paired Statistical Significance:** The 30.0 percentage-point recognition advantage is statistically significant under exact McNemar testing ($p = 0.0290$) across the 40 paired items.
2. **Additive Relationship:** Both semantic and opaque items exhibited a consistent ~2.7x discordant advantage in favor of forced choice (8 vs. 3 for semantic, 11 vs. 4 for opaque), indicating an additive response-mode effect without significant mode $\times$ identifier interaction.

### 3. Confidence Elicitation Intervention & Paired Contingency Table

| Paired Outcome Category | Trial Count ($N=20$) | Percentage |
|---|---|---|
| **Correct in Both Conditions** | 2 | 10.0% |
| **With-Confidence ONLY Correct** | 3 | 15.0% |
| **Without-Confidence ONLY Correct** | 2 | 10.0% |
| **Wrong in Both Conditions** | 13 | 65.0% |

- **With-Confidence Accuracy:** 25.0% (5 / 20)
- **Without-Confidence Accuracy:** 20.0% (4 / 20)
- **Exact McNemar Test:** $p = 1.00$ ($b=3, c=2$)
- *Conclusion:* Eliciting post-decision confidence produced no statistically significant accuracy distortion.

### 4. Interleaved Multi-Object Context Tracking

- **Overall Accuracy:** 15.0% (3 / 20)
- **Lag Performance Breakdown:**
  - $k=1$ intervening move: 50.0% (1 / 2)
  - $k=2$ intervening moves: 14.3% (1 / 7)
  - $k=3$ intervening moves: 14.3% (1 / 7)
  - $k=0$ (final move was target): 0.0% (0 / 4)
- **Error Distribution:**
  - Multi-object state confusion / hallucination: 11 / 20 (55.0%)
  - Response formatting noncompliance: 3 / 20 (15.0%)
  - Target immediate previous state substitution: 3 / 20 (15.0%)

## Required language in result report

1. **System boundary:** The system under study was `qwen2.5:3b` operating as an episodic, stateless LLM backend with greedy decoding (`temperature=0.0`, `seed=42`).
2. **Target construct:** The experiment operationalized associative recognition vs. free generation, measurement reactivity of confidence prompts, and interleaved multi-object state tracking.
3. **Strongest supported claim level:** The strongest claim supported is **Level 1 (Behavioral task validation & recognition/generation dissociation)** across matched counterbalanced items.
4. **Best alternative explanation:** Forced-choice accuracy benefits from candidate options acting as recognition cues rather than proving pre-existing latent memory availability.
5. **Consciousness relevance:** This result establishes measurement validity and null baselines; it has no direct consciousness implications.
6. **What it does not show:** The result does not establish privileged internal access, recurrent state persistence, or specific neural mechanisms.
7. **Next discriminating test:** Sprint S03 (E02) will implement the **Observer Ladder** to establish whether self-confidence provides any discrimination advantage over an equally informed non-recurrent observer reading the visible transcript.
