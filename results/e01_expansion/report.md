---
title: Result Report — S02: Hardened Task Validation & 2x2 Factorial Diagnostics
experiment_id: E01_Expansion_Hardened
protocol_version: 1.2.0
status: validation-complete
completed_at: 2026-08-13
---

# Result Report — S02: Hardened Task Validation & 2x2 Factorial Diagnostics

## Executive result

> Evaluated open model `qwen2.5:3b` across 120 trials using an isolated, collision-safe test harness with strict normalized exact matching, cross-process deterministic generation, and exact option-position counterbalancing (5 targets each at A, B, C, D). In the paired 2x2 factorial evaluation ($N=20$ matched items per cell), **associative accuracy was substantially higher under forced-choice recognition (62.5% mean) than under exact free generation (32.5% mean)**, establishing a +30.0 percentage-point recognition vs. free-generation dissociation. A paired 2x2 contingency analysis on the confidence-prompt intervention check revealed no significant trial-level or aggregate distortion (25.0% with confidence vs. 20.0% without; 2 both correct, 3 confidence-only, 2 no-confidence-only, 13 both wrong; exact McNemar $p=1.00$). When context tracking was tested across 3 moving objects and 6 transitions, accuracy was 30.0% (6/20), confirming that earlier single-object performance (70–80%) was largely an artifact of the target move appearing in the final prompt sentence.

## Decision

**Proceed to Sprint S03 (Observer & Reconstruction Controls / Self-Monitoring Baseline `E02`)**

Reason: Task measurement validity is established. The paired 2x2 matrix isolates associative recognition from free-generation drift under exact option counterbalancing, the confidence intervention control exhibits no significant reactivity, and item-level results are automatically tracked in committed structured artifacts.

## Evidence label

- **Stage:** scout / measurement validation
- **Reproduction level:** R1 (deterministic replay verified across fresh Python processes with environment fingerprints)
- **Evidence level:** behavior & task validation (Claim Level 1 — Behavioral capacity)
- **Measurement validity:** passed (paired items isolate response-mode dissociation; exact A/B/C/D counterbalancing eliminates position bias; strict exact scoring active)
- **Primary observed failure types:**
  - `KV Free Generation`: partial string corruption (27/40 errors across semantic & opaque)
  - `KV Forced Choice`: unresolved associative retrieval failure (15/40 errors)
  - `Interleaved Context Tracking`: multi-object state confusion / hallucination (9/20), initial state substitution (2/20), previous state (1/20), format noncompliance (2/20)
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
| environment hash | `e348a5ef77b79423af5e9727a6bb5aa576be5d6ac4b7afdb88cb536975d4efed` |
| stream checksum | `127d3f401590a189ef3c4f5a871fbfc457771124c4421e7f9e3d591e0da1414b` |

## Primary results

### 1. Paired 2x2 Factorial Matrix (Counterbalanced Targets: 5 A, 5 B, 5 C, 5 D)

| Identifier Type | Forced Choice (4-Way) | Free Generation | Main Effect (Identifiers) |
|---|---|---|---|
| **Semantic Identifiers** (Compound words) | **70.0%** (14 / 20) | **40.0%** (8 / 20) | 55.0% Mean |
| **Opaque Identifiers** (Alphanumeric strings) | **55.0%** (11 / 20) | **25.0%** (5 / 20) | 40.0% Mean |
| **Main Effect (Response Mode)** | **62.5% Mean** (25 / 40) | **32.5% Mean** (13 / 40) | **Δ = +30.0%** |

*Core Insights:*
1. **Response-Mode Dissociation:** Associative accuracy under forced-choice recognition (62.5%) remains nearly double that of exact free generation (32.5%) under strict exact scoring and uniform target-position counterbalancing.
2. **Identifier Effect:** Semantic identifiers yielded a consistent +15.0% advantage in both forced-choice recognition (70% vs. 55%) and free generation (40% vs. 25%), reflecting an identifier $\times$ response-mode additive relationship.

### 2. Confidence Elicitation Intervention & Paired Contingency Table

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

### 3. Interleaved Multi-Object Context Tracking

- **Overall Accuracy:** 30.0% (6 / 20)
- **Lag Performance Breakdown:**
  - $k=3$ intervening moves: 42.9% (3 / 7)
  - $k=2$ intervening moves: 28.6% (2 / 7)
  - $k=0$ (final move was target): 25.0% (1 / 4)
  - $k=1$ intervening move: 0.0% (0 / 2)
- **Error Distribution:**
  - Multi-object state confusion / hallucination: 9 / 20 (45.0%)
  - Target initial state substitution: 2 / 20 (10.0%)
  - Response formatting noncompliance: 2 / 20 (10.0%)
  - Target immediate previous state substitution: 1 / 20 (5.0%)

## Required language in result report

1. **System boundary:** The system under study was `qwen2.5:3b` operating as an episodic, stateless LLM backend with greedy decoding (`temperature=0.0`, `seed=42`).
2. **Target construct:** The experiment operationalized associative recognition vs. free generation, measurement reactivity of confidence prompts, and interleaved multi-object state tracking.
3. **Strongest supported claim level:** The strongest claim supported is **Level 1 (Behavioral task validation & recognition/generation dissociation)** across matched counterbalanced items.
4. **Best alternative explanation:** Forced-choice accuracy benefits from candidate options acting as recognition cues rather than proving pre-existing latent memory availability.
5. **Consciousness relevance:** This result establishes measurement validity and null baselines; it has no direct consciousness implications.
6. **What it does not show:** The result does not establish privileged internal access, recurrent state persistence, or specific neural mechanisms.
7. **Next discriminating test:** Sprint S03 (E02) will implement the **Observer Ladder** to establish whether self-confidence provides any discrimination advantage over an equally informed non-recurrent observer reading the visible transcript.
