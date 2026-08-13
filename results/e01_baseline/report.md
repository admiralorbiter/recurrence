---
title: Result Report — E01 Baseline Task Battery (Scout)
experiment_id: E01
protocol_version: 1.0.0
status: exploratory
completed_at: 2026-08-13
---

# Result Report — E01 Baseline Task Battery (Scout)

## Executive result

> On this 10-item exploratory scout, open model `qwen2.5:3b` achieved 1/5 (20%) on opaque key-value exact string reproduction and 4/5 (80%) on semantic multi-step entity location tracking (50% overall battery accuracy). Opaque-string errors frequently retained partial sub-sequences while corrupting or truncating others, and one error produced an explicit refusal/abstention (`None`). The single context-tracking error returned an intermediate state rather than the terminal state. These patterns motivate targeted measurement diagnostics in Sprint S02 before promoting the battery.

## Decision

**Proceed to Sprint S02 (Task Validation & Refinement)**

Reason: The harness reliably interfaces with local open models and captures deterministic execution streams. The observed error patterns highlight important task-design questions regarding tokenization, associative binding vs. surface realization, and state interference that must be resolved prior to observer-control experiments.

## Evidence label

- **Stage:** scout
- **Reproduction level:** R0 (exact scripted replay verified)
- **Evidence level:** behavior (Claim Level 1 — Behavioral capacity)
- **Measurement validity:** conditional (task executes reliably; error sources require diagnostic separation between surface generation and binding)
- **Primary observed failure types:**
  - `KVRetrievalTask`: tokenization / surface-form artifact (3/5), response-format noncompliance / refusal (1/5), target-process failure (unresolved)
  - `ContextTrackingTask`: candidate intermediate-state substitution (1/5)
- **Claim ceiling reached:** Level 1 (Behavioral capacity scout)
- **Confidence:** moderate (procedural harness validity); speculative (mechanistic explanations)

## Preregistered question and prediction

- **Question:** Does the automated test harness successfully execute deterministic task evaluations against local open models (`qwen2.5:3b`) and capture reproducible error distributions?
- **Prediction:** `qwen2.5:3b` will exhibit non-zero accuracy across both tasks under greedy decoding (`temperature=0.0`, `seed=42`).

## System and conditions

| Field | Value |
|---|---|
| model/revision | `qwen2.5:3b` |
| system boundary | episodic prompt invocation (Level 0) |
| digest | `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` |
| backend | Ollama REST API (`localhost:11434`) |
| tasks | `KVRetrievalTask` (5 items, 5 distractor pairs), `ContextTrackingTask` (5 items, 5 transition steps) |
| seed | 42 |
| run ID | `run_e01_001` |
| environment hash | `2d5e6870077b37c157b3b7e3885115c119d57e6abab4a0016df846f56291261a` |
| stream checksum | `5aa857f97ae706636c3a94a371f51f4dfb764f4b5b7426e480f5c52e10b1f162` |

## Primary results & error breakdown

### Task Performance Summary

| Task | Items | Conditions | Score | Accuracy |
|---|---|---|---|---|
| **Key-Value Retrieval** | 5 | Opaque alphanumeric strings, 5 distractors | 1 / 5 | 20.0% |
| **Context Tracking** | 5 | Semantic entity tracking, 5 transitions | 4 / 5 | 80.0% |
| **Overall Battery** | 10 | — | 5 / 10 | **50.0%** |

### Detailed Trial Observations

#### 1. Opaque-String Exact-Reproduction Failures (`KVRetrievalTask`)
- **Trial 1:** Target `key_30t9nt` $\to$ Ground Truth `val_ibljh7` $\to$ Model output `val_ibljh7` (**Correct**)
- **Trial 2:** Target `key_vqjt7y` $\to$ Ground Truth `val_iuc039` $\to$ Model output `iiooor39` (**Error**)
- **Trial 3:** Target `key_pq0y9d` $\to$ Ground Truth `val_89uzfk` $\to$ Model output `89uzz5` (**Error**)
- **Trial 4:** Target `key_ljylje` $\to$ Ground Truth `val_khmy4s` $\to$ Model output `val valhmy4s` (**Error**)
- **Trial 5:** Target `key_0ifznb` $\to$ Ground Truth `val_gbps3y` $\to$ Model output `None` (**Abstention / Refusal**)

*Candidate Explanations to Disentangle in S02:*
1. **Tokenization / Reassembly Burden:** Random alphanumeric strings are fragmented into multiple arbitrary BPE subwords; the model may struggle with exact subword copy-routing.
2. **Associative Retrieval Failure:** Attention heads fail to bind the target key to its associated value line under distractor competition.
3. **Surface-Realization Failure after Successful Retrieval:** The model successfully binds the association in latent state but errors during token generation (e.g. `val_89uzfk` $\to$ `89uzz5`). *Diagnostic: 4-way forced-choice vs. free-generation comparison.*

#### 2. State Tracking vs. Opaque Retrieval Differences
- Semantic location tracking was substantially easier (80% vs 20%) in this scout.
- *Open Question:* The performance difference conflates multiple factors simultaneously: semantic vocabulary vs. opaque random strings, small candidate space (5 known rooms) vs. open vocabulary, and single-token vs. multi-subword target representations. Factorial diagnostic matrices are required before attributing this to semantic priming.

#### 3. Candidate Intermediate-State Substitution (`ContextTrackingTask` Trial 8)
- **Sequence:** Initially garden $\to$ Bob moved to bedroom $\to$ Charlie moved to office $\to$ Bob moved to living room $\to$ Alice moved to kitchen.
- **Ground Truth:** `kitchen` | **Model Output:** `office`
- **Observation:** The model returned the 3rd transition state (`office`) rather than the 5th terminal state (`kitchen`). Rather than inferring generalized memory interference from a single instance, Sprint S02 will systematically track whether errors favor recent, midpoint, or initial states across a 50+ item sample.

## Scorer Normalization Rules

To prevent experimenter degrees of freedom, the automated scorer applies the following fixed normalization:
- Trim leading/trailing whitespace.
- Case-insensitive substring matching (`item.ground_truth.lower() in model_response.lower()`).
- Explicit noncompliance or abstention (`None`, empty string, refusal preamble without answer) is classified as an error with failure type `response-format/noncompliance`.

## Possible Interpretations and Relevant Literature

- **Induction Heads and Sequence Copying (Olsson et al., Anthropic 2022):** Describes 2-layer attention circuits capable of implementing in-context associative recall. Evaluates random token sequences, motivating our investigation into whether copy circuits or tokenization fragmentation limit opaque string retrieval.
- **In-Context Retrieval & Position Effects (Liu et al., 2023):** Documents performance variations depending on target position within context windows ("Lost in the Middle"), directly relevant to future distractor ordering controls in key-value retrieval.
- **Reality Monitoring & Provenance (Ranjan, Sokratous & Odegaard, July 2026):** Investigates self vs. external source attribution under multi-turn delays; motivates our upcoming source-ownership and provenance benchmarks (E11–E12).

## Required language in result report

1. **System boundary:** The system under study was `qwen2.5:3b` operating as an episodic, stateless LLM backend with greedy decoding (`temperature=0.0`, `seed=42`).
2. **Target construct:** The experiment operationalized baseline in-context memory lookup and multi-step state tracking competence.
3. **Strongest supported claim level:** The strongest claim supported is **Level 1 (Behavioral capacity)** on context tracking above baseline.
4. **Best alternative explanation:** Observed errors on key-value retrieval may stem from tokenization surface-form reassembly or formatting friction rather than failure of associative memory.
5. **Consciousness relevance:** This scout has no direct consciousness relevance; it establishes measurement validity and identifies task confounders.
6. **What it does not show:** The result does not establish recurrence, persistent state, metacognitive monitoring, privileged access, or specific neural circuit mechanisms.
7. **Next discriminating test:** Sprint S02 will implement a 2x2 task diagnostic matrix (Forced-Choice vs. Free-Generation $\times$ Opaque vs. Semantic Identifiers) and tokenizer strata analysis to isolate associative capacity from surface-generation artifacts.
