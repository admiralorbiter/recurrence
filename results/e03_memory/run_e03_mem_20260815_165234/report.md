# Experiment E03: Level 1 Explicit Memory Baseline Report

**Run ID:** `run_e03_mem_20260815_165234`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Date:** 2026-08-15T17:04:28.750134+00:00  
**Primary Endpoint:** Pure Answer-Only Forced-Choice Accuracy under Native JSON Schema  

---

## 1. Executive Summary & Core Results

Experiment E03 quantifies how much cognitive retention, delayed retrieval, and source attribution can be achieved across **6 Level-1 explicit memory representation configurations** without latent recurrent continuity.

### Memory Format Performance & Cost Pareto Table

| Memory Configuration | Micro Acc | Macro Acc | Delayed KV (4AFC) | Source Attr (3AFC) | Goal Resumption (4AFC) | Prompt Tok | Acc / 1k Tok | Pareto Status | Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fresh** | **35.7%** | 38.9% | 33.3% | 33.3% | 50.0% | 109 tok | 3.27 | Pareto Optimal | 100.0% |
| **transcript** | **81.0%** | 85.2% | 83.3% | 72.2% | 100.0% | 499 tok | 1.62 | Pareto Optimal | 100.0% |
| **deterministic_summary** | **61.9%** | 55.6% | 88.9% | 44.4% | 33.3% | 274 tok | 2.26 | Pareto Optimal | 100.0% |
| **model_summary** | **69.0%** | 75.9% | 77.8% | 50.0% | 100.0% | 469 tok | 1.47 | Pareto Optimal | 100.0% |
| **structured_state** | **64.3%** | 68.5% | 72.2% | 50.0% | 83.3% | 371 tok | 1.73 | Pareto Optimal | 100.0% |
| **combined** | **66.7%** | 74.1% | 88.9% | 33.3% | 100.0% | 730 tok | 0.91 | Dominated | 100.0% |

---

## 2. Serial Position Analysis (Delayed KV Retrieval Only)

Controlling for 'Lost-in-the-Middle' positional attention artifacts across early, middle, and late stream placements (isolated to Delayed KV probes):

| Memory Configuration | Early Placement Acc | Middle Placement Acc | Late Placement Acc | Positional Stability (Late - Early) |
| :--- | :---: | :---: | :---: | :---: |
| `fresh` | 16.7% | 33.3% | 50.0% | +33.3% |
| `transcript` | 100.0% | 50.0% | 100.0% | +0.0% |
| `deterministic_summary` | 100.0% | 100.0% | 66.7% | -33.3% |
| `model_summary` | 83.3% | 83.3% | 66.7% | -16.7% |
| `structured_state` | 100.0% | 50.0% | 66.7% | -33.3% |
| `combined` | 100.0% | 83.3% | 83.3% | -16.7% |

---

## 3. Two-Stage Consolidation Fidelity & Distortion Analysis

Model autobiographical summaries were generated in an isolated consolidation phase prior to evaluation probe trials.

- **Total Target Facts Evaluated:** 18
- **Retained Target Facts (Exact Key-Value Association):** 2 (11.1%)
- **Exact KV Omission Rate (Target Key String Absent from Summary):** 72.2% (13/18)
- **Exact Association Mutation Rate (Key Present but Bound Value Altered):** 16.7% (3/18)
- **Unsupported val_* Strings Detected:** 12
- **Partition Invariant Verified:** Retained (2) + Mutated (3) + Omitted (13) == Total (18)
- **Mean Consolidation Prompt Tokens:** 416.5
- **Mean Consolidation Output Tokens:** 345.7
- **Memory Fidelity vs. Utility Note:** While exact-string KV reproduction was low (2/18), Model Summary achieved 77.8% on Delayed KV forced choice, indicating that noisy or lossy narrative traces still preserve recognition cues.

---

## 4. Scientific Takeaways for Level 1 & Horizon 1

1. **Observed Static Explicit-Memory Profile:** Explicit external history substantially improves first-order performance over fresh invocation on this synthetic battery.
2. **Information Selection Policy:** The 6 conditions represent distinct memory-system configurations with differing information selection policies (e.g. deterministic summary focuses on factual bindings, while structured state includes an oracle goal registry), not pure encodings of identical information.
3. **Structured State Trade-off:** Structured State traded static read performance for a smaller, explicitly typed and manipulable representation. It is selected for S05 for semantic coverage and experimental controllability rather than static superiority.
4. **Autobiographical Narrative Reliability:** Unconstrained narrative consolidation showed high exact-string omission and mutation, though recognition remained viable under forced choice, demonstrating that exact fidelity and recognition utility diverge.
5. **Combined Condition Observation:** Combined modestly exceeded Structured State on aggregate accuracy (66.7% vs 64.3%), but at 730 prompt tokens it was Pareto-dominated by Full Transcript (81.0% at 499 tokens); the contributions of length, redundancy, ordering, and narrative conflict remain unresolved.
6. **Transition to Sprint S05:** S04 measured whether a model can read state; Sprint S05 will measure whether an autonomous system can maintain and update that state over time.