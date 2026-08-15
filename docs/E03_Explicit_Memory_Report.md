# Experiment E03 Research Memo: Level 1 Explicit Memory Baselines (Hardened S04.1)

**Sprint:** S04.1 (Horizon 1: Scaffolded Persistence)  
**Experiment ID:** `E03_Explicit_Memory_Baselines`  
**Run ID:** `run_e03_mem_20260815_165234`  
**Model:** `qwen2.5:3b` (`357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`)  
**Evaluation Protocol:** 252 Forced-Choice Trials under Temperature 0.0 with Native JSON Schema (`TARGET_4AFC_SCHEMA` & `TARGET_3AFC_SCHEMA`)

---

## 1. Executive Summary & Core Results

Experiment E03 quantifies how much cognitive retention, delayed retrieval, source attribution, and goal resumption can be achieved across **6 Level-1 explicit memory representation configurations** without latent recurrent continuity.

### Core Discoveries:
1. **Observed Static Explicit-Memory Profile**:
   - Adding explicit episodic context moves first-order performance far above the uninformed baseline ($35.7\% \to 61.9\% - 81.0\%$).
   - 100% of trials across all conditions satisfied native JSON Schema validation ($252/252$ schema compliant).
2. **Information Selection Policy Across Configurations**:
   - The 6 conditions represent distinct memory-system configurations with differing information selection policies, not pure encodings of identical information:
     - `Deterministic Summary` losslessly extracts factual key-value bindings and source assertions ($88.9\%$ Delayed KV), but excludes suspended goals by construction ($33.3\%$ Goal Resumption).
     - `Structured State` is an oracle state containing working memory, goal registry, and source ledgers ($72.2\%$ Delayed KV, $50.0\%$ Source Attribution, $83.3\%$ Goal Resumption).
     - `Full Transcript` retains complete chronological history, achieving $81.0\%$ micro-accuracy ($85.2\%$ macro-accuracy) at a comparatively high token footprint ($499$ prompt tokens, surpassed only by Combined at $730$ tokens).
3. **Autobiographical Narrative Consolidation Reliability**:
   - In the offline two-stage consolidation step, model-written summaries exhibited a **$72.2\%$ Exact KV Omission Rate** (13/18 target keys absent from narrative) and a **$16.7\%$ Exact Association Mutation Rate** (3/18 target keys bound to altered values), verifying the partition invariant:
     $$\text{Retained } (2) + \text{Mutated } (3) + \text{Omitted } (13) = \text{Total Facts } (18)$$
   - While exact-string reproduction was low (2/18), Model Summary achieved 77.8% on Delayed KV forced choice, demonstrating that memory fidelity (exact string reproduction) and memory utility (recognition under forced choice) diverge.
4. **Combined Condition Observation**:
   - Appending model narrative to structured state (`Combined`: $66.7\%$ Micro, $74.1\%$ Macro) modestly exceeded Structured State ($64.3\%$ Micro, $68.5\%$ Macro), but at $730$ prompt tokens it was Pareto-dominated by Full Transcript ($81.0\%$ at $499$ tokens); the contributions of length, redundancy, ordering, and narrative conflict remain unresolved.

---

## 2. Memory Format Performance & Cost Pareto Table

*Evaluated on 6 synthetic episodes (42 probes per memory format, 252 total evaluation trials):*

| Memory Configuration | Micro Acc | Macro Acc | Delayed KV (4AFC) | Source Attr (3AFC) | Goal Resumption (4AFC) | Prompt Tok | Acc / 1k Tok | Pareto Status | Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Fresh (No Memory)`** | **35.7%** | 38.9% | 33.3% | 33.3% | 50.0% | 109 tok | 3.27 | Pareto Optimal | 100.0% |
| **`Full Transcript`** | **81.0%** | **85.2%** | 83.3% | **72.2%** | **100.0%** | 499 tok | 1.62 | Pareto Optimal | 100.0% |
| **`Deterministic Summary`** | **61.9%** | 55.6% | **88.9%** | 44.4% | 33.3% | **274 tok** | **2.26** | Pareto Optimal | 100.0% |
| **`Model-Written Summary`** | **69.0%** | 75.9% | 77.8% | 50.0% | **100.0%** | 469 tok | 1.47 | Pareto Optimal | 100.0% |
| **`Structured Self-State`** | **64.3%** | 68.5% | 72.2% | 50.0% | 83.3% | 371 tok | 1.73 | Pareto Optimal | 100.0% |
| **`Combined (State + Narrative)`** | **66.7%** | 74.1% | **88.9%** | 33.3% | **100.0%** | 730 tok | 0.91 | Dominated | 100.0% |

*Note on Metrics:*
- **Micro Accuracy:** Unweighted accuracy across 42 trials per condition (18 Delayed KV + 18 Source + 6 Goal).
- **Macro Accuracy:** Equal-weighted average across the 3 cognitive subtasks: $\frac{1}{3}\text{KV} + \frac{1}{3}\text{Source} + \frac{1}{3}\text{Goal}$.
- **Pareto Optimality:** A configuration is Pareto optimal if no other tested condition achieves higher/equal accuracy at lower/equal token cost.

---

## 3. Serial Position Analysis (Delayed KV Retrieval Isolated)

To control for positional attention degradation ("Lost-in-the-Middle"), key-value bindings were counterbalanced across early, middle, and late stream placements and evaluated strictly on Delayed KV probes:

| Memory Configuration | Early Placement Acc | Middle Placement Acc | Late Placement Acc | Positional Stability ($\text{Late} - \text{Early}$) |
| :--- | :---: | :---: | :---: | :---: |
| `Fresh` | 16.7% | 33.3% | 50.0% | +33.3% |
| `Full Transcript` | 100.0% | 50.0% | 100.0% | +0.0% |
| `Deterministic Summary` | 100.0% | 100.0% | 66.7% | -33.3% |
| `Model-Written Summary` | 83.3% | 83.3% | 66.7% | -16.7% |
| `Structured Self-State` | 100.0% | 50.0% | 66.7% | -33.3% |
| `Combined State` | 100.0% | 83.3% | 83.3% | -16.7% |

*Key Takeaway:* In the Full Transcript condition, middle-placed items suffer from a marked dip ($50.0\%$ vs $100.0\%$ at edges), consistent with established long-context positional attention literature.

---

## 4. Consolidation Fidelity & Distortion Quantification

In Stage 1, `qwen2.5:3b` generated autobiographical summaries from the raw event stream at temperature 0.0:
* **Total Target Facts Evaluated:** 18
* **Retained Target Facts (Exact Key-Value Association):** 2 / 18 ($11.1\%$)
* **Exact KV Omission Rate (Target Key String Absent from Summary):** **$72.2\%$** (13 / 18 facts omitted)
* **Exact Association Mutation Rate (Key Present but Bound Value Altered):** **$16.7\%$** (3 / 18 facts mutated)
* **Unsupported val_* Strings Detected:** 12
* **Strict Partition Invariant:** $2 + 3 + 13 = 18$ ($\text{Retained} + \text{Mutated} + \text{Omitted} \equiv \text{Total}$)
* **Mean Consolidation Compute Overhead:** 416.5 prompt tokens + 345.7 completion tokens per consolidation step.

---

## 5. Architectural Implications & Roadmap Alignment

1. **Selection of `StructuredSelfState` as S05 Carrier**:
   - `StructuredSelfState` traded static read performance for a smaller, explicitly typed and manipulable representation. It is selected for S05 for semantic coverage and experimental controllability (working memory, goal registry, source ledger, unresolved queue, logical clock) rather than static superiority.
2. **Conceptual Distinction for Horizon 1**:
   - **Sprint S04:** Measured whether a model can *read* state from an externally constructed configuration.
   - **Sprint S05:** Will measure whether an autonomous system can *maintain and evolve* that state over multi-tick quiet intervals.
3. **Three Reference Controls for S05**:
   - **Oracle Updater:** Deterministic perfect update baseline.
   - **Model Updater:** Autonomous model-driven update loop subject to maintenance drift.
   - **Transcript Reference:** Raw event replay reference baseline.
4. **Exit Gate for Sprint S04**:
   - ✅ Task option leakage eliminated with format-matched morphological distractors.
   - ✅ Dynamic native JSON Schemas (`TARGET_4AFC_SCHEMA` & `TARGET_3AFC_SCHEMA`) enforced with 100% compliance.
   - ✅ Consolidation distortion calculation made association-aware and mathematically invariant.
   - ✅ Serial position table cleanly isolated to Delayed KV retrieval.
   - ✅ Dual micro/macro averages and true Pareto optimality calculated.
   - ✅ Provenance discipline enforced with canonical outputs saved in `results/e03_memory/run_e03_mem_20260815_165234/`.
