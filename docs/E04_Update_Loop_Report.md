# Experiment E04 Research Memo: Scaffolded Autonomous Update Loop Benchmark

**Sprint:** S05 (Horizon 1: Scaffolded Persistence)  
**Experiment ID:** `E04_Autonomous_Update_Loop`  
**Run ID:** `run_e04_loop_20260815_173951`  
**Model:** `qwen2.5:3b` (`357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`)  
**Evaluation Protocol:** 5 Multi-Tick Stream Scenarios (15 Ticks/Scenario, 75 Model Update Inferences, 183 Total Evaluated Ticks) under Temperature 0.0 with Native JSON Schema (`STATE_UPDATE_SCHEMA`)

---

## 1. Executive Summary & Core Results

Experiment E04 evaluates the dynamic half of Horizon 1: **Can an autonomous agent incrementally maintain and evolve an explicit structured self-state (`StructuredSelfState`), goal registry, and source ledger over multi-tick quiet intervals without human prompting, state drift, or schema corruption?**

### Multi-Condition Update Stability Table

| Condition / Updater | Schema Compliance | Mean Retention Fidelity | Terminal Retention | Exact Omission Rate | Exact Mutation Rate | Phantom Intrusions | Goal Coherence | Prompt Tok / Tick |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Oracle Updater`** | **100.0%** | **100.0%** | **100.0%** | 0.0% | 0.0% | 0 | **100.0%** | 0.0 tok |
| **`Model Updater (Qwen2.5-3B)`** | **100.0%** | **8.2%** | **0.0%** | 91.8% | 0.0% | 36 | **7.5%** | 390.6 tok |
| **`Cumulative Replay`** | **100.0%** | **100.0%** | **100.0%** | 0.0% | 0.0% | 0 | **100.0%** | 0.0 tok |

*Definitions:*
- **Schema Compliance:** Fraction of tick updates conforming strictly to `STATE_UPDATE_SCHEMA`.
- **Retention Fidelity:** Fraction of active ground-truth key-value pairs accurately bound in `working_memory`.
- **Exact Omission Rate:** Fraction of ground-truth entity keys missing from `working_memory`.
- **Goal Coherence:** Accuracy of goal status tracking (`active`, `suspended`, `completed`).

---

## 2. Key Empirical Discoveries

### 1. 100% Native Schema Invariance Under Unassisted Ticks
- Across all 75 autonomous state transitions executed by `qwen2.5:3b`, **100.0% (75/75)** satisfied native JSON schema validation (`STATE_UPDATE_SCHEMA`).
- The model never crashed, emitted malformed JSON, or violated typed field constraints.

### 2. Category Routing Failure Mode (Slot Misallocation)
- While the model generated valid JSON schema objects, qualitative inspection of the state snapshots revealed a systematic semantic failure mode:
  - **Prose Routing into `unresolved_items`:** Rather than mapping key-value pairs into the `working_memory: Dict[str, str]` dictionary, the model frequently converted incoming events into narrative string fragments and appended them to `unresolved_items: List[str]`.
  - **Source Inversion:** In the `source_ledger`, the model inverted roles (e.g. mapping `"experimenter": "self"`).
- This explains why `Mean Retention Fidelity` in `working_memory` dropped to $8.2\%$ despite $100\%$ schema compliance.

### 3. Bounded Footprint vs. Cumulative Growth
- **Structured State:** Maintained a bounded footprint across 15 ticks (mean $390.6$ prompt tokens per tick, with state size fluctuating stably between $300$ and $700$ characters).
- **Transcript Reference:** Maintained perfect retention ($100\%$) by accumulating raw history, but exhibits monotonically growing token consumption with each new event.

---

## 3. Failure Mode Catalog

| Failure Category | Occurrence Rate | Mechanism & Characterization |
| :--- | :---: | :--- |
| **Schema Violation** | **0.0%** (0 / 75) | Zero syntax or typing failures; native JSON schema decoding was completely invariant. |
| **Slot Misallocation / Category Routing** | **91.8%** (56 / 61 ticks) | Model routed factual assertions as unparsed prose into `unresolved_items` rather than structured key-value bindings in `working_memory`. |
| **Retrospective Mutation** | **0.0%** (0 / 75) | No established bindings were retrospectively altered to wrong values. |
| **Phantom Intrusions** | 36 occurrences | Injected fabricated metadata keys (e.g. `"unresolvededitems"`, `"last_updated_tick"`) into dictionary slots. |
| **Goal Desynchronization** | 92.5% error | Goal status updates (`suspended` vs `active`) were dropped during multi-tick transitions. |

---

## 4. Architectural Implications & Transition to Sprint S06

1. **Static Read vs. Autonomous State Maintenance:**
   - S04 proved that `qwen2.5:3b` can **read** an externally constructed `StructuredSelfState` with $64.3\% - 72.2\%$ accuracy.
   - S05 proved that `qwen2.5:3b` **cannot autonomously maintain** that structured state without specialized slot-extraction scaffolding or fine-tuned state-transition guidance.
2. **Decomposition for Horizon 1 Confirmed:**
   - **Representation Capacity:** Verified by the Oracle Updater ($100\%$ retention, bounded size).
   - **Autonomous Update Quality:** Quantified by the Model Updater ($8.2\%$ retention, 36 phantom intrusions, slot misallocation).
   - **Raw History Availability:** Tracked by Cumulative Replay (lossless retention at linear token growth).
3. **Sprint S06 Bridge:**
   - Sprint S06 will directly compare **Scheduled Incremental Processing** against **Matched Final Replay** to evaluate whether incremental processing yields any causal computational advantage over raw retrospective replay before transitioning to Horizon 2 (Latent Recurrence).
