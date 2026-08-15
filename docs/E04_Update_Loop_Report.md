# Experiment E04 Research Report: Scaffolded Autonomous Update Loop Benchmark (S05.2 Closeout)

**Sprint:** S05.2 (Horizon 1: Scaffolded Persistence Closeout)  
**Experiment ID:** `E04_Autonomous_Update_Loop_Benchmark`  
**Run ID:** `run_e04_loop_20260815_180935`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Scope:** 6 Benchmark Scenarios (3 Standard 15-tick scenarios, 1 Full Goal Lifecycle 16-tick scenario, 1 Capacity Overflow 28-tick scenario, 1 Long-Horizon 100-tick scenario; 189 logical ticks per condition, 69 active inference ticks, 756 total evaluated ticks).

---

## 1. Executive Summary & Comparative Results

Experiment E04 evaluates whether an autonomous agent can incrementally maintain an explicit structured self-state (`StructuredSelfState`), goal registry, and source ledger over multi-tick quiet intervals without human prompting, state drift, or schema corruption.

### Multi-Condition Update Stability Table

| Condition / Updater | Active Schema Compliance | Scenario-Macro Retention | Tick-Micro Retention | Terminal Retention | Macro Omission Rate | Macro Mutation Rate | Never-Seen Phantoms (Ticks / Unique) | Stale / Evicted Keys | Macro Goal Coherence | Tokens / Active Inference | Tokens / Logical Tick |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Oracle Grounded Update`** | **100.0%** (69/69) | **100.0%** | **100.0%** | **100.0%** | 0.0% | 0.0% | 0 / 0 | 0 | **100.0%** | 0.0 tok | 0.0 tok |
| **`Model Delta Updater (S05.1)`** | **100.0%** (69/69) | **13.2%** | **9.0%** | **11.1%** | 80.6% | 6.2% | 452 / 45 | 0 | **42.8%** | 848.5 tok | 309.8 tok |
| **`Model Full-State Updater (E04a)`** | **100.0%** (69/69) | **6.3%** | **5.4%** | **0.0%** | 92.0% | 1.7% | 56 / 25 | 0 | **16.7%** | 338.4 tok | 123.5 tok |
| **`Deterministic Grounded Reference`** | **100.0%** (69/69) | **100.0%** | **100.0%** | **100.0%** | 0.0% | 0.0% | 0 / 0 | 0 | **100.0%** | 0.0 tok | 0.0 tok |

*Note on Grounded Reference:* In E04b, this condition executes deterministic event processing identically to Oracle. Genuine retrospective replay (reconstructing from accumulated history at query time) is formally reserved for Sprint S06.

---

## 2. Core Scientific Discoveries

### 1. Delta Updating vs. Full-State Rewriting
- **Directional Advantage of Delta Updating:** Emitting structured deltas rather than regenerating the complete state doubled scenario-macro retention ($6.3\% \to 13.2\%$), produced non-zero terminal retention ($0.0\% \to 11.1\%$), and substantially improved goal coherence ($16.7\% \to 42.8\%$).
- **The Error Inheritance Phenomenon:** While full-state rewriting forgets aggressively (which often cleanses previously hallucinated keys), deterministic delta persistence protects erroneous model updates once they enter the state. As a result, `model_delta` exhibited 452 phantom tick instances across 45 unique keys. **Continuity preserves errors as well as truths.**

### 2. Token Footprint & Memory Degradation
- **Tokens per Active Inference:** `model_delta` consumed $848.5$ prompt tokens per active model call (due to structured event payloads and richer instructions), whereas `model_full_state` consumed $338.4$ tokens per call.
- **Why Broken Memory Looks Cheaper:** Because full-state rewriting rapidly forgets almost all entities, its subsequent state prompts become progressively shorter. Broken memory architectures can misleadingly appear computationally cheaper when measured by prompt size.
- **Operating Footprint:** Across the full 189-tick timeline (including 120 quiet ticks), total operating costs were $309.8$ tok/tick for delta vs $123.5$ tok/tick for full-state.

### 3. Quiet Tick Invariance & Idle Stability
- On logical ticks with no incoming events ($\Delta E_t = \emptyset$), the update loop executed an exact identity no-op ($0$ model calls, $0$ tokens, $0$ latency).
- In the **100-tick long-horizon scenario** (`scen_long_horizon_100_301`), the explicit scaffold maintained perfect temporal stability across extensive quiet spans without state drift.

### 4. Capacity Bounding & LRU Eviction Verified
- In the **capacity overflow scenario** (`scen_capacity_overflow_201`), 24 entities were asserted into $K_{\max}=16$ working memory.
- The state manager deterministically evicted the 8 least-recently-updated entities, bounding working memory size to exactly 16 items. The loop-level integration fix ensures recency order updates only on explicit key writes.

### 5. Goal Lifecycle State Machine
- Exercised goal transitions: `active` $\to$ `suspended` $\to$ `active/resumed` $\to$ `completed` + `secondary active` $\to$ `completed`.
- All 14 illegal demotion attempts by the model (e.g. `active` $\to$ `pending`, `suspended` $\to$ `pending`) were intercepted and rejected by `StateManager`, preserving goal integrity.

---

## 3. Formal S05 Scientific Gate Decisions

### 1. S05 Scaffold Gate: **PASS**
The Level-1 explicit persistence architecture is fully functional, auditable, and robust:
- Operates deterministically across active and quiet ticks.
- Enforces strict capacity bounding ($K_{\max}=16$, LRU eviction).
- Enforces goal lifecycle transition invariants and hash-chained audit logging (`state_trace.jsonl`).
- Passes all 65 unit and regression tests in `tests/`.

### 2. S05 Model-Autonomous Maintenance Gate: **FAIL**
`qwen2.5:3b` is not yet reliable as an unassisted semantic state maintainer ($13.2\%$ macro retention, $80.6\%$ omission, $42.8\%$ goal coherence). Small models cannot reliably parse and maintain multi-slot state without dedicated fine-tuning or deterministic parsing constraints.

### 3. Roadmap Fallback Formally Activated
As designed in the Horizon 1 roadmap (*"Use deterministic state-transition constraints or smaller state until the system is stable"*):
- **Canonical Level-1 State Maintainer for S06:** Deterministically maintained `StructuredSelfState` (via `OracleStateUpdater` / `apply_delta`).
- **Model-Maintained Updaters (`model_delta`, `model_full_state`):** Retained as diagnostic negative controls.

---

## 4. Provenance & Commit Lineage

- **S05.1 Code Freeze Commit:** `9cf2c3c`
- **S05.1 Canonical Results Commit:** `d2d816b`
- **S05.2 Closeout Documentation Commit:** (Current)
- **Canonical Artifacts Directory:** [`results/e04_update_loop/run_e04_loop_20260815_180935/`](file:///c:/Users/admir/Github/recurrence/results/e04_update_loop/run_e04_loop_20260815_180935/)
