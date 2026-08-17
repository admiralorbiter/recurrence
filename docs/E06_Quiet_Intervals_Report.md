# Experiment E06b: Available-Inference Null Consolidation Benchmark Report
## Sprint S07.1 — Horizon 1: Scaffolded Persistence

**Protocol Freeze Anchor:** Commit [`6ae34a2`](https://github.com/admiralorbiter/recurrence/commit/6ae34a2)  
**Canonical Confirmatory Run:** `run_e06b_quiet_20260817_010421_confirmatory` (Seed 1337, $N=1,248$ paired trials, 576 reflection audit traces)  
**Canonical Exploratory Run:** `run_e06b_quiet_20260817_001510_exploratory` (Seed 42, $N=624$ paired trials, 288 reflection audit traces)  
**Primary Research Question:**  
> *"When complete evidence is available pre-null, can scaffolded quiet processing cycles synthesize, verify, and persist task-relevant derived state to improve downstream performance?"*

---

## 1. Executive Summary & Core Scientific Findings

Experiment E06b evaluates a **two-regime derivability design with within-regime paired reflection controls**, comparing an **`available_inference`** regime (both premises $A \to B$ and $B \to C$ asserted pre-null) against a **`missing_premise_control`** regime (only $A \to B$ pre-null) across $K \in \{0, 1, 3, 6, 12\}$ quiet ticks.

### Core Confirmatory Findings ($N=1,248$ Trials, Seed 1337):
1. **Persistent Derivation Write Failure (0.0% Exact Derivation Consolidation):**
   - When all required premises ($A \to B$ and $B \to C$) are asserted in working memory before the quiet interval, `qwen2.5:3b` **fails to reliably externalize and persist the correct transitive deduction** ($A \to C$) into the explicit `derived_inferences` channel.
   - Across 192 selective reflection ticks in the available regime, the model generated **274 derived writes**, of which **zero** were the correct root $\to$ terminal derivation (Precision: **0.0%**, Recall: **0.0%**).
   - This failure is not an inability to perform relational reasoning generally: under Strict Identity, the model achieves **62.5%** multi-hop accuracy on the same available items, and the raw event transcript achieves **78.1%**.
   - Instead, the failure is specific to **derived-state consolidation**: copying and committing opaque symbolic identifiers into persistent state is fragile, leading to corrupted substrings and malformed keys (e.g. `"key_dist1_platinumatinum_bea"`, `"key_dist1_platinum_b_bea": "val_dist1_c_v_vortex"`).
2. **Self-Polluting Epistemic Channel & Retrieval Interference:**
   - Once a malformed derived write enters state, subsequent reflection ticks observe and amplify it, creating a self-polluting loop.
   - This added clutter degrades downstream multi-hop accuracy relative to `Strict Identity` (**31.2% vs 62.5%**, $\Delta = -31.2\%$, exact McNemar $p = 0.0020$, Bootstrap 95% CI: `[-50.0%, -12.5%]`, primary episode permutation $p = 0.0625$).
3. **Factual Integrity vs Epistemic State Quality:**
   - Protected evidence enforcement ensured that working memory and source ledgers experienced **0.0% mutation**.
   - This demonstrates that **factual integrity can remain perfect while epistemic-state quality degrades**: the true facts were not erased, but surrounding them with bad derived state impaired downstream readout.
4. **Severe State Drift in Unconstrained Reflection:**
   - Under `unconstrained_reflection`, the model exhibited **98.4% evidence drift**, collapsing into repetitive token loops (up to 1,850 completion tokens per tick) that wiped working memory and reduced accuracy to **42.7%**.
5. **Deterministic Preservation Remains the Strongest Tested Strategy:**
   - `Strict Identity Scaffold` remained stable across all quiet durations ($60.4\%$ accuracy flat across $K=1, 3, 6, 12$), outperforming all tested autonomous model reflection conditions under S05–S07.

---

## 2. Multi-Condition Performance & Cost Summary Table (Pooled Strictly Across $K > 0$)

| Condition | Group | Micro Acc ($K>0$) | Baseline ($K=0$) | Multi-Hop (Available) | Multi-Hop (Missing) | Goal State (4AFC) | Stable WM (4AFC) | Query Tok | Amortized Tok | Query Latency | Amortized Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strict Identity Scaffold** | No-Write Control | **60.4%** | 58.3% | 62.5% | 12.5% | 68.8% | 75.0% | 515.2 tok | 515.2 tok | 2,484.2 ms | 2,484.2 ms |
| **Replay Transcript (Raw)** | Retrospective Ref | **78.1%** | 75.0% | 78.1% | 68.8% | 64.1% | 96.9% | 593.9 tok | 593.9 tok | 2,520.6 ms | 2,520.6 ms |
| **Clock-Only (Timestamp Cue)** | No-Write Control | **57.8%** | — | 50.0% | 25.0% | 68.8% | 67.2% | 515.7 tok | 515.7 tok | 2,519.4 ms | 2,519.4 ms |
| **Semantic Reasoning (No-Write)** | Discarded Invocation Control | **55.7%** | — | 40.6% | 21.9% | 68.8% | 67.2% | 515.7 tok | 1,521.3 tok | 2,490.1 ms | 30,191.7 ms |
| **Selective Reflection (Derived)** | **Persistent Write (Primary)** | **53.1%** | — | 31.2% | 18.8% | 53.1% | 81.2% | 744.4 tok | 2,065.8 tok | 2,550.0 ms | 15,561.6 ms |
| **Unconstrained Full-State Rewrite** | Diagnostic Control | **42.7%** | — | 18.8% | 65.6% | 67.2% | 18.8% | 418.8 tok | 1,533.4 tok | 2,514.5 ms | 13,360.3 ms |

*Note on Controls:* `clock_only` and `semantic_no_write` produce bit-for-bit identical evaluation context strings (`context_hash` equality enforced by invariant test). Any performance difference between them reflects backend execution variability rather than representational difference.

---

## 3. Targeted Causal Estimands & Statistical Inference

| Causal Contrast | Target Domain | Regime | $\Delta$ Accuracy | 95% Bootstrap CI | Discordance ($b / c$) | Exact McNemar $p$ | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Delta_derivation-available`** | `derivation_multihop` | `available_inference` | **-31.2%** | [-50.0%, -12.5%] | 0 / 10 | $p = 0.0020$ | $p = 0.0625$ (`exact_exhaustive`) | **Large derivation deficit with mixed inferential evidence (primary permutation $p=.0625$)** |
| **`Delta_derivation-missing`** | `derivation_multihop` | `missing_premise_control` | **+6.2%** | [+0.0%, +15.6%] | 2 / 0 | $p = 0.5000$ | $p = 0.5000$ (`exact_exhaustive`) | **Null / Conservative Resistance** |
| **`Delta_derivation-nowrite-avail`** | `derivation_multihop` | `available_inference` | **-9.4%** | [-34.4%, +12.5%] | 4 / 7 | $p = 0.5488$ | $p = 0.6719$ (`exact_exhaustive`) | **No Resolved Storage Advantage** |
| **`Delta_goal-consolidation`** | `goal_activation` | `all` | **-15.6%** | [-37.5%, +7.8%] | 8 / 18 | $p = 0.0755$ | $p = 0.2620$ (`exact_exhaustive`) | **No Resolved Goal Difference** |
| **`Delta_clock-cue`** | `all` | `all` | **-2.6%** | [-14.1%, +8.3%] | 32 / 37 | $p = 0.6305$ | $p = 0.7238$ (`exact_exhaustive`) | **Null / Timing Cue Invariant** |
| **`Delta_evidence-integrity`** | `stable_kv` | `all` | **+6.2%** | [-9.4%, +23.4%] | 9 / 5 | $p = 0.4240$ | $p = 0.6016$ (`exact_exhaustive`) | **Evidence Invariance Confirmed** |
| **`Delta_unconstrained-drift`** | `all` | `all` | **-17.7%** | [-33.9%, -1.5%] | 33 / 67 | $p = 0.0009$ | $p = 0.0707$ (`exact_exhaustive`) | **Large descriptive state decay (drift: 98.4%, primary permutation $p=.0707$)** |

---

## 4. Mechanistic Reflection Quality & Audit Breakdown

From the 576 per-tick reflection audit traces (`ReflectionTickTrace`):
- **Selective Reflection Schema Validity:** **96.4%** across 192 selective reflection ticks.
- **Unconstrained Reflection Schema Validity:** **98.4%** across 192 unconstrained reflection ticks.
- **Semantic No-Write Schema Validity:** **100.0%** across 192 discarded semantic invocation ticks.
- **Overall Schema Validity:** **98.3%** across all 576 reflection ticks.
- **Available Regime Inferences Written:** **274 total writes** (Exact Correct: **0**).
- **Derived Inference Precision (Available Regime):** **0.0%**
- **Derived Inference Recall (Available Regime):** **0.0%**
- **Premature Hallucination Rate (Missing Premise Regime):** **2.38** premature ungrounded deductions per episode.
- **Selective Evidence Mutation Rate:** **0.0%** (strictly enforced and verified by SHA-256 assertions).
- **Unconstrained Evidence Drift Rate:** **98.4%** (repetitive token degeneration wiping working memory).

---

## 5. Scientific Synthesis & Gate Decision for Sprint S07

> **Sprint S07 / S07.1 Final Scientific Story:**  
> 1. **Under Informational Absence (E06a):** When nothing useful can yet be inferred, repeated Level-1 reflection mostly adds noise.
> 2. **Under Available Evidence (E06b):** Even when a valid relational inference is fully available pre-null, `qwen2.5:3b` repeatedly fails to externalize that inference into the explicit derived-state channel and instead generates self-reinforcing symbolic clutter.
> 3. **Factual Integrity vs Epistemic Quality:** Protected evidence prevents factual corruption of underlying working memory, but does not prevent epistemic interference from surrounding hallucinations.
> 4. **Level-1 Architectural Baseline:** Deterministic preservation remains the strongest tested Level-1 state-management strategy under S05–S07. This explicit reflection/write mechanism does not provide a successful Level-1 consolidation mechanism; whether native latent recurrence behaves differently remains an open H2 question.

Sprint S07 is fully closed and frozen. We proceed to **Sprint S08 (State Reset, Clone, and Swap Invariance)**.
