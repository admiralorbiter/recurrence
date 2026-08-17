# Experiment E06b: Available-Inference Null Consolidation Benchmark Report
## Sprint S07.1 — Horizon 1: Scaffolded Persistence

**Protocol Freeze Anchor:** Commit [`6ae34a2`](https://github.com/admiralorbiter/recurrence/commit/6ae34a2)  
**Canonical Confirmatory Run:** `run_e06b_quiet_20260817_010421_confirmatory` (Seed 1337, $N=1,248$ paired trials, 576 reflection audit traces)  
**Canonical Exploratory Run:** `run_e06b_quiet_20260817_001510_exploratory` (Seed 42, $N=624$ paired trials, 288 reflection audit traces)  
**Primary Research Question:**  
> *"When complete evidence is available pre-null, can scaffolded quiet processing cycles synthesize, verify, and persist task-relevant derived state to improve downstream performance?"*

---

## 1. Executive Summary & Core Scientific Findings

Experiment E06b evaluates the $2 \times 2$ **Derivability $\times$ Reflection** factorial, comparing an **`available_inference`** regime (both premises $A \to B$ and $B \to C$ asserted pre-null) against a **`missing_premise_control`** regime (only $A \to B$ pre-null).

### Core Confirmatory Findings ($N=1,248$ Trials, Seed 1337):
1. **The Synthetic Derivation Deficit (0.0% Precision & Recall Confirmed):**
   - Even when all premises ($A \to B$ and $B \to C$) are fully asserted in working memory before the quiet interval, `qwen2.5:3b` **completely fails to deduce and persist the true transitive link** ($A \to C$).
   - Across 192 selective reflection ticks in the available regime, the model generated **274 derived inferences**, of which **exactly zero** were the correct relational deduction (Precision: **0.0%**, Recall: **0.0%**).
   - In place of valid deductions, the model generated repetitive lexical distortions (e.g. `"key_dist1_platinumatinum_bea"`, `"key_dist1_platinum_b_bea": "val_dist1_c_v_vortex"`), actively polluting its structured context.
2. **Epistemic Clutter & Downstream Retrieval Interference:**
   - On the Multi-Hop task when evidence was fully available, `Selective Reflection` performed significantly worse than `Strict Identity` (**31.2% vs 62.5%**, $\Delta = -31.2\%$, exact McNemar $p = 0.0020$, Bootstrap 95% CI: `[-50.0%, -12.5%]`).
   - The model's own hallucinated reflections acted as in-context distractors, degrading its ability to extract facts from its own protected memory.
3. **Catastrophic Degeneration in Unconstrained Mode:**
   - Under `unconstrained_reflection`, the model suffered **98.4% evidence drift**, collapsing into token repetition loops (up to 1,850 completion tokens per tick) that frequently wiped working memory to `{}` and degraded accuracy to **42.7%**.
4. **Deterministic Preservation Invariant Re-Confirmed:**
   - `Strict Identity Scaffold` remained completely invariant across all quiet durations ($60.4\%$ accuracy flat across $K=1, 3, 6, 12$), outperforming all autonomous model reflection conditions.

---

## 2. Multi-Condition Performance & Cost Summary Table (Pooled Strictly Across $K > 0$)

| Condition | Group | Micro Acc ($K>0$) | Baseline ($K=0$) | Multi-Hop (Available) | Multi-Hop (Missing) | Goal State (4AFC) | Stable WM (4AFC) | Query Tok | Amortized Tok | Query Lat | Amortized Lat |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strict Identity Scaffold** | No Write Control | **60.4%** | 58.3% | 62.5% | 12.5% | 68.8% | 75.0% | 515.2 tok | 515.2 tok | 2,484.2 ms | 2,484.2 ms |
| **Replay Transcript (Raw)** | Retrospective Reference | **78.1%** | 75.0% | 78.1% | 68.8% | 64.1% | 96.9% | 593.9 tok | 593.9 tok | 2,520.6 ms | 2,520.6 ms |
| **Clock-Only (Timestamp Cue)** | No Write Control | **57.8%** | — | 50.0% | 25.0% | 68.8% | 67.2% | 515.7 tok | 515.7 tok | 2,519.4 ms | 2,519.4 ms |
| **Semantic Reasoning (No-Write)** | No Write Control | **55.7%** | — | 40.6% | 21.9% | 68.8% | 67.2% | 515.7 tok | 1,521.3 tok | 2,490.1 ms | 30,191.7 ms |
| **Selective Reflection (Derived Channel)** | **Persistent Write (Primary)** | **53.1%** | — | 31.2% | 18.8% | 53.1% | 81.2% | 744.4 tok | 2,065.8 tok | 2,550.0 ms | 15,561.6 ms |
| **Unconstrained Full-State Rewrite** | Persistent Write (Diagnostic) | **42.7%** | — | 18.8% | 65.6% | 67.2% | 18.8% | 418.8 tok | 1,533.4 tok | 2,514.5 ms | 13,360.3 ms |

---

## 3. Targeted Causal Estimands & Statistical Inference

| Causal Contrast | Target Domain | Regime | $\Delta$ Accuracy | 95% Bootstrap CI | Discordance ($b / c$) | Exact McNemar $p$ | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Delta_derivation-available`** | `derivation_multihop` | `available_inference` | **-31.2%** | [-50.0%, -12.5%] | 0 / 10 | $p = 0.0020$ | $p = 0.0625$ (`exact_exhaustive`) | **Statistically Significant Derivation Deficit** |
| **`Delta_derivation-missing`** | `derivation_multihop` | `missing_premise_control` | **+6.2%** | [+0.0%, +15.6%] | 2 / 0 | $p = 0.5000$ | $p = 0.5000$ (`exact_exhaustive`) | **Null / Conservative Resistance** |
| **`Delta_derivation-nowrite-avail`** | `derivation_multihop` | `available_inference` | **-9.4%** | [-34.4%, +12.5%] | 4 / 7 | $p = 0.5488$ | $p = 0.6719$ (`exact_exhaustive`) | **No Resolved Storage Advantage** |
| **`Delta_goal-consolidation`** | `goal_activation` | `all` | **-15.6%** | [-37.5%, +7.8%] | 8 / 18 | $p = 0.0755$ | $p = 0.2620$ (`exact_exhaustive`) | **No Resolved Goal Difference** |
| **`Delta_clock-cue`** | `all` | `all` | **-2.6%** | [-14.1%, +8.3%] | 32 / 37 | $p = 0.6305$ | $p = 0.7238$ (`exact_exhaustive`) | **Null / Timing Cue Invariant** |
| **`Delta_evidence-integrity`** | `stable_kv` | `all` | **+6.2%** | [-9.4%, +23.4%] | 9 / 5 | $p = 0.4240$ | $p = 0.6016$ (`exact_exhaustive`) | **Evidence Invariance Confirmed** |
| **`Delta_unconstrained-drift`** | `all` | `all` | **-17.7%** | [-33.9%, -1.5%] | 33 / 67 | $p = 0.0009$ | $p = 0.0707$ (`exact_exhaustive`) | **Severe Full-State Drift** |

---

## 4. Mechanistic Derived Inference Quality Metrics

- **Valid Reflection Schema Rate:** **96.4%** (192 total reflection ticks logged)
- **Available Regime Inferences Written:** **274** (Correct: **0**)
- **Derived Inference Precision (Available Regime):** **0.0%**
- **Derived Inference Recall (Available Regime):** **0.0%**
- **Premature Hallucination Rate (Missing Premise Regime):** **2.38** premature deductions / episode

---

## 5. Scientific Synthesis & Gate Decision for Sprint S07

> **Sprint S07 / S07.1 Final Scientific Conclusions:**  
> 1. **Informational Absence (E06a):** Quiet processing without new evidence injects representational noise and cannot beat deterministic preservation.
> 2. **Available Inferences (E06b):** Even when complete premises exist pre-null, explicit model reflection generates zero valid multi-hop deductions ($0.0\%$ precision), creating epistemic clutter that impairs downstream performance ($\Delta = -31.2\%$).
> 3. **The Architectural Finding:** Explicit scaffolded reflection cannot substitute for true latent state recurrence. Deterministic preservation of ground-truth evidence remains the optimal Level-1 strategy.

Sprint S07 is fully closed out and frozen. The research program is ready to proceed to **Sprint S08 (State Reset, Clone, and Swap Invariance)** and **Sprint S09 (Level-1 Metacognition & Ownership Battery)**.
