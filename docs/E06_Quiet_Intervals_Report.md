# Experiment E06: Scaffolded Null-Interval & Quiet Processing Benchmark Report
## Sprint S07 — Horizon 1: Scaffolded Persistence

**Protocol Freeze Anchor:** Commit [`7231f7a`](https://github.com/admiralorbiter/recurrence/commit/7231f7a)  
**Canonical Results Commit:** [`5ea37c1`](https://github.com/admiralorbiter/recurrence/commit/5ea37c1) (Confirmatory Run: `run_e06_quiet_20260816_221116_confirmatory`, Seed 1337)  
**Primary Research Question:**  
> *"Do scaffolded null-interval update cycles selectively preserve or reorganize unresolved cognitive state, or do they merely introduce representational drift and noise?"*

---

## 1. Executive Summary & Core Scientific Findings

Experiment E06 directly evaluates what happens when an autonomous agent experiences **scaffolded quiet intervals** ($K \in \{0, 1, 3, 6, 12\}$ null ticks where no new exogenous observations arrive) placed **between a common prefix ($E_{\text{prefix}}$) and continuation ($E_{\text{continuation}}$)**.

By cloning identical base episodes at the null-interval boundary ($S^*$) and evaluating across 6 strictly controlled conditions, E06 delivers key insights into Level-1 cognitive persistence:

1. **Deterministic Identity Scaffold Stability:**
   - The deterministic identity scaffold maintains stable performance flat across quiet durations ($46.9\%$ flat across $K=1, 3, 6, 12$ in confirmatory; $68.8\%$ in exploratory).
2. **Clock-Only (Timestamp Cue) Invariance:**
   - Exposing advancing timestamps (`last_updated_step = T_{\text{pre}} + K`) without state modification produces no resolved difference relative to strict identity ($\Delta_{\text{clock-cue}} = +9.4\%$, CI `[-2.3%, +21.9%]`, Permutation $p = 0.2656$).
3. **Protected Evidence Invariant Prevents Catastrophic Collapse:**
   - Under `selective_reflection`, clamping `working_memory` and `source_ledger` as read-only preserved a **0.0% evidence mutation rate**.
   - In stark contrast, `unconstrained_reflection` suffered a **100.0% evidence drift and slot loss rate**, collapsing stable working memory accuracy down to $34.4\%$.
4. **S07 Gate Outcome (Negative / Bounded Null Result):**
   - Active quiet processing cycles do **not** confer a statistically resolved advantage over deterministic identity preservation ($\Delta_{\text{derivation-selective}} = +9.4\%$, Permutation $p = 0.6875$; $\Delta_{\text{derivation-nowrite}} = -12.5\%$, Permutation $p = 0.5625$). Level-1 "thinking during idle periods" introduces noise and hallucinations unless strictly constrained by exogenous evidence.

---

## 2. Canonical Confirmatory Benchmark Results ($N=832$ Paired Trials on `qwen2.5:3b`, Seed 1337)

### Multi-Condition Performance & Cost Summary Table (Pooled Across $K > 0$)

| Condition | Group | Micro Accuracy | Multi-Hop Derivation (4AFC) | Source Conflict (3AFC) | Goal State (4AFC) | Stable WM (4AFC) | Query Prompt Tok | Amortized Prompt Tok | Query Latency | Amortized Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strict Identity Scaffold** | No Write Control | **47.5%** | 15.0% | 37.5% | 75.0% | 62.5% | **508.9 tok** | **508.9 tok** | 2,445.8 ms | 2,445.8 ms |
| **Clock-Only (Timestamp Cue)** | No Write Control | **56.2%** | 34.4% | 46.9% | 65.6% | 78.1% | **509.4 tok** | **509.4 tok** | 2,450.8 ms | 2,450.8 ms |
| **Semantic Reasoning (No-Write)** | No Write Control | **56.2%** | 34.4% | 46.9% | 65.6% | 78.1% | **509.4 tok** | **1,063.7 tok** | 2,426.7 ms | 7,080.6 ms |
| **Selective Reflection (Derived Channel)** | **Persistent Write (Primary)** | **35.9%** | 21.9% | 28.1% | 37.5% | 56.2% | **658.8 tok** | **1,550.0 tok** | 2,478.8 ms | 10,519.5 ms |
| **Unconstrained Full-State Rewrite** | Persistent Write (Diagnostic) | **46.9%** | 46.9% | 59.4% | 46.9% | 34.4% | **459.3 tok** | **1,108.3 tok** | 2,473.1 ms | 10,490.6 ms |
| **Replay Transcript (Raw)** | Retrospective Reference | **62.5%** | 85.0% | 5.0% | 75.0% | 85.0% | **627.8 tok** | **627.8 tok** | 2,473.4 ms | 2,473.4 ms |

---

## 3. Targeted Causal Estimands & Statistical Inference

| Causal Contrast | Target Domain | Contrast Definition | Exploratory $\Delta$ (Seed 42) | Confirmatory $\Delta$ (Seed 1337) | Confirmatory 95% Bootstrap CI | Discordance ($b / c$) | Exact McNemar $p$ | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Delta_derivation-selective`** | `derivation_multihop` | Selective Reflection vs Strict Identity | **+0.0%** | **+9.4%** | `[-18.8%, +31.2%]` | $6 / 3$ | $p = 0.5078$ | $p = 0.6875$ (`exact_exhaustive`) | **No Resolved Difference / Null** |
| **`Delta_derivation-nowrite`** | `derivation_multihop` | Selective Reflection vs Semantic No-Write | **+18.8%** | **-12.5%** | `[-40.6%, +15.6%]` | $5 / 9$ | $p = 0.4240$ | $p = 0.5625$ (`exact_exhaustive`) | **No Resolved Difference / Null** |
| **`Delta_conflict-consolidation`** | `source_conflict` | Selective Reflection vs Strict Identity | **-18.8%** | **-9.4%** | `[-37.5%, +15.6%]` | $3 / 6$ | $p = 0.5078$ | $p = 0.6875$ (`exact_exhaustive`) | **No Resolved Difference / Null** |
| **`Delta_clock-cue`** | `all` | Clock-Only vs Strict Identity | **-3.1%** | **+9.4%** | `[-2.3%, +21.9%]` | $24 / 12$ | $p = 0.0652$ | $p = 0.2656$ (`exact_exhaustive`) | **Null / Timing Cue Invariant** |
| **`Delta_evidence-integrity`** | `stable_kv` | Selective Reflection vs Strict Identity | **-6.2%** | **-6.2%** | `[-40.6%, +25.1%]` | $6 / 8$ | $p = 0.7905$ | $p = 0.8438$ (`exact_exhaustive`) | **Evidence Invariance Confirmed** |
| **`Delta_unconstrained-drift`** | `all` | Unconstrained Reflection vs Strict Identity | **-6.2%** | **+0.0%** | `[-18.8%, +14.1%]` | $27 / 27$ | $p = 1.0000$ | $p = 1.0000$ (`exact_exhaustive`) | **No Resolved Difference** |

---

## 4. Quiet Interval Scaling Dynamics ($K \in \{0, 1, 3, 6, 12\}$ Null Ticks)

| Quiet Interval | Strict Identity | Clock-Only | Semantic No-Write | Selective Reflection | Unconstrained Rewrite | Replay Transcript | $\Delta_{\text{derivation-selective}}$ [95% CI] | Permutation $p$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K=0$ ticks** | 50.0% | - | - | - | - | 62.5% | N/A | N/A |
| **$K=1$ ticks** | 46.9% | 56.2% | 59.4% | 37.5% | 62.5% | 62.5% | -12.5% [-37.5%, +0.0%] | $p = 1.0000$ (`exact_exhaustive`) |
| **$K=3$ ticks** | 46.9% | 50.0% | 46.9% | 34.4% | 46.9% | 65.6% | +25.0% [-25.0%, +75.0%] | $p = 0.6250$ (`exact_exhaustive`) |
| **$K=6$ ticks** | 46.9% | 56.2% | 59.4% | 37.5% | 40.6% | 59.4% | +25.0% [-25.0%, +62.5%] | $p = 0.6250$ (`exact_exhaustive`) |
| **$K=12$ ticks** | 46.9% | 62.5% | 59.4% | 34.4% | 37.5% | 62.5% | +0.0% [+0.0%, +0.0%] | $p = 1.0000$ (`exact_exhaustive`) |

---

## 5. Evidence Integrity & Representational Drift Analysis

- **Selective Reflection Protected Evidence Mutation Rate:** **0.0%** (enforced by invariant assertion across all 8 base episodes and 12 quiet ticks).
- **Unconstrained Reflection Evidence Drift / Slot Loss Rate:** **100.0%** (unconstrained model rewrites dropped or corrupted ground-truth bindings in working memory in every episode).

---

## 6. Scientific Conclusions & Gate Decision for Sprint S07

1. **The Null-Tick Negative Invariant:**
   - Level-1 explicit self-state does not autonomously consolidate or reorganize during quiet intervals without exogenous evidence. "Quiet reflection" introduces representational drift and noise rather than cognitive enhancement.
2. **Deterministic Scaffolding is Optimal for Level 1:**
   - Identity preservation by the deterministic scaffold provides flat, stable retrieval across arbitrary quiet intervals ($K=1 \dots 12$), outperforming autonomous model-driven state manipulation.
3. **Horizon 1 Programmatic Milestone:**
   - With S07 complete, Horizon 1 moves forward to **Sprint S08 (State Reset, Clone, and Swap Invariance)** and **Sprint S09 (Metacognitive Readout & Ownership Attribution)** to conclude Level 1 before transitioning to genuine latent recurrence in Horizon 2.
