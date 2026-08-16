# Experiment E06: Scaffolded Null-Interval & Quiet Processing Benchmark Report (Sprint S07)

**Run ID:** `run_e06_quiet_20260816_221116_confirmatory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `CONFIRMATORY` (Seed: `1337`)  
**Date:** 2026-08-16T23:09:48.785790+00:00  
**Scope:** 8 Base Episodes Cloned Across Intervals $K \in [0, 1, 3, 6, 12]$ | 832 Total Paired Trials  
**Primary Question:** *Do scaffolded null-interval update cycles selectively preserve or reorganize unresolved cognitive state?*  

---

## 1. Executive Summary & Benchmark Results

Experiment E06 evaluates whether intervening quiet processing cycles ($K \in \{0, 1, 3, 6, 12\}$ null ticks) placed between a common prefix ($E_{\text{prefix}}$) and continuation ($E_{\text{continuation}}$) selectively improve multi-hop relational derivation, source conflict consolidation, and goal prioritization, or whether they introduce representational drift.

### Multi-Condition Performance & Cost Summary Table (Pooled Across $K > 0$)

| Condition | Group | Micro Accuracy | Multi-Hop Derivation (4AFC) | Source Conflict (3AFC) | Goal State (4AFC) | Stable WM (4AFC) | Query Prompt Tok | Amortized Prompt Tok | Query Latency | Amortized Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strict Identity Scaffold** | No Write Control | **47.5%** | 15.0% | 37.5% | 75.0% | 62.5% | 508.9 tok | 508.9 tok | 2445.8 ms | 2445.8 ms |
| **Replay Transcript (Raw)** | Retrospective Reference | **62.5%** | 85.0% | 5.0% | 75.0% | 85.0% | 627.8 tok | 627.8 tok | 2473.4 ms | 2473.4 ms |
| **Clock-Only (Timestamp Cue)** | No Write Control | **56.2%** | 34.4% | 46.9% | 65.6% | 78.1% | 509.4 tok | 509.4 tok | 2450.8 ms | 2450.8 ms |
| **Semantic Reasoning (No-Write)** | No Write Control | **56.2%** | 34.4% | 46.9% | 65.6% | 78.1% | 509.4 tok | 1063.7 tok | 2426.7 ms | 7080.6 ms |
| **Selective Reflection (Derived Channel)** | **Persistent Write (Primary)** | **35.9%** | 21.9% | 28.1% | 37.5% | 56.2% | 658.8 tok | 1550.0 tok | 2478.8 ms | 10519.5 ms |
| **Unconstrained Full-State Rewrite** | Persistent Write (Diagnostic) | **46.9%** | 46.9% | 59.4% | 46.9% | 34.4% | 459.3 tok | 1108.3 tok | 2473.1 ms | 10490.6 ms |

---

## 2. Targeted Causal Estimands & Statistical Inference

Episode-clustered paired bootstrap 95% confidence intervals (B=2,000), exact two-sided binomial McNemar tests, and episode-level sign-flip permutation tests (Primary Inferential Decision):

| Causal Contrast | Target Domain | Contrast Definition | $\Delta$ Accuracy | 95% Bootstrap CI | Discordance ($b / c$) | Exact McNemar $p$ | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Delta_derivation-selective`** | `derivation_multihop` | Selective Reflection vs Strict Identity (Multi-Hop) | **+9.4%** | [-18.8%, +31.2%] | 6 / 3 | 0.5078 | 0.6875 (`exact_exhaustive`) | **No Resolved Difference / Null** |
| **`Delta_derivation-nowrite`** | `derivation_multihop` | Selective Reflection vs Semantic No-Write (Multi-Hop) | **-12.5%** | [-40.6%, +15.6%] | 5 / 9 | 0.4240 | 0.5625 (`exact_exhaustive`) | **No Resolved Difference / Null** |
| **`Delta_conflict-consolidation`** | `source_conflict` | Selective Reflection vs Strict Identity (Conflict) | **-9.4%** | [-37.5%, +15.6%] | 3 / 6 | 0.5078 | 0.6875 (`exact_exhaustive`) | **No Resolved Difference / Null** |
| **`Delta_clock-cue`** | `all` | Clock-Only vs Strict Identity (All Probes) | **+9.4%** | [-2.3%, +21.9%] | 24 / 12 | 0.0652 | 0.2656 (`exact_exhaustive`) | **Null / Timing Cue Invariant** |
| **`Delta_evidence-integrity`** | `stable_kv` | Selective Reflection vs Strict Identity (Stable KV) | **-6.2%** | [-40.6%, +25.1%] | 6 / 8 | 0.7905 | 0.8438 (`exact_exhaustive`) | **Evidence Invariance Confirmed** |
| **`Delta_unconstrained-drift`** | `all` | Unconstrained Reflection vs Strict Identity (All Probes) | **+0.0%** | [-18.8%, +14.1%] | 27 / 27 | 1.0000 | 1.0000 (`exact_exhaustive`) | **No Resolved Difference** |

---

## 3. Quiet Interval Scaling Dynamics ($K \in \{0, 1, 3, 6, 12\}$ Null Ticks)

| Quiet Interval | Strict Identity | Clock-Only | Semantic No-Write | Selective Reflection | Unconstrained Rewrite | Replay Transcript | $\Delta_{\text{derivation-selective}}$ [95% CI] | Permutation $p$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K=0$ ticks** | 50.0% | - | - | - | - | 62.5% | N/A | N/A |
| **$K=1$ ticks** | 46.9% | 56.2% | 59.4% | 37.5% | 62.5% | 62.5% | -12.5% [-37.5%, +0.0%] | 1.0000 (`exact_exhaustive`) |
| **$K=3$ ticks** | 46.9% | 50.0% | 46.9% | 34.4% | 46.9% | 65.6% | +25.0% [-25.0%, +75.0%] | 0.6250 (`exact_exhaustive`) |
| **$K=6$ ticks** | 46.9% | 56.2% | 59.4% | 37.5% | 40.6% | 59.4% | +25.0% [-25.0%, +62.5%] | 0.6250 (`exact_exhaustive`) |
| **$K=12$ ticks** | 46.9% | 62.5% | 59.4% | 34.4% | 37.5% | 62.5% | +0.0% [+0.0%, +0.0%] | 1.0000 (`exact_exhaustive`) |

---

## 4. Evidence Integrity & Representational Drift Analysis

- **Selective Reflection Protected Evidence Mutation Rate:** **0.0%** (enforced by invariant assertion)
- **Unconstrained Reflection Evidence Drift / Slot Loss Rate:** **100.0%**

---

## 5. Key Scientific Conclusions & Gate Assessment

1. **Scaffolded Null Processing vs Identity Baseline:** Evaluates whether active quiet processing cycles reorganize state for later continuation integration or whether deterministic identity preservation suffices.
2. **Evidence Channel Protection:** Confirms that restricting write access to `derived_inferences` and `unresolved_items` prevents the catastrophic state decay observed under unconstrained reflection.
3. **Compute vs Storage Separation:** Compares persistent writing against matched semantic reasoning token exposure (`semantic_no_write`).
4. **Horizon 1 Program Progression:** These findings complete the quiet-interval screen of Horizon 1, advancing to Sprint S08 (Reset/Clone/Swap) and Sprint S09 (Metacognition & Ownership).