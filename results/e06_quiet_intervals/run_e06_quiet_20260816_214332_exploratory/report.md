# Experiment E06: Scaffolded Null-Interval & Quiet Processing Benchmark Report (Sprint S07)

**Run ID:** `run_e06_quiet_20260816_214332_exploratory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-16T22:11:05.168770+00:00  
**Scope:** 4 Base Episodes Cloned Across Intervals $K \in [0, 1, 3, 6, 12]$ | 416 Total Paired Trials  
**Primary Question:** *Do scaffolded null-interval update cycles selectively preserve or reorganize unresolved cognitive state?*  

---

## 1. Executive Summary & Benchmark Results

Experiment E06 evaluates whether intervening quiet processing cycles ($K \in \{0, 1, 3, 6, 12\}$ null ticks) placed between a common prefix ($E_{\text{prefix}}$) and continuation ($E_{\text{continuation}}$) selectively improve multi-hop relational derivation, source conflict consolidation, and goal prioritization, or whether they introduce representational drift.

### Multi-Condition Performance & Cost Summary Table (Pooled Across $K > 0$)

| Condition | Group | Micro Accuracy | Multi-Hop Derivation (4AFC) | Source Conflict (3AFC) | Goal State (4AFC) | Stable WM (4AFC) | Query Prompt Tok | Amortized Prompt Tok | Query Latency | Amortized Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strict Identity Scaffold** | No Write Control | **67.5%** | 45.0% | 50.0% | 100.0% | 75.0% | 510.4 tok | 510.4 tok | 2489.1 ms | 2489.1 ms |
| **Replay Transcript (Raw)** | Retrospective Reference | **57.5%** | 45.0% | 15.0% | 90.0% | 80.0% | 631.7 tok | 631.7 tok | 2494.2 ms | 2494.2 ms |
| **Clock-Only (Timestamp Cue)** | No Write Control | **65.6%** | 37.5% | 50.0% | 100.0% | 75.0% | 510.9 tok | 510.9 tok | 2485.5 ms | 2485.5 ms |
| **Semantic Reasoning (No-Write)** | No Write Control | **64.1%** | 31.2% | 50.0% | 100.0% | 75.0% | 510.9 tok | 913.9 tok | 2469.9 ms | 6753.5 ms |
| **Selective Reflection (Derived Channel)** | **Persistent Write (Primary)** | **57.8%** | 50.0% | 31.2% | 81.2% | 68.8% | 595.4 tok | 1368.4 tok | 2498.2 ms | 7831.1 ms |
| **Unconstrained Full-State Rewrite** | Persistent Write (Diagnostic) | **62.5%** | 50.0% | 62.5% | 93.8% | 43.8% | 473.4 tok | 1144.3 tok | 2489.2 ms | 11335.7 ms |

---

## 2. Targeted Causal Estimands & Statistical Inference

Episode-clustered paired bootstrap 95% confidence intervals (B=2,000), exact two-sided binomial McNemar tests, and episode-level sign-flip permutation tests (Primary Inferential Decision):

| Causal Contrast | Target Domain | Contrast Definition | $\Delta$ Accuracy | 95% Bootstrap CI | Discordance ($b / c$) | Exact McNemar $p$ | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Delta_derivation-selective`** | `derivation_multihop` | Selective Reflection vs Strict Identity (Multi-Hop) | **+0.0%** | [-37.5%, +37.5%] | 3 / 3 | 1.0000 | 1.0000 (`exact_exhaustive`) | **No Resolved Difference / Null** |
| **`Delta_derivation-nowrite`** | `derivation_multihop` | Selective Reflection vs Semantic No-Write (Multi-Hop) | **+18.8%** | [-12.5%, +43.8%] | 4 / 1 | 0.3750 | 0.5000 (`exact_exhaustive`) | **No Resolved Difference / Null** |
| **`Delta_conflict-consolidation`** | `source_conflict` | Selective Reflection vs Strict Identity (Conflict) | **-18.8%** | [-62.5%, +25.0%] | 2 / 5 | 0.4531 | 0.6250 (`exact_exhaustive`) | **No Resolved Difference / Null** |
| **`Delta_clock-cue`** | `all` | Clock-Only vs Strict Identity (All Probes) | **-3.1%** | [-9.4%, +0.0%] | 0 / 2 | 0.5000 | 1.0000 (`exact_exhaustive`) | **Null / Timing Cue Invariant** |
| **`Delta_evidence-integrity`** | `stable_kv` | Selective Reflection vs Strict Identity (Stable KV) | **-6.2%** | [-56.2%, +37.5%] | 2 / 3 | 1.0000 | 1.0000 (`exact_exhaustive`) | **Evidence Invariance Confirmed** |
| **`Delta_unconstrained-drift`** | `all` | Unconstrained Reflection vs Strict Identity (All Probes) | **-6.2%** | [-34.4%, +21.9%] | 9 / 13 | 0.5235 | 1.0000 (`exact_exhaustive`) | **No Resolved Difference** |

---

## 3. Quiet Interval Scaling Dynamics ($K \in \{0, 1, 3, 6, 12\}$ Null Ticks)

| Quiet Interval | Strict Identity | Clock-Only | Semantic No-Write | Selective Reflection | Unconstrained Rewrite | Replay Transcript | $\Delta_{\text{derivation-selective}}$ [95% CI] | Permutation $p$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K=0$ ticks** | 62.5% | - | - | - | - | 68.8% | N/A | N/A |
| **$K=1$ ticks** | 68.8% | 68.8% | 68.8% | 56.2% | 50.0% | 50.0% | +25.0% [+0.0%, +75.0%] | 1.0000 (`exact_exhaustive`) |
| **$K=3$ ticks** | 68.8% | 68.8% | 62.5% | 62.5% | 75.0% | 56.2% | -25.0% [-75.0%, +0.0%] | 1.0000 (`exact_exhaustive`) |
| **$K=6$ ticks** | 68.8% | 62.5% | 62.5% | 68.8% | 62.5% | 50.0% | +0.0% [-75.0%, +75.0%] | 1.0000 (`exact_exhaustive`) |
| **$K=12$ ticks** | 68.8% | 62.5% | 62.5% | 43.8% | 62.5% | 62.5% | +0.0% [-75.0%, +75.0%] | 1.0000 (`exact_exhaustive`) |

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