# Experiment E06b: Available-Inference Null Consolidation Benchmark Report (Sprint S07.1)

**Run ID:** `run_e06b_quiet_20260817_001510_exploratory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-17T01:04:01.883603+00:00  
**Scope:** 8 Base Episodes Evaluated Across Regimes & Intervals $K \in [0, 1, 3, 6, 12]$ | 624 Total Paired Trials  
**Primary Question:** *When complete evidence is available pre-null, can scaffolded quiet processing cycles synthesize, verify, and persist task-relevant derived state?*  

---

## 1. Executive Summary & Core Results

Experiment E06b evaluates the $2 \times 2$ **Derivability $\times$ Reflection** factorial, comparing an **`available_inference`** regime (both premises $A \to B$ and $B \to C$ asserted pre-null) against a **`missing_premise_control`** regime (only $A \to B$ pre-null).

### Multi-Condition Performance & Cost Summary Table (Pooled Strictly Across $K > 0$)

| Condition | Group | Micro Acc ($K>0$) | Baseline ($K=0$) | Multi-Hop (Available) | Multi-Hop (Missing) | Goal State (4AFC) | Stable WM (4AFC) | Query Tok | Amortized Tok | Query Lat | Amortized Lat |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strict Identity Scaffold** | No Write Control | **66.7%** | 75.0% | 25.0% | 25.0% | 100.0% | 75.0% | 517.5 tok | 517.5 tok | 2478.2 ms | 2478.2 ms |
| **Replay Transcript (Raw)** | Retrospective Reference | **74.0%** | 91.7% | 62.5% | 93.8% | 53.1% | 90.6% | 596.0 tok | 596.0 tok | 2521.3 ms | 2521.3 ms |
| **Clock-Only (Timestamp Cue)** | No Write Control | **54.2%** | — | 25.0% | 37.5% | 65.6% | 65.6% | 518.0 tok | 518.0 tok | 2521.0 ms | 2521.0 ms |
| **Semantic Reasoning (No-Write)** | No Write Control | **57.3%** | — | 25.0% | 56.2% | 65.6% | 65.6% | 518.0 tok | 1340.9 tok | 2482.1 ms | 9385.9 ms |
| **Selective Reflection (Derived Channel)** | **Persistent Write (Primary)** | **52.1%** | — | 6.2% | 37.5% | 59.4% | 75.0% | 691.2 tok | 1968.1 tok | 2565.3 ms | 11386.4 ms |
| **Unconstrained Full-State Rewrite** | Persistent Write (Diagnostic) | **35.4%** | — | 12.5% | 56.2% | 43.8% | 28.1% | 408.3 tok | 1302.2 tok | 2509.9 ms | 12353.4 ms |

---

## 2. Targeted Causal Estimands & Statistical Inference

Episode-clustered paired bootstrap 95% confidence intervals (B=2,000), exact two-sided binomial McNemar tests, and episode-level sign-flip permutation tests (Primary Inferential Decision):

| Causal Contrast | Target Domain | Regime | $\Delta$ Accuracy | 95% Bootstrap CI | Discordance ($b / c$) | Exact McNemar $p$ | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Delta_derivation-available`** | `derivation_multihop` | `available_inference` | **-18.8%** | [-56.2%, +0.0%] | 0 / 3 | 0.2500 | 1.0000 (`exact_exhaustive`) | **No Resolved Difference / Null** |
| **`Delta_derivation-missing`** | `derivation_multihop` | `missing_premise_control` | **+12.5%** | [-43.8%, +50.0%] | 5 / 3 | 0.7266 | 0.8750 (`exact_exhaustive`) | **Null / Conservative Resistance** |
| **`Delta_derivation-nowrite-avail`** | `derivation_multihop` | `available_inference` | **-18.8%** | [-56.2%, +0.0%] | 0 / 3 | 0.2500 | 1.0000 (`exact_exhaustive`) | **No Resolved Storage Advantage** |
| **`Delta_goal-consolidation`** | `goal_activation` | `all` | **-40.6%** | [-59.4%, -21.9%] | 0 / 13 | 0.0002 | 0.0156 (`exact_exhaustive`) | **No Resolved Difference / Null** |
| **`Delta_clock-cue`** | `all` | `all` | **-12.5%** | [-27.1%, +2.1%] | 6 / 18 | 0.0227 | 0.2031 (`exact_exhaustive`) | **Null / Timing Cue Invariant** |
| **`Delta_evidence-integrity`** | `stable_kv` | `all` | **+0.0%** | [-31.2%, +31.2%] | 5 / 5 | 1.0000 | 1.0000 (`exact_exhaustive`) | **Evidence Invariance Confirmed** |
| **`Delta_unconstrained-drift`** | `all` | `all` | **-31.2%** | [-54.2%, -9.4%] | 9 / 39 | 0.0000 | 0.0938 (`exact_exhaustive`) | **No Resolved Difference** |

---

## 3. Mechanistic Derived Inference Quality Metrics

- **Valid Reflection Schema Rate:** **96.9%** (96 total reflection ticks logged)
- **Available Regime Inferences Written:** **133** (Correct: 0)
- **Derived Inference Precision (Available Regime):** **0.0%**
- **Derived Inference Recall (Available Regime):** **0.0%**
- **Premature Hallucination Rate (Missing Premise Regime):** **1.50** premature deductions / episode

---

## 4. Quiet Interval Scaling Dynamics ($K \in \{0, 1, 3, 6, 12\}$ Null Ticks)

| Quiet Duration | Strict Identity Scaffold | Clock-Only Cue | Semantic No-Write | Selective Reflection | Unconstrained Rewrite | Replay Transcript | $\Delta_{\text{derivation-avail}}$ [95% CI] | Permutation $p$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K=0$ ticks** | 75.0% | — | — | — | — | 91.7% | N/A | N/A |
| **$K=1$ ticks** | 66.7% | 62.5% | 62.5% | 66.7% | 33.3% | 75.0% | -25.0% [-75.0%, +0.0%] | 1.0000 (`exact_exhaustive`) |
| **$K=3$ ticks** | 66.7% | 50.0% | 58.3% | 58.3% | 45.8% | 70.8% | +0.0% [+0.0%, +0.0%] | 1.0000 (`exact_exhaustive`) |
| **$K=6$ ticks** | 66.7% | 50.0% | 50.0% | 50.0% | 29.2% | 75.0% | -25.0% [-75.0%, +0.0%] | 1.0000 (`exact_exhaustive`) |
| **$K=12$ ticks** | 66.7% | 54.2% | 58.3% | 33.3% | 33.3% | 75.0% | -25.0% [-75.0%, +0.0%] | 1.0000 (`exact_exhaustive`) |

---

## 5. Evidence Integrity & Drift Analysis

- **Selective Reflection Protected Evidence Mutation Rate:** **0.0%** (enforced by invariant assertion)
- **Unconstrained Reflection Evidence Drift Rate:** **100.0%**

---

## 6. Scientific Gate Assessment for Sprint S07.1

1. **Available Derivation vs Identity:** Evaluates whether quiet reflection can legitimately precompute and persist multi-hop links when complete evidence was asserted pre-null.
2. **Resistance to Premature Inference:** Evaluates whether the system avoids hallucinating ungrounded deductions in the missing premise control.
3. **Factual Integrity vs Epistemic Quality:** Assesses whether derived writing introduces epistemic interference on stable working memory retrieval.