# Experiment E06b: Available-Inference Null Consolidation Benchmark Report (Sprint S07.1)

**Run ID:** `run_e06b_quiet_20260817_010421_confirmatory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `CONFIRMATORY` (Seed: `1337`)  
**Date:** 2026-08-17T03:08:56.628676+00:00  
**Scope:** 16 Base Episodes Evaluated Across Regimes & Intervals $K \in [0, 1, 3, 6, 12]$ | 1248 Total Paired Trials  
**Primary Question:** *When complete evidence is available pre-null, can scaffolded quiet processing cycles synthesize, verify, and persist task-relevant derived state?*  

---

## 1. Executive Summary & Core Results

Experiment E06b evaluates the $2 \times 2$ **Derivability $\times$ Reflection** factorial, comparing an **`available_inference`** regime (both premises $A \to B$ and $B \to C$ asserted pre-null) against a **`missing_premise_control`** regime (only $A \to B$ pre-null).

### Multi-Condition Performance & Cost Summary Table (Pooled Strictly Across $K > 0$)

| Condition | Group | Micro Acc ($K>0$) | Baseline ($K=0$) | Multi-Hop (Available) | Multi-Hop (Missing) | Goal State (4AFC) | Stable WM (4AFC) | Query Tok | Amortized Tok | Query Lat | Amortized Lat |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strict Identity Scaffold** | No Write Control | **60.4%** | 58.3% | 62.5% | 12.5% | 68.8% | 75.0% | 515.2 tok | 515.2 tok | 2484.2 ms | 2484.2 ms |
| **Replay Transcript (Raw)** | Retrospective Reference | **78.1%** | 75.0% | 78.1% | 68.8% | 64.1% | 96.9% | 593.9 tok | 593.9 tok | 2520.6 ms | 2520.6 ms |
| **Clock-Only (Timestamp Cue)** | No Write Control | **57.8%** | — | 50.0% | 25.0% | 68.8% | 67.2% | 515.7 tok | 515.7 tok | 2519.4 ms | 2519.4 ms |
| **Semantic Reasoning (No-Write)** | No Write Control | **55.7%** | — | 40.6% | 21.9% | 68.8% | 67.2% | 515.7 tok | 1521.3 tok | 2490.1 ms | 30191.7 ms |
| **Selective Reflection (Derived Channel)** | **Persistent Write (Primary)** | **53.1%** | — | 31.2% | 18.8% | 53.1% | 81.2% | 744.4 tok | 2065.8 tok | 2550.0 ms | 15561.6 ms |
| **Unconstrained Full-State Rewrite** | Persistent Write (Diagnostic) | **42.7%** | — | 18.8% | 65.6% | 67.2% | 18.8% | 418.8 tok | 1533.4 tok | 2514.5 ms | 13360.3 ms |

---

## 2. Targeted Causal Estimands & Statistical Inference

Episode-clustered paired bootstrap 95% confidence intervals (B=2,000), exact two-sided binomial McNemar tests, and episode-level sign-flip permutation tests (Primary Inferential Decision):

| Causal Contrast | Target Domain | Regime | $\Delta$ Accuracy | 95% Bootstrap CI | Discordance ($b / c$) | Exact McNemar $p$ | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Delta_derivation-available`** | `derivation_multihop` | `available_inference` | **-31.2%** | [-50.0%, -12.5%] | 0 / 10 | 0.0020 | 0.0625 (`exact_exhaustive`) | **No Resolved Difference / Null** |
| **`Delta_derivation-missing`** | `derivation_multihop` | `missing_premise_control` | **+6.2%** | [+0.0%, +15.6%] | 2 / 0 | 0.5000 | 0.5000 (`exact_exhaustive`) | **Null / Conservative Resistance** |
| **`Delta_derivation-nowrite-avail`** | `derivation_multihop` | `available_inference` | **-9.4%** | [-34.4%, +12.5%] | 4 / 7 | 0.5488 | 0.6719 (`exact_exhaustive`) | **No Resolved Storage Advantage** |
| **`Delta_goal-consolidation`** | `goal_activation` | `all` | **-15.6%** | [-37.5%, +7.8%] | 8 / 18 | 0.0755 | 0.2620 (`exact_exhaustive`) | **No Resolved Difference / Null** |
| **`Delta_clock-cue`** | `all` | `all` | **-2.6%** | [-14.1%, +8.3%] | 32 / 37 | 0.6305 | 0.7238 (`exact_exhaustive`) | **Null / Timing Cue Invariant** |
| **`Delta_evidence-integrity`** | `stable_kv` | `all` | **+6.2%** | [-9.4%, +23.4%] | 9 / 5 | 0.4240 | 0.6016 (`exact_exhaustive`) | **Evidence Invariance Confirmed** |
| **`Delta_unconstrained-drift`** | `all` | `all` | **-17.7%** | [-33.9%, -1.5%] | 33 / 67 | 0.0009 | 0.0707 (`exact_exhaustive`) | **No Resolved Difference** |

---

## 3. Mechanistic Derived Inference Quality Metrics

- **Valid Reflection Schema Rate:** **96.4%** (192 total reflection ticks logged)
- **Available Regime Inferences Written:** **274** (Correct: 0)
- **Derived Inference Precision (Available Regime):** **0.0%**
- **Derived Inference Recall (Available Regime):** **0.0%**
- **Premature Hallucination Rate (Missing Premise Regime):** **2.38** premature deductions / episode

---

## 4. Quiet Interval Scaling Dynamics ($K \in \{0, 1, 3, 6, 12\}$ Null Ticks)

| Quiet Duration | Strict Identity Scaffold | Clock-Only Cue | Semantic No-Write | Selective Reflection | Unconstrained Rewrite | Replay Transcript | $\Delta_{\text{derivation-avail}}$ [95% CI] | Permutation $p$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K=0$ ticks** | 58.3% | — | — | — | — | 75.0% | N/A | N/A |
| **$K=1$ ticks** | 60.4% | 62.5% | 58.3% | 56.2% | 47.9% | 81.2% | -25.0% [-50.0%, +0.0%] | 0.5000 (`exact_exhaustive`) |
| **$K=3$ ticks** | 60.4% | 64.6% | 60.4% | 56.2% | 50.0% | 70.8% | -25.0% [-50.0%, +0.0%] | 0.5000 (`exact_exhaustive`) |
| **$K=6$ ticks** | 60.4% | 52.1% | 45.8% | 58.3% | 29.2% | 77.1% | -37.5% [-75.0%, +0.0%] | 0.2500 (`exact_exhaustive`) |
| **$K=12$ ticks** | 60.4% | 52.1% | 58.3% | 41.7% | 43.8% | 83.3% | -37.5% [-75.0%, +0.0%] | 0.2500 (`exact_exhaustive`) |

---

## 5. Evidence Integrity & Drift Analysis

- **Selective Reflection Protected Evidence Mutation Rate:** **0.0%** (enforced by invariant assertion)
- **Unconstrained Reflection Evidence Drift Rate:** **98.4%**

---

## 6. Scientific Gate Assessment for Sprint S07.1

1. **Available Derivation vs Identity:** Evaluates whether quiet reflection can legitimately precompute and persist multi-hop links when complete evidence was asserted pre-null.
2. **Resistance to Premature Inference:** Evaluates whether the system avoids hallucinating ungrounded deductions in the missing premise control.
3. **Factual Integrity vs Epistemic Quality:** Assesses whether derived writing introduces epistemic interference on stable working memory retrieval.