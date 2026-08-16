# Experiment E05d: Scheduled versus Replay Benchmark Report (Sprint S06.3 Final)

**Run ID:** `run_e05_sched_20260816_141108_exploratory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-16T14:22:16.589887+00:00  
**Scope:** 12 Episodes across Horizons [10, 25, 50] | 240 Total Paired Trials  
**Primary Endpoint:** Causal Estimands ($\Delta_{\text{online-direct}}$, $\Delta_{\text{reconstruction}}$, $\Delta_{\text{schedule}}$, $\Delta_{\text{representation}}$)  

---

## 1. Executive Summary & Benchmark Results

Experiment E05d evaluates whether an autonomous agent maintaining an explicit Level-1 state incrementally across discrete arrival ticks achieves superior accuracy, lower retrieval error, or computational efficiency compared to matched retrospective replay of uncompressed event history.

All measurement validity checks passed (in-context foils, explicit pending goal construct, exact prompt invariants, repaired reconstruction interface).

### Multi-Condition Performance & Cost Summary Table (Pooled across Horizons)

| Condition | Micro Accuracy | Macro Accuracy | Delayed KV (4AFC) | Source Attr (3AFC) | Goal State (4AFC) | Multi-Hop (4AFC) | Mean Query Prompt Tok | Mean Amortized Prompt Tok | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scheduled Incremental State** | **60.4%** | 60.4% | 58.3% | 41.7% | 83.3% | 58.3% | 421.4 tok | 421.4 tok | 2445.0 ms |
| **Deterministic Replay State** | **58.3%** | 58.3% | 50.0% | 41.7% | 83.3% | 58.3% | 421.4 tok | 421.4 tok | 2442.2 ms |
| **Replay Transcript (Raw)** | **72.9%** | 72.9% | 91.7% | 75.0% | 75.0% | 50.0% | 806.3 tok | 806.3 tok | 2428.5 ms |
| **Model Reconstructed Replay** | **33.3%** | 33.3% | 41.7% | 33.3% | 41.7% | 16.7% | 383.4 tok | 554.9 tok | 2446.1 ms |
| **Fresh (No History Floor)** | **29.2%** | 29.2% | 41.7% | 25.0% | 16.7% | 33.3% | 113.9 tok | 113.9 tok | 2378.5 ms |

---

## 2. Causal Estimand Contrasts & Statistical Inference

Episode-clustered paired bootstrap 95% confidence intervals (B=2,000), exact two-sided binomial McNemar tests, and sign-flip permutation tests:

| Causal Contrast | Contrast Definition | $\Delta$ Accuracy | 95% Bootstrap CI | Discordance ($b / c$) | Exact McNemar $p$ | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Delta_online-direct`** | Online State vs Raw Transcript | **-12.5%** | [-25.0%, +2.1%] | 9 / 15 | 0.3075 | 0.1826 (`exact_exhaustive`) | **Null / Indistinguishable** |
| **`Delta_reconstruction`** | Online State vs Model Recon State | **+27.1%** | [+12.5%, +41.7%] | 16 / 3 | 0.0044 | 0.0156 (`exact_exhaustive`) | **Statistically Significant** |
| **`Delta_schedule`** | Online State vs Retrospective State | **+2.1%** | [-4.2%, +8.3%] | 2 / 1 | 1.0000 | 1.0000 (`exact_exhaustive`) | **Null / Indistinguishable** |
| **`Delta_representation`** | Retrospective State vs Transcript | **-14.6%** | [-29.2%, +0.0%] | 10 / 17 | 0.2478 | 0.1465 (`exact_exhaustive`) | **Null / Indistinguishable** |

---

## 3. Horizon Breakdown & Scaling Analysis

| Horizon ($T$ ticks) | Incremental State | Replay Det State | Replay Transcript | Replay Model State | Fresh Floor | Incremental Prompt Tok | Transcript Prompt Tok | $\Delta_{\text{online-direct}}$ [95% CI] | Exact McNemar $p$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T=10$ ticks** | 50.0% | 56.2% | 81.2% | 50.0% | 25.0% | 417.7 tok | 572.4 tok | -31.2% [-43.8%, -25.0%] | 0.1250 |
| **$T=25$ ticks** | 62.5% | 50.0% | 68.8% | 18.8% | 25.0% | 423.6 tok | 783.9 tok | -6.2% [-25.0%, +12.5%] | 1.0000 |
| **$T=50$ ticks** | 68.8% | 68.8% | 68.8% | 31.2% | 37.5% | 422.8 tok | 1062.5 tok | +0.0% [-25.0%, +25.0%] | 1.0000 |

---

## 4. Object-Level Model Reconstruction Fidelity (Oracle Benchmark)

- **Working Memory Key-Value Retention Rate:** 16.7%
- **Goal Status Match Rate:** 0.0%
- **Source Ledger Attribution Accuracy:** 2.8%

---

## 5. Key Scientific Conclusions & Gate Assessment

1. **Deterministic Replay Invariant:** Online incremental state maintenance and retrospective deterministic replay state achieve bit-for-bit identical terminal state hashes and literal evaluation prompt hashes ($\Delta_{\text{state}} \equiv 0$, $\Delta_{\text{prompt}} \equiv 0$). Residual trial-level discordance reflects backend sampling stochasticity rather than a cognitive scheduling effect.
2. **The Model Retrospective Reconstruction Bottleneck:** When a compact structured state is required, incremental maintenance avoids the reconstruction loss observed with this single-pass Qwen2.5-3B retrospective procedure.
3. **Horizon Scaling & Token Bounding:** Incremental structured state querying maintains a bounded prompt size ($O(K)$) across horizons, reducing prompt token costs relative to uncompressed history ($O(T)$).
4. **Horizon 1 Program Progression:** These findings confirm that explicit Level-1 persistence is transcript-reconstructible under deterministic transition rules. The research program continues inside Horizon 1 with Sprint S07 (Quiet Interval & Null-Tick Screen), S08 (State Swap/Reset), and S09 (Metacognitive Readout).