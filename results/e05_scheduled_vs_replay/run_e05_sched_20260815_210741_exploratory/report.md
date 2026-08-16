# Experiment E05d: Scheduled versus Replay Benchmark Report (Sprint S06.3 Final)

**Run ID:** `run_e05_sched_20260815_210741_exploratory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-15T21:18:37.195536+00:00  
**Scope:** 12 Episodes across Horizons [10, 25, 50] | 240 Total Paired Trials  
**Primary Endpoint:** Causal Estimands ($\Delta_{\text{online-direct}}$, $\Delta_{\text{reconstruction}}$, $\Delta_{\text{schedule}}$, $\Delta_{\text{representation}}$)  

---

## 1. Executive Summary & Benchmark Results

Experiment E05d evaluates whether an autonomous agent maintaining an explicit Level-1 state incrementally across discrete arrival ticks achieves superior accuracy, lower retrieval error, or computational efficiency compared to matched retrospective replay of uncompressed event history.

All measurement validity checks passed (in-context foils, explicit pending goal construct, exact prompt invariants, repaired reconstruction interface).

### Multi-Condition Performance & Cost Summary Table (Pooled across Horizons)

| Condition | Micro Accuracy | Macro Accuracy | Delayed KV (4AFC) | Source Attr (3AFC) | Goal State (4AFC) | Multi-Hop (4AFC) | Query Prompt Tok | Amortized Prompt Tok | Query Latency | Amortized Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scheduled Incremental State** | **60.4%** | 60.4% | 83.3% | 58.3% | 66.7% | 33.3% | 406.9 tok | 406.9 tok | 2450.2 ms | 2450.2 ms |
| **Deterministic Replay State** | **62.5%** | 62.5% | 91.7% | 58.3% | 66.7% | 33.3% | 406.9 tok | 406.9 tok | 2413.9 ms | 2413.9 ms |
| **Replay Transcript (Raw)** | **62.5%** | 62.5% | 91.7% | 58.3% | 41.7% | 58.3% | 800.0 tok | 800.0 tok | 2435.0 ms | 2435.0 ms |
| **Model Reconstructed Replay** | **33.3%** | 33.3% | 25.0% | 41.7% | 25.0% | 41.7% | 144.6 tok | 315.1 tok | 2426.6 ms | 3928.1 ms |
| **Fresh (No History Floor)** | **20.8%** | 20.8% | 16.7% | 41.7% | 16.7% | 8.3% | 113.6 tok | 113.6 tok | 2377.0 ms | 2377.0 ms |

---

## 2. Causal Estimand Contrasts & Statistical Inference

Episode-clustered paired bootstrap 95% confidence intervals (B=2,000), exact two-sided binomial McNemar tests, and episode-level sign-flip permutation tests (Primary Inferential Decision):

| Causal Contrast | Contrast Definition | $\Delta$ Accuracy | 95% Bootstrap CI | Discordance ($b / c$) | Exact McNemar $p$ | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Delta_online-direct`** | Online State vs Raw Transcript | **-2.1%** | [-20.8%, +16.7%] | 8 / 9 | 1.0000 | 1.0000 (`exact_exhaustive`) | **No Resolved Pooled Difference** |
| **`Delta_reconstruction`** | Online State vs Model Recon State | **+27.1%** | [+6.3%, +45.8%] | 23 / 10 | 0.0351 | 0.0430 (`exact_exhaustive`) | **Statistically Significant (Reconstruction Bottleneck)** |
| **`Delta_schedule`** | Online State vs Retrospective State | **-2.1%** | [-6.2%, +0.0%] | 0 / 1 | 1.0000 | 1.0000 (`exact_exhaustive`) | **Null / Architectural Invariant Verified** |
| **`Delta_representation`** | Retrospective State vs Transcript | **+0.0%** | [-16.7%, +18.8%] | 8 / 8 | 1.0000 | 1.0000 (`exact_exhaustive`) | **Transcript-favoring point estimate with conflicting inferential evidence; not resolved by primary permutation test** |

---

## 3. Horizon Breakdown & Scaling Analysis

| Horizon ($T$ ticks) | Incremental State | Replay Det State | Replay Transcript | Replay Model State | Fresh Floor | Incremental Prompt Tok | Transcript Prompt Tok | $\Delta_{\text{online-direct}}$ [95% CI] | Exact McNemar $p$ | Permutation $p$ (Method) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T=10$ ticks** | 50.0% | 50.0% | 62.5% | 50.0% | 12.5% | 403.5 tok | 566.2 tok | -12.5% [-37.5%, +12.5%] | 0.6250 | 0.7500 (`exact_exhaustive`) |
| **$T=25$ ticks** | 68.8% | 68.8% | 75.0% | 18.8% | 25.0% | 409.5 tok | 777.8 tok | -6.2% [-37.5%, +31.2%] | 1.0000 | 1.0000 (`exact_exhaustive`) |
| **$T=50$ ticks** | 62.5% | 68.8% | 50.0% | 31.2% | 25.0% | 407.6 tok | 1055.9 tok | +12.5% [-12.5%, +37.5%] | 0.7266 | 0.7500 (`exact_exhaustive`) |

---

## 4. Object-Level Model Reconstruction Fidelity (Oracle Benchmark)

- **Working Memory Key-Value Retention Rate:** 0.0%
- **Goal Status Match Rate:** 0.0%
- **Source Ledger Attribution Accuracy:** 0.0%

---

## 5. Key Scientific Conclusions & Gate Assessment

1. **Deterministic Replay Invariant:** Online incremental state maintenance and retrospective deterministic replay state achieve bit-for-bit identical terminal state hashes and literal evaluation prompt hashes ($\Delta_{\text{state}} \equiv 0$, $\Delta_{\text{prompt}} \equiv 0$). Residual trial-level discordance reflects backend sampling stochasticity rather than a cognitive scheduling effect.
2. **The Model Retrospective Reconstruction Bottleneck:** When a compact structured state is required, incremental maintenance avoids the reconstruction loss observed with this single-pass Qwen2.5-3B retrospective procedure.
3. **Horizon Scaling & Token Bounding:** Incremental structured state querying maintains a bounded prompt size ($O(K)$) across horizons, reducing prompt token costs relative to uncompressed history ($O(T)$).
4. **Horizon 1 Program Progression:** These findings confirm that explicit Level-1 persistence is transcript-reconstructible under deterministic transition rules. The research program continues inside Horizon 1 with Sprint S07 (Quiet Interval & Null-Tick Screen), S08 (State Swap/Reset), and S09 (Metacognitive Readout).