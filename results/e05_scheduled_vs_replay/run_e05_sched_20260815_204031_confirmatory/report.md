# Experiment E05d: Scheduled versus Replay Benchmark Report (Sprint S06.3 Final)

**Run ID:** `run_e05_sched_20260815_204031_confirmatory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `CONFIRMATORY` (Seed: `1337`)  
**Date:** 2026-08-15T20:55:00.813839+00:00  
**Scope:** 12 Episodes across Horizons [10, 25, 50] | 300 Total Paired Trials  
**Primary Endpoint:** Causal Estimands ($\Delta_{\text{online-direct}}$, $\Delta_{\text{reconstruction}}$, $\Delta_{\text{schedule}}$, $\Delta_{\text{representation}}$)  

---

## 1. Executive Summary & Benchmark Results

Experiment E05d evaluates whether an autonomous agent maintaining an explicit Level-1 state incrementally across discrete arrival ticks achieves superior accuracy, lower retrieval error, or computational efficiency compared to matched retrospective replay of uncompressed event history.

All measurement validity checks passed (in-context foils, explicit pending goal construct, exact prompt invariants, repaired reconstruction interface).

### Multi-Condition Performance & Cost Summary Table (Pooled across Horizons)

| Condition | Micro Accuracy | Macro Accuracy | Delayed KV (4AFC) | Source Attr (3AFC) | Goal State (4AFC) | Multi-Hop (4AFC) | Query Prompt Tok | Amortized Prompt Tok | Query Latency | Amortized Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scheduled Incremental State** | **80.0%** | 80.0% | 83.3% | 83.3% | 100.0% | 83.3% | 474.6 tok | 474.6 tok | 2467.1 ms | 2467.1 ms |
| **Deterministic Replay State** | **83.3%** | 83.3% | 100.0% | 83.3% | 100.0% | 83.3% | 474.6 tok | 474.6 tok | 2446.1 ms | 2446.1 ms |
| **Replay Transcript (Raw)** | **66.7%** | 66.7% | 91.7% | 66.7% | 50.0% | 91.7% | 883.9 tok | 883.9 tok | 2429.1 ms | 2429.1 ms |
| **Model Reconstructed Replay** | **40.0%** | 40.0% | 25.0% | 25.0% | 0.0% | 75.0% | 910.0 tok | 910.0 tok | 13631.5 ms | 13631.5 ms |
| **Fresh (No History Floor)** | **36.7%** | 36.7% | 58.3% | 16.7% | 0.0% | 41.7% | 123.9 tok | 123.9 tok | 2415.9 ms | 2415.9 ms |

---

## 2. Causal Estimand Contrasts & Statistical Inference

Episode-clustered paired bootstrap 95% confidence intervals (B=2,000), exact two-sided binomial McNemar tests, and episode-level sign-flip permutation tests (Primary Inferential Decision):

| Causal Contrast | Contrast Definition | $\Delta$ Accuracy | 95% Bootstrap CI | Discordance ($b / c$) | Exact McNemar $p$ | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Delta_online-direct`** | Online State vs Raw Transcript | **+13.3%** | [-3.3%, +26.7%] | 15 / 7 | 0.1338 | 0.1924 (`exact_exhaustive`) | **No Resolved Pooled Difference** |
| **`Delta_reconstruction`** | Online State vs Model Recon State | **+40.0%** | [+28.3%, +51.7%] | 28 / 4 | 0.0000 | 0.0010 (`exact_exhaustive`) | **Statistically Significant (Reconstruction Bottleneck)** |
| **`Delta_schedule`** | Online State vs Retrospective State | **-3.3%** | [-8.3%, +0.0%] | 0 / 2 | 0.5000 | 0.5000 (`exact_exhaustive`) | **Null / Architectural Invariant Verified** |
| **`Delta_representation`** | Retrospective State vs Transcript | **+16.7%** | [+3.3%, +30.0%] | 15 / 5 | 0.0414 | 0.0752 (`exact_exhaustive`) | **Transcript-favoring point estimate with conflicting inferential evidence; not resolved by primary permutation test** |

---

## 3. Horizon Breakdown & Scaling Analysis

| Horizon ($T$ ticks) | Incremental State | Replay Det State | Replay Transcript | Replay Model State | Fresh Floor | Incremental Prompt Tok | Transcript Prompt Tok | $\Delta_{\text{online-direct}}$ [95% CI] | Exact McNemar $p$ | Permutation $p$ (Method) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T=10$ ticks** | 80.0% | 85.0% | 70.0% | 35.0% | 30.0% | 474.1 tok | 672.9 tok | +10.0% [-10.0%, +20.0%] | 0.6875 | 0.6250 (`exact_exhaustive`) |
| **$T=25$ ticks** | 75.0% | 80.0% | 65.0% | 30.0% | 45.0% | 477.2 tok | 829.7 tok | +10.0% [-35.0%, +40.0%] | 0.7266 | 0.8750 (`exact_exhaustive`) |
| **$T=50$ ticks** | 85.0% | 85.0% | 65.0% | 55.0% | 35.0% | 472.3 tok | 1149.0 tok | +20.0% [+5.0%, +35.0%] | 0.2891 | 0.2500 (`exact_exhaustive`) |

---

## 5. Key Scientific Conclusions & Gate Assessment

1. **Deterministic Replay Invariant:** Online incremental state maintenance and retrospective deterministic replay state achieve bit-for-bit identical terminal state hashes and literal evaluation prompt hashes ($\Delta_{\text{state}} \equiv 0$, $\Delta_{\text{prompt}} \equiv 0$). Residual trial-level discordance reflects backend sampling stochasticity rather than a cognitive scheduling effect.
2. **The Model Retrospective Reconstruction Bottleneck:** When a compact structured state is required, incremental maintenance avoids the reconstruction loss observed with this single-pass Qwen2.5-3B retrospective procedure.
3. **Horizon Scaling & Token Bounding:** Incremental structured state querying maintains a bounded prompt size ($O(K)$) across horizons, reducing prompt token costs relative to uncompressed history ($O(T)$).
4. **Horizon 1 Program Progression:** These findings confirm that explicit Level-1 persistence is transcript-reconstructible under deterministic transition rules. The research program continues inside Horizon 1 with Sprint S07 (Quiet Interval & Null-Tick Screen), S08 (State Swap/Reset), and S09 (Metacognitive Readout).