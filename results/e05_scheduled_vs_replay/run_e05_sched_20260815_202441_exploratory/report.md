# Experiment E05d: Scheduled versus Replay Benchmark Report (Sprint S06.3 Final)

**Run ID:** `run_e05_sched_20260815_202441_exploratory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-15T20:40:21.697337+00:00  
**Scope:** 12 Episodes across Horizons [10, 25, 50] | 300 Total Paired Trials  
**Primary Endpoint:** Causal Estimands ($\Delta_{\text{online-direct}}$, $\Delta_{\text{reconstruction}}$, $\Delta_{\text{schedule}}$, $\Delta_{\text{representation}}$)  

---

## 1. Executive Summary & Benchmark Results

Experiment E05d evaluates whether an autonomous agent maintaining an explicit Level-1 state incrementally across discrete arrival ticks achieves superior accuracy, lower retrieval error, or computational efficiency compared to matched retrospective replay of uncompressed event history.

All measurement validity checks passed (in-context foils, explicit pending goal construct, exact prompt invariants, repaired reconstruction interface).

### Multi-Condition Performance & Cost Summary Table (Pooled across Horizons)

| Condition | Micro Accuracy | Macro Accuracy | Delayed KV (4AFC) | Source Attr (3AFC) | Goal State (4AFC) | Multi-Hop (4AFC) | Query Prompt Tok | Amortized Prompt Tok | Query Latency | Amortized Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scheduled Incremental State** | **81.7%** | 81.7% | 100.0% | 66.7% | 100.0% | 83.3% | 474.1 tok | 474.1 tok | 2479.3 ms | 2479.3 ms |
| **Deterministic Replay State** | **81.7%** | 81.7% | 100.0% | 66.7% | 100.0% | 83.3% | 474.1 tok | 474.1 tok | 2498.7 ms | 2498.7 ms |
| **Replay Transcript (Raw)** | **78.3%** | 78.3% | 91.7% | 83.3% | 83.3% | 83.3% | 890.0 tok | 890.0 tok | 2470.0 ms | 2470.0 ms |
| **Model Reconstructed Replay** | **40.0%** | 40.0% | 58.3% | 58.3% | 0.0% | 25.0% | 1022.5 tok | 1022.5 tok | 18475.5 ms | 18475.5 ms |
| **Fresh (No History Floor)** | **45.0%** | 45.0% | 58.3% | 33.3% | 0.0% | 50.0% | 123.6 tok | 123.6 tok | 2451.1 ms | 2451.1 ms |

---

## 2. Causal Estimand Contrasts & Statistical Inference

Episode-clustered paired bootstrap 95% confidence intervals (B=2,000), exact two-sided binomial McNemar tests, and episode-level sign-flip permutation tests (Primary Inferential Decision):

| Causal Contrast | Contrast Definition | $\Delta$ Accuracy | 95% Bootstrap CI | Discordance ($b / c$) | Exact McNemar $p$ | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Delta_online-direct`** | Online State vs Raw Transcript | **+3.3%** | [-11.7%, +16.7%] | 10 / 8 | 0.8145 | 0.8438 (`exact_exhaustive`) | **No Resolved Pooled Difference** |
| **`Delta_reconstruction`** | Online State vs Model Recon State | **+41.7%** | [+28.3%, +55.0%] | 32 / 7 | 0.0001 | 0.0010 (`exact_exhaustive`) | **Statistically Significant (Reconstruction Bottleneck)** |
| **`Delta_schedule`** | Online State vs Retrospective State | **+0.0%** | [+0.0%, +0.0%] | 0 / 0 | 1.0000 | 1.0000 (`exact_exhaustive`) | **Null / Architectural Invariant Verified** |
| **`Delta_representation`** | Retrospective State vs Transcript | **+3.3%** | [-11.7%, +16.7%] | 10 / 8 | 0.8145 | 0.8438 (`exact_exhaustive`) | **Transcript-favoring point estimate with conflicting inferential evidence; not resolved by primary permutation test** |

---

## 3. Horizon Breakdown & Scaling Analysis

| Horizon ($T$ ticks) | Incremental State | Replay Det State | Replay Transcript | Replay Model State | Fresh Floor | Incremental Prompt Tok | Transcript Prompt Tok | $\Delta_{\text{online-direct}}$ [95% CI] | Exact McNemar $p$ | Permutation $p$ (Method) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T=10$ ticks** | 85.0% | 85.0% | 80.0% | 50.0% | 55.0% | 471.1 tok | 648.3 tok | +5.0% [-15.0%, +30.0%] | 1.0000 | 1.0000 (`exact_exhaustive`) |
| **$T=25$ ticks** | 65.0% | 65.0% | 75.0% | 40.0% | 35.0% | 473.6 tok | 843.3 tok | -10.0% [-45.0%, +15.0%] | 0.7266 | 1.0000 (`exact_exhaustive`) |
| **$T=50$ ticks** | 95.0% | 95.0% | 80.0% | 30.0% | 45.0% | 477.6 tok | 1178.3 tok | +15.0% [+0.0%, +30.0%] | 0.3750 | 0.5000 (`exact_exhaustive`) |

---

## 5. Key Scientific Conclusions & Gate Assessment

1. **Deterministic Replay Invariant:** Online incremental state maintenance and retrospective deterministic replay state achieve bit-for-bit identical terminal state hashes and literal evaluation prompt hashes ($\Delta_{\text{state}} \equiv 0$, $\Delta_{\text{prompt}} \equiv 0$). Residual trial-level discordance reflects backend sampling stochasticity rather than a cognitive scheduling effect.
2. **The Model Retrospective Reconstruction Bottleneck:** When a compact structured state is required, incremental maintenance avoids the reconstruction loss observed with this single-pass Qwen2.5-3B retrospective procedure.
3. **Horizon Scaling & Token Bounding:** Incremental structured state querying maintains a bounded prompt size ($O(K)$) across horizons, reducing prompt token costs relative to uncompressed history ($O(T)$).
4. **Horizon 1 Program Progression:** These findings confirm that explicit Level-1 persistence is transcript-reconstructible under deterministic transition rules. The research program continues inside Horizon 1 with Sprint S07 (Quiet Interval & Null-Tick Screen), S08 (State Swap/Reset), and S09 (Metacognitive Readout).