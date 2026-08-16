# Experiment E05d: Scheduled versus Replay Benchmark Report (Sprint S06.3 Final)

**Run ID:** `run_e05_sched_20260815_211846_confirmatory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `CONFIRMATORY` (Seed: `1337`)  
**Date:** 2026-08-15T22:03:58.968114+00:00  
**Scope:** 24 Episodes across Horizons [10, 25, 50] | 480 Total Paired Trials  
**Primary Endpoint:** Causal Estimands ($\Delta_{\text{online-direct}}$, $\Delta_{\text{reconstruction}}$, $\Delta_{\text{schedule}}$, $\Delta_{\text{representation}}$)  

---

## 1. Executive Summary & Benchmark Results

Experiment E05d evaluates whether an autonomous agent maintaining an explicit Level-1 state incrementally across discrete arrival ticks achieves superior accuracy, lower retrieval error, or computational efficiency compared to matched retrospective replay of uncompressed event history.

All measurement validity checks passed (in-context foils, explicit pending goal construct, exact prompt invariants, repaired reconstruction interface).

### Multi-Condition Performance & Cost Summary Table (Pooled across Horizons)

| Condition | Micro Accuracy | Macro Accuracy | Delayed KV (4AFC) | Source Attr (3AFC) | Goal State (4AFC) | Multi-Hop (4AFC) | Query Prompt Tok | Amortized Prompt Tok | Query Latency | Amortized Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scheduled Incremental State** | **69.8%** | 69.8% | 91.7% | 45.8% | 66.7% | 75.0% | 407.0 tok | 407.0 tok | 5732.1 ms | 5732.1 ms |
| **Deterministic Replay State** | **68.8%** | 68.8% | 87.5% | 45.8% | 66.7% | 75.0% | 407.0 tok | 407.0 tok | 5076.3 ms | 5076.3 ms |
| **Replay Transcript (Raw)** | **70.8%** | 70.8% | 83.3% | 54.2% | 66.7% | 79.2% | 801.7 tok | 801.7 tok | 5066.0 ms | 5066.0 ms |
| **Model Reconstructed Replay** | **36.5%** | 36.5% | 50.0% | 33.3% | 16.7% | 45.8% | 147.5 tok | 341.8 tok | 5111.8 ms | 7061.6 ms |
| **Fresh (No History Floor)** | **28.1%** | 28.1% | 33.3% | 45.8% | 25.0% | 8.3% | 114.1 tok | 114.1 tok | 5286.0 ms | 5286.0 ms |

---

## 2. Causal Estimand Contrasts & Statistical Inference

Episode-clustered paired bootstrap 95% confidence intervals (B=2,000), exact two-sided binomial McNemar tests, and episode-level sign-flip permutation tests (Primary Inferential Decision):

| Causal Contrast | Contrast Definition | $\Delta$ Accuracy | 95% Bootstrap CI | Discordance ($b / c$) | Exact McNemar $p$ | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Delta_online-direct`** | Online State vs Raw Transcript | **-1.0%** | [-13.5%, +11.5%] | 17 / 18 | 1.0000 | 1.0000 (`monte_carlo_50k`) | **No Resolved Pooled Difference** |
| **`Delta_reconstruction`** | Online State vs Model Recon State | **+33.3%** | [+18.8%, +46.9%] | 40 / 8 | 0.0000 | 0.0001 (`monte_carlo_50k`) | **Statistically Significant (Reconstruction Bottleneck)** |
| **`Delta_schedule`** | Online State vs Retrospective State | **+1.0%** | [+0.0%, +3.1%] | 1 / 0 | 1.0000 | 1.0000 (`monte_carlo_50k`) | **Null / Architectural Invariant Verified** |
| **`Delta_representation`** | Retrospective State vs Transcript | **-2.1%** | [-14.6%, +10.4%] | 17 / 19 | 0.8679 | 0.8735 (`monte_carlo_50k`) | **Transcript-favoring point estimate with conflicting inferential evidence; not resolved by primary permutation test** |

---

## 3. Horizon Breakdown & Scaling Analysis

| Horizon ($T$ ticks) | Incremental State | Replay Det State | Replay Transcript | Replay Model State | Fresh Floor | Incremental Prompt Tok | Transcript Prompt Tok | $\Delta_{\text{online-direct}}$ [95% CI] | Exact McNemar $p$ | Permutation $p$ (Method) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T=10$ ticks** | 65.6% | 65.6% | 75.0% | 37.5% | 31.2% | 404.6 tok | 570.4 tok | -9.4% [-25.0%, +6.2%] | 0.5078 | 0.4531 (`exact_exhaustive`) |
| **$T=25$ ticks** | 65.6% | 65.6% | 78.1% | 34.4% | 28.1% | 406.0 tok | 776.9 tok | -12.5% [-34.4%, +6.2%] | 0.3877 | 0.5000 (`exact_exhaustive`) |
| **$T=50$ ticks** | 78.1% | 75.0% | 59.4% | 37.5% | 25.0% | 410.5 tok | 1057.8 tok | +18.8% [+0.0%, +37.5%] | 0.1796 | 0.1875 (`exact_exhaustive`) |

---

## 4. Object-Level Model Reconstruction Fidelity (Oracle Benchmark)

- **Working Memory Key-Value Retention Rate:** 0.0%
- **Goal Status Match Rate:** 0.0%
- **Source Ledger Attribution Accuracy:** 0.7%

---

## 5. Key Scientific Conclusions & Gate Assessment

1. **Deterministic Replay Invariant:** Online incremental state maintenance and retrospective deterministic replay state achieve bit-for-bit identical terminal state hashes and literal evaluation prompt hashes ($\Delta_{\text{state}} \equiv 0$, $\Delta_{\text{prompt}} \equiv 0$). Residual trial-level discordance reflects backend sampling stochasticity rather than a cognitive scheduling effect.
2. **The Model Retrospective Reconstruction Bottleneck:** When a compact structured state is required, incremental maintenance avoids the reconstruction loss observed with this single-pass Qwen2.5-3B retrospective procedure.
3. **Horizon Scaling & Token Bounding:** Incremental structured state querying maintains a bounded prompt size ($O(K)$) across horizons, reducing prompt token costs relative to uncompressed history ($O(T)$).
4. **Horizon 1 Program Progression:** These findings confirm that explicit Level-1 persistence is transcript-reconstructible under deterministic transition rules. The research program continues inside Horizon 1 with Sprint S07 (Quiet Interval & Null-Tick Screen), S08 (State Swap/Reset), and S09 (Metacognitive Readout).