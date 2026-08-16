# Experiment E05c: Scheduled versus Replay Benchmark Report (Sprint S06.2 Final Freeze)

**Run ID:** `run_e05_sched_20260816_040451_confirmatory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `CONFIRMATORY` (Seed: `1337`)  
**Date:** 2026-08-16T04:42:10.785472+00:00  
**Scope:** 24 Episodes across Horizons [10, 25, 50] | 480 Total Paired Trials  
**Primary Endpoint:** Causal Estimands ($\Delta_{\text{online-direct}}$, $\Delta_{\text{reconstruction}}$, $\Delta_{\text{schedule}}$, $\Delta_{\text{representation}}$)  

---

## 1. Executive Summary & Hardened Results

Experiment E05c evaluates whether an autonomous agent maintaining an explicit Level-1 state incrementally across discrete arrival ticks achieves superior accuracy, lower retrieval error, or computational efficiency compared to matched retrospective replay of uncompressed event history.

All probe measurement shortcuts have been completely eradicated:
- **In-Context Foils:** All candidate foils for Delayed KV and Multi-Hop are drawn from other actual values in the same episode (isolating binding and path traversal from candidate familiarity).
- **Explicit Pending Goal:** Pending secondary goals are explicitly queued in state and transcripts.
- **Exact Prompt Equality:** Literal prompt hashes match bit-for-bit between online and deterministic replay.

### Multi-Condition Performance & Cost Summary Table

| Condition | Micro Accuracy | Macro Accuracy | Delayed KV (4AFC) | Source Attr (3AFC) | Goal State (4AFC) | Multi-Hop (4AFC) | Query Prompt Tok | Amortized Prompt Tok | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scheduled Incremental State** | **61.5%** | 61.5% | 79.2% | 58.3% | 75.0% | 33.3% | 420.9 tok | 420.9 tok | 4370.8 ms |
| **Deterministic Replay State** | **61.5%** | 61.5% | 79.2% | 58.3% | 75.0% | 33.3% | 420.9 tok | 420.9 tok | 4297.5 ms |
| **Replay Transcript (Raw)** | **60.4%** | 60.4% | 83.3% | 50.0% | 50.0% | 58.3% | 807.4 tok | 807.4 tok | 4288.8 ms |
| **Model Reconstructed Replay** | **30.2%** | 30.2% | 25.0% | 37.5% | 25.0% | 33.3% | 146.1 tok | 331.0 tok | 4333.9 ms |
| **Fresh (No History Floor)** | **27.1%** | 27.1% | 33.3% | 37.5% | 16.7% | 20.8% | 113.8 tok | 113.8 tok | 4202.2 ms |

---

## 2. Causal Estimand Contrasts & Statistical Inference

Episode-clustered paired bootstrap 95% confidence intervals (B=2,000), exact two-sided binomial McNemar tests, and sign-flip permutation tests:

| Causal Contrast | Contrast Definition | $\Delta$ Accuracy | 95% Bootstrap CI | Discordance ($b / c$) | Exact McNemar $p$ | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Delta_online-direct`** | Online State vs Raw Transcript | **+1.0%** | [-9.4%, +11.5%] | 22 / 21 | 1.0000 | 1.0000 (`monte_carlo_50k`) | **Null / Indistinguishable** |
| **`Delta_reconstruction`** | Online State vs Model Recon State | **+31.3%** | [+19.8%, +43.8%] | 40 / 10 | 0.0000 | 0.0001 (`monte_carlo_50k`) | **Statistically Significant** |
| **`Delta_schedule`** | Online State vs Retrospective State | **+0.0%** | [-3.1%, +3.1%] | 1 / 1 | 1.0000 | 1.0000 (`exact_exhaustive`) | **Null / Indistinguishable** |
| **`Delta_representation`** | Retrospective State vs Transcript | **+1.0%** | [-9.4%, +11.5%] | 22 / 21 | 1.0000 | 1.0000 (`monte_carlo_50k`) | **Null / Indistinguishable** |

---

## 3. Horizon Breakdown & Horizon-Specific Contrasts

| Horizon ($T$ ticks) | Incremental State | Replay Det State | Replay Transcript | Replay Model State | Fresh Floor | $\Delta_{\text{online-direct}}$ [95% CI] | Exact McNemar $p$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T=10$ ticks** | 59.4% | 59.4% | 53.1% | 34.4% | 25.0% | +6.2% [-12.5%, +25.0%] | 0.7744 |
| **$T=25$ ticks** | 59.4% | 59.4% | 68.8% | 31.2% | 31.2% | -9.4% [-28.1%, +9.4%] | 0.5811 |
| **$T=50$ ticks** | 65.6% | 65.6% | 59.4% | 25.0% | 25.0% | +6.2% [-12.5%, +25.0%] | 0.8145 |

---

## 4. Object-Level Model Reconstruction Fidelity (Oracle Benchmark)

- **Working Memory Key-Value Retention Rate:** 0.0%
- **Goal Status Match Rate:** 0.0%
- **Source Ledger Attribution Accuracy:** 0.0%

---

## 5. Key Scientific Conclusions & Gate Assessment

1. **Deterministic Replay Invariant:** Online incremental state maintenance and retrospective deterministic replay state achieve bit-for-bit identical terminal state hashes and literal evaluation prompt hashes ($\Delta_{\text{state}} \equiv 0$, $\Delta_{\text{prompt}} \equiv 0$). Residual trial-level discordance reflects backend sampling stochasticity rather than a cognitive scheduling effect.
2. **The Model Retrospective Reconstruction Bottleneck:** When compact structured state is required, maintaining it incrementally avoids the severe multi-slot information loss produced by this single-pass Qwen2.5-3B retrospective reconstruction procedure.
3. **Horizon Scaling & Token Bounding:** Incremental structured state querying maintains a bounded prompt size ($O(K)$) and preserves memory fidelity over extended horizons.
4. **Horizon 1 Program Progression:** These findings confirm that explicit Level-1 persistence is transcript-reconstructible under deterministic transition rules. The research program continues inside Horizon 1 with Sprint S07 (Quiet Interval & Null-Tick Screen), S08 (State Swap/Reset), and S09 (Metacognitive Readout).