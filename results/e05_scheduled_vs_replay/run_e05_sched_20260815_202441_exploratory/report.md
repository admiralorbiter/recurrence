# Experiment E05: Scheduled versus Replay Benchmark Report (Sprint S06)

**Run ID:** `run_e05_sched_20260815_202441_exploratory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-15T20:40:21.697337+00:00  
**Scope:** 12 Episodes across Horizons [10, 25, 50] | 300 Total Paired Trials  
**Primary Endpoint:** Causal Estimands ($\Delta_{\text{online-direct}}$, $\Delta_{\text{reconstruction}}$, $\Delta_{\text{schedule}}$, $\Delta_{\text{representation}}$)  

---

## 1. Executive Summary & Core Results

Experiment E05 evaluates whether an autonomous agent maintaining an explicit Level-1 state incrementally across discrete arrival ticks achieves superior accuracy, lower retrieval error, or computational efficiency compared to matched retrospective replay of uncompressed event history.

### Multi-Condition Performance & Cost Summary Table

| Condition | Micro Accuracy | Macro Accuracy | Delayed KV | Source Attr | Goal State | Goal Action | Multi-Hop | Mean Prompt Tok | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scheduled Incremental State** | **81.7%** | 81.7% | 100.0% | 66.7% | 100.0% | 58.3% | 83.3% | 474.1 tok | 2479.3 ms |
| **Deterministic Replay State** | **81.7%** | 81.7% | 100.0% | 66.7% | 100.0% | 58.3% | 83.3% | 474.1 tok | 2498.7 ms |
| **Replay Transcript (Raw)** | **78.3%** | 78.3% | 91.7% | 83.3% | 83.3% | 50.0% | 83.3% | 890.0 tok | 2470.0 ms |
| **Model Reconstructed Replay** | **40.0%** | 40.0% | 58.3% | 58.3% | 0.0% | 58.3% | 25.0% | 1022.5 tok | 18475.5 ms |
| **Fresh (No History Floor)** | **45.0%** | 45.0% | 58.3% | 33.3% | 0.0% | 83.3% | 50.0% | 123.6 tok | 2451.1 ms |

---

## 2. Causal Estimand Contrasts & Statistical Testing

Episode-clustered paired bootstrap 95% confidence intervals (B=2,000) and paired McNemar exact tests:

| Causal Contrast | Contrast Definition | $\Delta$ Accuracy | 95% Bootstrap CI | McNemar Stat | p-value | Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`Delta_online-direct`** | Online State vs Raw Transcript | **+3.3%** | [-11.7%, +16.7%] | 0.06 | 0.8137 | **Null / Indistinguishable** |
| **`Delta_reconstruction`** | Online State vs Model Recon State | **+41.7%** | [+28.3%, +55.0%] | 14.77 | 0.0001 | **Statistically Significant** |
| **`Delta_schedule`** | Online State vs Retrospective State | **+0.0%** | [+0.0%, +0.0%] | 0.00 | 1.0000 | **Null / Indistinguishable** |
| **`Delta_representation`** | Retrospective State vs Transcript | **+3.3%** | [-11.7%, +16.7%] | 0.06 | 0.8137 | **Null / Indistinguishable** |

---

## 3. Horizon Breakdown & Scaling Analysis

| Horizon ($T$ ticks) | Incremental State | Replay Det State | Replay Transcript | Replay Model State | Fresh Floor |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$T=10$ ticks** | 85.0% | 85.0% | 80.0% | 50.0% | 55.0% |
| **$T=25$ ticks** | 65.0% | 65.0% | 75.0% | 40.0% | 35.0% |
| **$T=50$ ticks** | 95.0% | 95.0% | 80.0% | 30.0% | 45.0% |

---

## 4. Key Scientific Conclusions & Gate Assessment

1. **Pure Scheduling Invariant ($\Delta_{\text{schedule}} = +0.0%$):** Confirms that when deterministic Level-1 state transitions are replayed retrospectively, terminal state is bit-for-bit identical to online state maintenance.
2. **Representation vs Scheduling Effect:** Isolates whether observed performance advantages stem from structured state compactness (representation) or incremental temporal execution (scheduling).
3. **Retrospective Reconstruction Loss ($\Delta_{\text{reconstruction}}$):** Quantifies the degradation introduced when an unassisted LLM compresses multi-tick raw history into structured memory in a single retrospective pass.
4. **Transition to Horizon 2 (Latent Recurrence):** These results establish the empirical boundary of scaffolded persistence, directly motivating the transition to true continuous latent state recurrence in Horizon 2.