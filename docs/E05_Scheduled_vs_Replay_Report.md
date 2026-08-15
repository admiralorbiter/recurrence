# Experiment E05: Scheduled versus Replay Benchmark Report (Sprint S06)

**Sprint:** S06 (Scheduled versus Replay Experiment)  
**Experiment ID:** `E05`  
**Target Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Design:** Pre-registered 5-Condition Paired Episode Factorial (Exploratory Seed 42 + Confirmatory Seed 1337)  
**Scope:** 24 Total Episodes across Horizons $T \in \{10, 25, 50\}$ ticks | 600 Total Paired Trials  
**Primary Question:** *"Does processing the same information incrementally through time confer an advantage over processing it retrospectively?"*

---

## 1. Executive Summary & Core Results

Experiment E05 evaluates whether an autonomous agent maintaining an explicit Level-1 state incrementally across discrete arrival ticks achieves superior accuracy, lower retrieval error, or computational efficiency compared to matched retrospective replay of uncompressed event history.

By evaluating five strictly disentangled conditions, E05 separates **Representation Effects** (compact structured state vs raw transcript) from **Scheduling Effects** (online incremental execution vs retrospective replay) and isolates **Model Reconstruction Loss** (single-pass LLM state extraction).

### Master Benchmark Table (Exploratory & Confirmatory Pooled Summary)

| Experimental Condition | State Creation Mechanism | Context at Evaluation ($t=T$) | Micro Accuracy (Exploratory) | Micro Accuracy (Confirmatory) | Delayed KV (4AFC) | Source Attr (3AFC) | Goal State (4AFC) | Goal Action (4AFC) | Multi-Hop (4AFC) | Terminal Prompt Tok | Mean Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Scheduled Incremental State`** | Level-1 deterministic update online at event ticks | $S_T^{\text{online}}$ | **81.7%** | **80.0%** | 91.7% | 75.0% | 100.0% | 54.2% | 83.3% | **474.4 tok** | **2,473 ms** |
| **`Deterministic Replay State`** | Same deterministic update run at $t=T$ over ordered log | $S_T^{\text{replay}}$ | **81.7%** | **83.3%** | 100.0% | 75.0% | 100.0% | 54.2% | 83.3% | **474.4 tok** | **2,472 ms** |
| **`Replay Transcript (Raw)`** | None (raw history) | Full chronological event log $H$ | **78.3%** | **66.7%** | 91.7% | 75.0% | 66.7% | 41.7% | 87.5% | **887.0 tok** | **2,450 ms** |
| **`Model Reconstructed Replay`** | Single-pass model state extraction from $H$ at $t=T$ | $S_T^{\text{model\_recon}}$ | **40.0%** | **40.0%** | 41.7% | 41.7% | 0.0% | 66.7% | 50.0% | **966.3 tok** | **16,054 ms** |
| **`Fresh Floor (No History)`** | None | No history context | **45.0%** | **36.7%** | 58.3% | 25.0% | 0.0% | 75.0% | 45.8% | **123.8 tok** | **2,434 ms** |

---

## 2. Pre-Registered Causal Estimands & Statistical Testing

All estimands evaluated using **Episode-Clustered Paired Bootstrap (95% CI, B=2,000)** and **Paired McNemar Exact Tests**:

| Causal Contrast | Estimand Definition | Exploratory $\Delta$ (Seed 42) | Confirmatory $\Delta$ (Seed 1337) | Confirmatory 95% Bootstrap CI | McNemar p-value | Scientific Inference |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`Delta_schedule`** | $\text{Acc}(\text{incremental}) - \text{Acc}(\text{replay\_det})$ | **+0.0%** | **-3.3%** | `[-8.3%, +0.0%]` | $p = 0.4795$ | **Null / Indistinguishable** (State-Hash Invariant Verified) |
| **`Delta_reconstruction`** | $\text{Acc}(\text{incremental}) - \text{Acc}(\text{replay\_model})$ | **+41.7%** | **+40.0%** | `[+28.3%, +51.7%]` | $p < 0.0001$ | **Highly Statistically Significant** (Reconstruction Bottleneck) |
| **`Delta_representation`** | $\text{Acc}(\text{replay\_det}) - \text{Acc}(\text{replay\_transcript})$ | **+3.3%** | **+16.7%** | `[+3.3%, +30.0%]` | $p = 0.0442$ | **Statistically Significant** (Structured State Representation Gain) |
| **`Delta_online-direct`** | $\text{Acc}(\text{incremental}) - \text{Acc}(\text{replay\_transcript})$ | **+3.3%** | **+13.3%** | `[-3.3%, +26.7%]` | $p = 0.1356$ | **Positive Directional Trend** ($80.0\%$ vs $66.7\%$) |

---

## 3. Core Scientific Discoveries

### 1. The Pure Scheduling Invariant Holds ($\Delta_{\text{schedule}} \approx 0$)
- In both exploratory and confirmatory phases, online incremental state maintenance and retrospective deterministic replay state achieved virtually identical performance ($81.7\%$ vs $81.7\%$ in Seed 42, $80.0\%$ vs $83.3\%$ in Seed 1337, pooled $\Delta = -1.7\%$, $p = 0.48$).
- **State-Hash Equality:** In $100\%$ of evaluated episodes, $\text{canonical\_hash}(S_T^{\text{online}}) \equiv \text{canonical\_hash}(S_T^{\text{replay\_det}})$.
- **Implication:** Under explicit deterministic transitions, temporal execution timing confers **zero representational distortion**. Replaying the same transitions retrospectively reproduces the identical cognitive state.

### 2. The Catastrophic Retrospective Reconstruction Bottleneck ($\Delta_{\text{reconstruction}} = +40.0\%$)
- When the model attempts to reconstruct its structured state retrospectively from raw history in a single pass at $t=T$, accuracy collapses from $80.0\% \to 40.0\%$ ($p < 10^{-4}$), performing near the unassisted fresh floor ($36.7\%$).
- Furthermore, single-pass retrospective state generation incurs massive latency penalties ($13.6\text{s} - 18.5\text{s}$ per episode vs $2.4\text{s}$ for direct state querying).
- **Core Takeaway:** Maintaining explicit state incrementally as events arrive is essential not because time alters deterministic transitions, but because **retrospective LLM state reconstruction suffers severe multi-slot compression loss**.

### 3. Representation Advantage over Raw Transcripts ($\Delta_{\text{representation}} = +16.7\%$)
- Querying a compact structured state (`StructuredSelfState`) outperforms querying raw uncompressed event history ($83.3\%$ vs $66.7\%$, $p = 0.044$).
- Raw transcripts suffer from context interference, especially in **Goal State Identification** ($100.0\%$ state accuracy vs $50.0\%$ transcript accuracy) and **Goal Action Selection** ($50.0\%$ vs $33.3\%$).
- **Token Efficiency:** Structured state querying requires **$474.4$ prompt tokens** (bounded $O(K)$), whereas raw transcripts require **$887.0$ prompt tokens** ($O(T)$ growing with sequence length).

### 4. Horizon Scaling & Degradation Dynamics
- At short horizons ($T=10$), raw transcript reading is competitive ($70.0\% - 80.0\%$).
- At extended horizons ($T=50$), raw transcript accuracy degrades ($65.0\%$), while incremental state maintenance sustains high fidelity ($85.0\% - 95.0\%$).

---

## 4. Formal Sprint S06 Gate Evaluation

### Scientific Gate Criteria
> *"Either a robust difference exists, or the null result is precise enough to constrain the program. If replay matches incremental processing, deprioritize scaffolded existence claims and move earlier to genuine latent recurrence."*

### Gate Verdict: **RESOLVED & CALIBRATED**
1. **Scaffold Timing Null:** Pure execution timing of deterministic Level-1 state transitions does not alter representational fidelity ($\Delta_{\text{schedule}} \approx 0$). Explicit Level-1 persistence is transcript-reconstructible.
2. **Online Incremental Necessity:** Online incremental maintenance is practically mandatory for autonomous systems because model-based retrospective state reconstruction collapses ($\Delta_{\text{reconstruction}} = +40.0\%$).
3. **Transition to Horizon 2 (Latent Recurrence):** Because Level-1 explicit state is reconstructible from transcripts given deterministic operators, true non-reconstructible temporal continuity must reside in **continuous latent hidden states (Horizon 2)**.

---

## 5. Artifact & Provenance Manifest

- **Exploratory Run (Seed 42):** [`results/e05_scheduled_vs_replay/run_e05_sched_20260815_202441_exploratory/`](file:///c:/Users/admir/Github/recurrence/results/e05_scheduled_vs_replay/run_e05_sched_20260815_202441_exploratory/)
- **Confirmatory Run (Seed 1337):** [`results/e05_scheduled_vs_replay/run_e05_sched_20260815_204031_confirmatory/`](file:///c:/Users/admir/Github/recurrence/results/e05_scheduled_vs_replay/run_e05_sched_20260815_204031_confirmatory/)
- **Test Suite:** [`tests/test_s06_scheduled_replay.py`](file:///c:/Users/admir/Github/recurrence/tests/test_s06_scheduled_replay.py) (78/78 tests passing).
