# Experiment E05d: Scheduled versus Replay Benchmark Report (Sprint S06.3 Final)

**Sprint:** S06.3 (Reconstruction Interface Repair & Final S06 Closeout)  
**Experiment ID:** `E05d`  
**Target Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Design:** Pre-specified 5-Condition Paired Factorial with In-Context Foils, Explicit Pending Goals, and Validated Model Reconstruction  
**Scope:** 36 Total Episodes (12 Exploratory Seed 42 + 24 Confirmatory Seed 1337) across Horizons $T \in \{10, 25, 50\}$ | 720 Total Paired Trials  
**Primary Question:** *"Does processing the same information incrementally through time confer an advantage over processing it retrospectively?"*

---

## 1. Executive Summary & Core Results

Experiment E05d evaluates whether an autonomous agent maintaining an explicit Level-1 state incrementally across discrete arrival ticks achieves superior accuracy, lower retrieval error, or computational efficiency compared to matched retrospective replay of uncompressed event history.

Sprint S06.3 completed the final interface repair:
- **Repaired Reconstruction Interface:** Aligned `STATE_RECONSTRUCTION_SCHEMA` with `ReconstructedSelfState`, ensuring that single-pass LLM state extraction validates into structured state objects without timestamp confounds or silent empty fallbacks. A runtime assertion (`reconstruction_valid`) prevents unhandled invalid state fallbacks in future executions.
- **In-Context Foils:** All candidate distractor foils for Delayed KV and Multi-Hop are selected from other actual entity values in the same episode, isolating associative binding and path traversal from candidate familiarity.
- **Explicit Pending Goals:** Queued secondary objectives are explicitly represented as pending in state and transcripts.
- **Exact State & Prompt Invariants:** Online and deterministic replay conditions share bit-for-bit identical terminal state hashes and literal evaluation prompt hashes ($\Delta_{\text{state}} \equiv 0$, $\Delta_{\text{prompt}} \equiv 0$).

### Master Benchmark Table (Canonical Confirmatory Run: 24 Episodes / 480 Paired Trials)

| Experimental Condition | Evaluated Context at $t=T$ | Micro Accuracy | Macro Accuracy | Delayed KV (4AFC) | Source Attr (3AFC) | Goal State (4AFC) | Multi-Hop (4AFC) | Query Prompt Tok | Amortized Prompt Tok | Query Latency | Amortized Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Scheduled Incremental State`** | $S_T^{\text{online}}$ | **60.4%** | 60.4% | 66.7% | 58.3% | 79.2% | 37.5% | **420.9 tok** | **420.9 tok** | **6,497 ms** | **6,497 ms** |
| **`Deterministic Replay State`** | $S_T^{\text{replay}}$ | **59.4%** | 59.4% | 62.5% | 58.3% | 79.2% | 37.5% | **420.9 tok** | **420.9 tok** | **6,589 ms** | **6,589 ms** |
| **`Replay Transcript (Raw)`** | Raw log $H = [E_0 \dots E_T]$ | **67.7%** | 67.7% | 83.3% | 66.7% | 58.3% | 62.5% | **807.4 tok** | **807.4 tok** | **6,495 ms** | **6,495 ms** |
| **`Model Reconstructed Replay`** | $S_T^{\text{model\_recon}}$ | **39.6%** | 39.6% | 45.8% | 41.7% | 41.7% | 29.2% | **378.7 tok** | **558.4 tok** | **6,621 ms** | **9,197 ms** |
| **`Fresh Floor (No History)`** | None | **27.1%** | 27.1% | 33.3% | 20.8% | 29.2% | 25.0% | **113.8 tok** | **113.8 tok** | **6,353 ms** | **6,353 ms** |

---

## 2. Pre-Specified Causal Estimands & Statistical Inference

Contrasts evaluated using **Episode-Clustered Paired Bootstrap (95% CI, B=2,000)**, **Exact Two-Sided Binomial McNemar Tests** (supplementary trial-level inference), and **Episode-Level Sign-Flip Permutation Tests** (primary inferential decision criterion):

| Causal Contrast | Estimand Definition | Exploratory $\Delta$ (Seed 42) | Confirmatory $\Delta$ (Seed 1337) | Confirmatory 95% Bootstrap CI | Discordance ($b / c$) | Exact McNemar $p$ | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Delta_schedule`** | $\text{Acc}(\text{incremental}) - \text{Acc}(\text{replay\_det})$ | **+2.1%** | **+1.0%** | `[+0.0%, +3.1%]` | $1 / 0$ | $p = 1.0000$ | $p = 1.0000$ (`monte_carlo_50k`) | **Null / Architectural Invariant Verified** ($\Delta_{\text{state}} \equiv 0$, $\Delta_{\text{prompt}} \equiv 0$) |
| **`Delta_reconstruction`** | $\text{Acc}(\text{incremental}) - \text{Acc}(\text{replay\_model})$ | **+27.1%** | **+20.8%** | `[+9.4%, +32.3%]` | $34 / 14$ | $p = 0.0055$ | $p = 0.0025$ (`monte_carlo_50k`) | **Statistically Significant (Reconstruction Bottleneck)** |
| **`Delta_online-direct`** | $\text{Acc}(\text{incremental}) - \text{Acc}(\text{replay\_transcript})$ | **-12.5%** | **-7.3%** | `[-15.6%, +0.0%]` | $17 / 24$ | $p = 0.3489$ | $p = 0.1469$ (`monte_carlo_50k`) | **No Resolved Pooled Difference** |
| **`Delta_representation`** | $\text{Acc}(\text{replay\_det}) - \text{Acc}(\text{replay\_transcript})$ | **-14.6%** | **-8.3%** | `[-16.7%, -1.0%]` | $17 / 25$ | $p = 0.2800$ | $p = 0.0965$ (`monte_carlo_50k`) | **Transcript-favoring point estimate with conflicting inferential evidence; not resolved by primary permutation test** |

---

## 3. Horizon Breakdown & Token Scaling

| Horizon ($T$ ticks) | Incremental State | Replay Det State | Replay Transcript | Replay Model State | Fresh Floor | Incremental Prompt Tok | Transcript Prompt Tok | $\Delta_{\text{online-direct}}$ [95% CI] | Exact McNemar $p$ | Permutation $p$ (Method) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T=10$ ticks** | 65.6% | 62.5% | 71.9% | 40.6% | 28.1% | **418.4 tok** | 576.2 tok | -6.2% [-21.9%, +6.2%] | $p = 0.7539$ | $p = 0.7500$ (`exact_exhaustive`) |
| **$T=25$ ticks** | 56.2% | 56.2% | 71.9% | 40.6% | 28.1% | **419.5 tok** | 782.4 tok | -15.6% [-31.2%, -3.1%] | $p = 0.3018$ | $p = 0.2500$ (`exact_exhaustive`) |
| **$T=50$ ticks** | **59.4%** | **59.4%** | **59.4%** | 37.5% | 25.0% | **424.7 tok** | **1063.6 tok** | **+0.0% [-9.4%, +9.4%]** | **$p = 1.0000$** | **$p = 1.0000$ (`exact_exhaustive`)** |

*Note on $T=25$:* While the bootstrap CI for $\Delta_{\text{online-direct}}$ is $[-31.2\%, -3.1\%]$, neither the exact McNemar test ($p = 0.3018$) nor the exact cluster-level permutation test ($p = 0.2500$) resolves this point difference as statistically significant across the 8 episodes. The effect is unresolved.

---

## 4. Final Scientific Synthesis for Sprint S06

1. **Deterministic Level-1 Persistence is Algorithmically Reconstructible ($\Delta_{\text{schedule}} \equiv 0$):**  
   Under deterministic transitions over ordered events, online incremental execution and retrospective replay produce bit-for-bit identical terminal state hashes and literal evaluation prompt hashes ($\text{hash}(S_T^{\text{online}}) \equiv \text{hash}(S_T^{\text{replay}})$). The minute +1.0pp trial discrepancy reflects backend sampling stochasticity.
2. **The Model Retrospective Reconstruction Bottleneck ($\Delta_{\text{reconstruction}} = +20.8\%$):**  
   Under this protocol, Qwen2.5-3B is substantially worse at reconstructing the required compact multi-slot state retrospectively in one pass ($39.6\%$) than the deterministic scaffold is at maintaining that state incrementally ($60.4\%$, $p = 0.0025$). When a compact structured state is required, incremental maintenance avoids this single-pass reconstruction loss.
3. **No Resolved Accuracy Advantage Over Direct History; Clear Systems Advantage in Prompt Bounding:**  
   At the tested horizons, deterministic online explicit-state maintenance does not provide a resolved overall accuracy advantage over direct access to the raw history ($60.4\%$ vs $67.7\%$, pooled permutation $p = 0.1469$). Its clear systems advantage is bounded representation size ($418 - 425$ tokens, $O(K)$) relative to linear transcript growth ($1,063.6$ tokens at $T=50$, $O(T)$), saving $60.1\%$ of prompt compute while reaching identical accuracy ($59.4\%$ vs $59.4\%$) at $T=50$.
4. **Horizon 1 Program Progression:**  
   Sprint S06 is fully closed. The research program proceeds within Horizon 1 to:
   - **Sprint S07:** Quiet Interval and Null-Tick Screen (testing whether intervening silent periods alter state or policy dynamics).
   - **Sprint S08:** State Swap/Reset/Clone Invariance.
   - **Sprint S09:** Metacognitive Readout & Ownership Attribution.

---

## 5. Artifact Provenance

- **Protocol Freeze Commit:** [`db7273c`](https://github.com/admiralorbiter/recurrence/commit/db7273c)
- **Canonical Results Commit:** [`e75a963`](https://github.com/admiralorbiter/recurrence/commit/e75a963)
- **Note on Commit Sequence:** Intervening commits between `db7273c` and `e75a963` were unrelated H0 psychophysics work and an orthogonal backend helper that did not alter the E05d execution path.
- **Canonical Results Directory:** [`results/e05_scheduled_vs_replay/run_e05_sched_20260816_142239_confirmatory/`](file:///c:/Users/admir/Github/recurrence/results/e05_scheduled_vs_replay/run_e05_sched_20260816_142239_confirmatory/)
- **Test Suite:** [`tests/test_s06_scheduled_replay.py`](file:///c:/Users/admir/Github/recurrence/tests/test_s06_scheduled_replay.py) (**92/92 tests passing in 23.3s**).
