# Sprint S09 Walkthrough: Source Attribution, Self/Other Ownership & Metacognitive Continuity (Experiments E08 & E09)

## 1. Executive Summary

Sprint S09 completed the final closing battery of **Horizon 1 (Level 1 Recurrence: Scaffolded Persistence)** on live `qwen2.5:3b`:
- **Experiment E08 (S09a):** Source Attribution, Self/Other Memory Ownership, and Agency Boundaries ($N=16$ Multi-Source Episodes, 320 Intervention Trials, Seed 1337).
- **Experiment E09 (S09b):** Metacognitive Continuity & Item-Paired Post-Choice Error Prediction Screen ($N=16$ Multi-Source Episodes, 320 Metacognitive Probes, Seed 1337).

All 119 repository tests are passing cleanly, and all pre-registered estimands have been evaluated with exact permutation tests and cluster-bootstrap 95% confidence intervals.

---

## 2. Confirmatory Benchmark Results

### A. Experiment E08: Source Attribution Breakdown & $5 \times 5$ Confusion Matrix

| Estimand / Contrast | Point Estimate | 95% Clustered CI | Permutation $p$ (Method) | Primary Scientific Inference |
| :--- | :---: | :---: | :---: | :--- |
| **`Overall_SAA_5AFC`** | **31.2%** | [22.5%, 40.0%] | **$p = 0.0059$** (`within_episode_source_shuffle_50000_mc`) | Above 20% Chance Baseline |
| **`Self_SAA_5AFC`** | **81.2%** | [62.5%, 100.0%] | N/A (`cluster_bootstrap_ci_only`) | High Apparent Self Recognition |
| **`Environment_SAA_5AFC`** | **6.2%** | [0.0%, 18.8%] | N/A (`cluster_bootstrap_ci_only`) | Unresolved Sensory Provenance |
| **`Experimenter_SAA_5AFC`** | **31.2%** | [12.5%, 56.2%] | N/A (`cluster_bootstrap_ci_only`) | Moderate Controller Resolution |
| **`Peer_Agent_SAA_5AFC`** | **31.2%** | [12.5%, 56.2%] | N/A (`cluster_bootstrap_ci_only`) | Moderate Peer Resolution |
| **`Observer_SAA_5AFC`** | **6.2%** | [0.0%, 18.8%] | N/A (`cluster_bootstrap_ci_only`) | Unresolved Observer Resolution |
| **`Self_Other_Confusion_Rate`** | **50.0%** | [25.0%, 75.0%] | N/A (`cluster_bootstrap_ci_only`) | **50.0% Peer->Self Bleed (Egocentric Bias)** |

#### $5 \times 5$ Empirical Attribution Confusion Matrix (True Source $\rightarrow$ Attributed Actor)
| True Source Class | agent_alpha (Self) | telemetry_sensor (Env) | human_controller (Exp) | agent_beta (Peer) | auditor_gamma (Obs) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`self`** | **81.2%** | 6.2% | 0.0% | 12.5% | 0.0% |
| **`environment`** | **37.5%** | 6.2% | 12.5% | 31.2% | 12.5% |
| **`experimenter`** | **56.2%** | 0.0% | 31.2% | 6.2% | 6.2% |
| **`peer_agent`** | **50.0%** | 6.2% | 12.5% | 31.2% | 0.0% |
| **`observer`** | **56.2%** | 6.2% | 18.8% | 12.5% | 6.2% |

Across all 4 non-self categories, `agent_alpha` is the modal attributed actor (37.5% to 56.2%), indicating an **egocentric response attractor** (accounting for 56.2% of all responses and 50.0% of non-self trials).

---

### B. Cue-Conflict & Channel Factorials

- **Cue-Conflict Contrast ($2 \times 2$ Tag $\times$ Narrative):** **-34.4%** (95% CI: [-59.4%, -12.5%], $p = 0.0312$).
  - **Narrative Leverage (62.5%)** is more than double **Tag Leverage (28.1%)**. Narrative actor mentions in text dominate explicit metadata tags.
- **Channel Factorial ($2 \times 2$ Tags $\times$ Ledger Across Balanced Sources):**
  - Tags Present + Ledger Present: **50.0%**
  - Tags Present + Ledger Stripped: **31.2%**
  - Tags Stripped + Ledger Present: **25.0%**
  - Tags Stripped + Ledger Stripped: **12.5%**
  - Transcript Tag Marginal Effect: **+21.9%** (95% CI: [+3.1%, +37.6%], $p = 0.0625$).
  - Source Ledger Marginal Effect: **+15.6%** (95% CI: [+0.0%, +31.2%], $p = 0.1250$).
- **Framing & Challenge Reprobe:**
  - Framing Accuracy Gap (*"You"* vs *"agent_alpha"*): **+6.2%** ($p = 1.0000$).
  - Framing Response Disagreement Rate: **18.8%** (95% CI: [0.0%, 37.5%], $p = 0.2500$).
  - Unconditional Shift Toward Self ($\Delta_{\text{challenge-self}}$): **+0.0%** ($p = 1.0000$).

---

### C. Experiment E09: Item-Paired Metacognitive Continuity Screen

| Evaluator | Memory Format | Trials | Accuracy | Mean Confidence | Brier Score | AUROC Error Prediction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Agent (Self / alpha)** | `Transcript-Only` | 80 | 37.5% | 59.2% | **0.3674** | **0.641** |
| **Primary Agent (Self / alpha)** | `Scaffolded Persistence` | 80 | 32.5% | 53.1% | **0.5440** | **0.440** |
| **Auditing Observer (gamma)** | `Transcript-Only` | 80 | 37.5% | 67.4% | **0.4643** | **0.560** |
| **Auditing Observer (gamma)** | `Scaffolded Persistence` | 80 | 32.5% | 61.2% | **0.4507** | **0.594** |

| Item-Paired Estimand | Point Estimate | 95% Clustered CI | Permutation $p$ (Method) | Primary Scientific Inference |
| :--- | :---: | :---: | :---: | :--- |
| **`Delta_AUROC_Transcript`** | **+0.081** | [-0.106, +0.250] | $p = 0.3778$ (`exact_confidence_swap_65k`) | **Null / Invariant** |
| **`Delta_AUROC_Scaffolded`** | **-0.154** | [-0.308, -0.029] | $p = 0.0615$ (`exact_confidence_swap_65k`) | **Null / Invariant** |
| **`Delta_Brier_Transcript`** | **+0.0969** | [-0.0710, +0.2517] | $p = 0.2658$ (`exact_exhaustive`) | **Null / Invariant** |
| **`Delta_Brier_Scaffolded`** | **-0.0934** | [-0.2115, +0.0233] | $p = 0.1525$ (`exact_exhaustive`) | **Null / Invariant** |
| **`Scaffolding_Metacognitive_Interaction`** | **-0.235** | [-0.423, -0.052] | **$p = 0.0286$** (`exact_format_block_swap_65k`) | **Format-Dependent Metacognitive Reversal** |

Under matched public information, self-framing provides **no positive privileged metacognitive advantage** over an external observer. A format-dependent reversal occurs between raw transcript and scaffolded contexts.

---

## 3. Horizon 1 Closeout Summary

With the completion of Sprints S04 through S09, the full empirical foundation of Horizon 1 is complete:
- Complete Horizon 1 synthesis document created at [`H1_Level1_Synthesis.md`](file:///c:/Users/admir/Github/recurrence/H1_Level1_Synthesis.md).
- The codebase, tests (119 passing), testbeds, analysis pipelines, and confirmatory results are frozen and synchronized on `main`.
