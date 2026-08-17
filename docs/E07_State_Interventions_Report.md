# Experiment E07: State $\times$ Memory Conflict & Causal State Intervention Report
## Sprint S08 — Horizon 1: Scaffolded Persistence

**Protocol Freeze Anchor:** Commit [`a0159b6`](https://github.com/admiralorbiter/recurrence/commit/a0159b6)  
**Canonical Confirmatory Run:** `run_e07_interv_20260817_144606_confirmatory` (Seed 1337, $N=16$ matched twin pairs, 800 paired trials)  
**Canonical Exploratory Run:** `run_e07_interv_20260817_143732_exploratory` (Seed 42, $N=4$ matched twin pairs, 200 paired trials)  
**Primary Research Question:**  
> *"Holding the model's explicit memory/transcript and final question constant, does changing only the explicit `StructuredSelfState` causally redirect downstream behavior?"*

---

## 1. Executive Summary & Core Scientific Findings

Across S04–S07, we demonstrated that deterministic `StructuredSelfState` is an explicit, inspectable control surface. Experiment E07 directly tested whether explicit state acts as an **authoritative causal control surface** or whether the model remains governed by historical episodic memory.

Using structurally matched twin episodes ($A$ and $B$) with balanced candidate vocabularies (both $V_{\text{red}}$ and $V_{\text{blue}}$ appear in both histories), we evaluated the model across a full State $\times$ Memory intervention matrix.

### Master Confirmatory Estimands ($N=16$ Twin Pairs, 800 Trials, Seed 1337)

| Causal Estimand | Description | Point Estimate | 95% Bootstrap CI | Permutation $p$ (Method) | Primary Scientific Inference |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`Delta_allegiance`** | Primary Conflict Contrast ($SAR - MAR$) | **-32.0%** | [-43.0%, -21.1%] | $p = 0.0002$ (`exact_exhaustive`) | **Episodic Memory Dominates Conflict ($p < .001$)** |
| **`Delta_state_given_memory`** | Effect of swapping State ($S_A \to S_B$) holding Memory ($M_A$) fixed | **+3.1%** | [+0.0%, +9.4%] | $p = 1.0000$ (`exact_exhaustive`) | **Transcript-Equivalent Null (State swap inert)** |
| **`Delta_memory_given_state`** | Effect of swapping Memory ($M_A \to M_B$) holding State ($S_A$) fixed | **+90.6%** | [+81.2%, +100.0%] | $p = 0.0000$ (`exact_exhaustive`) | **Memory Governs Downstream Choice ($p < .001$)** |
| **`Reset_Dependence`** | Drop in target accuracy when state is reset to empty ($M_A + S_0$) | **-3.1%** | [-9.4%, +0.0%] | $p = 1.0000$ (`exact_exhaustive`) | **Direct Memory Fully Compensates** |

---

## 2. State $\times$ Memory Conflict 3-Way Partition & Directional Breakdown

Under direct State–Memory conflict ($N=128$ conflict trials across 16 twin pairs):
- **Follows Memory Value Rate ($MAR$):** **64.1%** (82 / 128 trials)
- **Follows State Value Rate ($SAR$):** **32.0%** (41 / 128 trials)
- **Chooses Neither / Foil Rate:** **3.9%** (5 / 128 trials)
- **Conditional State Preference ($P(\text{State} \mid \text{State or Memory})$):** **33.3%** (41 / 123 resolved trials)
- **Primary Conflict Contrast ($\Delta_{\text{allegiance}} = SAR - MAR$):** **-32.0%** ($p = 0.0002$)

### Directional Breakdown
- **Direction 1 ($M_A + S_B$):** State Allegiance = **50.0%** | Memory Allegiance = **46.9%** (balanced / mixed)
- **Direction 2 ($M_B + S_A$):** State Allegiance = **14.1%** | Memory Allegiance = **81.2%** (strong memory allegiance)

---

## 3. Multi-Condition Intervention Matrix Breakdown

| Condition | Presentation Order | Trials | State Allegiance | Memory Allegiance | Target Acc (Congruent) | Control Acc | Goal Acc | Prompt Tokens | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Congruent Baseline A ($M_A + S_A$)** | `memory_first` | 48 | **56.2%** | **56.2%** | 93.8% | 100.0% | 75.0% | 574.9 tok | 2,537.0 ms |
| **Congruent Baseline A ($M_A + S_A$)** | `state_first` | 48 | **54.2%** | **54.2%** | 100.0% | 100.0% | 62.5% | 574.9 tok | 2,521.9 ms |
| **Congruent Baseline B ($M_B + S_B$)** | `memory_first` | 48 | **64.6%** | **64.6%** | 93.8% | 93.8% | 100.0% | 575.9 tok | 2,514.9 ms |
| **Congruent Baseline B ($M_B + S_B$)** | `state_first` | 48 | **66.7%** | **66.7%** | 100.0% | 100.0% | 100.0% | 575.9 tok | 2,512.0 ms |
| **State/Memory Conflict ($M_A + S_B$)** | `memory_first` | 48 | **35.4%** | **29.2%** | — | 93.8% | 100.0% | 575.9 tok | 2,515.4 ms |
| **State/Memory Conflict ($M_A + S_B$)** | `state_first` | 48 | **33.3%** | **33.3%** | — | 93.8% | 93.8% | 575.9 tok | 2,496.3 ms |
| **State/Memory Conflict ($M_B + S_A$)** | `memory_first` | 48 | **14.6%** | **50.0%** | — | 100.0% | 31.2% | 574.9 tok | 2,502.6 ms |
| **State/Memory Conflict ($M_B + S_A$)** | `state_first` | 48 | **4.2%** | **58.3%** | — | 100.0% | 6.2% | 574.9 tok | 2,499.9 ms |
| **Reset with Memory Preserved ($M_A + S_0$)** | `memory_first` | 48 | **0.0%** | **45.8%** | — | 93.8% | 0.0% | 381.3 tok | 2,487.4 ms |
| **Surgical Slot Inversion ($M_A + S_A'$)** | `memory_first` | 48 | **10.4%** | **52.1%** | — | 93.8% | 12.5% | 574.9 tok | 2,509.2 ms |
| **State-Only Calibration ($S_A$)** | `state_only` | 48 | **54.2%** | **0.0%** | — | 81.2% | 81.2% | 360.3 tok | 2,474.7 ms |
| **State-Only Calibration ($S_B$)** | `state_only` | 48 | **58.3%** | **0.0%** | — | 87.5% | 100.0% | 361.3 tok | 2,478.5 ms |
| **State-Only Calibration ($S_0$)** | `state_only` | 48 | **0.0%** | **0.0%** | — | 18.8% | 0.0% | 166.7 tok | 2,496.9 ms |
| **Memory-Only Calibration ($M_A$)** | `memory_only` | 48 | **0.0%** | **56.2%** | — | 87.5% | 0.0% | 331.3 tok | 2,442.3 ms |
| **Memory-Only Calibration ($M_B$)** | `memory_only` | 48 | **0.0%** | **62.5%** | — | 87.5% | 0.0% | 331.3 tok | 2,472.0 ms |
| **Clone Fork A (Congruent)** | `memory_first` | 16 | **100.0%** | **100.0%** | 100.0% | 0.0% | 0.0% | 383.8 tok | 2,548.7 ms |
| **Clone Fork A (Cross-Swap $S_B$)** | `memory_first` | 16 | **75.0%** | **25.0%** | — | 0.0% | 0.0% | 383.2 tok | 2,466.6 ms |
| **Clone Fork B (Congruent)** | `memory_first` | 16 | **100.0%** | **100.0%** | 100.0% | 0.0% | 0.0% | 382.1 tok | 2,511.2 ms |
| **Reconverged Branch A ($E_{\text{sync}}$)** | `memory_first` | 16 | **100.0%** | **100.0%** | — | 0.0% | 0.0% | 435.6 tok | 2,534.2 ms |
| **Reconverged Branch B ($E_{\text{sync}}$)** | `memory_first` | 16 | **93.8%** | **93.8%** | — | 0.0% | 0.0% | 434.5 tok | 2,520.8 ms |

---

## 4. Surgical Inversion & Presentation Order Effects

- **Target Slot Intervention Uptake:** **12.5%** (2 / 16 twin pairs followed surgical state modification).
- **Control Slot Preservation:** **93.8%** (15 / 16 twin pairs preserved control slot).
- **Joint Local Causal Precision:** **12.5%** (both target uptake AND control preservation).
- **Order Sensitivity Gap:** **-6.2%** (State Allegiance was 25.0% under Memory-first order vs 18.8% under State-first order).
- **Reconvergence Behavioral Concordance:** **93.8%** agreement across independent branches post-synchronization.

---

## 5. Scientific Gate & Synthesis for Level 1 Recurrence

1. **Explicit State is Not an Authoritative Internal Controller:**
   When explicit state contradicts historical episodic memory, Qwen2.5-3B systematically defaults to the episodic transcript ($\Delta_{\text{allegiance}} = -32.0\%, p = 0.0002$).
2. **Transcript Equivalence Confirmed:**
   Holding state fixed and swapping memory shifts behavior by $+90.6\%$ ($p < .001$), whereas holding memory fixed and swapping state shifts behavior by only $+3.1\%$ ($p = 1.0$). Wiping state with memory intact causes no drop in accuracy ($\text{Reset Dependence} = -3.1\%$).
3. **Horizon 1 Takeaway:**
   `StructuredSelfState` operates as an inspectable **external scaffolding and recording format**, but does not act as an authoritative causal mediator over in-context memory. This establishes the clean transcript-equivalent baseline and motivates testing genuine latent recurrence in Horizon 2.
