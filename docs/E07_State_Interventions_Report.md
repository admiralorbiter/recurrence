# Experiment E07: State $\times$ Memory Conflict & Causal State Intervention Report
## Sprint S08 — Horizon 1: Scaffolded Persistence

**Protocol Freeze Anchor:** Commit [`a0159b6`](https://github.com/admiralorbiter/recurrence/commit/a0159b6)  
**Analysis & Reporting Hardening:** Commit [`4bb2019`](https://github.com/admiralorbiter/recurrence/commit/4bb2019)  
**Canonical Confirmatory Run:** `run_e07_interv_20260817_144606_confirmatory` (Seed 1337, $N=16$ matched twin pairs, 800 paired trials)  
**Canonical Exploratory Run:** `run_e07_interv_20260817_143732_exploratory` (Seed 42, $N=4$ matched twin pairs, 200 paired trials)  
**Primary Research Question:**  
> *"Holding the model's explicit memory/transcript and final question constant, does changing only the explicit `StructuredSelfState` causally redirect downstream behavior?"*

---

## 1. Executive Summary & Core Scientific Findings

Across S04–S07, we established that deterministic `StructuredSelfState` is an explicit, inspectable control surface. Experiment E07 evaluated whether explicit state operates as an **authoritative causal control surface** or whether the model remains governed by historical episodic memory when the two sources conflict.

Using structurally matched twin episodes ($A$ and $B$) with balanced candidate vocabularies (both $V_{\text{red}}$ and $V_{\text{blue}}$ appear in both histories), we evaluated the model across the full $2 \times 2$ State $\times$ Memory intervention matrix.

### Master Confirmatory Estimands ($N=16$ Twin Pairs, 800 Trials, Seed 1337)

| Causal Estimand | Description | Point Estimate | 95% Bootstrap CI | Permutation $p$ (Method) | Primary Scientific Inference |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`Delta_allegiance`** | Primary Conflict Contrast ($SAR - MAR$) | **-32.0%** | [-43.0%, -21.1%] | **$p = 0.0002$** (`exact_exhaustive`) | **Episodic Memory Favored Under Conflict ($p < .001$)** |
| **`Delta_state_given_memory_A`** | State Swap ($S_A \to S_B$) holding Memory ($M_A$) fixed | **+3.1%** | [+0.0%, +9.4%] | **$p = 1.0000$** (`exact_exhaustive`) | **No Resolved Independent State Leverage** |
| **`Delta_state_given_memory_B`** | State Swap ($S_B \to S_A$) holding Memory ($M_B$) fixed | **+6.2%** | [-6.2%, +18.8%] | **$p = 0.6250$** (`exact_exhaustive`) | **No Resolved Independent State Leverage** |
| **`Delta_memory_given_state_A`** | Memory Swap ($M_A \to M_B$) holding State ($S_A$) fixed | **+90.6%** | [+81.2%, +100.0%] | **$p = 0.0000$** (`exact_exhaustive`) | **Strong Transcript Dominance ($p < .001$)** |
| **`Delta_memory_given_state_B`** | Memory Swap ($M_B \to M_A$) holding State ($S_B$) fixed | **+87.5%** | [+68.8%, +100.0%] | **$p = 0.0001$** (`exact_exhaustive`) | **Strong Transcript Dominance ($p < .001$)** |
| **`Average_Marginal_State_Effect`** | Pooled Marginal Effect of State Swaps ($\bar{\Delta}_{\text{state}}$) | **+4.7%** | [+0.0%, +9.4%] | **$p = 0.2500$** (`exact_exhaustive`) | **No Resolved Independent State Leverage** |
| **`Average_Marginal_Memory_Effect`** | Pooled Marginal Effect of Memory Swaps ($\bar{\Delta}_{\text{memory}}$) | **+89.1%** | [+78.1%, +96.9%] | **$p = 0.0000$** (`exact_exhaustive`) | **Strong Transcript Dominance ($p < .001$)** |
| **`Reset_Dependence`** | Drop in target consistency when state is reset ($M_A + S_0$) | **-3.1%** | [-9.4%, +0.0%] | **$p = 1.0000$** (`exact_exhaustive`) | **Direct Memory Fully Compensates** |

---

## 2. State $\times$ Memory Conflict 3-Way Partition & Directional Breakdown

Under direct State–Memory conflict ($N=128$ conflict trials across 16 twin pairs):
- **Follows Memory Value Rate ($MAR$):** **64.1%** (82 / 128 trials)
- **Follows State Value Rate ($SAR$):** **32.0%** (41 / 128 trials)
- **Chooses Neither / Foil Rate:** **3.9%** (5 / 128 trials)
- **Conditional State Preference ($P(\text{State} \mid \text{State or Memory})$):** **33.3%** (41 / 123 resolved binary trials)
- **Primary Conflict Contrast ($\Delta_{\text{allegiance}} = SAR - MAR$):** **-32.0%** ($p = 0.0002$)

### Directional Breakdown
- **Direction 1 ($M_A + S_B$):** State Allegiance = **50.0%** | Memory Allegiance = **46.9%** (mixed / balanced)
- **Direction 2 ($M_B + S_A$):** State Allegiance = **14.1%** | Memory Allegiance = **81.2%** (strong memory allegiance, driven by suspended goal semantics)

---

## 3. Disaggregated Multi-Condition Matrix Breakdown

| Condition | Presentation Order | Trials | Target State Alleg. | Target Mem Alleg. | Goal State Alleg. | Goal Mem Alleg. | Control Correctness | Prompt Tokens | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clone Fork A (Congruent)** | `memory_first` | 16 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 383.8 tok | 2,548.7 ms |
| **Clone Fork A (Cross-Swap $S_B$)** | `memory_first` | 16 | 75.0% | 25.0% | 0.0% | 0.0% | — | 383.2 tok | 2,466.6 ms |
| **Clone Fork B (Congruent)** | `memory_first` | 16 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 382.1 tok | 2,511.2 ms |
| **State/Memory Conflict ($M_A + S_B$)** | `memory_first` | 48 | 6.2% | 87.5% | 100.0% | 0.0% | 93.8% | 575.9 tok | 2,515.4 ms |
| **State/Memory Conflict ($M_A + S_B$)** | `state_first` | 48 | 0.0% | 93.8% | 93.8% | 6.2% | 93.8% | 575.9 tok | 2,496.3 ms |
| **State/Memory Conflict ($M_B + S_A$)** | `memory_first` | 48 | 12.5% | 87.5% | 31.2% | 62.5% | 100.0% | 574.9 tok | 2,502.6 ms |
| **State/Memory Conflict ($M_B + S_A$)** | `state_first` | 48 | 6.2% | 93.8% | 6.2% | 81.2% | 100.0% | 574.9 tok | 2,499.9 ms |
| **Congruent Baseline A ($M_A + S_A$)** | `memory_first` | 48 | 93.8% | 93.8% | 75.0% | 75.0% | 100.0% | 574.9 tok | 2,537.0 ms |
| **Congruent Baseline A ($M_A + S_A$)** | `state_first` | 48 | 100.0% | 100.0% | 62.5% | 62.5% | 100.0% | 574.9 tok | 2,521.9 ms |
| **Congruent Baseline B ($M_B + S_B$)** | `memory_first` | 48 | 93.8% | 93.8% | 100.0% | 100.0% | 93.8% | 575.9 tok | 2,514.9 ms |
| **Congruent Baseline B ($M_B + S_B$)** | `state_first` | 48 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 575.9 tok | 2,512.0 ms |
| **Memory-Only Calibration ($M_A$)** | `memory_only` | 48 | 0.0% | 100.0% | 0.0% | 62.5% | 87.5% | 331.3 tok | 2,442.3 ms |
| **Memory-Only Calibration ($M_B$)** | `memory_only` | 48 | 0.0% | 100.0% | 0.0% | 87.5% | 87.5% | 331.3 tok | 2,472.0 ms |
| **Reset with Memory Preserved ($M_A + S_0$)** | `memory_first` | 48 | 0.0% | 100.0% | 0.0% | 37.5% | 93.8% | 381.3 tok | 2,487.4 ms |
| **State-Only Calibration ($S_A$)** | `state_only` | 48 | 75.0% | 0.0% | 81.2% | 0.0% | 81.2% | 360.3 tok | 2,474.7 ms |
| **State-Only Calibration ($S_B$)** | `state_only` | 48 | 75.0% | 0.0% | 100.0% | 0.0% | 87.5% | 361.3 tok | 2,478.5 ms |
| **State-Only Calibration ($S_0$)** | `state_only` | 48 | 0.0% | 0.0% | 0.0% | 0.0% | 18.8% | 166.7 tok | 2,496.9 ms |
| **Surgical Slot Inversion ($M_A + S_A'$)** | `memory_first` | 48 | 12.5% | 87.5% | 12.5% | 68.8% | 93.8% | 574.9 tok | 2,509.2 ms |
| **Reconverged Branch A ($E_{\text{sync}}$)** | `memory_first` | 16 | 0.0% | 0.0% | 0.0% | 0.0% | — | 435.6 tok | 2,534.2 ms |
| **Reconverged Branch B ($E_{\text{sync}}$)** | `memory_first` | 16 | 0.0% | 0.0% | 0.0% | 0.0% | — | 434.5 tok | 2,520.8 ms |

---

## 4. Surgical Single-Slot Inversion, Order Effects & Clone Nuance

1. **Local Surgical Inversion ($M_A + S_A'$):**
   - **Target Slot Intervention Uptake:** **12.5%** (only 2 / 16 pairs adopted the counterfactual state slot).
   - **Control Slot Preservation:** **93.8%** (15 / 16 pairs preserved the unedited control binding).
   - **Joint Local Causal Precision:** **12.5%**.
2. **Order Sensitivity Gap:**
   - Memory $\to$ State: **25.0%** State Allegiance
   - State $\to$ Memory: **18.8%** State Allegiance
   - Order Gap: **-6.2%** (Memory dominance persists in both presentation orders).
3. **Paired Reconvergence Concordance:**
   - **93.8%** pairwise behavioral agreement across independent branches post-synchronization ($E_{\text{sync}}$).
4. **The Clone Cross-Swap Nuance:**
   - In the clone cross-swap condition, where the swapped state contributes a novel, out-of-history value (`fork_B`), state allegiance reaches **75.0%**.
   - In the balanced matched-twin experiment, where both candidate values appear in both context histories, episodic memory dominates (**64.1% vs 32.0%**).
   - *Scientific Interpretation:* Structured state is causally readable and can guide behavior when it introduces distinctive information, but is not treated as an authoritative epistemic source when directly contradicting a rich episodic record.

---

## 5. Pre-S09 Level-1 Working Synthesis

1. **Operationally Transcript-Dominant Under E07:**
   Under balanced target-key conflicts, Qwen2.5-3B behavior is strongly governed by episodic transcript information ($\bar{\Delta}_{\text{memory}} = +89.1\%, p = 0.0000$) and shows no resolved independent leverage from swapping ($\bar{\Delta}_{\text{state}} = +4.7\%, p = 0.2500$) or removing ($\text{Reset Dependence} = -3.1\%, p = 1.0$) explicit state.
2. **External Control Surface vs Internal Epistemic Authority:**
   `StructuredSelfState` functions effectively as an inspectable **external serialization and experimenter control surface**, but does not act as an authoritative internal epistemic representation that overrides episodic history.
3. **Bridge to Sprint S09:**
   Because explicit state is causally readable without being epistemically authoritative under conflict, Sprint S09 can now investigate **Source Attribution & Ownership Boundaries**: What does the model actually treat as "its own" when explicit self-state and episodic evidence diverge?
