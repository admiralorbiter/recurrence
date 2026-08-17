# Sprint S08 Walkthrough: State $\times$ Memory Conflict & Causal Interventions (Experiment E07)

## 1. Executive Summary

Sprint S08 completed the canonical **State $\times$ Memory Conflict & Causal State Intervention Battery (Experiment E07)** on `qwen2.5:3b`.

Across 16 matched twin episode pairs (32 total episodes, 800 paired trials, Seed 1337), we evaluated whether explicit `StructuredSelfState` functions as an independently dominant causal control surface when placed in direct competition with historical episodic memory.

---

## 2. Master Confirmatory Results Summary

### A. Primary Causal Estimands ($N=16$ Twin Pairs, 800 Trials, Seed 1337)

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

### B. State $\times$ Memory Conflict 3-Way Partition & Directional Breakdown

Under direct State–Memory conflict ($N=128$ conflict trials):
- **Follows Memory Value Rate ($MAR$):** **64.1%** (82 / 128 trials)
- **Follows State Value Rate ($SAR$):** **32.0%** (41 / 128 trials)
- **Chooses Neither / Foil Option:** **3.9%** (5 / 128 trials)
- **Conditional State Preference ($P(\text{State} \mid \text{State or Memory})$):** **33.3%**
- **Directional Breakdown:**
  - $M_A + S_B$: State Allegiance = **50.0%** | Memory Allegiance = **46.9%**
  - $M_B + S_A$: State Allegiance = **14.1%** | Memory Allegiance = **81.2%**

---

### C. Surgical Single-Slot Inversion & Order Invariance

- **Target Slot Intervention Uptake:** **12.5%** (2 / 16 twin pairs followed surgical state edit)
- **Control Slot Preservation:** **93.8%** (15 / 16 twin pairs preserved control slot)
- **Joint Local Causal Precision:** **12.5%**
- **Order Sensitivity Gap:** **-6.2%** (State Allegiance was 25.0% under Memory-first order vs 18.8% under State-first order)
- **Reconvergence Behavioral Concordance:** **93.8%** across independent branch trajectories post-synchronization

---

## 3. Scientific Synthesis Across Level 1 Recurrence (S04–S08)

The completion of S08 completes the empirical characterization of Level-1 Scaffolded Persistence:

1. **S04 (State Readout):** Structured state can be parsed and read when provided.
2. **S05 (State Maintenance):** Structured state can be maintained deterministically without factual drift under prefix updates.
3. **S06 (State Reconstruction):** Structured state can be reconstructed by external observer models from history.
4. **S07 (Consolidation Deficit):** Autonomous model-generated self-updates suffer **Persistent Derivation Write Failure** (0.0% derived write precision).
5. **S08 (Causal Conflict & Control):** When explicit state is manipulated into conflict with raw episodic history, the model **overwhelmingly defaults to the historical episodic transcript** ($\bar{\Delta}_{\text{memory}} = +89.1\%, p < .001$ vs $\bar{\Delta}_{\text{state}} = +4.7\%, p = 0.25$). Wiping state with memory intact causes no drop in accuracy ($\text{Reset Dependence} = -3.1\%$).

### Pre-S09 Level-1 Working Synthesis:
`StructuredSelfState` operates as a **causally readable external recording and serialization format for the user/system**, but **does not act as an authoritative internal epistemic controller for the model** when directly conflicting with a rich episodic record.

This establishes the operationally transcript-dominant baseline for Level-1 recurrence and sets up the core questions for **Sprint S09 (Source Attribution & Ownership Boundaries)**.
