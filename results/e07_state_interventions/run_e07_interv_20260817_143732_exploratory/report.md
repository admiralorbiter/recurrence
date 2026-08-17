# Experiment E07: State $\times$ Memory Conflict & Causal Intervention Report (Sprint S08)

**Run ID:** `run_e07_interv_20260817_143732_exploratory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-17T14:45:52.378420+00:00  
**Scope:** 4 Matched Twin Episode Pairs | 200 Total Paired Intervention Trials  
**Primary Question:** *Holding the model's explicit memory and final question constant, does changing only the explicit StructuredSelfState causally redirect downstream behavior?*  

---

## 1. Executive Summary & Causal Steering Estimands

| Causal Estimand | Description | Point Estimate | 95% Bootstrap CI | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`Delta_allegiance`** | Primary Conflict Contrast (State Allegiance Rate - Memory Allegiance Rate) | **-21.9%** | [-43.8%, +0.0%] | 0.5000 (`exact_exhaustive`) | **No Resolved Conflict Preference (Null)** |
| **`Delta_state_given_memory`** | Effect of swapping State (S_A -> S_B) on target choice holding Memory (M_A) fixed | **+12.5%** | [+0.0%, +37.5%] | 1.0000 (`exact_exhaustive`) | **Transcript-Equivalent Null** |
| **`Delta_memory_given_state`** | Effect of swapping Memory (M_A -> M_B) on target choice holding State (S_A) fixed | **+87.5%** | [+62.5%, +100.0%] | 0.1250 (`exact_exhaustive`) | **Memory Invariant / Null** |
| **`Reset_Dependence`** | Drop in target answer consistency when state is reset to empty with memory preserved | **+0.0%** | [+0.0%, +0.0%] | 1.0000 (`exact_exhaustive`) | **Direct Memory Fully Compensates** |

---

## 2. State $\times$ Memory Conflict 3-Way Partition & Directional Breakdown

- **Total Conflict Trials Evaluated:** 32
- **Follows State Value Rate ($SAR$):** **37.5%**
- **Follows Memory Value Rate ($MAR$):** **59.4%**
- **Chooses Neither / Foil Option Rate:** **3.1%**
- **Conditional State Preference ($P(\text{State} \mid \text{State or Memory})$):** **38.7%**
- **Primary Conflict Contrast ($\Delta_{\text{allegiance}} = SAR - MAR$):** **-21.9%**

### Directional Conflict Breakdown:
- **Direction 1 ($M_A + S_B$):** State Allegiance = **50.0%** | Memory Allegiance = **43.8%**
- **Direction 2 ($M_B + S_A$):** State Allegiance = **25.0%** | Memory Allegiance = **75.0%**

---

## 3. Multi-Condition Intervention Matrix Breakdown

| Condition | Presentation Order | Trials | State Allegiance | Memory Allegiance | Target Acc (Congruent) | Control Acc | Goal Acc | Prompt Tokens | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clone Fork A (Congruent)** | `memory_first` | 4 | **100.0%** | **100.0%** | 100.0% | 0.0% | 0.0% | 385.0 tok | 2554.4 ms |
| **Clone Fork A (Cross-Swap $S_B$)** | `memory_first` | 4 | **25.0%** | **75.0%** | — | 0.0% | 0.0% | 384.2 tok | 2429.2 ms |
| **Clone Fork B (Congruent)** | `memory_first` | 4 | **100.0%** | **100.0%** | 100.0% | 0.0% | 0.0% | 382.8 tok | 2458.9 ms |
| **State/Memory Conflict ($M_A + S_B$)** | `memory_first` | 12 | **41.7%** | **25.0%** | — | 100.0% | 100.0% | 580.0 tok | 2504.2 ms |
| **State/Memory Conflict ($M_A + S_B$)** | `state_first` | 12 | **33.3%** | **33.3%** | — | 75.0% | 75.0% | 580.0 tok | 2438.2 ms |
| **State/Memory Conflict ($M_B + S_A$)** | `memory_first` | 12 | **16.7%** | **50.0%** | — | 100.0% | 25.0% | 579.0 tok | 2495.1 ms |
| **State/Memory Conflict ($M_B + S_A$)** | `state_first` | 12 | **16.7%** | **50.0%** | — | 100.0% | 50.0% | 579.0 tok | 2513.4 ms |
| **Congruent Baseline A ($M_A + S_A$)** | `memory_first` | 12 | **66.7%** | **66.7%** | 100.0% | 100.0% | 100.0% | 579.0 tok | 2530.0 ms |
| **Congruent Baseline A ($M_A + S_A$)** | `state_first` | 12 | **66.7%** | **66.7%** | 100.0% | 100.0% | 100.0% | 579.0 tok | 2514.7 ms |
| **Congruent Baseline B ($M_B + S_B$)** | `memory_first` | 12 | **50.0%** | **50.0%** | 100.0% | 100.0% | 50.0% | 580.0 tok | 2524.5 ms |
| **Congruent Baseline B ($M_B + S_B$)** | `state_first` | 12 | **58.3%** | **58.3%** | 75.0% | 75.0% | 100.0% | 580.0 tok | 2541.5 ms |
| **Memory-Only Calibration ($M_A$)** | `memory_only` | 12 | **0.0%** | **50.0%** | — | 100.0% | 0.0% | 334.0 tok | 2421.5 ms |
| **Memory-Only Calibration ($M_B$)** | `memory_only` | 12 | **0.0%** | **66.7%** | — | 100.0% | 0.0% | 334.0 tok | 2431.3 ms |
| **Reconverged Branch A ($E_{\text{sync}}$)** | `memory_first` | 4 | **100.0%** | **100.0%** | — | 0.0% | 0.0% | 435.5 tok | 2514.8 ms |
| **Reconverged Branch B ($E_{\text{sync}}$)** | `memory_first` | 4 | **100.0%** | **100.0%** | — | 0.0% | 0.0% | 434.0 tok | 2571.0 ms |
| **Reset with Memory Preserved ($M_A + S_0$)** | `memory_first` | 12 | **0.0%** | **50.0%** | — | 100.0% | 0.0% | 384.0 tok | 2466.5 ms |
| **State-Only Calibration ($S_A$)** | `state_only` | 12 | **58.3%** | **0.0%** | — | 50.0% | 100.0% | 362.5 tok | 2462.9 ms |
| **State-Only Calibration ($S_B$)** | `state_only` | 12 | **66.7%** | **0.0%** | — | 50.0% | 100.0% | 363.5 tok | 2471.2 ms |
| **State-Only Calibration ($S_0$)** | `state_only` | 12 | **0.0%** | **0.0%** | — | 25.0% | 0.0% | 167.5 tok | 2499.7 ms |
| **Surgical Slot Inversion ($M_A + S_A'$)** | `memory_first` | 12 | **8.3%** | **58.3%** | — | 100.0% | 25.0% | 578.2 tok | 2512.7 ms |

---

## 4. Surgical Single-Slot Edit & Local Causal Precision

- **Target Slot Intervention Uptake (P(Target = Injected)):** **0.0%**
- **Control Slot Preservation (P(Control = Gold)):** **100.0%**
- **Joint Local Causal Precision (P(Target Uptake and Control Preserved)):** **0.0%**

---

## 5. Presentation Order Sensitivity & Infrastructure Invariants

- **State Allegiance (Memory -> State Order):** **29.2%**
- **State Allegiance (State -> Memory Order):** **25.0%**
- **Order Sensitivity Gap:** **-4.2%**
- **Reconvergence Behavioral Concordance Rate:** **100.0%**

---

## 6. Scientific Gate Decision for Sprint S08

1. **Causal State Steering vs Transcript Equivalence:** Does explicit state intervention reliably steer the model's output away from historical memory?
2. **Reset Dependence:** Does removing state with memory intact impair performance, proving explicit state provides non-redundant operational utility?
3. **Local Surgical Precision:** Does single-slot editing steer the targeted behavior without causing collateral representation drift?