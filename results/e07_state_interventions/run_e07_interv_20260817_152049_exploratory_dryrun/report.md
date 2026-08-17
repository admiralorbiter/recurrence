# Experiment E07: State $\times$ Memory Conflict & Causal Intervention Report (Sprint S08)

**Run ID:** `run_e07_interv_20260817_152049_exploratory_dryrun`  
**Model:** `qwen2.5:3b` (`mock_digest_...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-17T15:20:49.317075+00:00  
**Scope:** 2 Matched Twin Episode Pairs | 100 Total Paired Intervention Trials  
**Primary Question:** *Holding the model's explicit memory and final question constant, does changing only the explicit StructuredSelfState causally redirect downstream behavior?*  

---

## 1. Executive Summary & Causal Steering Estimands

| Causal Estimand | Description | Point Estimate | 95% Bootstrap CI | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`Delta_allegiance`** | Primary Conflict Contrast (State Allegiance Rate - Memory Allegiance Rate) | **+0.0%** | [+0.0%, +0.0%] | 1.0000 (`exact_exhaustive`) | **No Resolved Conflict Preference (Null)** |
| **`Delta_state_given_memory`** | Effect of swapping State (S_A -> S_B) on target choice holding Memory (M_A) fixed | **+0.0%** | [+0.0%, +0.0%] | 1.0000 (`exact_exhaustive`) | **Transcript-Equivalent Null** |
| **`Delta_memory_given_state`** | Effect of swapping Memory (M_A -> M_B) on target choice holding State (S_A) fixed | **+0.0%** | [+0.0%, +0.0%] | 1.0000 (`exact_exhaustive`) | **Memory Invariant / Null** |
| **`Reset_Dependence`** | Drop in target answer consistency when state is reset to empty with memory preserved | **+0.0%** | [+0.0%, +0.0%] | 1.0000 (`exact_exhaustive`) | **Direct Memory Fully Compensates** |

---

## 2. State $\times$ Memory Conflict 3-Way Partition & Directional Breakdown

- **Total Conflict Trials Evaluated:** 16
- **Follows State Value Rate ($SAR$):** **37.5%**
- **Follows Memory Value Rate ($MAR$):** **37.5%**
- **Chooses Neither / Foil Option Rate:** **25.0%**
- **Conditional State Preference ($P(\text{State} \mid \text{State or Memory})$):** **50.0%**
- **Primary Conflict Contrast ($\Delta_{\text{allegiance}} = SAR - MAR$):** **+0.0%**

### Directional Conflict Breakdown:
- **Direction 1 ($M_A + S_B$):** State Allegiance = **25.0%** | Memory Allegiance = **50.0%**
- **Direction 2 ($M_B + S_A$):** State Allegiance = **50.0%** | Memory Allegiance = **25.0%**

---

## 3. Multi-Condition Intervention Matrix Breakdown

| Condition | Presentation Order | Trials | State Allegiance | Memory Allegiance | Target Acc (Congruent) | Control Acc | Goal Acc | Prompt Tokens | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clone Fork A (Congruent)** | `memory_first` | 2 | **0.0%** | **0.0%** | 0.0% | 0.0% | 0.0% | 314.0 tok | 0.0 ms |
| **Clone Fork A (Cross-Swap $S_B$)** | `memory_first` | 2 | **100.0%** | **0.0%** | — | 0.0% | 0.0% | 314.0 tok | 0.0 ms |
| **Clone Fork B (Congruent)** | `memory_first` | 2 | **100.0%** | **100.0%** | 100.0% | 0.0% | 0.0% | 313.0 tok | 0.0 ms |
| **State/Memory Conflict ($M_A + S_B$)** | `memory_first` | 6 | **16.7%** | **33.3%** | — | 0.0% | 0.0% | 517.8 tok | 0.0 ms |
| **State/Memory Conflict ($M_A + S_B$)** | `state_first` | 6 | **16.7%** | **33.3%** | — | 0.0% | 0.0% | 517.8 tok | 0.0 ms |
| **State/Memory Conflict ($M_B + S_A$)** | `memory_first` | 6 | **33.3%** | **16.7%** | — | 0.0% | 50.0% | 516.0 tok | 0.0 ms |
| **State/Memory Conflict ($M_B + S_A$)** | `state_first` | 6 | **33.3%** | **16.7%** | — | 0.0% | 50.0% | 516.0 tok | 0.0 ms |
| **Congruent Baseline A ($M_A + S_A$)** | `memory_first` | 6 | **33.3%** | **33.3%** | 50.0% | 0.0% | 50.0% | 517.0 tok | 0.0 ms |
| **Congruent Baseline A ($M_A + S_A$)** | `state_first` | 6 | **33.3%** | **33.3%** | 50.0% | 0.0% | 50.0% | 517.0 tok | 0.0 ms |
| **Congruent Baseline B ($M_B + S_B$)** | `memory_first` | 6 | **16.7%** | **16.7%** | 50.0% | 0.0% | 0.0% | 517.0 tok | 0.0 ms |
| **Congruent Baseline B ($M_B + S_B$)** | `state_first` | 6 | **16.7%** | **16.7%** | 50.0% | 0.0% | 0.0% | 517.0 tok | 0.0 ms |
| **Memory-Only Calibration ($M_A$)** | `memory_only` | 6 | **0.0%** | **33.3%** | — | 0.0% | 0.0% | 299.8 tok | 0.0 ms |
| **Memory-Only Calibration ($M_B$)** | `memory_only` | 6 | **0.0%** | **16.7%** | — | 0.0% | 0.0% | 299.2 tok | 0.0 ms |
| **Reconverged Branch A ($E_{\text{sync}}$)** | `memory_first` | 2 | **0.0%** | **0.0%** | — | 0.0% | 0.0% | 369.0 tok | 0.0 ms |
| **Reconverged Branch B ($E_{\text{sync}}$)** | `memory_first` | 2 | **0.0%** | **0.0%** | — | 0.0% | 0.0% | 368.5 tok | 0.0 ms |
| **Reset with Memory Preserved ($M_A + S_0$)** | `memory_first` | 6 | **0.0%** | **33.3%** | — | 0.0% | 0.0% | 344.7 tok | 0.0 ms |
| **State-Only Calibration ($S_A$)** | `state_only` | 6 | **33.3%** | **0.0%** | — | 0.0% | 50.0% | 298.3 tok | 0.0 ms |
| **State-Only Calibration ($S_B$)** | `state_only` | 6 | **16.7%** | **0.0%** | — | 0.0% | 0.0% | 298.8 tok | 0.0 ms |
| **State-Only Calibration ($S_0$)** | `state_only` | 6 | **0.0%** | **0.0%** | — | 0.0% | 0.0% | 126.0 tok | 0.0 ms |
| **Surgical Slot Inversion ($M_A + S_A'$)** | `memory_first` | 6 | **16.7%** | **33.3%** | — | 0.0% | 0.0% | 517.5 tok | 0.0 ms |

---

## 4. Surgical Single-Slot Edit & Local Causal Precision

- **Target Slot Intervention Uptake (P(Target = Injected)):** **50.0%**
- **Control Slot Preservation (P(Control = Gold)):** **0.0%**
- **Joint Local Causal Precision (P(Target Uptake and Control Preserved)):** **0.0%**

---

## 5. Presentation Order Sensitivity & Infrastructure Invariants

- **State Allegiance (Memory -> State Order):** **25.0%**
- **State Allegiance (State -> Memory Order):** **25.0%**
- **Order Sensitivity Gap:** **+0.0%**
- **Reconvergence Behavioral Concordance Rate:** **100.0%**

---

## 6. Scientific Gate Decision for Sprint S08

1. **Causal State Steering vs Transcript Equivalence:** Does explicit state intervention reliably steer the model's output away from historical memory?
2. **Reset Dependence:** Does removing state with memory intact impair performance, proving explicit state provides non-redundant operational utility?
3. **Local Surgical Precision:** Does single-slot editing steer the targeted behavior without causing collateral representation drift?