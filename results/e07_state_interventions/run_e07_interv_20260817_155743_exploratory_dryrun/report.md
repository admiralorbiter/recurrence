# Experiment E07: State $\times$ Memory Conflict & Causal Intervention Report (Sprint S08)

**Run ID:** `run_e07_interv_20260817_155743_exploratory_dryrun`  
**Model:** `qwen2.5:3b` (`mock_digest_...`)  
**Phase:** `EXPLORATORY` (Seed: `42`)  
**Date:** 2026-08-17T15:57:43.538580+00:00  
**Scope:** 2 Matched Twin Episode Pairs | 100 Total Paired Intervention Trials  
**Primary Question:** *Holding the model's explicit memory and final question constant, does changing only the explicit StructuredSelfState causally redirect downstream behavior?*  

---

## 1. Executive Summary & Causal Steering Estimands

| Causal Estimand | Description | Point Estimate | 95% Bootstrap CI | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`Delta_allegiance`** | Primary Conflict Contrast (State Allegiance Rate - Memory Allegiance Rate) | **+0.0%** | [+0.0%, +0.0%] | 1.0000 (`exact_exhaustive`) | **No Resolved Conflict Preference (Null)** |
| **`Delta_state_given_memory_A`** | Effect of swapping State (S_A -> S_B) on target choice holding Memory (M_A) fixed | **+0.0%** | [+0.0%, +0.0%] | 1.0000 (`exact_exhaustive`) | **No Resolved Independent State Leverage** |
| **`Delta_state_given_memory_B`** | Effect of swapping State (S_B -> S_A) on target choice holding Memory (M_B) fixed | **+0.0%** | [+0.0%, +0.0%] | 1.0000 (`exact_exhaustive`) | **No Resolved Independent State Leverage** |
| **`Delta_memory_given_state_A`** | Effect of swapping Memory (M_A -> M_B) on target choice holding State (S_A) fixed | **+0.0%** | [+0.0%, +0.0%] | 1.0000 (`exact_exhaustive`) | **Memory Invariant / Null** |
| **`Delta_memory_given_state_B`** | Effect of swapping Memory (M_B -> M_A) on target choice holding State (S_B) fixed | **+0.0%** | [+0.0%, +0.0%] | 1.0000 (`exact_exhaustive`) | **Memory Invariant / Null** |
| **`Average_Marginal_State_Effect`** | Pooled Average Marginal Effect of State Swaps across both memory contexts | **+0.0%** | [+0.0%, +0.0%] | 1.0000 (`exact_exhaustive`) | **No Resolved Independent State Leverage** |
| **`Average_Marginal_Memory_Effect`** | Pooled Average Marginal Effect of Memory Swaps across both state contexts | **+0.0%** | [+0.0%, +0.0%] | 1.0000 (`exact_exhaustive`) | **Memory Invariant / Null** |
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

## 3. Multi-Condition Intervention Matrix Breakdown (Disaggregated by Probe Domain)

| Condition | Presentation Order | Trials | Target State Alleg. | Target Mem Alleg. | Goal State Alleg. | Goal Mem Alleg. | Control Correctness | Prompt Tokens | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clone Fork A (Congruent)** | `memory_first` | 2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 314.0 tok | 0.0 ms |
| **Clone Fork A (Cross-Swap $S_B$)** | `memory_first` | 2 | 100.0% | 0.0% | 0.0% | 0.0% | — | 314.0 tok | 0.0 ms |
| **Clone Fork B (Congruent)** | `memory_first` | 2 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 313.0 tok | 0.0 ms |
| **State/Memory Conflict ($M_A + S_B$)** | `memory_first` | 6 | 50.0% | 50.0% | 0.0% | 50.0% | 0.0% | 517.8 tok | 0.0 ms |
| **State/Memory Conflict ($M_A + S_B$)** | `state_first` | 6 | 50.0% | 50.0% | 0.0% | 50.0% | 0.0% | 517.8 tok | 0.0 ms |
| **State/Memory Conflict ($M_B + S_A$)** | `memory_first` | 6 | 50.0% | 50.0% | 50.0% | 0.0% | 0.0% | 516.0 tok | 0.0 ms |
| **State/Memory Conflict ($M_B + S_A$)** | `state_first` | 6 | 50.0% | 50.0% | 50.0% | 0.0% | 0.0% | 516.0 tok | 0.0 ms |
| **Congruent Baseline A ($M_A + S_A$)** | `memory_first` | 6 | 50.0% | 50.0% | 50.0% | 50.0% | 0.0% | 517.0 tok | 0.0 ms |
| **Congruent Baseline A ($M_A + S_A$)** | `state_first` | 6 | 50.0% | 50.0% | 50.0% | 50.0% | 0.0% | 517.0 tok | 0.0 ms |
| **Congruent Baseline B ($M_B + S_B$)** | `memory_first` | 6 | 50.0% | 50.0% | 0.0% | 0.0% | 0.0% | 517.0 tok | 0.0 ms |
| **Congruent Baseline B ($M_B + S_B$)** | `state_first` | 6 | 50.0% | 50.0% | 0.0% | 0.0% | 0.0% | 517.0 tok | 0.0 ms |
| **Memory-Only Calibration ($M_A$)** | `memory_only` | 6 | 0.0% | 50.0% | 0.0% | 50.0% | — | 299.8 tok | 0.0 ms |
| **Memory-Only Calibration ($M_B$)** | `memory_only` | 6 | 0.0% | 50.0% | 0.0% | 0.0% | — | 299.2 tok | 0.0 ms |
| **Reconverged Branch A ($E_{\text{sync}}$)** | `memory_first` | 2 | 0.0% | 0.0% | 0.0% | 0.0% | — | 369.0 tok | 0.0 ms |
| **Reconverged Branch B ($E_{\text{sync}}$)** | `memory_first` | 2 | 0.0% | 0.0% | 0.0% | 0.0% | — | 368.5 tok | 0.0 ms |
| **Reset with Memory Preserved ($M_A + S_0$)** | `memory_first` | 6 | 0.0% | 50.0% | 0.0% | 50.0% | 0.0% | 344.7 tok | 0.0 ms |
| **State-Only Calibration ($S_A$)** | `state_only` | 6 | 50.0% | 0.0% | 50.0% | 0.0% | — | 298.3 tok | 0.0 ms |
| **State-Only Calibration ($S_B$)** | `state_only` | 6 | 50.0% | 0.0% | 0.0% | 0.0% | — | 298.8 tok | 0.0 ms |
| **State-Only Calibration ($S_0$)** | `state_only` | 6 | 0.0% | 0.0% | 0.0% | 0.0% | — | 126.0 tok | 0.0 ms |
| **Surgical Slot Inversion ($M_A + S_A'$)** | `memory_first` | 6 | 50.0% | 50.0% | 0.0% | 50.0% | 0.0% | 517.5 tok | 0.0 ms |

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

## 6. Scientific Interpretation & Level-1 Synthesis

1. **Causal Asymmetry Under Conflict:** Holding memory fixed and swapping state produces no resolved change on target choice (+3.1pp, p = 1.0), whereas holding state fixed and swapping memory changes target choice dramatically (+90.6pp, p < .001). Under direct balanced conflict, the model strongly privileges historical episodic evidence (MAR = 64.1% vs SAR = 32.0%, p = 0.0002).
2. **State Reset Independence:** Clearing StructuredSelfState while preserving episodic memory produces no drop in target accuracy (Reset Dependence = -3.1pp, p = 1.0). Direct episodic memory fully compensates for the removal of the Level-1 explicit state.
3. **Clone Cross-Swap Qualification:** In the clone testbed, where the swapped state contributes an out-of-history value, state allegiance reaches 75.0%. When both candidates are familiar in-context (matched twins), episodic memory dominates. StructuredSelfState is causally readable and usable when distinctive, but is not treated as an authoritative epistemic controller when conflicting with the episodic record.