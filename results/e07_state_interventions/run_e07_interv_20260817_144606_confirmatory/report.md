# Experiment E07: State $\times$ Memory Conflict & Causal Intervention Report (Sprint S08)

**Run ID:** `run_e07_interv_20260817_144606_confirmatory`  
**Model:** `qwen2.5:3b` (`357c53fb659c...`)  
**Phase:** `CONFIRMATORY` (Seed: `1337`)  
**Date:** 2026-08-17T15:19:28.481169+00:00  
**Scope:** 16 Matched Twin Episode Pairs | 800 Total Paired Intervention Trials  
**Primary Question:** *Holding the model's explicit memory and final question constant, does changing only the explicit StructuredSelfState causally redirect downstream behavior?*  

---

## 1. Executive Summary & Causal Steering Estimands

| Causal Estimand | Description | Point Estimate | 95% Bootstrap CI | Permutation $p$ (Method) | Scientific Inference |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`Delta_allegiance`** | Primary Conflict Contrast (State Allegiance Rate - Memory Allegiance Rate) | **-32.0%** | [-43.0%, -21.1%] | 0.0002 (`exact_exhaustive`) | **Statistically Distinguishable Conflict Preference (Memory Favored)** |
| **`Delta_state_given_memory_A`** | Effect of swapping State (S_A -> S_B) on target choice holding Memory (M_A) fixed | **+3.1%** | [+0.0%, +9.4%] | 1.0000 (`exact_exhaustive`) | **No Resolved Independent State Leverage** |
| **`Delta_state_given_memory_B`** | Effect of swapping State (S_B -> S_A) on target choice holding Memory (M_B) fixed | **+6.2%** | [-6.2%, +18.8%] | 0.6250 (`exact_exhaustive`) | **No Resolved Independent State Leverage** |
| **`Delta_memory_given_state_A`** | Effect of swapping Memory (M_A -> M_B) on target choice holding State (S_A) fixed | **+90.6%** | [+81.2%, +100.0%] | 0.0000 (`exact_exhaustive`) | **Strong Transcript Dominance** |
| **`Delta_memory_given_state_B`** | Effect of swapping Memory (M_B -> M_A) on target choice holding State (S_B) fixed | **+87.5%** | [+68.8%, +100.0%] | 0.0001 (`exact_exhaustive`) | **Strong Transcript Dominance** |
| **`Average_Marginal_State_Effect`** | Pooled Average Marginal Effect of State Swaps across both memory contexts | **+4.7%** | [+0.0%, +9.4%] | 0.2500 (`exact_exhaustive`) | **No Resolved Independent State Leverage** |
| **`Average_Marginal_Memory_Effect`** | Pooled Average Marginal Effect of Memory Swaps across both state contexts | **+89.1%** | [+78.1%, +96.9%] | 0.0000 (`exact_exhaustive`) | **Strong Transcript Dominance** |
| **`Reset_Dependence`** | Drop in target answer consistency when state is reset to empty with memory preserved | **-3.1%** | [-9.4%, +0.0%] | 1.0000 (`exact_exhaustive`) | **Direct Memory Fully Compensates** |

---

## 2. State $\times$ Memory Conflict 3-Way Partition & Directional Breakdown

- **Total Conflict Trials Evaluated:** 128
- **Follows State Value Rate ($SAR$):** **32.0%**
- **Follows Memory Value Rate ($MAR$):** **64.1%**
- **Chooses Neither / Foil Option Rate:** **3.9%**
- **Conditional State Preference ($P(\text{State} \mid \text{State or Memory})$):** **33.3%**
- **Primary Conflict Contrast ($\Delta_{\text{allegiance}} = SAR - MAR$):** **-32.0%**

### Directional Conflict Breakdown:
- **Direction 1 ($M_A + S_B$):** State Allegiance = **50.0%** | Memory Allegiance = **46.9%**
- **Direction 2 ($M_B + S_A$):** State Allegiance = **14.1%** | Memory Allegiance = **81.2%**

---

## 3. Multi-Condition Intervention Matrix Breakdown (Disaggregated by Probe Domain)

| Condition | Presentation Order | Trials | Target State Alleg. | Target Mem Alleg. | Goal State Alleg. | Goal Mem Alleg. | Control Correctness | Prompt Tokens | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clone Fork A (Congruent)** | `memory_first` | 16 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 383.8 tok | 2548.7 ms |
| **Clone Fork A (Cross-Swap $S_B$)** | `memory_first` | 16 | 75.0% | 25.0% | 0.0% | 0.0% | — | 383.2 tok | 2466.6 ms |
| **Clone Fork B (Congruent)** | `memory_first` | 16 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 382.1 tok | 2511.2 ms |
| **State/Memory Conflict ($M_A + S_B$)** | `memory_first` | 48 | 6.2% | 87.5% | 100.0% | 0.0% | 93.8% | 575.9 tok | 2515.4 ms |
| **State/Memory Conflict ($M_A + S_B$)** | `state_first` | 48 | 0.0% | 93.8% | 93.8% | 6.2% | 93.8% | 575.9 tok | 2496.3 ms |
| **State/Memory Conflict ($M_B + S_A$)** | `memory_first` | 48 | 12.5% | 87.5% | 31.2% | 62.5% | 100.0% | 574.9 tok | 2502.6 ms |
| **State/Memory Conflict ($M_B + S_A$)** | `state_first` | 48 | 6.2% | 93.8% | 6.2% | 81.2% | 100.0% | 574.9 tok | 2499.9 ms |
| **Congruent Baseline A ($M_A + S_A$)** | `memory_first` | 48 | 93.8% | 93.8% | 75.0% | 75.0% | 100.0% | 574.9 tok | 2537.0 ms |
| **Congruent Baseline A ($M_A + S_A$)** | `state_first` | 48 | 100.0% | 100.0% | 62.5% | 62.5% | 100.0% | 574.9 tok | 2521.9 ms |
| **Congruent Baseline B ($M_B + S_B$)** | `memory_first` | 48 | 93.8% | 93.8% | 100.0% | 100.0% | 93.8% | 575.9 tok | 2514.9 ms |
| **Congruent Baseline B ($M_B + S_B$)** | `state_first` | 48 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 575.9 tok | 2512.0 ms |
| **Memory-Only Calibration ($M_A$)** | `memory_only` | 48 | 0.0% | 100.0% | 0.0% | 62.5% | 87.5% | 331.3 tok | 2442.3 ms |
| **Memory-Only Calibration ($M_B$)** | `memory_only` | 48 | 0.0% | 100.0% | 0.0% | 87.5% | 87.5% | 331.3 tok | 2472.0 ms |
| **Reconverged Branch A ($E_{\text{sync}}$)** | `memory_first` | 16 | 0.0% | 0.0% | 0.0% | 0.0% | — | 435.6 tok | 2534.2 ms |
| **Reconverged Branch B ($E_{\text{sync}}$)** | `memory_first` | 16 | 0.0% | 0.0% | 0.0% | 0.0% | — | 434.5 tok | 2520.8 ms |
| **Reset with Memory Preserved ($M_A + S_0$)** | `memory_first` | 48 | 0.0% | 100.0% | 0.0% | 37.5% | 93.8% | 381.3 tok | 2487.4 ms |
| **State-Only Calibration ($S_A$)** | `state_only` | 48 | 75.0% | 0.0% | 81.2% | 0.0% | 81.2% | 360.3 tok | 2474.7 ms |
| **State-Only Calibration ($S_B$)** | `state_only` | 48 | 75.0% | 0.0% | 100.0% | 0.0% | 87.5% | 361.3 tok | 2478.5 ms |
| **State-Only Calibration ($S_0$)** | `state_only` | 48 | 0.0% | 0.0% | 0.0% | 0.0% | 18.8% | 166.7 tok | 2496.9 ms |
| **Surgical Slot Inversion ($M_A + S_A'$)** | `memory_first` | 48 | 12.5% | 87.5% | 12.5% | 68.8% | 93.8% | 574.9 tok | 2509.2 ms |

---

## 4. Surgical Single-Slot Edit & Local Causal Precision

- **Target Slot Intervention Uptake (P(Target = Injected)):** **12.5%**
- **Control Slot Preservation (P(Control = Gold)):** **93.8%**
- **Joint Local Causal Precision (P(Target Uptake and Control Preserved)):** **12.5%**

---

## 5. Presentation Order Sensitivity & Infrastructure Invariants

- **State Allegiance (Memory -> State Order):** **25.0%**
- **State Allegiance (State -> Memory Order):** **18.8%**
- **Order Sensitivity Gap:** **-6.2%**
- **Reconvergence Behavioral Concordance Rate:** **93.8%**

---

## 6. Scientific Interpretation & Level-1 Synthesis

1. **Causal Asymmetry Under Conflict:** Holding memory fixed and swapping state produces no resolved change on target choice (+3.1pp, p = 1.0), whereas holding state fixed and swapping memory changes target choice dramatically (+90.6pp, p < .001). Under direct balanced conflict, the model strongly privileges historical episodic evidence (MAR = 64.1% vs SAR = 32.0%, p = 0.0002).
2. **State Reset Independence:** Clearing StructuredSelfState while preserving episodic memory produces no drop in target accuracy (Reset Dependence = -3.1pp, p = 1.0). Direct episodic memory fully compensates for the removal of the Level-1 explicit state.
3. **Clone Cross-Swap Qualification:** In the clone testbed, where the swapped state contributes an out-of-history value, state allegiance reaches 75.0%. When both candidates are familiar in-context (matched twins), episodic memory dominates. StructuredSelfState is causally readable and usable when distinctive, but is not treated as an authoritative epistemic controller when conflicting with the episodic record.