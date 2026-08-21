# Q16b.1: Transitive Ancestry Composition & Laundering Corroboration Report

========================================================================================================================
Q16b.1 REPORT (16 SEEDS, RUNTIME: 2.0165335s)
1. Designated Local & Composed Edge Validation: 16/16 seeds passed (100.0%)
2. Causal Transmission Spectrum: A -> B = +69.2%, B -> C = +61.1%, Composed Ancestry Score A -> C = +42.3%, A -> D = +0.0%
========================================================================================================================

## 1. Provenance Laundering & Corroboration Battery Results:
| Scenario Name | Target Action | Realized Return | Target Action Acc | Secondary Error Rate | Arrow-Sign Acc | Transposed Target Acc | Transposed Return | Paired ΔAcc Drop (±STE) | Seed Promotion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. DIRECT COPY CONFLICT (A != B, VERIFY=1.00)** | Commit Parent A | +1.42 | +100.0% | +0.0% | +100.0% | +0.0% | -4.42 | +100.0% (±0.0%) | 16/16 seeds |
| **2. TRANSITIVE MULTI-HOP CONFLICT (A != C, COMPOSE A->B->C)** | Commit Root A (Composed) | +1.44 | +100.0% | +0.0% | +100.0% | +0.0% | -4.44 | +100.0% (±0.0%) | 16/16 seeds |
| **3. LAUNDERED REDUNDANT AGREEMENT (A == C, THRESHOLD=1.60)** | VERIFY (Redundant Copy) | +1.60 | +100.0% | +0.0% | +100.0% | +100.0% | +1.60 | +0.0% (±0.0%) | 16/16 seeds |
| **4. INDEPENDENT CORROBORATION (A == D, THRESHOLD=1.60)** | COMMIT (True Corroboration) | +1.92 | +93.8% | +6.2% | +93.8% | +93.8% | +1.92 | +0.0% (±0.0%) | 15/16 seeds |
| **5. INDEPENDENT CONFLICT (A != D, VERIFY=1.00)** | VERIFY (Indep Conflict) | +0.85 | +93.8% | +6.2% | +93.8% | +93.8% | +0.84 | +0.0% (±0.0%) | 15/16 seeds |

========================================================================================================================
## 2. SCIENTIFIC DIAGNOSTIC CONCLUSIONS:
- **Transitive Ancestry Composition from Autonomous Local Relations:**
  * By measuring ONLY local neighbor shocks (do(A)->B and do(B)->C) and algebraically composing paths via generic transitive composition, the system constructs a **+42.3% composed ancestry score (A => C)** without ever directly observing do(A)->C.
  * Direct Copy Conflict (A != B)    : **+100.0% Parent Choice Accuracy** (Return = +1.42, 16/16 seeds)
  * Transitive Multi-Hop (A != C)    : **+100.0% Root Originator Choice** (Return = +1.44, Transposed = -4.44, 16/16 seeds)
- **Double Dissociation in Laundering Corroboration (High-Threshold Regime VERIFY = +1.60):**
  * **Laundered Redundant Agreement (A == C):** Organism detects redundant ancestry score and selects **VERIFY with +100.0% accuracy** (Return = +1.60, 16/16 seeds), avoiding the overconfidence trap (+1.44 commit < +1.60 verify).
  * **Truly Independent Corroboration (A == D):** Organism detects absence of ancestry and confidently **COMMITS with +93.8% accuracy** (Return = +1.92 > +1.60 threshold, 15/16 seeds).
  * **Independent Conflict (A != D):** Organism falls back to **VERIFY with +93.8% accuracy** (Return = +0.85, 15/16 seeds).
- **Epistemic Laundering Discrimination:** Provenance tracking successfully discriminates direct copying, multi-hop laundering, redundant corroboration, and genuine independent confirmation under a fixed Bayes decision policy.
========================================================================================================================
