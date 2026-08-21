# Q16b.1: Transitive Ancestry Composition & Laundering Corroboration Report

========================================================================================================================
Q16b.1 REPORT (16 SEEDS, RUNTIME: 2.0165335s)
1. Transitive Composition Accuracy: +100.0% correct orientations (Direct A -> C shocks MASKED during development)
2. Causal Transmission Spectrum   : A -> B = +69.2%, B -> C = +61.1%, Composed A -> C = +42.3%, A -> D = +0.0%
========================================================================================================================

## 1. Provenance Laundering & Corroboration Battery Results:
| Scenario Name | Target Action | Realized Return | Target Action Acc | Secondary Error Rate | Arrow-Sign Acc | Transposed Target Acc | Transposed Return | Paired ΔAcc Drop (±STE) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. DIRECT COPY CONFLICT (A != B, VERIFY=1.00)** | Commit Parent A | +1.42 | +100.0% | +0.0% | +100.0% | +0.0% | -4.42 | +100.0% (±0.0%) |
| **2. TRANSITIVE MULTI-HOP CONFLICT (A != C, COMPOSE A->B->C)** | Commit Root A (Composed) | +1.44 | +100.0% | +0.0% | +100.0% | +0.0% | -4.44 | +100.0% (±0.0%) |
| **3. LAUNDERED REDUNDANT AGREEMENT (A == C, THRESHOLD=1.60)** | VERIFY (Redundant Copy) | +1.60 | +100.0% | +0.0% | +100.0% | +100.0% | +1.60 | +0.0% (±0.0%) |
| **4. INDEPENDENT CORROBORATION (A == D, THRESHOLD=1.60)** | COMMIT (True Corroboration) | +1.92 | +93.8% | +6.2% | +93.8% | +93.8% | +1.92 | +0.0% (±0.0%) |
| **5. INDEPENDENT CONFLICT (A != D, VERIFY=1.00)** | VERIFY (Indep Conflict) | +0.85 | +93.8% | +6.2% | +93.8% | +93.8% | +0.84 | +0.0% (±0.0%) |

========================================================================================================================
## 2. SCIENTIFIC DIAGNOSTIC CONCLUSIONS:
- **Transitive Ancestry Composition Without Direct Observation:**
  * By measuring ONLY local neighbor shocks (do(A)->B and do(B)->C) and algebraically composing paths, the system achieves **+100.0% accurate transitive reachability (A => C)** without ever directly observing do(A)->C.
  * Direct Copy Conflict (A != B)    : **+100.0% Parent Choice Accuracy** (Return = +1.42)
  * Transitive Multi-Hop (A != C)    : **+100.0% Root Originator Choice** (Return = +1.44, Transposed = -4.44)
- **Double Dissociation in Laundering Corroboration (High-Threshold Regime VERIFY = +1.60):**
  * **Laundered Redundant Agreement (A == C):** Organism recognizes shared ancestry and selects **VERIFY with +100.0% accuracy** (Return = +1.60), avoiding the overconfidence trap (+1.44 commit < +1.60 verify).
  * **Truly Independent Corroboration (A == D):** Organism recognizes true independence and confidently **COMMITS with +93.8% accuracy** (Return = +1.92 > +1.60 threshold).
  * **Independent Conflict (A != D):** Organism falls back to **VERIFY with +93.8% accuracy** (Return = +0.85).
- **Epistemic Laundering Solved:** Provenance tracking successfully discriminates direct copying, multi-hop laundering, redundant corroboration, and genuine independent confirmation in an unmasked, unforced Bayesian world.
========================================================================================================================
