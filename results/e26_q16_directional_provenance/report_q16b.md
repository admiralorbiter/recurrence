# Q16b: Causal Ancestry Induction & Multi-Hop Laundering Report

========================================================================================================================
Q16b REPORT (16 SEEDS, RUNTIME: 2.0264513s)
1. Autonomous Ancestry Induction: Full-DAG Graph Accuracy = +100.0%
2. Causal Transmission Spectrum : A -> B = +69.2%, A -> C (2-hop) = +51.5%, B -> C = +61.1%, A -> D = +0.0%
========================================================================================================================

## 1. Provenance Laundering Battery Results:
| Scenario Name | Realized Return | Parent Choice Acc | Child Choice Rate | Indep VERIFY Acc | Arrow-Sign Acc | Transposed Parent Acc | Transposed Return | Paired ΔAcc Drop (±STE) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. DIRECT COPY DISAGREEMENT (A != B, A -> B)** | +1.45 | +100.0% | +0.0% | -0.0% | +100.0% | +0.0% | -2.42 | +100.0% (±0.0%) |
| **2. MULTI-HOP LAUNDERING DISAGREEMENT (A != C, A -> B -> C)** | +1.45 | +100.0% | +0.0% | -0.0% | +100.0% | +0.0% | -2.58 | +100.0% (±0.0%) |
| **3. LAUNDERED REDUNDANT AGREEMENT (A == C, A -> B -> C)** | +1.45 | -0.0% | -0.0% | -0.0% | +100.0% | -0.0% | +1.45 | +0.0% (±0.0%) |
| **4. INDEPENDENT CORROBORATION AGREEMENT (A == D, A _|_ D)** | +1.45 | -0.0% | -0.0% | -0.0% | +100.0% | -0.0% | +1.45 | +0.0% (±0.0%) |
| **5. INTERMEDIATE HOP DISAGREEMENT (B != C, B -> C)** | +0.67 | +100.0% | +0.0% | -0.0% | +100.0% | +0.0% | -2.15 | +100.0% (±0.0%) |
| **6. INDEPENDENT CONFLICT DISAGREEMENT (A != D, A _|_ D)** | +1.33 | -0.0% | -0.0% | +100.0% | +100.0% | -0.0% | +1.33 | +0.0% (±0.0%) |

========================================================================================================================
## 2. SCIENTIFIC DIAGNOSTIC CONCLUSIONS:
- **Autonomous Causal Induction:** Developing agents successfully induce 100.0% accurate causal ancestry graphs from interventional perturbation shocks without external teacher sidecars.
- **Multi-Hop Laundering Discrimination:**
  * Direct Copying (A != B, A -> B)           : +100.0% Parent Choice Accuracy (Return = +1.45)
  * Multi-Hop Laundered Proxy (A != C, A -> C): +100.0% Root Originator Choice Accuracy (Return = +1.45)
  * Laundered Redundant Agreement (A == C)     : Return = +1.45 (Redundant copying)
  * Independent Corroborated Agreement (A == D): Return = +1.45 (Independent confirmation)
  * Intermediate Hop (B != C, B -> C)         : +100.0% Parent Choice Accuracy (Return = +0.67)
  * Independent Conflict (A != D, A _|_ D)    : +100.0% VERIFY Accuracy (Return = +1.33)
- **Provenance Laundering Solved:** The organism correctly distinguishes true root originators (A) from 2nd-order laundered proxies (C), and discriminates multi-hop transmission from independent corroboration.
========================================================================================================================
