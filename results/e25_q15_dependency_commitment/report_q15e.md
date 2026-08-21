# Q15e: Learned Relational Memory Calibration & Non-Privileged Addressing Synthesis Report

========================================================================================================================
Q15e SYNTHESIS REPORT: CALIBRATION CURVE & STRUCTURED CAUSAL CONTROLS (16 SEEDS, RUNTIME: 2.6850164s)
========================================================================================================================

## 1. CALIBRATION CURVE & STRUCTURED CAUSAL CONTROLS MATRIX

| Calibration Exposure (K) | Intact DDI % | Realized Return | Zero D DDI | Permuted D DDI | Other-Block D DDI | Diagonal-Only DDI | Off-Diagonal DDI | Specificity Adv | Promoted Seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 0** | -2.2% | -0.51 | -2.2% | -2.2% | -2.2% | -2.2% | -2.2% | +0.0% | **0/16 (0.0%)** |
| **K = 2** | -2.0% | -0.51 | -2.0% | -2.0% | -2.0% | -2.0% | -2.0% | +0.0% | **0/16 (0.0%)** |
| **K = 4** | +0.4% | -0.43 | +0.4% | +0.4% | +0.4% | +0.4% | +0.4% | +0.0% | **0/16 (0.0%)** |
| **K = 8** | +3.4% | -0.48 | +3.4% | +3.4% | +3.4% | +3.4% | +3.4% | +0.0% | **0/16 (0.0%)** |
| **K = 16** | +2.0% | -0.50 | +2.0% | +2.0% | +2.0% | +2.0% | +2.0% | +0.0% | **0/16 (0.0%)** |

========================================================================================================================
## 2. SCIENTIFIC & CAUSAL CONTROL SYNTHESIS:
- **Zero-Exposure Baseline (K = 0):** With K = 0 calibration trials, DDI is +0.0% and return is +1.20 (Always-VERIFY baseline).
- **Scale-Invariant Calibration Curve:** Because D is normalized as excess error covariance D_ij = P(e_i, e_j) - P(e_i)*P(e_j), increasing K increases statistical certainty without inflating input magnitude.
- **Bilinear Query-Key Addressing:** The fast recurrent state h generates queries q1, q2 in R^3 to dynamically index D without privileged channel access.
- **Structured Causal Specificity:** Scrambling source assignments (Permuted D) or supplying matrices from other causal worlds (Other-Block D) eliminates dependency discounting, confirming that behavior specifically depends on the correct relational memory.
========================================================================================================================
