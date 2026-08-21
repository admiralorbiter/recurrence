# Q17A Endogenous Transitive Composition Verification Report

- **Contract**: `CONTRACT-E-Q17A-R1`
- **Method**: Endogenous 2-hop Neural Composition Kernel $f_\theta(e_{AB}, e_{BC})$
- **Evaluation Topology**: 16 Random Seeds ($101 \dots 116$) over 300 stochastic Bayesian challenge episodes

## 1. Primary Empirical Outcomes

| Gate / Estimand | Pre-registered Floor | Observed Result | Status |
| :--- | :--- | :--- | :--- |
| **Gate 1 (Zero-Shot Multi-Hop Conflict)** | $\ge 12/16$ seeds ($75.0\%$) | **16/16 seeds** | **PASS** |
| **Gate 2 (Zero-Shot Laundering Discrimination)** | $\ge 11/16$ seeds ($68.75\%$) | **16/16 seeds** | **PASS** |
| **Gate 3 (Independent Corroboration)** | $\ge 15/16$ seeds ($93.75\%$) | **16/16 seeds** | **PASS** |
| **Gate 4 (Independent Conflict)** | $\ge 15/16$ seeds ($93.75\%$) | **16/16 seeds** | **PASS** |
| **Gate 5 (Composition Ablation Floor)** | $n_{10} - n_{01} \ge 3$ | **$n_{10}=16, n_{01}=0, \Delta=16$** ($p=3.0518e-5$) | **PASS** |
| **Gate 6 (Exact Permutation Test)** | $p < 0.01$ | **$p = 1.525879e-5$** | **PASS** |
| **Transposition Falsification ($A \neq C$)** | $\le 2/16$ seeds, return $< 0.00$ | **0/16 seeds, mean return = -0.995** | **PASS** |
| **Transposition Laundering ($A = C$)** | $\ge 10/16$ seeds | **16/16 seeds** | **PASS** |

## 2. Claim Ceiling
The neural composition kernel $f_\theta$ successfully composes 2-hop causal reachability from specifically addressed local evidence $(e_{AB}, e_{BC})$ without intermediate search loops or algorithmic path traversal under an engineered downstream decision mapping.
