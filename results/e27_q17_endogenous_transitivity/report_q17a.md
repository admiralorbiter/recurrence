# Q17A-R1: Endogenous Transitive Composition Verification Report

- **Contract ID**: `CONTRACT-E-Q17A-R1`
- **Execution Timestamp**: 181.4226ms
- **Total Evaluation Seeds**: 16 (`101..=116`)

## 1. Success Gates Audit Summary

| Gate | Description | Threshold | Observed Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1** | Zero-Shot Multi-Hop Conflict Accuracy | $\ge 12/16$ seeds | 16/16 seeds (100.0%) | **PASS** |
| **Gate 2** | Zero-Shot Laundering Discrimination | $\ge 11/16$ seeds | 14/16 seeds (87.5%) | **PASS** |
| **Gate 3** | Independent Corroboration ($A = D$) | $\ge 15/16$ seeds | 16/16 seeds (100.0%) | **PASS** |
| **Gate 4** | Independent Conflict ($A \neq D$) | $\ge 15/16$ seeds | 15/16 seeds (93.8%) | **PASS** |
| **Gate 5** | Composition Ablation Floor | $n_{10} - n_{01} \ge 3$ | $n_{10}=14, n_{01}=0, \Delta=14$ ($p=1.2207e-4$) | **PASS** |
| **Gate 6** | Exact Paired Permutation Test | $p < 0.01$ ($2^{16}$ perms) | $p = 1.5259e-5$ | **PASS** |

## 2. Directional Controls & Lesion Audit
- **Directional Transposition**: Transposition collapses multi-hop conflict choice significantly, demonstrating strict reliance on transmission directionality $A \to B \to C$.
- **Path Breaks ($e_{AB}=0, e_{BC}=0$)**: Specifically collapses $a_{AC}$ without disturbing independent baseline ($A/D$).

## 3. Scientific Conclusion
The endogenous neural composition kernel $f_\theta$ successfully computes transitive reachability from specifically addressed local representations $(e_{AB}, e_{BC})$ without intermediate node search or algorithmic path traversal, transferring zero-shot to $(A, C)$ across matched stochastic Bayesian challenge episodes.
