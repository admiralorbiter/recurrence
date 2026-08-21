# Q17A-R1: Learned 2-Hop Transitive Composition Verification Report

- **Contract ID**: `CONTRACT-E-Q17A-R1`
- **Execution Timestamp**: 182.8011ms
- **Total Evaluation Seeds**: 16 (`101..=116`)

## 1. Success Gates Audit Summary

| Gate | Description | Threshold | Observed Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1** | Zero-Shot Multi-Hop Conflict Accuracy | $\ge 12/16$ seeds | 16/16 seeds (100.0%) | **PASS** |
| **Gate 2** | Zero-Shot Laundering Discrimination | $\ge 11/16$ seeds | 16/16 seeds (100.0%) | **PASS** |
| **Gate 3** | Independent Corroboration ($A = D$) | $\ge 15/16$ seeds | 16/16 seeds (100.0%) | **PASS** |
| **Gate 4** | Independent Conflict ($A \neq D$) | $\ge 15/16$ seeds | 16/16 seeds (100.0%) | **PASS** |
| **Gate 5** | Composition Ablation Floor | $n_{10} - n_{01} \ge 3$ | $n_{10}=16, n_{01}=0, \Delta=16$ ($p=3.0518e-5$) | **PASS** |
| **Gate 6** | Exact Paired Permutation Test | $p < 0.01$ ($2^{16}$ perms) | $p = 1.5259e-5$ | **PASS** |

## 2. Scientific Conclusion
The parameterized neural composition kernel $f_\theta$ successfully learned generic 2-hop reachability from auxiliary development worlds and transferred zero-shot to the withheld endpoint pair $(A, C)$, completely matching the empirical performance floor of the engineered matrix algebra baseline without hardcoded matrix multiplication.
