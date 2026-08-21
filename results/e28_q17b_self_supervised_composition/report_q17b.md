# Q17B Self-Supervised Endogenous Composition Experiment Report (Matched Control Hardened)

- **Protocol**: `CONTRACT-E-Q17B`
- **Total Seeds**: 16
- **Dataset Size per Seed**: 2500 Empirical Samples
- **Matched Negative Control**: Exact Permuted Pairings (Target Sum: 10072 Intact vs 10072 Shuffled)
- **Execution Duration**: 47.15ms
- **All Gates Passed**: **PASS**

## Empirical Gate Results

| Gate / Estimand | Pre-registered Floor | Observed Empirical Result | Verdict |
| :--- | :--- | :--- | :--- |
| **Gate 1 (Zero-Shot Multi-Hop Conflict)** | >= 10/16 seeds | **16/16 seeds** | PASS |
| **Gate 2 (Laundering Discrimination)** | >= 10/16 seeds | **16/16 seeds** | PASS |
| **Gate 3 (Temporal Shuffle Control Superiority)** | n10 - n01 >= 3, p < 0.05 | **n10=13, n01=0, Delta=13, p=1.2207e-4** | PASS |
| **Gate 4 (Directional Transposition Falsification)** | <= 2/16 seeds, return < 0.00 | **0/16 seeds passed, mean return = -1.000** | PASS |
| **Gate 5 (Transposition Laundering Invariant)** | >= 10/16 seeds | **16/16 seeds** | PASS |
| **Gate 6 (Mechanistic Path-Break Continuous Permutation)** | p < 0.01 | **p = 1.5259e-5** | PASS |
