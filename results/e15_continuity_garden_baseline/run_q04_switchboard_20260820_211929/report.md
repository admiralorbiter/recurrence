# Synchronization Report: Gate B / Q04 Hidden Switchboard Baseline

================================================================================
SYNCHRONIZATION REPORT: GATE B / Q04
================================================================================
1. QUESTION:                  Can a small recurrent organism use persistent latent state for delayed prediction in a partially observable world without target construct leakage?
2. WHAT WAS FROZEN:           Protocol: `docs/Gate_B_Environment_Contract.md`. Models: Oracle, Feedforward MLP (64-unit), GRU Organism (64-unit). Seeds: [42, 43, 44, 45, 46, 47, 48, 49].
3. WHAT WAS RUN:              8 seeds x 2 delay regimes (Short 8-16, Long 32-64) x 4 models/interventions. 500 train episodes, 200 test episodes per seed.
4. PRIMARY ESTIMAND:          Delta(GRU - MLP) >= 0.30 and GRU_reset <= 0.55.
5. RESULT + UNCERTAINTY:
   - SHORT DELAY (8-16):
     * Oracle:               100.0% (+/- 0.0%)
     * Feedforward MLP:      51.4% (+/- 2.1%)
     * GRU Organism:         100.0% (+/- 0.0%)
     * GRU Causal Reset:     48.4% (+/- 4.6%)
     * Recurrent Margin:     +48.6 percentage points
   - LONG DELAY (32-64):
     * Oracle:               100.0% (+/- 0.0%)
     * Feedforward MLP:      50.1% (+/- 1.6%)
     * GRU Organism:         100.0% (+/- 0.0%)
     * GRU Causal Reset:     50.2% (+/- 2.1%)
     * Recurrent Margin:     +49.9 percentage points
6. CONTROL RESULTS:           Feedforward remains at chance (50%), confirming zero environment leakage (Q06). GRU Reset collapses to 50%, proving latent state is causally required.
7. FAILURES / INVALID CELLS:  None. All 8 seeds trained to stability with zero optimizer divergence.
8. STRONGEST ALTERNATIVE:     Supervised sequence learning might not transfer to RL control policy. (Addressed in Gate B bridge).
9. CLAIM CEILING:             Establishes minimal temporal POMDP memory substrate for Gate B; does not yet establish agency or self/world boundary.
10. DECISION:                 PROMOTE (Gate B / Q04 Gate Passed).
================================================================================
