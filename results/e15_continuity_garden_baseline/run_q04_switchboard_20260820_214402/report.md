# Synchronization Report: Gate B / Q04 Hidden Switchboard Baseline

================================================================================
SYNCHRONIZATION REPORT: GATE B / Q04 (EVIDENCE MODE: TRAINED_MODEL)
================================================================================
1. QUESTION:                  Can a small recurrent organism use persistent latent state for 
                              delayed prediction in a partially observable world without target 
                              construct leakage?
2. WHAT WAS FROZEN:           Protocol: `docs/Gate_B_Environment_Contract.md`. Models: Oracle, 
                              Current-Input MLP (64-unit), History-Window MLP K=4 (64-unit), 
                              GRU Organism (64-unit, 20K params). Seeds: [42, 43, 44, 45, 46, 47, 48, 49].
3. WHAT WAS RUN:              8 seeds x 2 delay regimes (Short 8-16, Long 32-64 steps) x 4 models 
                              x 2 interventions. 500 train, 200 test episodes per seed.
4. PRIMARY ESTIMAND:          Delta(GRU - MLP) >= 0.30, Hist-MLP <= 0.55, GRU_reset <= 0.55, GRU_sham >= 0.85.
5. RESULT + UNCERTAINTY:
   - SHORT DELAY (8-16 steps):
     * Oracle Ceiling:          100.0% (+/- 0.0%)
     * Current-Input MLP:       51.4% (+/- 2.1%)
     * History-Window MLP K=4:  51.8% (+/- 1.8%)
     * GRU Organism:            100.0% (+/- 0.0%)
     * GRU Causal Reset:        53.8% (+/- 3.5%)
     * GRU Sham Reset:          100.0% (+/- 0.0%)
     * Recurrent Margin (MLP):  +48.6 percentage points
     * Recurrent Margin (Hist): +48.2 percentage points
   - LONG DELAY (32-64 steps):
     * Oracle Ceiling:          100.0% (+/- 0.0%)
     * Current-Input MLP:       50.1% (+/- 1.6%)
     * History-Window MLP K=4:  49.3% (+/- 0.8%)
     * GRU Organism:            100.0% (+/- 0.0%)
     * GRU Causal Reset:        51.3% (+/- 1.9%)
     * GRU Sham Reset:          100.0% (+/- 0.0%)
     * Recurrent Margin (MLP):  +49.9 percentage points
     * Recurrent Margin (Hist): +50.7 percentage points
6. CONTROL RESULTS:           Current-Input and History-Window (K=4) MLPs remain at chance (~50%), 
                              confirming no direct target-field leakage and that finite context cannot bridge delay. 
                              Sham reset preserves 100% accuracy, while true state reset collapses to ~50%, 
                              proving latent recurrent state is causally required.
7. FAILURES / INVALID CELLS:  None. All 8 seeds trained to stability with zero divergence.
8. STRONGEST ALTERNATIVE:     Supervised sequence learning might not transfer to RL control policy.
9. CLAIM CEILING:             Establishes minimal temporal POMDP memory substrate for Gate B; 
                              does not yet establish agency or self/world boundary.
10. DECISION:                 SCOUT_GATE_PASS (Gate B Baseline Validated across 8 seeds).
================================================================================
