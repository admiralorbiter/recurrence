# Synchronization Report: Gate C / Q08a Within-Organism Controllability Diagnostic

================================================================================
SYNCHRONIZATION REPORT: GATE C / Q08a (EVIDENCE MODE: OFFLINE_ANALYSIS)
================================================================================
1. QUESTION:                  Is environmental controllability (W_ctrl vs W_yoked) linearly decodable 
                              within each individual organism's native recurrent state h_{T_exp}, 
                              resolving the cross-organism raw-coordinate alignment confound in Q08?
2. WHAT WAS FROZEN:           Within-Organism Repeated Stratified 5-Fold Cross-Validation with 
                              1,000-draw Monte Carlo label permutation null tests across all 8 independent organisms:
                              (1) Goal Only, (2) Matched Action History (5 steps), (3) Matched Effect History (5 steps), 
                              (4) Matched Joint Observer (5 steps), (5) Native Latent State h_{T_exp}.
                              Source Dataset: `results/e20_garden_q08_representation/run_q08_representation_20260820_225836/raw_trials.jsonl`.
3. WHAT WAS RUN:              8 organisms x 100 trials = 800 trials analyzed across 50 CV iterations and 1,000 Monte Carlo permutations each.
4. PRIMARY ESTIMAND:          AUC(Probe 5) >= 0.80 across seeds -> Outcome A (Within-Organism Linear Decodability Confirmed);
                              AUC(Probe 5) <= 0.58 across seeds -> Outcome B (True Absence of Trajectory Information in h);
                              Mixed significance -> Outcome C (Solution Multiplicity).
5. RESULT + UNCERTAINTY (WITHIN-ORGANISM PROBE LADDER ACROSS 8 SEEDS):
   - Probe 1 (Goal Only):                      AUC = 0.4147 (+/- 0.0085),  Significant: 0/8 seeds (p > 0.05)  [Chance Floor]
   - Probe 2 (Matched Action History 5-step):  AUC = 0.6089 (+/- 0.1117),  Significant: 4/8 seeds  [Weak Motor Drift]
   - Probe 3 (Matched Effect History 5-step):  AUC = 0.8999 (+/- 0.0311),  Significant: 8/8 seeds (p <= 0.001)  [Effect History Signal]
   - Probe 4 (Matched Joint Observer 5-step):  AUC = 0.9206 (+/- 0.0193),  Significant: 8/8 seeds (p <= 0.001)  [Matched External Ceiling]
   - Probe 5 (Within-Organism Latent State h): AUC = 0.9160 (+/- 0.0182),  Significant: 8/8 seeds (p <= 0.001)  [HIGH LINEAR RECOVERABILITY]
6. PER-SEED LATENT STATE BREAKDOWN:
   - Seed 42: Matched Observer AUC = 0.9439 | Latent h AUC = 0.9407 (p_mc <= 0.001, Acc = 90.8%)
   - Seed 43: Matched Observer AUC = 0.9014 | Latent h AUC = 0.8966 (p_mc <= 0.001, Acc = 81.2%)
   - Seed 44: Matched Observer AUC = 0.9292 | Latent h AUC = 0.9222 (p_mc <= 0.001, Acc = 85.0%)
   - Seed 45: Matched Observer AUC = 0.8891 | Latent h AUC = 0.8909 (p_mc <= 0.001, Acc = 82.8%)
   - Seed 46: Matched Observer AUC = 0.9055 | Latent h AUC = 0.9081 (p_mc <= 0.001, Acc = 83.0%)
   - Seed 47: Matched Observer AUC = 0.9455 | Latent h AUC = 0.9239 (p_mc <= 0.001, Acc = 86.5%)
   - Seed 48: Matched Observer AUC = 0.9175 | Latent h AUC = 0.9029 (p_mc <= 0.001, Acc = 83.2%)
   - Seed 49: Matched Observer AUC = 0.9325 | Latent h AUC = 0.9423 (p_mc <= 0.001, Acc = 85.0%)
7. DIAGNOSTIC VERDICT:
   - Classification:                          OUTCOME_A_WITHIN_ORGANISM_LINEAR_RECOVERABILITY
   - Mechanistic Account:                     Within every independently trained organism, the controllable-vs-yoked 
                                              regime is strongly linearly decodable from its native recurrent state 
                                              (mean AUC = 0.9160, 8/8 seeds p <= 0.001), approximately matching 
                                              the history-matched external observer (0.9206).
8. THEORETICAL LESSON (REPRESENTATION-POLICY DISSOCIATION):
                                              Together with Q07, this demonstrates a clear representation-policy dissociation: 
                                              world-regime information is available in the recurrent state, while exploitation 
                                              remains weakly contingency-sensitive. The failure in Q07 was not due to 
                                              an absence of latent information, but an absence of behavioral coupling.
9. CLAIM CEILING:                             Establishes that joint sensorimotor/actor-critic training produces recurrent 
                                              states from which environmental regime is strongly linearly recoverable. 
                                              The assay does not yet establish a dedicated or invariant second-order macro-variable, 
                                              causal policy use, purely forward-prediction origin, or rotational basis alignment.
10. DECISION:                                 SCOUT_GATE_PASS (Gate C / Q08a Diagnostic Concluded).
================================================================================
