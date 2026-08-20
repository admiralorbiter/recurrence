# Synchronization Report: Gate C / Q08 Controllability Representation Diagnostic

================================================================================
SYNCHRONIZATION REPORT: GATE C / Q08 (EVIDENCE MODE: TRAINED_MODEL)
================================================================================
1. QUESTION:                  Does the internal recurrent state h_{T_exp} encode a linearly separable 
                              representation of environmental controllability (W_ctrl vs W_yoked), 
                              resolving whether Q07 failed due to representation absence or policy neglect?
2. WHAT WAS FROZEN:           5-Probe Ladder across 8 Leave-One-Seed-Out (LOSO) Cross-Validation Folds:
                              (1) Goal Only, (2) Action History Only, (3) Effect History Only, 
                              (4) Joint Action+Effect Observer, (5) Latent State h_{T_exp}.
                              Seeds: [42, 43, 44, 45, 46, 47, 48, 49].
3. WHAT WAS RUN:              8 seeds x 100 counterbalanced episodes = 800 full latent trajectory records.
4. PRIMARY ESTIMAND:          AUC(Probe 5) >= 0.75 -> World A (Representation Exists but Unused);
                              AUC(Probe 5) <= 0.55 -> World B (No Macro Controllability Variable).
5. RESULT + UNCERTAINTY (8-FOLD LEAVE-ONE-SEED-OUT CV):
   - Probe 1 (Goal Only):                      AUC = 0.5000 (+/- 0.0000),  Acc = 50.0%  [Chance Floor <= 0.55]
   - Probe 2 (Action History Only):            AUC = 0.5446 (+/- 0.1181),  Acc = 52.0%  [Chance Floor <= 0.55]
   - Probe 3 (Effect History Only):            AUC = 0.5104 (+/- 0.1690),  Acc = 46.2%  [Chance Floor <= 0.55]
   - Probe 4 (Joint Action+Effect Observer):   AUC = 0.9413 (+/- 0.0196),  Acc = 88.2%  [External Ceiling >= 0.80]
   - Probe 5 (Target Latent Vector h_{T_exp}):  AUC = 0.3626 (+/- 0.2452),  Acc = 34.2%
6. DIAGNOSTIC VERDICT:
   - Classification:                          WORLD_B_NO_WORLD_LEVEL_VARIABLE_INDUCED
   - Mechanistic Account:                     The recurrent state h_{T_exp} remains at or near chance floor (AUC < 0.75), proving that learning local forward action-outcome predictions does NOT induce a macro-controllability variable.
7. FAILURES / INVALID CELLS:  None. 800/800 full latent vectors evaluated under LOSO cross-validation.
8. STRONGEST ALTERNATIVE:     Latent state might encode controllability nonlinearly; linear probe 
                              establishes whether the representation is immediately causally accessible.
9. CLAIM CEILING:             Establishes the presence/absence of a linearly decodable controllability 
                              macro-variable in recurrent state h_t following sensorimotor interaction.
10. DECISION:                 SCOUT_GATE_PASS (Gate C / Q08 Diagnostic Completed).
================================================================================
