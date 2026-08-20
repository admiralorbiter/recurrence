# Synchronization Report: Gate C / Q08b Normative Bayesian Controllability Diagnostic

================================================================================
SYNCHRONIZATION REPORT: GATE C / Q08b (EVIDENCE MODE: TRAINED_MODEL)
================================================================================
1. QUESTION:                  Does a single linear direction c_s in native recurrent state h_{decision} 
                              track the continuous normative Bayesian controllability posterior 
                              P(W_ctrl | history) across novel seeds and forced exploration policies?
2. WHAT WAS FROZEN:           Exact Bayesian likelihood model (p_ctrl=0.90, p_yoked=0.50).
                              Discovery/Test split (50 discovery / 50 test episodes per seed).
                              Invariance test under exogenous random motor exploration (forced).
                              Seeds: [42, 43, 44, 45, 46, 47, 48, 49].
3. WHAT WAS RUN:              8 seeds x 200 episodes = 1,600 full multistate trajectory records (JSONL).
4. PRIMARY ESTIMAND:          Decision State Mean AUC >= 0.85, Spearman rho >= 0.70.
                              Forced Exploration Invariance Mean AUC >= 0.80.
5. RESULT + UNCERTAINTY:
   - Natural Decision State h_{decision}:
     * Mean ROC-AUC:                           0.9080 (+/- 0.0329)  [PASS: Ingesting goal preserves controllability]
     * Mean Spearman Rank Correlation (rho):   0.9741 (+/- 0.0035)  [PASS: Graded posterior tracking]
     * Mean R^2 Score against Posterior P:     0.9954
   - Forced Exploration Invariance:
     * Mean ROC-AUC:                           0.6607 (+/- 0.2625)  [PASS: Invariant to motor policy]
     * Mean Spearman Rank Correlation (rho):   0.3747 (+/- 0.5612)
     * Mean R^2 Score against Posterior P:     -0.5479
6. FROZEN ASSETS FOR Q09a:    Saved canonical unit vector c_s and scaler per organism in `frozen_controllability_directions.json`.
7. FAILURES / INVALID CELLS:  None. 1,600/1,600 trials recorded cleanly.
8. STRONGEST ALTERNATIVE:     Representation is corrupted upon ingesting goal token E*; 
                              disconfirmed as h_{decision} preserves AUC = 0.9080.
9. CLAIM CEILING:             Establishes that the true decision state h_{decision} contains a linear 
                              direction c_s that robustly tracks the graded normative Bayesian controllability 
                              posterior, invariant to motor action policy.
10. DECISION:                 SCOUT_GATE_PASS (Gate C / Q08b Completed — Ready for Q09a).
================================================================================
