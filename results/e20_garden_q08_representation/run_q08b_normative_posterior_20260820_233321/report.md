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
4. PRIMARY ESTIMAND:          Decision State Mean AUC >= 0.85, Spearman rho >= 0.70 (Natural).
                              Forced Exploration Invariance Mean AUC >= 0.80 (Invariance Gate).
5. RESULT + UNCERTAINTY:
   - Natural Decision State h_{decision}:
     * Mean ROC-AUC:                           0.9080 (+/- 0.0329)  [PASS: Linear decodability preserved post-goal]
     * Mean Spearman Rank Correlation (rho):   0.9741 (+/- 0.0035)  [PASS: Near-perfect graded match-count tracking]
     * Mean R^2 Score against Posterior P:     0.9954
   - Forced Exploration Invariance:
     * Mean ROC-AUC:                           0.6607 (+/- 0.2625)  [MISS / GATE FAIL: Target >= 0.80]
     * Mean Spearman Rank Correlation (rho):   0.3747 (+/- 0.5612)  [High Cross-Seed Heterogeneity]
     * Mean R^2 Score against Posterior P:     -0.5479
6. THEORETICAL IMPLICATION (POLICY-DEPENDENT CONTROLLABILITY):
                              Within the organism's native policy regime, the decision state contains a 
                              remarkably precise linear correlate of the action-effect agreement statistic. 
                              However, that representation does not reliably transfer under exogenous action 
                              selection (AUC = 0.661, rho = 0.375), proving that the internal representation 
                              is policy-dependent (P(control | H, pi_self)) rather than abstractly observer-invariant.
7. FROZEN ASSETS FOR Q09a:    Saved canonical unit vector c_s and scaler per organism in `frozen_controllability_directions.json`.
8. FAILURES / INVALID CELLS:  None. 1,600/1,600 trials recorded cleanly.
9. CLAIM CEILING:             Establishes that recurrent organisms develop a precise, graded controllability 
                              statistic under their native interaction policy, but this representation is 
                              policy-dependent and does not constitute a policy-invariant macro-variable.
10. DECISION:                 PARTIAL_SCOUT_PASS (Natural Graded Statistic Validated / Invariance Gate Missed).
================================================================================
