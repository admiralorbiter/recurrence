# Synchronization Report: Gate C / Q07 Learned Functional Controllability

================================================================================
SYNCHRONIZATION REPORT: GATE C / Q07 (EVIDENCE MODE: TRAINED_MODEL)
================================================================================
1. QUESTION:                  Does interaction with sensorimotor contingencies cause a recurrent 
                              organism to learn an internal distinction between events it can causally 
                              control and events that merely happen around it, without explicit agency supervision?
2. WHAT WAS FROZEN:           Protocol: `docs/Q07_Functional_Controllability_Spec.md`.
                              Payoff matrix: +0.90 (success), -1.10 (failure), 0.00 (abstain).
                              Objective: Forward dynamics prediction + actor-critic return learning (zero policy labels).
                              Seeds: [42, 43, 44, 45, 46, 47, 48, 49].
3. WHAT WAS RUN:              8 seeds x 100 counterbalanced evaluation episodes = 800 trial records (JSONL).
4. PRIMARY ESTIMAND:          E[R | W_ctrl] >= 0.50, P(Abstain | W_yoked) >= 0.70, Sensitivity >= 0.50,
                              Observer 1 <= 0.55, Observer 2 <= 0.55, Observer 3 >= 0.80.
5. RESULT + UNCERTAINTY:
   - Controllable Return E[R | W_ctrl]:        +0.73 (+/- 0.05)   [PASS: Goal Mastery at Physical Upper Bound]
   - Uncontrollable Return E[R | W_yoked]:     -0.01 (+/- 0.18)   [Sub-Zero Regime]
   - P(Exploit | W_ctrl):                      100.0% (+/- 0.0%)
   - P(Abstain | W_yoked):                       3.5% (+/- 9.3%)  [MISS: Target >= 70%]
   - Contingency Sensitivity (Exploit Delta):  +3.5% (+/- 9.3%)  [MISS: Target >= 50%]
   - Exploit Success Rate in W_ctrl:            91.5% (+/- 2.9%)  [Matches 90% physical action-effect reliability]
6. THREE-TIER OBSERVER SANITY LADDER:
   - Observer 1 (Goal Only):                    50.0%  [Target: <= 55% -> PASS: No sensory artifact]
   - Observer 2 (Effect History Only):          37.5%  [Target: <= 55% -> PASS: Matched marginals]
   - Observer 3 (Joint Action+Effect History):  88.0%  [Target: >= 80% -> PASS: True contingency signal exists]
7. FAILURES / INVALID CELLS:  None. 800/800 individual trial rows committed to `raw_trials.jsonl`.
8. STRONGEST SURVIVING EXPLANATION:
                              The organism learned a robust first-order forward control law (a -> E), 
                              enabling 91.5% goal attainment in W_ctrl. However, because always trying 
                              yields a positive overall expected return (+0.36 > 0.00), the policy adopted 
                              a coarse "always try" heuristic and failed to learn second-order behavioral 
                              arbitration, despite an external observer extracting 88.0% contingency signal.
9. CLAIM CEILING:             Q07 establishes learned goal-directed action-effect control in W_ctrl and 
                              verifies that controllability is strongly inferable from joint action-effect history. 
                              However, the trained organisms did not use that information to condition 
                              exploitation on environmental controllability; the preregistered behavioral 
                              selectivity criteria were not met.
10. DECISION:                 PARTIAL / PRIMARY_GATE_MISS (PROMOTE Q08 AS DIAGNOSTIC).
================================================================================
