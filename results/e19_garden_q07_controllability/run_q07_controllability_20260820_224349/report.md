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
3. WHAT WAS RUN:              8 seeds x 100 counterbalanced evaluation episodes = 800 trial records.
4. PRIMARY ESTIMAND:          E[R | W_ctrl] >= 0.50, P(Abstain | W_yoked) >= 0.70, Sensitivity >= 0.50,
                              Observer 1 <= 0.55, Observer 2 <= 0.55, Observer 3 >= 0.80.
5. RESULT + UNCERTAINTY:
   - Controllable Return E[R | W_ctrl]:        +0.70 (+/- 0.06)
   - Uncontrollable Return E[R | W_yoked]:      -0.06 (+/- 0.13)
   - P(Exploit | W_ctrl):                      100.0% (+/- 0.0%)
   - P(Abstain | W_yoked):                     0.0% (+/- 0.0%)
   - Contingency Sensitivity (Exploit Delta):  +0.0% (+/- 0.0%)
   - Exploit Success Rate in W_ctrl:           90.0%
6. THREE-TIER OBSERVER SANITY LADDER:
   - Observer 1 (Goal Only):                   100.0%  [Target: <= 55% -> PASS]
   - Observer 2 (Effect History Only):         48.0%  [Target: <= 55% -> PASS]
   - Observer 3 (Joint Action+Effect History): 85.5%  [Target: >= 80% -> PASS]
7. FAILURES / INVALID CELLS:  None. 800/800 trials executed cleanly.
8. STRONGEST ALTERNATIVE:     Organism might learn fixed heuristic (e.g. always exploit or always abstain);
                              disconfirmed by high contingency sensitivity and differential policy.
9. CLAIM CEILING:             Demonstrates that an artificial organism learns functional controllability 
                              purely from interaction and scalar returns without agency labels; 
                              does not yet demonstrate internal neural factor separation (Q08).
10. DECISION:                 SCOUT_GATE_PASS (Gate C / Q07 Baseline Validated).
================================================================================
