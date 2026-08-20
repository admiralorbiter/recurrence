# Synchronization Report: Gate C / Q07b Harsher Economic Selection Pressure

================================================================================
SYNCHRONIZATION REPORT: GATE C / Q07b (EVIDENCE MODE: TRAINED_MODEL)
================================================================================
1. QUESTION:                  Does increasing developmental selection pressure (making always-trying 
                              actively loss-making: E[R] = -0.05 < 0.00) recruit the latent 
                              controllability representation into downstream behavioral action regulation?
2. WHAT WAS FROZEN:           Harsher Payoff Matrix: R_succ = +1.00, R_fail = -1.50, c_try = 0.30, R_abstain = 0.00.
                              Bayes-optimal threshold: P(W_ctrl | history) > 0.55.
                              Seeds: [42, 43, 44, 45, 46, 47, 48, 49].
3. WHAT WAS RUN:              8 seeds x 100 episodes = 800 trial records (JSONL).
4. PRIMARY ESTIMAND:          P(Abstain | W_yoked) >= 0.70, P(Exploit | W_ctrl) >= 0.70, Sensitivity >= 0.50.
5. RESULT + UNCERTAINTY:
   - Controllable Return E[R | W_ctrl]:        +0.46 (+/- 0.10)   [PASS: High Goal Attainment in W_ctrl]
   - Uncontrollable Return E[R | W_yoked]:     -0.48 (+/- 0.17)   [Unconditional TRY Losses Incurred]
   - P(Exploit | W_ctrl):                      100.0% (+/- 0.0%)
   - P(Abstain | W_yoked):                     2.0% (+/- 3.3%)    [MISS / GATE FAIL: Target >= 70%]
   - Contingency Sensitivity (Exploit Delta):  +2.0% (+/- 3.3%)    [MISS / GATE FAIL: Target >= 50%]
   - Exploit Success Rate in W_ctrl:           90.2%
6. PER-SEED BREAKDOWN:
   - Seed 42: P(Exploit|Ctrl) = 100.0% | P(Abstain|Yoked) = 2.0% | Sensitivity = +2.0% | Return(Ctrl) = +0.55
   - Seed 43: P(Exploit|Ctrl) = 100.0% | P(Abstain|Yoked) = 10.0% | Sensitivity = +10.0% | Return(Ctrl) = +0.45
   - Seed 44: P(Exploit|Ctrl) = 100.0% | P(Abstain|Yoked) = 4.0% | Sensitivity = +4.0% | Return(Ctrl) = +0.40
   - Seed 45: P(Exploit|Ctrl) = 100.0% | P(Abstain|Yoked) = 0.0% | Sensitivity = +0.0% | Return(Ctrl) = +0.25
   - Seed 46: P(Exploit|Ctrl) = 100.0% | P(Abstain|Yoked) = 0.0% | Sensitivity = +0.0% | Return(Ctrl) = +0.50
   - Seed 47: P(Exploit|Ctrl) = 100.0% | P(Abstain|Yoked) = 0.0% | Sensitivity = +0.0% | Return(Ctrl) = +0.60
   - Seed 48: P(Exploit|Ctrl) = 100.0% | P(Abstain|Yoked) = 0.0% | Sensitivity = +0.0% | Return(Ctrl) = +0.45
   - Seed 49: P(Exploit|Ctrl) = 100.0% | P(Abstain|Yoked) = 0.0% | Sensitivity = +0.0% | Return(Ctrl) = +0.45
7. FAILURES / INVALID CELLS:  None. 800/800 trials recorded cleanly.
8. THEORETICAL CONCLUSION:    In this training architecture, substantially increasing the economic penalty 
                              for acting without control is INSUFFICIENT by itself to produce reliable behavioral 
                              abstention. The failure of recruitment is an empirical null for this actor-critic training regime.
9. CLAIM CEILING:             Establishes that stronger economic incentives alone do not automatically overcome 
                              the optimization barrier to coupling latent representations into terminal policy actions.
10. DECISION:                 PRIMARY_GATE_FAIL / EMPIRICAL_NULL (Promote Q09b Mechanistic Closure).
================================================================================
