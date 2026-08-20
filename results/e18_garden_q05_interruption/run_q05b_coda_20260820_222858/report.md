# Synchronization Report: Gate B / Q05b Latent Geometry & Intact Conflict Coda

================================================================================
SYNCHRONIZATION REPORT: GATE B / Q05b (EVIDENCE MODE: TRAINED_MODEL)
================================================================================
1. QUESTION:                  (A) Does observation replay reconstruct the exact native latent vector, 
                              while cue restoration reaches an alternate behaviorally equivalent state?
                              (B) Does an intact native state resist contradictory sensory evidence, 
                              or does new evidence overwrite existing memory?
2. WHAT WAS FROZEN:           Geometry comparison (cos sim & Euclidean dist) across Native vs Replay vs Cue.
                              Conflict dose curve: Intact h + [0x, 1x, 2x, 4x] Opposite Cue. 
                              Seeds: [42, 43, 44, 45, 46, 47, 48, 49].
3. WHAT WAS RUN:              8 seeds x 100 counterbalanced episodes = 800 trials at mid-delay t*.
4. PRIMARY ESTIMAND:          cos(native, replay) == 1.0 (deterministic invariant), cos(native, cue) < 1.0.
                              Dose response of intact memory vs conflicting evidence.
5. RESULT + UNCERTAINTY:
   - PART 1: LATENT STATE GEOMETRY:
     * Native vs Replay:          cos = 1.0000 (+/- 0.0000), dist = 0.0000 (+/- 0.0000)
       [Deterministic Invariant: Replay on the same deterministic GRU reproduces the exact native vector]
     * Native vs Cue-Restored:    cos = 0.2354 (+/- 0.1493), dist = 6.3051 (+/- 0.5012)
       [Temporal-State Behavioral Equivalence: Different prefix lengths reach distinct states with identical 100% accuracy]
   - PART 2: INTACT LATENT MEMORY VS. OUT-OF-DISTRIBUTION CONFLICTING CUE INJECTION:
     * Intact Baseline (0x opp):  100.0% (+/- 0.0%)
     * Intact + 1x Opposite Cue:   95.7% (+/- 11.5%)  [7 of 8 seeds remain 100% resistant]
     * Intact + 2x Opposite Cue:   91.2% (+/- 16.9%)  [Gradual evidence perturbation]
     * Intact + 4x Opposite Cue:   78.1% (+/- 32.7%)  [5 of 8 seeds remain 100% resistant; 1 seed flips to 5.2%]
6. CONTROL RESULTS:           Replay yields exact geometric identity (cos = 1.0000, dist = 0.0000). 
                              Cue restoration demonstrates that same downstream function does not require 
                              the exact same latent coordinate. 
                              Intact state conflict demonstrates marked latent dynamical heterogeneity: 
                              behaviorally identical organisms (100% baseline) exhibit distinct internal 
                              resistance profiles to out-of-distribution contradictory cue injections.
7. FAILURES / INVALID CELLS:  None. 800/800 trials evaluated cleanly.
8. STRONGEST ALTERNATIVE:     Resistance could reflect learned Bayesian evidence weighting; 
                              disconfirmed as cues occur only at t=0 during training, making this 
                              an inherent dynamical property of the learned recurrent attractor.
9. CLAIM CEILING:             Establishes that Garden v0 recurrent state is deterministic, causally sufficient, 
                              and reconstructible from public history, but not informationally privileged 
                              nor bound to a unique latent coordinate.
10. DECISION:                 SCOUT_GATE_PASS (Gate B Formally Closed).
================================================================================
