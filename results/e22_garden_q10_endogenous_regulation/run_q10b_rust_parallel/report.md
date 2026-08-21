# Synchronization Report: Gate D / Q10b Reservoir Scout & Architectural Availability

================================================================================
SYNCHRONIZATION REPORT: GATE D / Q10b (EVIDENCE MODE: FIXED_RESERVOIR_RUST_SCOUT)
================================================================================
1. QUESTION:                  Under structured precursor evidence, what temporal information is available in 
                              a recurrent reservoir, and does a linear-head TD policy recruit it for anticipatory regulation?
2. WHAT WAS EXECUTED:         - Garden Rust Reservoir Engine v0.1.
                              - Fixed randomly initialized GRU reservoir (untrained recurrent weights).
                              - Linear policy and value readout training across 8 seeds x 3,200 episodes.
                              - Checkpoint battery across T in [0, 25, 50, 100, 200, 400, 800, 1600, 3200].
3. CORE EMPIRICAL FINDINGS:
   - ARCHITECTURAL AVAILABILITY:               R^2(h_0 -> q_t) = 0.41 at T=0 before any task training.
                                               A random recurrent reservoir natively preserves historical precursor 
                                               evidence across blank steps.
   - DEVELOPMENTAL INVARIANCE:                 R^2(h_T -> q_t) remains ~0.41 across all T because recurrent weights were fixed.
   - MOTOR COMPETENCE:                         100% target hit rate acquired on linear readout by T=25.
   - RECRUITMENT NULL:                         Policy readout does not recruit anticipatory maintenance (p_maint = 0.0).
4. CONCEPTUAL CLASSIFICATION:
   - Architectural Availability != Learned Internal Model != Behavioral Recruitment.
   - Decodability of risk at T=0 establishes temporal basis availability in random recurrent reservoirs, 
     not learned developmental emergence.
5. TRANSITION TO Q10c:        Full Plastic GRU (BPTT) vs. Paired Frozen Reservoir vs. Feedforward Control.
================================================================================
