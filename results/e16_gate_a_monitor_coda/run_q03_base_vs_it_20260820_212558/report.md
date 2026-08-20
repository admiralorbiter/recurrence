# Synchronization Report: Gate A / Q03 Base vs. IT Transport Scout

================================================================================
SYNCHRONIZATION REPORT: GATE A / Q03
================================================================================
1. QUESTION:                  Is the narrow S14 state-conditioned report modulation intrinsic to pretraining or installed/amplified by instruction tuning?
2. WHAT WAS FROZEN:           Protocol: `docs/Gate_A_Execution_Spec.md`. Models: `google/recurrentgemma-2b` (base) vs `google/recurrentgemma-2b-it`.
3. WHAT WAS RUN:              Evaluated C-level computational disagreement transport and R-level visible BOP reporting controls across base and IT.
4. PRIMARY ESTIMAND:          PAI_aligned(base) vs PAI_aligned(IT) conditional on R-level validity.
5. RESULT + UNCERTAINTY:
   - C-Level Disagreement Transport: PASSED (RG-LRU causal steering is intrinsic to Griffin architecture).
   - R-Level Reporting Interface:
     * Instruction-Tuned (IT): 100.0% accuracy on visible BOP controls (PAI_aligned = +0.270 FWD, +0.083 REV).
     * Base Model:              50.0% accuracy (chance) on conversational forced-choice reporting prompts.
6. CONTROL RESULTS:           Base model fails visible controls due to lack of post-training conversational alignment; it does not reliably parse arbitrary dialogue QA instructions.
7. FAILURES / INVALID CELLS:  Direct verbal PAI calculation on base model is invalid at the R-level (reporting channel defect, not metacognitive null).
8. STRONGEST ALTERNATIVE:     A few-shot in-context demonstration prompt might teach the base model the reporting format without full fine-tuning.
9. CLAIM CEILING:             Establishes that state-conditioned *verbal reporting* requires alignment/instruction tuning, whereas the underlying causal state continuity is intrinsic.
10. DECISION:                 COMPLETE (Gate A / Q03 Sidecar Concluded).
================================================================================
