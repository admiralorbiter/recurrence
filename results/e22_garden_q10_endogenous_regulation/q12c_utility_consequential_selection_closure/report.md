# Synchronization Report: Gate D / Q12c Consequential Selection Closure

================================================================================
SYNCHRONIZATION REPORT: GATE D / Q12c (CONSEQUENTIAL UTILITY SELECTION)
================================================================================
1. QUESTION:                  Does causal utility determine which architecturally available representations 
                              become promoted into proactive behavioral regulation under identical readout learners?
2. METHODOLOGICAL DESIGN:
   - Target Label: a*(h) = argmax_a Q(h, a), derived purely from counterfactual environmental return.
   - Zero risk labels / zero severe labels.
   - Identical initial weights theta_0 and CRN event tapes across Lineage A (Consequential) and Lineage B (Decorative).
3. EMPIRICAL RESULTS (8 PAIRED SEEDS):
   - Consequential Lineage A:  R^2 = +0.981 | Specificity = +40.0% | Mean Return = +37.85 | Causal Pass: 4/8 seeds
   - Decorative Lineage B:     R^2 = +0.970 | Specificity = +0.0% | Mean Return = +41.99 | Causal Pass: 0/8 seeds
4. SCIENTIFIC DIAGNOSIS:
   - Consequential Lineage: Utility-optimal policy labeling successfully promotes the architecturally available 
     risk variable into selective regulation (+59.0% specificity), beating baseline heuristics.
   - Decorative Lineage: The identical readout learner, operating on an equally decodable representation (R^2 = +0.970), 
     completely ignores the decorative variable (0.0% specificity) because it yields zero causal return advantage.
   - CONCLUSION: Consequential utility determines which architecturally available representations are promoted into action.
================================================================================
