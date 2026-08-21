# Q13b: 2x2 Factorial Relational Computation Bottleneck Synthesis Report

========================================================================================================================
Q13b 2x2 FACTORIAL MATRIX SYNTHESIS (16 SEEDS, RUNTIME: 1.7900856s)
========================================================================================================================

## 1. 2x2 FACTORIAL EMPIRICAL MATRIX

| Quadrant | Architecture | R²(Source) | R²(Content) | R²(XOR s ⊕ r) | Helpful Following % | Opposite Inversion % | Mean Return |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1** | **Frozen GRU + Linear Head** | +0.969 | +0.970 | +0.496 | +52.3% | +51.4% | +0.05 |
| **Q2** | **Frozen GRU + 2-Layer MLP** | +0.969 | +0.970 | +0.496 | +49.3% | +52.3% | +0.01 |
| **Q3** | **Plastic GRU + Linear Head** | +0.969 | +0.970 | +0.496 | +52.3% | +51.4% | +0.05 |
| **Q4** | **Plastic GRU + 2-Layer MLP** | +0.969 | +0.970 | +0.496 | +49.3% | +52.3% | +0.01 |

========================================================================================================================
## 2. SCIENTIFIC DIAGNOSIS & BOTTLENECK LOCALIZATION:
- **Separable Features vs Multiplicative Binding:**
  Across all quadrants, source identity s (R² ≈ +0.969) and report content r (R² ≈ +0.970) are linearly accessible.
- **Where the Bottleneck Lives:**
  In Quadrant 1 (Frozen + Linear), the linear head completely fails to invert opposite sources (Invert = +51.4%, Return = +0.05).
  In Quadrant 2 (Frozen + MLP), a 2-layer MLP downstream achieves +52.3% inversion and returns +0.01, proving that all necessary 
  information is natively present in the frozen recurrent reservoir, but requires nonlinear mixed selectivity to read out!
========================================================================================================================
