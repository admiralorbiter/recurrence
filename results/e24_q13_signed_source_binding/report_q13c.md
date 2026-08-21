# Q13c: Definitive 2x2 Factorial Relational Computation Synthesis Report

========================================================================================================================
Q13c 2x2 FACTORIAL MATRIX SYNTHESIS (16 SEEDS, RUNTIME: 2.6362875s)
========================================================================================================================

## 1. CONTROL VERIFICATION
- **Sanity Check:** `[s, r] -> 2-Layer MLP -> XOR` Accuracy = **100.0%** (Baseline verification of MLP solver).

## 2. 2x2 FACTORIAL EMPIRICAL MATRIX

| Quadrant | Architecture | R²(Source) | R²(Content) | R²(Ridge XOR) | MLP Probe h→XOR | Helpful Following % | Opposite Inversion % | Return | ||Δθ_GRU|| |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1** | **Frozen GRU + Linear Head** | +0.969 | +0.970 | +0.496 | +49.9% | +54.0% | +51.0% | +0.02 | +0.000 |
| **Q2** | **Frozen GRU + 2-Layer MLP** | +0.969 | +0.970 | +0.496 | +49.9% | +53.8% | +51.4% | +0.04 | +0.000 |
| **Q3** | **Plastic GRU + Linear Head** | +0.969 | +0.970 | +0.485 | +49.9% | +54.0% | +51.0% | +0.02 | +0.069 |
| **Q4** | **Plastic GRU + 2-Layer MLP** | +0.969 | +0.970 | +0.502 | +49.9% | +53.8% | +51.4% | +0.04 | +0.025 |

========================================================================================================================
## 3. SCIENTIFIC LOCALIZATION:
- **Optimizer-Independent Probes on Frozen h:**
  Linear Ridge regression achieves R² = +0.496 on XOR.
  An optimizer-independent 2-layer MLP classifier on frozen h achieves **49.9% accuracy** on decoding XOR (s ⊕ r)!
- **Plasticity Effects:**
  Plastic GRU updates displace weights by ||Δθ|| = +0.069 (Q3) and +0.025 (Q4).
========================================================================================================================
