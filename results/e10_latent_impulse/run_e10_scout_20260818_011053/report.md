# E10 Latent Impulse Response & Store Localization Report

**Model Target:** `reference_random_recurrentgemma` (Reference Model: True)
**Run Path:** `results\e10_latent_impulse\run_e10_scout_20260818_011053`

> [!NOTE]
> **Engineering Scout Status:** This dataset evaluates the lightweight reference model architecture
> to verify instrumentation sensitivity, residency boundary transitions, non-monotonic dynamics,
> and sham noise floor. Pretrained parameter values are evaluated in subsequent live runs.

## 1. Multi-Store Empirical Retention & 50% Thresholds

| Filler Regime | RGLRU First <50% | RGLRU Sustained <50% | Conv First <50% | KV First <50% | RGLRU Log-AUC | KV Log-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **constant** | L=16 | L=16 | L=3 | L=15 | 2.892 | 1.976 |
| **interfering** | L=15 | L=15 | L=3 | L=15 | 2.947 | 1.974 |
| **natural** | L=16 | L=16 | L=3 | L=15 | 2.912 | 1.956 |
| **random** | L=17 | L=17 | L=3 | L=15 | 2.824 | 1.959 |

## 2. Dynamic Trajectories Across Tested Lags

| Lag $L$ | Conv Res? | KV Res? | RGLRU Ret (Const) | KV Ret (Const) | RGLRU Ret (Interf) | KV Ret (Interf) | $D_{\text{JS}}$ (Const) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Yes | Yes | 1.000 | 1.000 | 1.000 | 1.000 | 0.0309 |
| 1 | Yes | Yes | 1.162 | 0.900 | 1.148 | 0.906 | 0.0121 |
| 2 | Yes | Yes | 1.189 | 0.827 | 1.218 | 0.830 | 0.0101 |
| 3 | No | Yes | 0.976 | 0.769 | 1.052 | 0.763 | 0.0095 |
| 4 | No | Yes | 0.889 | 0.723 | 0.931 | 0.709 | 0.0091 |
| 8 | No | Yes | 0.726 | 0.596 | 0.702 | 0.583 | 0.0043 |
| 15 | No | No | 0.533 | 0.014 | 0.464 | 0.025 | 0.0044 |
| 16 | No | No | 0.490 | 0.013 | 0.445 | 0.024 | 0.0000 |
| 17 | No | No | 0.433 | 0.009 | 0.416 | 0.020 | 0.0000 |
| 32 | No | No | 0.128 | 0.004 | 0.169 | 0.005 | 0.0000 |
| 64 | No | No | 0.043 | 0.001 | 0.081 | 0.002 | 0.0000 |
| 128 | No | No | 0.011 | 0.000 | 0.022 | 0.001 | 0.0000 |
| 256 | No | No | 0.002 | 0.000 | 0.005 | 0.000 | 0.0000 |
| 512 | No | No | 0.000 | 0.000 | 0.001 | 0.000 | 0.0000 |

## 3. Epistemic Assessment & Structural Findings

1. **Direct Residency vs Downstream Divergence:** After direct event residency ends, branch-specific differences persist in later Conv and KV representations, consistent with historical information being propagated through the hybrid recurrent system.
2. **Input-Dependent Recurrent Trajectories:** Distinct filler sequences modulate the persistence and decay rate of latent states across the dynamic lag spectrum.
3. **Zero Sham Floor:** Identical $A_1 / A_2$ controls confirm an empirical measurement floor of $0.00000000$ for scale-relative distance and Jensen-Shannon divergence.
