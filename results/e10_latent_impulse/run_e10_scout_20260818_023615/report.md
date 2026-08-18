# E10 Latent Impulse Response & Store Localization Report

**Model Target:** `google/recurrentgemma-2b` (Reference Model: False)
**Run Path:** `results\e10_latent_impulse\run_e10_scout_20260818_023615`

## 1. Multi-Store Empirical Retention & 50% Thresholds

| Filler Regime | RGLRU First <50% | RGLRU Sustained <50% | Conv First <50% | KV First <50% | RGLRU Log-AUC | KV Log-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **constant** | L=8 | L=8 | L=1 | L=64 | 3.258 | 4.212 |
| **interfering** | L=8 | L=8 | L=1 | L=128 | 3.039 | 4.767 |
| **natural** | L=8 | L=8 | L=1 | L=128 | 2.889 | 4.733 |
| **random** | L=16 | L=16 | L=1 | L=128 | 2.762 | 4.623 |

## 2. Dynamic Trajectories Across Tested Lags

| Lag $L$ | Conv Res? | KV Res? | RGLRU Ret (Const) | KV Ret (Const) | RGLRU Ret (Interf) | KV Ret (Interf) | $D_{\text{JS}}$ (Const) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Yes | Yes | 1.000 | 1.000 | 1.000 | 1.000 | 0.0077 |
| 1 | Yes | Yes | 0.867 | 0.990 | 0.911 | 0.983 | 0.1799 |
| 2 | Yes | Yes | 0.759 | 0.968 | 0.765 | 0.954 | 0.1721 |
| 3 | No | Yes | 0.692 | 0.933 | 0.724 | 0.939 | 0.2428 |
| 4 | No | Yes | 0.631 | 0.899 | 0.796 | 0.924 | 0.1769 |
| 8 | No | Yes | 0.477 | 0.800 | 0.490 | 0.855 | 0.0348 |
| 16 | No | Yes | 0.398 | 0.662 | 0.473 | 0.783 | 0.0172 |
| 32 | No | Yes | 0.317 | 0.503 | 0.332 | 0.700 | 0.0030 |
| 64 | No | Yes | 0.284 | 0.414 | 0.254 | 0.584 | 0.0003 |
| 128 | No | Yes | 0.261 | 0.335 | 0.179 | 0.463 | 0.0003 |
| 256 | No | Yes | 0.225 | 0.276 | 0.148 | 0.360 | 0.0028 |
| 512 | No | Yes | 0.167 | 0.230 | 0.099 | 0.277 | 0.0001 |
| 1024 | No | Yes | 0.153 | 0.189 | 0.079 | 0.212 | 0.0002 |
| 2040 | No | Yes | 0.219 | 0.152 | 0.069 | 0.162 | 0.0006 |
| 2047 | No | No | 0.218 | 0.137 | 0.070 | 0.142 | 0.0002 |
| 2048 | No | No | 0.218 | 0.136 | 0.068 | 0.140 | 0.0007 |
| 2049 | No | No | 0.218 | 0.136 | 0.068 | 0.140 | 0.0003 |
| 4096 | No | No | 0.219 | 0.094 | 0.057 | 0.075 | 0.0011 |

## 3. Epistemic Assessment & Structural Findings

1. **Direct Residency vs Downstream Divergence:** After direct event residency ends, branch-specific differences persist in later Conv and KV representations, consistent with historical information being propagated through the hybrid recurrent system.
2. **Input-Dependent Recurrent Trajectories:** Distinct filler sequences modulate the persistence and decay rate of latent states across the dynamic lag spectrum.
3. **Zero Sham Floor:** Identical $A_1 / A_2$ controls confirm an empirical measurement floor of $0.00000000$ for scale-relative distance and Jensen-Shannon divergence.
