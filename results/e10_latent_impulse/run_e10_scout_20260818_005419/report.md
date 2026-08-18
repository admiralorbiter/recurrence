# E10 Latent Impulse Response & Store Localization Report

**Run Path:** `results\e10_latent_impulse\run_e10_scout_20260818_005419`

## 1. Summary of Empirical Retention Trajectories

| Filler Regime | RGLRU 50% Crossing ($L_{50\%}$) | Conv1D 50% Crossing ($L_{50\%}$) | KV 50% Crossing ($L_{50\%}$) | RGLRU AUC | KV AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **constant** | L=8 | L=3 | L=15 | 14.37 | 8.58 |
| **interfering** | L=32 | L=3 | L=15 | 27.15 | 8.67 |
| **natural** | L=15 | L=3 | L=15 | 26.06 | 8.48 |
| **random** | L=15 | L=3 | L=15 | 27.36 | 8.73 |

## 2. Retention Trajectories Table (Constant vs Natural vs Interfering)

| Lag $L$ | Conv Res? | KV Res? | RGLRU Ret (Const) | KV Ret (Const) | RGLRU Ret (Interf) | KV Ret (Interf) | $D_{\text{JS}}$ (Const) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Yes | Yes | 1.000 | 1.000 | 1.000 | 1.000 | 0.0309 |
| 1 | Yes | Yes | 1.103 | 0.900 | 1.087 | 0.896 | 0.0051 |
| 2 | Yes | Yes | 1.079 | 0.825 | 1.116 | 0.824 | 0.0037 |
| 3 | No | Yes | 0.859 | 0.764 | 0.975 | 0.758 | 0.0026 |
| 4 | No | Yes | 0.724 | 0.714 | 0.860 | 0.727 | 0.0018 |
| 8 | No | Yes | 0.464 | 0.576 | 0.680 | 0.582 | 0.0004 |
| 15 | No | No | 0.219 | 0.027 | 0.524 | 0.027 | 0.0001 |
| 16 | No | No | 0.198 | 0.026 | 0.552 | 0.023 | 0.0000 |
| 17 | No | No | 0.178 | 0.021 | 0.534 | 0.021 | 0.0000 |
| 32 | No | No | 0.067 | 0.004 | 0.167 | 0.007 | 0.0000 |
| 64 | No | No | 0.027 | 0.002 | 0.082 | 0.002 | 0.0000 |
| 128 | No | No | 0.008 | 0.001 | 0.021 | 0.001 | 0.0000 |
| 256 | No | No | 0.002 | 0.000 | 0.004 | 0.000 | 0.0000 |
| 512 | No | No | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 |
