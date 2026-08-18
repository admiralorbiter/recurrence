# E10 Latent Impulse Response & Store Localization Report

**Model Target:** `google/recurrentgemma-2b` (Reference Model: False)
**Run Path:** `results\e10_latent_impulse\run_e10_confirmatory_20260818_033015`

## 1. Multi-Store Empirical Retention & 50% Thresholds

| Filler Regime | RGLRU First <50% | RGLRU Sustained <50% | Conv First <50% | KV First <50% | RGLRU Log-AUC | KV Log-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **constant** | L=8 | L=8 | L=2 | L=32 | 3.369 | 3.963 |
| **interfering** | L=16 | L=16 | L=8 | L=128 | 3.241 | 4.538 |
| **natural** | L=8 | L=8 | L=2 | L=64 | 2.779 | 4.148 |
| **random** | L=8 | L=8 | L=3 | L=64 | 2.825 | 4.099 |

## 2. Primary S11b Estimands & 95% Pair-Cluster Bootstrap CIs

| Estimand | Point Estimate / Mean | 95% Bootstrap CI |
| :--- | :---: | :---: |
| `constant_cloze_acc_2w` | 0.5 | [0.5, 0.5] |
| `constant_cloze_margin_2w` | -0.0265 | [-0.1109, 0.0656] |
| `constant_cloze_margin_w1` | 0.5187 | [0.3812, 0.6719] |
| `constant_rglru_ret_2w` | 0.3384 | [0.2484, 0.4401] |
| `constant_rglru_ret_w1` | 0.2845 | [0.2204, 0.3582] |
| `delta_rglru_ret_interf_minus_const_2w` | -0.2586 | [-0.3566, -0.1697] |
| `interfering_cloze_acc_2w` | 0.4497 | [0.375, 0.5] |
| `interfering_cloze_margin_2w` | 0.0044 | [-0.075, 0.0844] |
| `interfering_cloze_margin_w1` | 0.0307 | [-0.0281, 0.0953] |
| `interfering_rglru_ret_2w` | 0.0798 | [0.0734, 0.0864] |
| `interfering_rglru_ret_w1` | 0.0983 | [0.0896, 0.1075] |
| `natural_cloze_acc_2w` | 0.476 | [0.425, 0.5] |
| `natural_cloze_margin_2w` | 0.0123 | [-0.0125, 0.0375] |
| `natural_cloze_margin_w1` | -0.0182 | [-0.0703, 0.0344] |
| `natural_rglru_ret_2w` | 0.0514 | [0.0461, 0.0571] |
| `natural_rglru_ret_w1` | 0.0636 | [0.0584, 0.0694] |
| `random_cloze_acc_2w` | 0.5 | [0.5, 0.5] |
| `random_cloze_margin_2w` | -0.003 | [-0.0203, 0.0125] |
| `random_cloze_margin_w1` | 0.0667 | [0.0219, 0.1094] |
| `random_rglru_ret_2w` | 0.0453 | [0.0402, 0.0501] |
| `random_rglru_ret_w1` | 0.0551 | [0.0489, 0.0613] |

## 3. Dynamic Trajectories Across Tested Lags

| Lag $L$ | Conv Res? | KV Res? | RGLRU Ret (Const) | KV Ret (Const) | RGLRU Ret (Interf) | KV Ret (Interf) | Cloze Margin (Const) | Cloze Acc (Const) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Yes | Yes | 1.000 | 1.000 | 1.000 | 1.000 | +10.78 | 1.00 |
| 1 | Yes | Yes | 0.842 | 0.962 | 0.833 | 0.968 | +11.03 | 1.00 |
| 2 | Yes | Yes | 0.767 | 0.925 | 0.776 | 0.940 | +11.14 | 0.97 |
| 3 | No | Yes | 0.692 | 0.885 | 0.797 | 0.924 | +10.86 | 1.00 |
| 4 | No | Yes | 0.633 | 0.849 | 0.701 | 0.901 | +10.47 | 1.00 |
| 8 | No | Yes | 0.460 | 0.748 | 0.602 | 0.824 | +12.26 | 1.00 |
| 16 | No | Yes | 0.386 | 0.623 | 0.426 | 0.712 | +11.11 | 0.95 |
| 32 | No | Yes | 0.307 | 0.494 | 0.366 | 0.601 | +7.85 | 0.95 |
| 64 | No | Yes | 0.292 | 0.399 | 0.270 | 0.501 | +3.73 | 0.82 |
| 128 | No | Yes | 0.254 | 0.309 | 0.234 | 0.407 | +4.96 | 0.90 |
| 256 | No | Yes | 0.234 | 0.244 | 0.195 | 0.339 | +0.54 | 0.55 |
| 512 | No | Yes | 0.196 | 0.196 | 0.184 | 0.280 | +1.85 | 0.72 |
| 1024 | No | Yes | 0.194 | 0.153 | 0.121 | 0.228 | +0.62 | 0.55 |
| 2040 | No | Yes | 0.291 | 0.124 | 0.105 | 0.186 | +0.50 | 0.57 |
| 2047 | No | No | 0.286 | 0.106 | 0.101 | 0.168 | +0.47 | 0.50 |
| 2048 | No | No | 0.285 | 0.105 | 0.101 | 0.167 | +0.52 | 0.50 |
| 2049 | No | No | 0.285 | 0.105 | 0.098 | 0.167 | +0.52 | 0.50 |
| 4096 | No | No | 0.340 | 0.086 | 0.080 | 0.097 | -0.03 | 0.50 |

## 4. Epistemic Assessment & Structural Findings

1. **Direct Residency vs Downstream Divergence:** After direct event residency ends, branch-specific differences persist in later Conv and KV representations, consistent with historical information being propagated through the hybrid recurrent system.
2. **Factual Usability Across Windows:** The cloze log-likelihood margin measures the usable factual trace surviving in the recurrent state even after sliding-window attention eviction ($L \ge 2047$).
3. **Zero Sham Floor:** Identical $A_1 / A_2$ controls confirm an empirical measurement floor of $0.00000000$ for scale-relative distance and Jensen-Shannon divergence.
