# E10 Latent Impulse Response & Store Localization Report

**Model Target:** `google/recurrentgemma-2b` (Reference Model: False)
**Run Path:** `results\e10_latent_impulse\run_e10_scout_20260818_025459`

## 1. Multi-Store Empirical Retention & 50% Thresholds

| Filler Regime | RGLRU First <50% | RGLRU Sustained <50% | Conv First <50% | KV First <50% | RGLRU Log-AUC | KV Log-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **constant** | L=8 | L=8 | L=1 | L=64 | 3.28 | 4.008 |
| **interfering** | L=8 | L=8 | L=1 | L=64 | 2.926 | 4.383 |
| **natural** | L=16 | L=16 | L=2 | L=64 | 2.892 | 4.275 |
| **random** | L=8 | L=8 | L=2 | L=64 | 2.642 | 4.085 |

## 2. Primary S11b Estimands & 95% Pair-Cluster Bootstrap CIs

| Estimand | Point Estimate / Mean | 95% Bootstrap CI |
| :--- | :---: | :---: |
| `constant_cloze_acc_2w` | 0.5 | [0.5, 0.5] |
| `constant_cloze_margin_2w` | 0.0715 | [-0.2422, 0.3906] |
| `constant_rglru_ret_2w` | 0.3943 | [0.1444, 0.7277] |
| `constant_rglru_ret_w1` | 0.2362 | [0.1459, 0.3449] |
| `interfering_cloze_acc_2w` | 0.5 | [0.5, 0.5] |
| `interfering_cloze_margin_2w` | -0.0077 | [-0.0469, 0.0234] |
| `interfering_rglru_ret_2w` | 0.0599 | [0.0544, 0.0696] |
| `interfering_rglru_ret_w1` | 0.0752 | [0.066, 0.0845] |
| `natural_cloze_acc_2w` | 0.5 | [0.5, 0.5] |
| `natural_cloze_margin_2w` | 0.0402 | [-0.0781, 0.1797] |
| `natural_rglru_ret_2w` | 0.0673 | [0.0494, 0.0968] |
| `natural_rglru_ret_w1` | 0.0718 | [0.0598, 0.0851] |
| `random_cloze_acc_2w` | 0.5 | [0.5, 0.5] |
| `random_cloze_margin_2w` | -0.0317 | [-0.0703, 0.0] |
| `random_rglru_ret_2w` | 0.044 | [0.0346, 0.055] |
| `random_rglru_ret_w1` | 0.0512 | [0.0409, 0.067] |

## 3. Dynamic Trajectories Across Tested Lags

| Lag $L$ | Conv Res? | KV Res? | RGLRU Ret (Const) | KV Ret (Const) | RGLRU Ret (Interf) | KV Ret (Interf) | Cloze Margin (Const) | Cloze Acc (Const) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Yes | Yes | 1.000 | 1.000 | 1.000 | 1.000 | +12.15 | 1.00 |
| 1 | Yes | Yes | 0.836 | 0.965 | 0.866 | 0.970 | +12.51 | 1.00 |
| 2 | Yes | Yes | 0.753 | 0.929 | 0.761 | 0.942 | +11.84 | 1.00 |
| 3 | No | Yes | 0.670 | 0.890 | 0.725 | 0.917 | +11.06 | 1.00 |
| 4 | No | Yes | 0.611 | 0.855 | 0.651 | 0.889 | +11.01 | 1.00 |
| 8 | No | Yes | 0.455 | 0.761 | 0.468 | 0.811 | +13.92 | 1.00 |
| 16 | No | Yes | 0.376 | 0.637 | 0.365 | 0.703 | +13.53 | 1.00 |
| 32 | No | Yes | 0.297 | 0.504 | 0.297 | 0.581 | +8.97 | 0.88 |
| 64 | No | Yes | 0.289 | 0.412 | 0.239 | 0.471 | +5.75 | 1.00 |
| 128 | No | Yes | 0.247 | 0.316 | 0.188 | 0.374 | +4.76 | 0.88 |
| 256 | No | Yes | 0.207 | 0.246 | 0.180 | 0.313 | +0.95 | 0.62 |
| 512 | No | Yes | 0.189 | 0.198 | 0.150 | 0.255 | +2.62 | 0.88 |
| 1024 | No | Yes | 0.186 | 0.153 | 0.108 | 0.202 | +0.87 | 0.62 |
| 2040 | No | Yes | 0.242 | 0.120 | 0.075 | 0.158 | +0.65 | 0.50 |
| 2047 | No | No | 0.238 | 0.100 | 0.076 | 0.138 | +0.59 | 0.50 |
| 2048 | No | No | 0.238 | 0.099 | 0.075 | 0.137 | +0.62 | 0.50 |
| 2049 | No | No | 0.237 | 0.098 | 0.075 | 0.136 | +0.68 | 0.50 |
| 4096 | No | No | 0.397 | 0.091 | 0.060 | 0.075 | +0.07 | 0.50 |

## 4. Epistemic Assessment & Structural Findings

1. **Direct Residency vs Downstream Divergence:** After direct event residency ends, branch-specific differences persist in later Conv and KV representations, consistent with historical information being propagated through the hybrid recurrent system.
2. **Factual Usability Across Windows:** The cloze log-likelihood margin measures the usable factual trace surviving in the recurrent state even after sliding-window attention eviction ($L \ge 2047$).
3. **Zero Sham Floor:** Identical $A_1 / A_2$ controls confirm an empirical measurement floor of $0.00000000$ for scale-relative distance and Jensen-Shannon divergence.
