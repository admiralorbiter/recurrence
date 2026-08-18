# E11 Multi-Store Surgical State Swaps Causal Attribution Report

**Model Target:** `google/recurrentgemma-2b`
**Phase:** `confirmatory`
**Run Path:** `results\e11_surgical_swaps\run_e11_confirmatory_20260818_152553`

**Bootstrap Inference:** Pair-Cluster Bootstrap ($B=10,000$) conditional on frozen filler panel / deterministic seed assignment.

## 1. Primary S12 Estimands & 95% Pair-Cluster Bootstrap CIs

| Estimand | Description | Observed Estimate | 95% Bootstrap CI |
| :--- | :--- | :---: | :---: |
| `alpha_kv_2w` | KV Relative Directional Share: $\alpha_{\text{KV}}^{\text{logit}}(2W)$ | +0.6320 | [+0.5966, +0.6670] |
| `alpha_match_2w` | RG-LRU Relative Directional Share: $\alpha_{\text{RGLRU}}^{\text{logit}}(2W)$ | +0.3680 | [+0.3330, +0.4034] |
| `alpha_unrel_2w` | alpha_unrel_2w | +0.3300 | [+0.2299, +0.4198] |
| `delta_kv_2w` | delta_kv_2w | +0.0465 | [+0.0176, +0.0781] |
| `delta_match_2w` | delta_match_2w | +0.0199 | [-0.0336, +0.0699] |
| `delta_p_growth_2w_minus_w1` | Temporal Causal Growth: $P_{\text{match}}(2W) - P_{\text{match}}(W+1)$ | +52.6587 | [+26.6603, +83.7810] |
| `delta_p_kv_minus_rglru_2w` | Store Causal Contrast: $P_{\text{KV}}(2W) - P_{\text{match}}(2W)$ | -11.6512 | [-49.0193, +20.1108] |
| `delta_p_spec_noise_2w` | Matched Frobenius Noise Contrast: $P_{\text{match}}(2W) - P_{\text{noise}}(2W)$ | +56.4601 | [+29.4490, +89.4735] |
| `delta_p_spec_perm_2w` | Secondary Paired Specificity Contrast: $P_{\text{match}}(2W) - P_{\text{perm}}(2W)$ | +29.6404 | [+11.8234, +52.4697] |
| `delta_p_spec_unrel_2w` | Primary Paired Specificity Contrast: $P_{\text{match}}(2W) - P_{\text{unrel}}(2W)$ | +19.6759 | [+1.8384, +39.1219] |
| `p_kv_2w` | p_kv_2w | +62.4483 | [+54.7684, +69.7399] |
| `p_match_2w` | Primary Physical Causal Endpoint: $P_{\text{RGLRU}}(2W)$ | +74.0994 | [+46.7899, +106.7161] |
| `p_match_l8` | RG-LRU Displacement at $L=8$: $P_{\text{RGLRU}}(L=8)$ | +18.2964 | [+13.5355, +23.1668] |
| `p_match_w1` | RG-LRU Displacement at $W+1$: $P_{\text{RGLRU}}(W+1)$ | +21.4408 | [+17.3425, +25.8959] |
| `p_noise_2w` | p_noise_2w | +17.6393 | [+10.7672, +25.4122] |
| `p_perm_2w` | p_perm_2w | +44.4590 | [+32.0241, +57.5830] |
| `p_unrel_2w` | p_unrel_2w | +54.4236 | [+32.2609, +77.1805] |
| `p_whole_2w` | p_whole_2w | +136.5477 | [+111.7998, +165.1752] |

## 2. Causal Factorial Panel & Directional Logit Displacement Across Lags

| Lag $L$ | Condition | Signed Graft $\bar{\Delta}_C$ | Directional Displacement $P_C$ | Logit Proj $\alpha_C^{\text{logit}}$ | Attrib Index $\alpha_C^{\text{cloze}}$ | Eligible N | Donor Concord |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 8 | `conv_only_a_into_b` | +0.00 | +0.00 | +0.000 | 0.000 | 80/80 | 1.2% |
| 8 | `conv_only_b_into_a` | +0.00 | +0.00 | +0.000 | 0.000 | 80/80 | 0.0% |
| 8 | `intact_a` | +0.00 | +0.00 | +0.000 | N/A | 0/80 | 100.0% |
| 8 | `intact_b` | +0.00 | +0.00 | +0.000 | N/A | 0/80 | 98.8% |
| 8 | `kv_only_a_into_b` | +20.89 | +981.49 | +0.981 | 0.989 | 80/80 | 100.0% |
| 8 | `kv_only_b_into_a` | +20.84 | +983.72 | +0.983 | 0.986 | 80/80 | 98.8% |
| 8 | `noise_rglru_a_into_b_s1` | -0.05 | -3.79 | -0.003 | -0.002 | 80/80 | 0.0% |
| 8 | `noise_rglru_a_into_b_s2` | +0.03 | +0.74 | +0.000 | 0.001 | 80/80 | 0.0% |
| 8 | `noise_rglru_b_into_a_s1` | +0.06 | +6.06 | +0.005 | 0.003 | 80/80 | 0.0% |
| 8 | `noise_rglru_b_into_a_s2` | +0.05 | +3.78 | +0.004 | 0.002 | 80/80 | 0.0% |
| 8 | `permuted_rglru_a_into_b` | +0.14 | +10.74 | +0.012 | 0.008 | 80/80 | 0.0% |
| 8 | `permuted_rglru_b_into_a` | +0.07 | +18.43 | +0.019 | 0.005 | 80/80 | 0.0% |
| 8 | `recurrent_core_a_into_b` | +0.29 | +17.18 | +0.017 | 0.014 | 80/80 | 1.2% |
| 8 | `recurrent_core_b_into_a` | +0.24 | +19.41 | +0.019 | 0.011 | 80/80 | 0.0% |
| 8 | `rglru_only_a_into_b` | +0.29 | +17.18 | +0.017 | 0.014 | 80/80 | 1.2% |
| 8 | `rglru_only_b_into_a` | +0.24 | +19.41 | +0.019 | 0.011 | 80/80 | 0.0% |
| 8 | `sham_a2_into_a1` | +0.00 | +0.00 | +0.000 | N/A | 0/80 | 0.0% |
| 8 | `sham_b2_into_b1` | +0.00 | +0.00 | +0.000 | N/A | 0/80 | 0.0% |
| 8 | `unrelated_rglru_a_into_b` | +0.07 | +11.87 | +0.013 | 0.004 | 80/80 | 0.0% |
| 8 | `unrelated_rglru_b_into_a` | +0.02 | +23.32 | +0.021 | 0.002 | 80/80 | 0.0% |
| 8 | `whole_swap_a_into_b` | +21.13 | +1000.90 | +1.000 | 1.000 | 80/80 | 100.0% |
| 8 | `whole_swap_b_into_a` | +21.13 | +1000.90 | +1.000 | 1.000 | 80/80 | 98.8% |
| 2049 | `conv_only_a_into_b` | +0.00 | +0.00 | +0.000 | 0.000 | 28/80 | 48.8% |
| 2049 | `conv_only_b_into_a` | +0.00 | +0.00 | +0.000 | 0.000 | 28/80 | 45.0% |
| 2049 | `intact_a` | +0.00 | +0.00 | +0.000 | N/A | 0/80 | 55.0% |
| 2049 | `intact_b` | +0.00 | +0.00 | +0.000 | N/A | 0/80 | 51.2% |
| 2049 | `kv_only_a_into_b` | +0.40 | +89.06 | +0.805 | 1.005 | 28/80 | 55.0% |
| 2049 | `kv_only_b_into_a` | +0.42 | +84.95 | +0.775 | 1.030 | 28/80 | 50.0% |
| 2049 | `noise_rglru_a_into_b_s1` | -0.02 | +28.12 | +0.216 | -0.123 | 28/80 | 0.0% |
| 2049 | `noise_rglru_a_into_b_s2` | +0.01 | +32.89 | +0.256 | 0.020 | 28/80 | 0.0% |
| 2049 | `noise_rglru_b_into_a_s1` | +0.03 | -0.31 | +0.067 | 0.151 | 28/80 | 0.0% |
| 2049 | `noise_rglru_b_into_a_s2` | -0.01 | -6.73 | +0.026 | -0.065 | 28/80 | 0.0% |
| 2049 | `permuted_rglru_a_into_b` | +0.00 | +21.58 | +0.214 | -0.000 | 28/80 | 0.0% |
| 2049 | `permuted_rglru_b_into_a` | +0.01 | +12.99 | +0.144 | 0.029 | 28/80 | 0.0% |
| 2049 | `recurrent_core_a_into_b` | -0.00 | +23.50 | +0.225 | -0.029 | 28/80 | 50.0% |
| 2049 | `recurrent_core_b_into_a` | +0.01 | +19.38 | +0.195 | -0.005 | 28/80 | 45.0% |
| 2049 | `rglru_only_a_into_b` | -0.00 | +23.50 | +0.225 | -0.029 | 28/80 | 50.0% |
| 2049 | `rglru_only_b_into_a` | +0.01 | +19.38 | +0.195 | -0.005 | 28/80 | 45.0% |
| 2049 | `sham_a2_into_a1` | +0.00 | +0.00 | +0.000 | N/A | 0/80 | 0.0% |
| 2049 | `sham_b2_into_b1` | +0.00 | +0.00 | +0.000 | N/A | 0/80 | 0.0% |
| 2049 | `unrelated_rglru_a_into_b` | -0.02 | +19.67 | +0.191 | -0.103 | 28/80 | 0.0% |
| 2049 | `unrelated_rglru_b_into_a` | +0.04 | +19.33 | +0.202 | 0.061 | 28/80 | 0.0% |
| 2049 | `whole_swap_a_into_b` | +0.41 | +108.45 | +1.000 | 1.000 | 28/80 | 55.0% |
| 2049 | `whole_swap_b_into_a` | +0.41 | +108.45 | +1.000 | 1.000 | 28/80 | 51.2% |
| 4096 | `conv_only_a_into_b` | +0.00 | +0.00 | +0.000 | 0.000 | 5/80 | 51.2% |
| 4096 | `conv_only_b_into_a` | +0.00 | +0.00 | +0.000 | 0.000 | 5/80 | 50.0% |
| 4096 | `intact_a` | +0.00 | +0.00 | +0.000 | N/A | 0/80 | 50.0% |
| 4096 | `intact_b` | +0.00 | +0.00 | +0.000 | N/A | 0/80 | 48.8% |
| 4096 | `kv_only_a_into_b` | +0.03 | +62.33 | +0.614 | 0.395 | 5/80 | 51.2% |
| 4096 | `kv_only_b_into_a` | +0.06 | +62.56 | +0.650 | 0.570 | 5/80 | 50.0% |
| 4096 | `noise_rglru_a_into_b_s1` | +0.01 | +20.78 | +0.207 | 0.118 | 5/80 | 0.0% |
| 4096 | `noise_rglru_a_into_b_s2` | +0.07 | +66.59 | +0.269 | 0.733 | 5/80 | 0.0% |
| 4096 | `noise_rglru_b_into_a_s1` | +0.02 | +15.40 | +0.217 | 0.302 | 5/80 | 0.0% |
| 4096 | `noise_rglru_b_into_a_s2` | -0.10 | -32.22 | +0.142 | -0.512 | 5/80 | 0.0% |
| 4096 | `permuted_rglru_a_into_b` | -0.01 | +49.17 | +0.225 | 0.037 | 5/80 | 0.0% |
| 4096 | `permuted_rglru_b_into_a` | +0.07 | +39.75 | +0.354 | 0.887 | 5/80 | 0.0% |
| 4096 | `recurrent_core_a_into_b` | +0.00 | +73.98 | +0.350 | 0.430 | 5/80 | 50.0% |
| 4096 | `recurrent_core_b_into_a` | +0.04 | +74.22 | +0.386 | 0.605 | 5/80 | 48.8% |
| 4096 | `rglru_only_a_into_b` | +0.00 | +73.98 | +0.350 | 0.430 | 5/80 | 50.0% |
| 4096 | `rglru_only_b_into_a` | +0.04 | +74.22 | +0.386 | 0.605 | 5/80 | 48.8% |
| 4096 | `sham_a2_into_a1` | +0.00 | +0.00 | +0.000 | N/A | 0/80 | 0.0% |
| 4096 | `sham_b2_into_b1` | +0.00 | +0.00 | +0.000 | N/A | 0/80 | 0.0% |
| 4096 | `unrelated_rglru_a_into_b` | +0.10 | +47.19 | +0.259 | 0.243 | 5/80 | 0.0% |
| 4096 | `unrelated_rglru_b_into_a` | +0.03 | +61.65 | +0.401 | 0.895 | 5/80 | 0.0% |
| 4096 | `whole_swap_a_into_b` | +0.07 | +136.55 | +1.000 | 1.000 | 5/80 | 50.0% |
| 4096 | `whole_swap_b_into_a` | +0.07 | +136.55 | +1.000 | 1.000 | 5/80 | 48.8% |

## 3. Mediational Forward Dynamic Propagation ($R^B \to K_{\text{future}}^B$)

| Pair ID | Regime | Init Lag | Future Tokens | Turnover? | Raw Post Dist Rec A | Raw Post Dist Don B | Post Migr $\mathcal{M}_{\text{post}}$ | Full Migr $\mathcal{M}_{\text{full}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `item_material_01` | `constant` | 8 | 512 | No | 11.7524 | 24.8055 | -0.4916 | -0.6139 |
| `item_material_01` | `constant` | 8 | 2048 | Yes | 19.6750 | 27.9048 | -0.2525 | -0.2525 |
| `item_material_01` | `interfering` | 8 | 512 | No | 8.5344 | 18.5517 | -0.5160 | -0.6653 |
| `item_material_01` | `interfering` | 8 | 2048 | Yes | 23.0954 | 27.8078 | -0.1427 | -0.1427 |
| `item_material_01` | `natural` | 8 | 512 | No | 7.3285 | 10.1025 | -0.2346 | -0.5865 |
| `item_material_01` | `natural` | 8 | 2048 | Yes | 14.6224 | 17.0674 | -0.1273 | -0.1273 |
| `item_material_01` | `random` | 8 | 512 | No | 6.0987 | 13.4175 | -0.5082 | -0.7014 |
| `item_material_01` | `random` | 8 | 2048 | Yes | 10.9301 | 16.5858 | -0.3081 | -0.3081 |
| `item_material_02` | `constant` | 8 | 512 | No | 10.0164 | 15.6329 | -0.3113 | -0.4945 |
| `item_material_02` | `constant` | 8 | 2048 | Yes | 21.0733 | 19.5621 | +0.0574 | +0.0574 |
| `item_material_02` | `interfering` | 8 | 512 | No | 10.6937 | 23.0214 | -0.4847 | -0.5892 |
| `item_material_02` | `interfering` | 8 | 2048 | Yes | 18.0555 | 32.1476 | -0.4070 | -0.4070 |
| `item_material_02` | `natural` | 8 | 512 | No | 7.1179 | 12.1201 | -0.3714 | -0.5944 |
| `item_material_02` | `natural` | 8 | 2048 | Yes | 18.1940 | 18.5943 | -0.0167 | -0.0167 |
| `item_material_02` | `random` | 8 | 512 | No | 6.1880 | 12.5192 | -0.4669 | -0.6583 |
| `item_material_02` | `random` | 8 | 2048 | Yes | 11.5810 | 15.6342 | -0.2244 | -0.2244 |
| `item_material_03` | `constant` | 8 | 512 | No | 11.2206 | 20.3659 | -0.3933 | -0.5849 |
| `item_material_03` | `constant` | 8 | 2048 | Yes | 20.0825 | 26.1835 | -0.1952 | -0.1952 |
| `item_material_03` | `interfering` | 8 | 512 | No | 11.4366 | 24.8076 | -0.4929 | -0.6209 |
| `item_material_03` | `interfering` | 8 | 2048 | Yes | 22.8391 | 35.6408 | -0.3418 | -0.3418 |
| `item_material_03` | `natural` | 8 | 512 | No | 7.8803 | 14.2365 | -0.4179 | -0.6597 |
| `item_material_03` | `natural` | 8 | 2048 | Yes | 15.5975 | 19.2042 | -0.1591 | -0.1591 |
| `item_material_03` | `random` | 8 | 512 | No | 7.0911 | 17.8251 | -0.5579 | -0.7121 |
| `item_material_03` | `random` | 8 | 2048 | Yes | 12.1227 | 21.0634 | -0.3826 | -0.3826 |
| `item_material_04` | `constant` | 8 | 512 | No | 10.0305 | 19.5108 | -0.4358 | -0.5949 |
| `item_material_04` | `constant` | 8 | 2048 | Yes | 17.7542 | 25.1944 | -0.2518 | -0.2518 |
| `item_material_04` | `interfering` | 8 | 512 | No | 10.9749 | 22.8873 | -0.5113 | -0.6375 |
| `item_material_04` | `interfering` | 8 | 2048 | Yes | 24.8434 | 35.6582 | -0.2781 | -0.2781 |
| `item_material_04` | `natural` | 8 | 512 | No | 9.7217 | 14.4943 | -0.2699 | -0.5196 |
| `item_material_04` | `natural` | 8 | 2048 | Yes | 14.8965 | 20.7418 | -0.2507 | -0.2507 |
| `item_material_04` | `random` | 8 | 512 | No | 6.5613 | 16.2648 | -0.5604 | -0.7148 |
| `item_material_04` | `random` | 8 | 2048 | Yes | 11.4409 | 20.0007 | -0.3920 | -0.3920 |
| `item_material_05` | `constant` | 8 | 512 | No | 10.7765 | 17.7154 | -0.3483 | -0.4942 |
| `item_material_05` | `constant` | 8 | 2048 | Yes | 23.4181 | 21.9169 | +0.0496 | +0.0496 |
| `item_material_05` | `interfering` | 8 | 512 | No | 8.8822 | 18.8256 | -0.5039 | -0.6259 |
| `item_material_05` | `interfering` | 8 | 2048 | Yes | 20.5163 | 30.2294 | -0.2939 | -0.2939 |
| `item_material_05` | `natural` | 8 | 512 | No | 6.8821 | 11.4818 | -0.3424 | -0.5617 |
| `item_material_05` | `natural` | 8 | 2048 | Yes | 15.8335 | 16.3226 | -0.0251 | -0.0251 |
| `item_material_05` | `random` | 8 | 512 | No | 6.2314 | 13.3139 | -0.4942 | -0.6516 |
| `item_material_05` | `random` | 8 | 2048 | Yes | 11.1988 | 16.5061 | -0.2924 | -0.2924 |
| `item_material_06` | `constant` | 8 | 512 | No | 16.0685 | 31.8436 | -0.4226 | -0.5661 |
| `item_material_06` | `constant` | 8 | 2048 | Yes | 30.5295 | 36.3225 | -0.1210 | -0.1210 |
| `item_material_06` | `interfering` | 8 | 512 | No | 15.2156 | 43.3495 | -0.6282 | -0.7073 |
| `item_material_06` | `interfering` | 8 | 2048 | Yes | 25.0556 | 58.7148 | -0.5363 | -0.5363 |
| `item_material_06` | `natural` | 8 | 512 | No | 10.1172 | 28.0897 | -0.6030 | -0.7271 |
| `item_material_06` | `natural` | 8 | 2048 | Yes | 16.0363 | 42.4726 | -0.5892 | -0.5892 |
| `item_material_06` | `random` | 8 | 512 | No | 7.7152 | 23.7960 | -0.6356 | -0.7704 |
| `item_material_06` | `random` | 8 | 2048 | Yes | 12.3091 | 28.0361 | -0.5225 | -0.5225 |
| `item_material_07` | `constant` | 8 | 512 | No | 11.3703 | 22.2377 | -0.4228 | -0.5595 |
| `item_material_07` | `constant` | 8 | 2048 | Yes | 20.1423 | 28.1004 | -0.2273 | -0.2273 |
| `item_material_07` | `interfering` | 8 | 512 | No | 14.2248 | 32.4949 | -0.5071 | -0.5926 |
| `item_material_07` | `interfering` | 8 | 2048 | Yes | 23.2674 | 45.6716 | -0.4618 | -0.4618 |
| `item_material_07` | `natural` | 8 | 512 | No | 7.7656 | 15.6998 | -0.4625 | -0.6478 |
| `item_material_07` | `natural` | 8 | 2048 | Yes | 15.2777 | 22.1894 | -0.2793 | -0.2793 |
| `item_material_07` | `random` | 8 | 512 | No | 6.7941 | 22.0872 | -0.6557 | -0.7551 |
| `item_material_07` | `random` | 8 | 2048 | Yes | 12.1614 | 25.7630 | -0.4832 | -0.4832 |
| `item_material_08` | `constant` | 8 | 512 | No | 10.1773 | 18.0303 | -0.3887 | -0.5466 |
| `item_material_08` | `constant` | 8 | 2048 | Yes | 20.4249 | 21.9200 | -0.0524 | -0.0524 |
| `item_material_08` | `interfering` | 8 | 512 | No | 10.1482 | 21.1916 | -0.4705 | -0.5951 |
| `item_material_08` | `interfering` | 8 | 2048 | Yes | 24.9398 | 36.9193 | -0.2938 | -0.2938 |
| `item_material_08` | `natural` | 8 | 512 | No | 8.0787 | 11.7587 | -0.2853 | -0.5568 |
| `item_material_08` | `natural` | 8 | 2048 | Yes | 22.1320 | 21.3809 | +0.0325 | +0.0325 |
| `item_material_08` | `random` | 8 | 512 | No | 6.2567 | 14.3345 | -0.5246 | -0.6876 |
| `item_material_08` | `random` | 8 | 2048 | Yes | 10.8592 | 17.4594 | -0.3428 | -0.3428 |
| `item_material_09` | `constant` | 8 | 512 | No | 11.0445 | 22.5855 | -0.4633 | -0.5912 |
| `item_material_09` | `constant` | 8 | 2048 | Yes | 18.6068 | 27.5145 | -0.2804 | -0.2804 |
| `item_material_09` | `interfering` | 8 | 512 | No | 11.9657 | 29.7424 | -0.5641 | -0.6457 |
| `item_material_09` | `interfering` | 8 | 2048 | Yes | 24.7193 | 44.9058 | -0.4148 | -0.4148 |
| `item_material_09` | `natural` | 8 | 512 | No | 10.1540 | 16.1311 | -0.3448 | -0.5584 |
| `item_material_09` | `natural` | 8 | 2048 | Yes | 17.2285 | 23.9258 | -0.2688 | -0.2688 |
| `item_material_09` | `random` | 8 | 512 | No | 7.0635 | 19.6879 | -0.6000 | -0.7231 |
| `item_material_09` | `random` | 8 | 2048 | Yes | 12.2100 | 23.2121 | -0.4329 | -0.4329 |
| `item_material_10` | `constant` | 8 | 512 | No | 12.5993 | 18.7764 | -0.2795 | -0.4623 |
| `item_material_10` | `constant` | 8 | 2048 | Yes | 31.5468 | 24.3225 | +0.1989 | +0.1989 |
| `item_material_10` | `interfering` | 8 | 512 | No | 10.9951 | 25.3083 | -0.5579 | -0.6597 |
| `item_material_10` | `interfering` | 8 | 2048 | Yes | 23.2674 | 37.8443 | -0.3422 | -0.3422 |
| `item_material_10` | `natural` | 8 | 512 | No | 8.8765 | 14.4845 | -0.3415 | -0.5636 |
| `item_material_10` | `natural` | 8 | 2048 | Yes | 16.2348 | 22.2931 | -0.2477 | -0.2477 |
| `item_material_10` | `random` | 8 | 512 | No | 6.5419 | 16.4081 | -0.5691 | -0.7106 |
| `item_material_10` | `random` | 8 | 2048 | Yes | 11.1687 | 19.6600 | -0.3961 | -0.3961 |
| `item_material_11` | `constant` | 8 | 512 | No | 13.6466 | 26.5691 | -0.4220 | -0.5793 |
| `item_material_11` | `constant` | 8 | 2048 | Yes | 20.9213 | 33.8488 | -0.3249 | -0.3249 |
| `item_material_11` | `interfering` | 8 | 512 | No | 12.5486 | 29.5695 | -0.5549 | -0.6606 |
| `item_material_11` | `interfering` | 8 | 2048 | Yes | 24.1594 | 41.9912 | -0.4067 | -0.4067 |
| `item_material_11` | `natural` | 8 | 512 | No | 8.6540 | 13.9762 | -0.3319 | -0.6074 |
| `item_material_11` | `natural` | 8 | 2048 | Yes | 18.7656 | 21.5029 | -0.1121 | -0.1121 |
| `item_material_11` | `random` | 8 | 512 | No | 7.0298 | 20.0783 | -0.6154 | -0.7444 |
| `item_material_11` | `random` | 8 | 2048 | Yes | 11.8258 | 23.4266 | -0.4563 | -0.4563 |
| `item_material_12` | `constant` | 8 | 512 | No | 11.1305 | 16.5169 | -0.2849 | -0.4685 |
| `item_material_12` | `constant` | 8 | 2048 | Yes | 22.3061 | 19.8521 | +0.0869 | +0.0869 |
| `item_material_12` | `interfering` | 8 | 512 | No | 16.3027 | 20.0043 | -0.1712 | -0.3627 |
| `item_material_12` | `interfering` | 8 | 2048 | Yes | 25.4455 | 29.8964 | -0.1337 | -0.1337 |
| `item_material_12` | `natural` | 8 | 512 | No | 9.7615 | 14.2708 | -0.3344 | -0.5662 |
| `item_material_12` | `natural` | 8 | 2048 | Yes | 16.4905 | 21.6699 | -0.2245 | -0.2245 |
| `item_material_12` | `random` | 8 | 512 | No | 6.3530 | 14.7735 | -0.5340 | -0.6863 |
| `item_material_12` | `random` | 8 | 2048 | Yes | 11.4637 | 18.0176 | -0.3300 | -0.3300 |
| `item_material_13` | `constant` | 8 | 512 | No | 11.1760 | 18.2710 | -0.3333 | -0.5003 |
| `item_material_13` | `constant` | 8 | 2048 | Yes | 35.7735 | 23.0605 | +0.3087 | +0.3087 |
| `item_material_13` | `interfering` | 8 | 512 | No | 10.0171 | 25.0894 | -0.5795 | -0.6705 |
| `item_material_13` | `interfering` | 8 | 2048 | Yes | 22.4836 | 35.8036 | -0.3337 | -0.3337 |
| `item_material_13` | `natural` | 8 | 512 | No | 7.4060 | 11.5125 | -0.3068 | -0.5703 |
| `item_material_13` | `natural` | 8 | 2048 | Yes | 14.7114 | 17.2119 | -0.1307 | -0.1307 |
| `item_material_13` | `random` | 8 | 512 | No | 6.3603 | 13.8772 | -0.4966 | -0.6698 |
| `item_material_13` | `random` | 8 | 2048 | Yes | 11.1988 | 17.0148 | -0.3028 | -0.3028 |
| `item_material_14` | `constant` | 8 | 512 | No | 12.1904 | 25.4685 | -0.4869 | -0.6056 |
| `item_material_14` | `constant` | 8 | 2048 | Yes | 19.7979 | 30.6448 | -0.3132 | -0.3132 |
| `item_material_14` | `interfering` | 8 | 512 | No | 12.9890 | 25.0310 | -0.4217 | -0.5476 |
| `item_material_14` | `interfering` | 8 | 2048 | Yes | 27.9128 | 41.2295 | -0.2757 | -0.2757 |
| `item_material_14` | `natural` | 8 | 512 | No | 8.0925 | 14.8418 | -0.3960 | -0.6213 |
| `item_material_14` | `natural` | 8 | 2048 | Yes | 15.6648 | 22.3201 | -0.2526 | -0.2526 |
| `item_material_14` | `random` | 8 | 512 | No | 6.5057 | 16.3538 | -0.5643 | -0.7195 |
| `item_material_14` | `random` | 8 | 2048 | Yes | 11.2332 | 19.6721 | -0.3925 | -0.3925 |
| `item_material_15` | `constant` | 8 | 512 | No | 10.2576 | 14.7995 | -0.2616 | -0.4679 |
| `item_material_15` | `constant` | 8 | 2048 | Yes | 18.4497 | 19.4272 | -0.0393 | -0.0393 |
| `item_material_15` | `interfering` | 8 | 512 | No | 16.0348 | 24.2599 | -0.2940 | -0.4187 |
| `item_material_15` | `interfering` | 8 | 2048 | Yes | 22.7483 | 31.4564 | -0.2322 | -0.2322 |
| `item_material_15` | `natural` | 8 | 512 | No | 8.1594 | 12.7724 | -0.3121 | -0.5580 |
| `item_material_15` | `natural` | 8 | 2048 | Yes | 15.0800 | 17.7387 | -0.1300 | -0.1300 |
| `item_material_15` | `random` | 8 | 512 | No | 6.3206 | 13.4971 | -0.4883 | -0.6650 |
| `item_material_15` | `random` | 8 | 2048 | Yes | 11.2501 | 16.7526 | -0.2960 | -0.2960 |
| `item_material_16` | `constant` | 8 | 512 | No | 11.0562 | 20.6704 | -0.4200 | -0.5743 |
| `item_material_16` | `constant` | 8 | 2048 | Yes | 20.5537 | 25.8656 | -0.1702 | -0.1702 |
| `item_material_16` | `interfering` | 8 | 512 | No | 11.0628 | 24.1890 | -0.5002 | -0.6287 |
| `item_material_16` | `interfering` | 8 | 2048 | Yes | 20.7143 | 32.3555 | -0.3202 | -0.3202 |
| `item_material_16` | `natural` | 8 | 512 | No | 9.1667 | 14.3345 | -0.3256 | -0.5808 |
| `item_material_16` | `natural` | 8 | 2048 | Yes | 15.7013 | 21.2694 | -0.2406 | -0.2406 |
| `item_material_16` | `random` | 8 | 512 | No | 6.8216 | 17.6829 | -0.5860 | -0.7251 |
| `item_material_16` | `random` | 8 | 2048 | Yes | 12.0518 | 20.5688 | -0.3761 | -0.3761 |
| `item_material_17` | `constant` | 8 | 512 | No | 10.6750 | 18.4504 | -0.3740 | -0.5375 |
| `item_material_17` | `constant` | 8 | 2048 | Yes | 17.4191 | 23.6388 | -0.2216 | -0.2216 |
| `item_material_17` | `interfering` | 8 | 512 | No | 15.6876 | 27.0868 | -0.4309 | -0.5419 |
| `item_material_17` | `interfering` | 8 | 2048 | Yes | 27.7278 | 40.7976 | -0.3203 | -0.3203 |
| `item_material_17` | `natural` | 8 | 512 | No | 7.2010 | 11.6468 | -0.3488 | -0.5959 |
| `item_material_17` | `natural` | 8 | 2048 | Yes | 16.5019 | 19.7034 | -0.1480 | -0.1480 |
| `item_material_17` | `random` | 8 | 512 | No | 6.5288 | 14.7658 | -0.5120 | -0.6785 |
| `item_material_17` | `random` | 8 | 2048 | Yes | 13.5705 | 19.0283 | -0.2686 | -0.2686 |
| `item_material_18` | `constant` | 8 | 512 | No | 11.5430 | 22.0811 | -0.4385 | -0.5838 |
| `item_material_18` | `constant` | 8 | 2048 | Yes | 24.0142 | 26.4043 | -0.0711 | -0.0711 |
| `item_material_18` | `interfering` | 8 | 512 | No | 15.3505 | 25.8569 | -0.3844 | -0.5199 |
| `item_material_18` | `interfering` | 8 | 2048 | Yes | 22.2505 | 37.4271 | -0.3756 | -0.3756 |
| `item_material_18` | `natural` | 8 | 512 | No | 7.4820 | 15.8793 | -0.4890 | -0.6648 |
| `item_material_18` | `natural` | 8 | 2048 | Yes | 14.7674 | 24.6870 | -0.3565 | -0.3565 |
| `item_material_18` | `random` | 8 | 512 | No | 7.0889 | 15.8854 | -0.5146 | -0.6843 |
| `item_material_18` | `random` | 8 | 2048 | Yes | 11.7911 | 19.1760 | -0.3450 | -0.3450 |
| `item_material_19` | `constant` | 8 | 512 | No | 11.9366 | 24.2502 | -0.4488 | -0.5938 |
| `item_material_19` | `constant` | 8 | 2048 | Yes | 19.0594 | 31.2654 | -0.3352 | -0.3352 |
| `item_material_19` | `interfering` | 8 | 512 | No | 14.8799 | 31.4661 | -0.5228 | -0.6238 |
| `item_material_19` | `interfering` | 8 | 2048 | Yes | 29.6492 | 45.2317 | -0.3455 | -0.3455 |
| `item_material_19` | `natural` | 8 | 512 | No | 8.8342 | 19.0968 | -0.5174 | -0.6771 |
| `item_material_19` | `natural` | 8 | 2048 | Yes | 17.7342 | 25.2850 | -0.2784 | -0.2784 |
| `item_material_19` | `random` | 8 | 512 | No | 7.1632 | 21.7995 | -0.6364 | -0.7451 |
| `item_material_19` | `random` | 8 | 2048 | Yes | 12.7368 | 28.3435 | -0.5230 | -0.5230 |
| `item_material_20` | `constant` | 8 | 512 | No | 12.1167 | 32.6730 | -0.6006 | -0.6862 |
| `item_material_20` | `constant` | 8 | 2048 | Yes | 20.3334 | 39.2227 | -0.4433 | -0.4433 |
| `item_material_20` | `interfering` | 8 | 512 | No | 20.0662 | 32.2201 | -0.3344 | -0.4674 |
| `item_material_20` | `interfering` | 8 | 2048 | Yes | 26.5851 | 46.6641 | -0.4247 | -0.4247 |
| `item_material_20` | `natural` | 8 | 512 | No | 9.2256 | 21.8807 | -0.5487 | -0.6914 |
| `item_material_20` | `natural` | 8 | 2048 | Yes | 18.0426 | 27.3454 | -0.3046 | -0.3046 |
| `item_material_20` | `random` | 8 | 512 | No | 7.3822 | 27.4957 | -0.6961 | -0.7784 |
| `item_material_20` | `random` | 8 | 2048 | Yes | 11.8495 | 32.8118 | -0.6032 | -0.6032 |

## 4. Epistemic Assessment & Causal Framework

1. **Directional Displacement ($P_C$) as Primary Causal Endpoint:** Directional displacement $P_C = (z_G - z_R) \cdot \frac{z_D - z_R}{\|z_D - z_R\|}$ distinguishes true causal steering magnitude from normalized share $\alpha_C^{\text{logit}}$ when the total donor-recipient contrast $\|z_D - z_R\|$ collapses at deep lags.
2. **Historical Specificity Contrast:** Primary inference tests $P_{\text{matching}} > P_{\text{unrelated}}$ across balanced cyclic derangements conditional on frozen filler streams.
3. **Dynamic Post-Graft KV Mediation:** Measures distances strictly over newly generated post-graft cache entries to determine whether continuous recurrent state propagates historical steering into downstream attention representations.
