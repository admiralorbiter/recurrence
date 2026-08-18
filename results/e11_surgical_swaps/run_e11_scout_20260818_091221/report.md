# E11 Multi-Store Surgical State Swaps Causal Attribution Report

**Model Target:** `google/recurrentgemma-2b`
**Run Path:** `results\e11_surgical_swaps\run_e11_scout_20260818_091221`

## 1. Causal Transfer & Directional Logit Displacement Across Lags

| Lag $L$ | Condition | Signed Graft $\bar{\Delta}_C$ | Abs Displacement $P_C$ | Logit Proj $\alpha_C^{\text{logit}}$ | Attrib Index $\alpha_C^{\text{cloze}}$ | Eligible N | Donor Concord |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 8 | `conv_only_a_into_b` | +0.00 | +0.00 | +0.000 | 0.000 | 16/16 | 0.0% |
| 8 | `conv_only_b_into_a` | +0.00 | +0.00 | +0.000 | 0.000 | 16/16 | 0.0% |
| 8 | `intact_a` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 100.0% |
| 8 | `intact_b` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 100.0% |
| 8 | `kv_only_a_into_b` | +22.81 | +1032.24 | +0.987 | 0.986 | 16/16 | 100.0% |
| 8 | `kv_only_b_into_a` | +22.84 | +1035.41 | +0.989 | 0.986 | 16/16 | 100.0% |
| 8 | `noise_control_rglru_seed1` | -0.09 | +1.86 | +0.002 | -0.003 | 16/16 | 0.0% |
| 8 | `noise_control_rglru_seed2` | -0.02 | -3.42 | -0.003 | -0.002 | 16/16 | 0.0% |
| 8 | `permuted_donor_rglru` | -0.10 | +0.17 | +0.001 | -0.004 | 16/16 | 0.0% |
| 8 | `recurrent_core_a_into_b` | +0.28 | +11.37 | +0.011 | 0.014 | 16/16 | 0.0% |
| 8 | `rglru_only_a_into_b` | +0.28 | +11.37 | +0.011 | 0.014 | 16/16 | 0.0% |
| 8 | `rglru_only_b_into_a` | +0.32 | +14.53 | +0.013 | 0.014 | 16/16 | 0.0% |
| 8 | `sham_a2_into_a1` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 0.0% |
| 8 | `unrelated_donor_rglru` | -0.07 | +6.13 | +0.006 | -0.003 | 16/16 | 0.0% |
| 8 | `whole_swap_a_into_b` | +23.13 | +1046.78 | +1.000 | 1.000 | 16/16 | 100.0% |
| 8 | `whole_swap_b_into_a` | +23.13 | +1046.78 | +1.000 | 1.000 | 16/16 | 100.0% |
| 2049 | `conv_only_a_into_b` | +0.00 | +0.00 | +0.000 | 0.000 | 6/16 | 43.8% |
| 2049 | `conv_only_b_into_a` | +0.00 | +0.00 | +0.000 | 0.000 | 6/16 | 50.0% |
| 2049 | `intact_a` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 50.0% |
| 2049 | `intact_b` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 56.2% |
| 2049 | `kv_only_a_into_b` | +0.54 | +79.18 | +0.806 | 1.033 | 6/16 | 50.0% |
| 2049 | `kv_only_b_into_a` | +0.58 | +69.37 | +0.800 | 1.094 | 6/16 | 50.0% |
| 2049 | `noise_control_rglru_seed1` | -0.04 | -3.38 | +0.067 | -0.098 | 6/16 | 0.0% |
| 2049 | `noise_control_rglru_seed2` | -0.02 | +6.09 | +0.136 | -0.008 | 6/16 | 0.0% |
| 2049 | `permuted_donor_rglru` | -0.03 | +29.36 | +0.266 | -0.049 | 6/16 | 0.0% |
| 2049 | `recurrent_core_a_into_b` | -0.03 | +21.80 | +0.200 | -0.094 | 6/16 | 50.0% |
| 2049 | `rglru_only_a_into_b` | -0.03 | +21.80 | +0.200 | -0.094 | 6/16 | 50.0% |
| 2049 | `rglru_only_b_into_a` | +0.02 | +12.00 | +0.194 | -0.033 | 6/16 | 50.0% |
| 2049 | `sham_a2_into_a1` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 0.0% |
| 2049 | `unrelated_donor_rglru` | -0.02 | +27.97 | +0.244 | -0.032 | 6/16 | 0.0% |
| 2049 | `whole_swap_a_into_b` | +0.55 | +91.17 | +1.000 | 1.000 | 6/16 | 50.0% |
| 2049 | `whole_swap_b_into_a` | +0.55 | +91.17 | +1.000 | 1.000 | 6/16 | 56.2% |
| 4096 | `conv_only_a_into_b` | +0.00 | +0.00 | +0.000 | 0.000 | 1/16 | 50.0% |
| 4096 | `conv_only_b_into_a` | +0.00 | +0.00 | +0.000 | 0.000 | 1/16 | 56.2% |
| 4096 | `intact_a` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 43.8% |
| 4096 | `intact_b` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 50.0% |
| 4096 | `kv_only_a_into_b` | -0.03 | +77.70 | +0.677 | 0.100 | 1/16 | 50.0% |
| 4096 | `kv_only_b_into_a` | +0.04 | +67.81 | +0.624 | 0.100 | 1/16 | 56.2% |
| 4096 | `noise_control_rglru_seed1` | -0.16 | +54.58 | +0.367 | 0.800 | 1/16 | 0.0% |
| 4096 | `noise_control_rglru_seed2` | -0.09 | +70.07 | +0.354 | 1.500 | 1/16 | 0.0% |
| 4096 | `permuted_donor_rglru` | +0.06 | -13.96 | +0.047 | -0.200 | 1/16 | 0.0% |
| 4096 | `recurrent_core_a_into_b` | -0.10 | +51.89 | +0.376 | 0.900 | 1/16 | 43.8% |
| 4096 | `rglru_only_a_into_b` | -0.10 | +51.89 | +0.376 | 0.900 | 1/16 | 43.8% |
| 4096 | `rglru_only_b_into_a` | -0.04 | +41.99 | +0.323 | 0.900 | 1/16 | 50.0% |
| 4096 | `sham_a2_into_a1` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 0.0% |
| 4096 | `unrelated_donor_rglru` | -0.01 | -35.41 | -0.177 | -0.200 | 1/16 | 0.0% |
| 4096 | `whole_swap_a_into_b` | -0.06 | +119.70 | +1.000 | 1.000 | 1/16 | 43.8% |
| 4096 | `whole_swap_b_into_a` | -0.06 | +119.70 | +1.000 | 1.000 | 1/16 | 50.0% |

## 2. Mediational Forward Dynamic Propagation ($R^B \to K_{\text{future}}^B$)

| Pair ID | Regime | Init Lag | Future Tokens | Turnover? | Post Dist Rec A | Post Dist Don B | Post Migr $\mathcal{M}_{\text{post}}$ | Full Migr $\mathcal{M}_{\text{full}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `item_material_01` | `constant` | 8 | 512 | No | 0.0170 | 0.0358 | -0.4878 | -0.6165 |
| `item_material_01` | `interfering` | 8 | 512 | No | 0.0166 | 0.0368 | -0.5288 | -0.6645 |
| `item_material_01` | `natural` | 8 | 512 | No | 0.0147 | 0.0209 | -0.2571 | -0.5780 |
| `item_material_01` | `random` | 8 | 512 | No | 0.0116 | 0.0260 | -0.5205 | -0.6972 |
| `item_material_02` | `constant` | 8 | 512 | No | 0.0144 | 0.0223 | -0.3053 | -0.5007 |
| `item_material_02` | `interfering` | 8 | 512 | No | 0.0216 | 0.0470 | -0.4906 | -0.5825 |
| `item_material_02` | `natural` | 8 | 512 | No | 0.0143 | 0.0251 | -0.3931 | -0.5894 |
| `item_material_02` | `random` | 8 | 512 | No | 0.0117 | 0.0241 | -0.4802 | -0.6551 |
| `item_material_03` | `constant` | 8 | 512 | No | 0.0171 | 0.0314 | -0.4001 | -0.5932 |
| `item_material_03` | `interfering` | 8 | 512 | No | 0.0223 | 0.0496 | -0.5053 | -0.6201 |
| `item_material_03` | `natural` | 8 | 512 | No | 0.0158 | 0.0295 | -0.4395 | -0.6564 |
| `item_material_03` | `random` | 8 | 512 | No | 0.0136 | 0.0352 | -0.5727 | -0.7099 |
| `item_material_04` | `constant` | 8 | 512 | No | 0.0154 | 0.0305 | -0.4450 | -0.6054 |
| `item_material_04` | `interfering` | 8 | 512 | No | 0.0215 | 0.0457 | -0.5210 | -0.6349 |
| `item_material_04` | `natural` | 8 | 512 | No | 0.0201 | 0.0304 | -0.2774 | -0.5023 |
| `item_material_04` | `random` | 8 | 512 | No | 0.0125 | 0.0320 | -0.5783 | -0.7146 |

## 3. Causal Interpretation & Control Framework

1. **Absolute Displacement ($P_C$) as Primary Causal Metric:** Directional displacement $P_C = (z_G - z_R) \cdot \frac{z_D - z_R}{\|z_D - z_R\|}$ distinguishes true causal steering magnitude from normalized share $\alpha_C^{\text{logit}}$ when the total donor-recipient contrast $\|z_D - z_R\|$ collapses at deep lags.
2. **Historical Specificity vs Matched Perturbations:** Matching donor RG-LRU is compared against unrelated-donor, permuted-donor, and Frobenius-matched Gaussian noise controls projected along the real donor direction ($P_{\text{match}} > P_{\text{control}}$).
3. **Dynamic Post-Graft KV Mediation:** Measures distances strictly over newly generated post-graft cache entries to determine whether continuous recurrent state propagates historical steering into downstream attention representations.
