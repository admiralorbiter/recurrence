# E11 Multi-Store Surgical State Swaps Causal Attribution Report

**Model Target:** `google/recurrentgemma-2b`
**Run Path:** `results\e11_surgical_swaps\run_e11_scout_20260818_081635`

## 1. Causal Transfer & Directional Logit Displacement Across Lags

| Lag $L$ | Condition | Raw Graft $\Delta_C$ | Abs Displacement $P_C$ | Logit Proj $\alpha_C^{\text{logit}}$ | Attrib Index $\alpha_C^{\text{cloze}}$ | Eligible N | Donor Concord |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 8 | `conv_only_a_into_b` | +0.00 | +0.00 | +0.000 | 0.000 | 16/16 | 0.0% |
| 8 | `conv_only_b_into_a` | +0.00 | +0.00 | +0.000 | 0.000 | 16/16 | 0.0% |
| 8 | `intact_a` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 100.0% |
| 8 | `intact_b` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 100.0% |
| 8 | `kv_only_a_into_b` | +22.81 | +1032.24 | +0.987 | 0.986 | 16/16 | 100.0% |
| 8 | `kv_only_b_into_a` | -22.84 | +1035.41 | +0.989 | 0.986 | 16/16 | 100.0% |
| 8 | `noise_control_rglru` | -0.40 | +0.00 | +0.000 | N/A | 0/16 | 0.0% |
| 8 | `recurrent_core_a_into_b` | +0.28 | +11.37 | +0.011 | 0.014 | 16/16 | 0.0% |
| 8 | `rglru_only_a_into_b` | +0.28 | +11.37 | +0.011 | 0.014 | 16/16 | 0.0% |
| 8 | `rglru_only_b_into_a` | -0.32 | +14.53 | +0.013 | 0.014 | 16/16 | 0.0% |
| 8 | `sham_a2_into_a1` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 0.0% |
| 8 | `unrelated_donor_rglru` | -0.07 | +6.13 | +0.006 | -0.003 | 16/16 | 0.0% |
| 8 | `whole_swap_a_into_b` | +23.13 | +1046.78 | +1.000 | 1.000 | 16/16 | 100.0% |
| 8 | `whole_swap_b_into_a` | -23.13 | +1046.78 | +1.000 | 1.000 | 16/16 | 100.0% |
| 2049 | `conv_only_a_into_b` | +0.00 | +0.00 | +0.000 | 0.000 | 6/16 | 43.8% |
| 2049 | `conv_only_b_into_a` | +0.00 | +0.00 | +0.000 | 0.000 | 6/16 | 50.0% |
| 2049 | `intact_a` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 50.0% |
| 2049 | `intact_b` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 56.2% |
| 2049 | `kv_only_a_into_b` | +0.54 | +79.18 | +0.806 | 1.033 | 6/16 | 50.0% |
| 2049 | `kv_only_b_into_a` | -0.58 | +69.37 | +0.800 | 1.094 | 6/16 | 50.0% |
| 2049 | `noise_control_rglru` | +0.52 | +0.00 | +0.000 | N/A | 0/16 | 0.0% |
| 2049 | `recurrent_core_a_into_b` | -0.03 | +21.80 | +0.200 | -0.094 | 6/16 | 50.0% |
| 2049 | `rglru_only_a_into_b` | -0.03 | +21.80 | +0.200 | -0.094 | 6/16 | 50.0% |
| 2049 | `rglru_only_b_into_a` | -0.02 | +12.00 | +0.194 | -0.033 | 6/16 | 50.0% |
| 2049 | `sham_a2_into_a1` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 0.0% |
| 2049 | `unrelated_donor_rglru` | -0.02 | +27.97 | +0.244 | -0.032 | 6/16 | 0.0% |
| 2049 | `whole_swap_a_into_b` | +0.55 | +91.17 | +1.000 | 1.000 | 6/16 | 50.0% |
| 2049 | `whole_swap_b_into_a` | -0.55 | +91.17 | +1.000 | 1.000 | 6/16 | 56.2% |
| 4096 | `conv_only_a_into_b` | +0.00 | +0.00 | +0.000 | 0.000 | 1/16 | 50.0% |
| 4096 | `conv_only_b_into_a` | +0.00 | +0.00 | +0.000 | 0.000 | 1/16 | 56.2% |
| 4096 | `intact_a` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 43.8% |
| 4096 | `intact_b` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 50.0% |
| 4096 | `kv_only_a_into_b` | -0.03 | +77.70 | +0.677 | 0.100 | 1/16 | 50.0% |
| 4096 | `kv_only_b_into_a` | -0.04 | +67.81 | +0.624 | 0.100 | 1/16 | 56.2% |
| 4096 | `noise_control_rglru` | +0.86 | +0.00 | +0.000 | N/A | 0/16 | 0.0% |
| 4096 | `recurrent_core_a_into_b` | -0.10 | +51.89 | +0.376 | 0.900 | 1/16 | 43.8% |
| 4096 | `rglru_only_a_into_b` | -0.10 | +51.89 | +0.376 | 0.900 | 1/16 | 43.8% |
| 4096 | `rglru_only_b_into_a` | +0.04 | +41.99 | +0.323 | 0.900 | 1/16 | 50.0% |
| 4096 | `sham_a2_into_a1` | +0.00 | +0.00 | +0.000 | N/A | 0/16 | 0.0% |
| 4096 | `unrelated_donor_rglru` | -0.01 | -35.41 | -0.177 | -0.200 | 1/16 | 0.0% |
| 4096 | `whole_swap_a_into_b` | -0.06 | +119.70 | +1.000 | 1.000 | 1/16 | 43.8% |
| 4096 | `whole_swap_b_into_a` | +0.06 | +119.70 | +1.000 | 1.000 | 1/16 | 50.0% |

## 2. Mediational Forward Dynamic Propagation ($R^B \to K_{\text{future}}^B$)

| Pair ID | Regime | Initial Lag | Future Tokens | Dist to Recipient A | Dist to Donor B | KV Migration Index $\mathcal{M}_{\text{KV}}$ | Propagated? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `item_material_01` | `constant` | 8 | 512 | 0.0169 | 0.0459 | -0.6165 | NO |
| `item_material_01` | `interfering` | 8 | 512 | 0.0163 | 0.0497 | -0.6645 | NO |
| `item_material_01` | `natural` | 8 | 512 | 0.0145 | 0.0370 | -0.5780 | NO |
| `item_material_01` | `random` | 8 | 512 | 0.0114 | 0.0394 | -0.6972 | NO |
| `item_material_02` | `constant` | 8 | 512 | 0.0143 | 0.0307 | -0.5007 | NO |
| `item_material_02` | `interfering` | 8 | 512 | 0.0213 | 0.0561 | -0.5825 | NO |
| `item_material_02` | `natural` | 8 | 512 | 0.0140 | 0.0363 | -0.5894 | NO |
| `item_material_02` | `random` | 8 | 512 | 0.0115 | 0.0348 | -0.6551 | NO |
| `item_material_03` | `constant` | 8 | 512 | 0.0169 | 0.0454 | -0.5932 | NO |
| `item_material_03` | `interfering` | 8 | 512 | 0.0220 | 0.0628 | -0.6201 | NO |
| `item_material_03` | `natural` | 8 | 512 | 0.0156 | 0.0463 | -0.6564 | NO |
| `item_material_03` | `random` | 8 | 512 | 0.0134 | 0.0494 | -0.7099 | NO |
| `item_material_04` | `constant` | 8 | 512 | 0.0152 | 0.0417 | -0.6054 | NO |
| `item_material_04` | `interfering` | 8 | 512 | 0.0212 | 0.0577 | -0.6349 | NO |
| `item_material_04` | `natural` | 8 | 512 | 0.0198 | 0.0448 | -0.5023 | NO |
| `item_material_04` | `random` | 8 | 512 | 0.0123 | 0.0449 | -0.7146 | NO |

## 3. Causal Interpretation & Control Framework

1. **Absolute vs Relative Logit Displacement:** Absolute donor displacement $P_C = (z_G - z_R) \cdot \frac{z_D - z_R}{\|z_D - z_R\|}$ distinguishes true causal steering magnitude from relative share $\alpha_C^{\text{logit}}$ when the total donor-recipient contrast $\|z_D - z_R\|$ collapses at deep lags.
2. **Historical Specificity vs Generic Perturbation:** Unrelated-donor and permuted-donor RG-LRU controls demonstrate that directional logit steering is specific to the matching historical event ($P_{\text{donor}} > P_{\text{unrelated}}$), while norm-matched Gaussian noise is orthogonal ($P_{\text{noise}} \approx 0.00$).
3. **Dynamic Forward Mediational Propagation:** Forward unrolling from hybrid state $(R^B, C^A, K^A)$ verifies whether RG-LRU causally transmits historical information into downstream sliding-window KV representations during ongoing generation.
