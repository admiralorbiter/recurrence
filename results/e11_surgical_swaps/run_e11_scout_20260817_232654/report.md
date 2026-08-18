# E11 Multi-Store Surgical State Swaps Causal Attribution Report

**Model Target:** `google/recurrentgemma-2b`
**Run Path:** `results\e11_surgical_swaps\run_e11_scout_20260817_232654`

## 1. Causal Transfer & Directional Logit Projection Across Lags

| Lag $L$ | Condition | Raw Graft $\Delta_C$ (Primary) | Logit Projection $\alpha_C^{\text{logit}}$ | Attrib Index $\alpha_C^{\text{cloze}}$ | Donor Concordance |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 8 | `conv_only_a_into_b` | +0.00 | +0.000 | 0.000 | 0.0% |
| 8 | `conv_only_b_into_a` | +0.00 | +0.000 | 0.000 | 0.0% |
| 8 | `intact_a` | +0.00 | +0.000 | N/A (ineligible) | 100.0% |
| 8 | `intact_b` | +0.00 | +0.000 | N/A (ineligible) | 100.0% |
| 8 | `kv_only_a_into_b` | +22.81 | +0.987 | 0.986 | 100.0% |
| 8 | `kv_only_b_into_a` | -22.84 | +0.989 | 0.986 | 100.0% |
| 8 | `noise_control_rglru` | -0.40 | +0.000 | N/A (ineligible) | 0.0% |
| 8 | `recurrent_core_a_into_b` | +0.28 | +0.011 | 0.014 | 0.0% |
| 8 | `rglru_only_a_into_b` | +0.28 | +0.011 | 0.014 | 0.0% |
| 8 | `rglru_only_b_into_a` | -0.32 | +0.013 | 0.014 | 0.0% |
| 8 | `sham_a2_into_a1` | +0.00 | +0.000 | N/A (ineligible) | 0.0% |
| 8 | `whole_swap_a_into_b` | +23.13 | +1.000 | 1.000 | 100.0% |
| 8 | `whole_swap_b_into_a` | -23.13 | +1.000 | 1.000 | 100.0% |
| 2049 | `conv_only_a_into_b` | +0.00 | +0.000 | 0.000 | 43.8% |
| 2049 | `conv_only_b_into_a` | +0.00 | +0.000 | 0.000 | 50.0% |
| 2049 | `intact_a` | +0.00 | +0.000 | N/A (ineligible) | 50.0% |
| 2049 | `intact_b` | +0.00 | +0.000 | N/A (ineligible) | 56.2% |
| 2049 | `kv_only_a_into_b` | +0.54 | +0.806 | 1.033 | 50.0% |
| 2049 | `kv_only_b_into_a` | -0.58 | +0.800 | 1.094 | 50.0% |
| 2049 | `noise_control_rglru` | +0.52 | +0.000 | N/A (ineligible) | 0.0% |
| 2049 | `recurrent_core_a_into_b` | -0.03 | +0.200 | -0.094 | 50.0% |
| 2049 | `rglru_only_a_into_b` | -0.03 | +0.200 | -0.094 | 50.0% |
| 2049 | `rglru_only_b_into_a` | -0.02 | +0.194 | -0.033 | 50.0% |
| 2049 | `sham_a2_into_a1` | +0.00 | +0.000 | N/A (ineligible) | 0.0% |
| 2049 | `whole_swap_a_into_b` | +0.55 | +1.000 | 1.000 | 50.0% |
| 2049 | `whole_swap_b_into_a` | -0.55 | +1.000 | 1.000 | 56.2% |
| 4096 | `conv_only_a_into_b` | +0.00 | +0.000 | 0.000 | 50.0% |
| 4096 | `conv_only_b_into_a` | +0.00 | +0.000 | 0.000 | 56.2% |
| 4096 | `intact_a` | +0.00 | +0.000 | N/A (ineligible) | 43.8% |
| 4096 | `intact_b` | +0.00 | +0.000 | N/A (ineligible) | 50.0% |
| 4096 | `kv_only_a_into_b` | -0.03 | +0.677 | 0.100 | 50.0% |
| 4096 | `kv_only_b_into_a` | -0.04 | +0.624 | 0.100 | 56.2% |
| 4096 | `noise_control_rglru` | +0.86 | +0.000 | N/A (ineligible) | 0.0% |
| 4096 | `recurrent_core_a_into_b` | -0.10 | +0.376 | 0.900 | 43.8% |
| 4096 | `rglru_only_a_into_b` | -0.10 | +0.376 | 0.900 | 43.8% |
| 4096 | `rglru_only_b_into_a` | +0.04 | +0.323 | 0.900 | 50.0% |
| 4096 | `sham_a2_into_a1` | +0.00 | +0.000 | N/A (ineligible) | 0.0% |
| 4096 | `whole_swap_a_into_b` | -0.06 | +1.000 | 1.000 | 43.8% |
| 4096 | `whole_swap_b_into_a` | +0.06 | +1.000 | 1.000 | 50.0% |

## 2. Causal Interpretation & Gates

1. **Whole-State Swap Equivalence:** Whole-state transplantation establishes the baseline total dynamic range for complete behavioral reversal.
2. **RG-LRU Causal Sufficiency:** If $\bar{\Delta}_{\text{RGLRU}} > 0$ and $\alpha_{\text{RGLRU}}^{\text{logit}} \approx 1.0$ post-window, RG-LRU is confirmed as the causal substrate.
3. **Sham Floor & Noise Control:** Sham transplantation ($A_2 \to A_1$) and norm-matched noise perturbation verify that state grafting introduces zero artifactual logit distortion and distinguishes specific historical information from generic state disruption.
