# Sprint S12c Specificity Microscope Causal Attribution Report

**Model Target:** `google/recurrentgemma-2b`  
**Phase:** `confirmatory`  
**Run Path:** `results\e12_specificity_microscope\run_e12_confirmatory_20260818_155145`  

**Inference:** Pair-Cluster Bootstrap ($B=10,000$) across 24 value pairs conditional on frozen filler panel.

## 1. Primary S12c Estimands & 95% Pair-Cluster Bootstrap CIs

| Estimand | Description | Observed Estimate | 95% Bootstrap CI | Confirmatory Inference |
| :--- | :--- | :---: | :---: | :--- |
| `delta_p_value_spec` | **Value-Specific Retention Contrast:** $P_{\text{match}} - P_{\text{same\_template\_wrong\_val}}$ | **+38.4939** | **[+25.8180, +50.8524]** | Positive; excludes zero |
| `delta_p_template_align` | **Template Alignment Contrast:** $P_{\text{same\_template\_wrong\_val}} - P_{\text{cross}}$ | **+7.3798** | **[-8.2553, +24.7325]** | Unresolved; spans zero |
| `delta_p_template_vs_noise` | **Template vs. Noise Contrast:** $P_{\text{same\_template\_wrong\_val}} - P_{\text{noise}}$ | **+34.8949** | **[+5.9699, +69.6188]** | Positive; excludes zero |
| `delta_p_match_vs_cross` | Matching vs. Cross-Template Contrast: $P_{\text{match}} - P_{\text{cross}}$ | **+45.8737** | **[+30.7178, +61.6130]** | Positive; excludes zero |
| `delta_p_match_vs_noise` | Matching vs. Noise Contrast: $P_{\text{match}} - P_{\text{noise}}$ | **+73.3888** | **[+39.7344, +111.8121]** | Positive; excludes zero |
| `p_match` | Matching Historical State: $P_{\text{match}}(2W)$ | **+121.6190** | **[+105.9816, +138.2551]** | Positive; excludes zero |
| `p_wrong_val` | Same-Template Wrong-Value State: $P_{\text{wrong\_val}}(2W)$ | **+83.1252** | **[+71.7718, +95.1869]** | Positive; excludes zero |
| `p_cross` | Cross-Template Historical State: $P_{\text{cross}}(2W)$ | **+75.7454** | **[+58.9245, +92.0811]** | Positive; excludes zero |
| `p_noise` | Matched Frobenius Gaussian Noise: $P_{\text{noise}}(2W)$ | **+48.2302** | **[+15.5731, +74.7612]** | Positive; excludes zero |
| `p_whole` | Whole State Positive Control: $P_{\text{whole}}(2W)$ | **+218.7596** | **[+197.1265, +241.8022]** | Positive; excludes zero |

## 2. Scientific Interpretation

- **Value-Specific Retention Confirmed:** Matching historical state provides a resolved directional increment of **+38.49** (95% CI [+25.82, +50.85]) over same-template wrong-value states. Recurrent state carries token-level historical binding beyond syntactic template alignment.

