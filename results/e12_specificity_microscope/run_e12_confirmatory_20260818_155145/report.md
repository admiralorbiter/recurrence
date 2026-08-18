# Sprint S12c Specificity Microscope Causal Attribution Report

**Model Target:** `google/recurrentgemma-2b`  
**Phase:** `confirmatory`  
**Run Path:** `results\e12_specificity_microscope\run_e12_confirmatory_20260818_155145`  

**Inference:** Pair-Cluster Bootstrap ($B=10,000$) across 24 value pairs (6 pairs per family across 4 template families) conditional on frozen filler panel.

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
| `p_whole` | Whole-State Positive Reference: $P_{\text{whole}}(2W)$ | **+218.7596** | **[+197.1265, +241.8022]** | Positive; excludes zero |
| `delta_proj_value_spec` | **Normalized Value-Specific Projection:** $\Delta \alpha_{\text{value\_spec}}$ | **+0.1744** | **[+0.1001, +0.2536]** | Positive; excludes zero |

## 2. Regime-Specific Sensitivity Breakdown

| Regime | $P_{\text{match}}$ | $P_{\text{wrong\_val}}$ | $P_{\text{noise}}$ | $\Delta P_{\text{value\_spec}}$ | 95% Bootstrap CI | $\Delta \alpha_{\text{value\_spec}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `constant` | +265.94 | +152.34 | +20.56 | **+113.60** | **[+79.82, +148.81]** | **+0.3504** |
| `interfering` | +17.93 | +15.30 | +12.12 | **+2.63** | **[-0.16, +5.53]** | **+0.0194** |
| `natural` | +75.39 | +77.23 | +65.37 | **-1.83** | **[-35.92, +28.48]** | **+0.0344** |
| `random` | +127.22 | +87.64 | +94.87 | **+39.58** | **[+4.19, +76.49]** | **+0.2933** |

## 3. Template Family Breakdown & Leave-One-Family-Out (LOFO) Robustness

### Family-Specific Value Contrasts

| Family | $N_{\text{pairs}}$ | $P_{\text{match}}$ | $P_{\text{wrong\_val}}$ | $\Delta P_{\text{value\_spec}}$ | 95% Bootstrap CI | $\Delta \alpha_{\text{value\_spec}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `archived_artifact` | 6 | +94.44 | +80.10 | **+14.35** | **[-9.33, +42.21]** | **+0.0372** |
| `marked_object` | 6 | +95.50 | +62.25 | **+33.25** | **[+21.55, +45.60]** | **+0.2050** |
| `monitored_signal` | 6 | +143.82 | +79.57 | **+64.24** | **[+52.59, +76.28]** | **+0.2778** |
| `sealed_container` | 6 | +152.72 | +110.58 | **+42.14** | **[+18.94, +67.07]** | **+0.1774** |

### Leave-One-Family-Out (LOFO) Analysis

| Left-Out Family | Remaining Pairs | $\Delta P_{\text{value\_spec}}$ (LOFO) | 95% Bootstrap CI | Status |
| :--- | :---: | :---: | :---: | :--- |
| `archived_artifact` | 18 | **+46.54** | **[+34.73, +58.24]** | Robustly Positive |
| `marked_object` | 18 | **+40.24** | **[+24.27, +55.83]** | Robustly Positive |
| `monitored_signal` | 18 | **+29.91** | **[+16.37, +44.11]** | Robustly Positive |
| `sealed_container` | 18 | **+37.28** | **[+23.35, +50.80]** | Robustly Positive |

## 4. Scientific Interpretation

- **Value-Specific Retention Confirmed:** Matching historical state provides a resolved directional increment of **+38.49** (95% CI [+25.82, +50.85]) over same-template wrong-value states. In normalized projection units, the contrast is **+0.1744** (95% CI [+0.1001, +0.2536]). Recurrent state carries value-specific historical information beyond syntactic template alignment.

- **Descriptive Contrast Ladder:**

  - Matched-norm noise control: $P_{\text{noise}} = +48.23$

  - Same-template wrong-value history: $P_{\text{wrong\_val}} = +83.13$ (contrast over noise: $\Delta P = +34.89$ [+5.97, +69.62])

  - Matching historical value: $P_{\text{match}} = +121.62$ (contrast over wrong-value: $\Delta P = +38.49$ [+25.82, +50.85])

  - Whole-state reference: $P_{\text{whole}} = +218.76$

- **Template Alignment Contrast:** The contrast between same-template wrong-value and cross-template historical states is $\Delta P = +7.38$ (95% CI [-8.26, +24.73]). Because this interval spans zero, we do not resolve an additional template increment over the cross-template control used here; structured nonmatching histories steer substantially more than noise, while the matching value provides a sharp, selective advantage.

