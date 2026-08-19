# Sprint S13 Controlled Recurrent Dynamics Report

**Model Target:** `google/recurrentgemma-2b`  
**Phase:** `scout`  
**Run Path:** `results\e13_controlled_recurrent_dynamics\run_e13_scout_20260818_193353`  

**Standardized Origin:** Random 2W Baseline ($L_0 = 4096$)  
**Longitudinal Ruler:** Frozen Baseline Axis $u_0 = (z_D(0) - z_R(0)) / \|z_D(0) - z_R(0)\|$  
**Inference:** Longitudinal Pair-Cluster Bootstrap ($B=10,000$) across 4 value pairs.

## Preregistered Confirmatory Endpoint Hierarchy

1. **Primary Endpoint A (Fate of Historical Representation):** $V_{\text{intact}}^{(0)}(2048)$
   - **Result:** $V_{\text{intact}}^{(0)}(2048) = \mathbf{-23.42}$ [-37.77, -5.82]

2. **Primary Endpoint B (Causal Carry-Evolution Effect):** $\Delta V_{\text{carry\_effect}}^{(0)}(2048) = V_{\text{intact}}^{(0)}(2048) - V_{\text{clamped}}^{(0)}(2048)$
   - **Result:** $\Delta V_{\text{carry\_effect}}^{(0)}(2048) = \mathbf{-38.98}$ [-60.30, -12.13]

4. **Trajectory Anatomy:** Horizons $N \in \{16, 64, 256, 1024\}$ on frozen baseline axis $u_0$.

5. **Secondary Re-expression:** Contemporaneous steerability $V^{(N)}(N)$ and logit axis cosine $C_{\text{logit}}(N)$.


## 1. Primary Value Specificity Trajectory $V^{(0)}(N)$ across Horizons

| Horizon $N$ | $P_{\text{match}}^{(0)}(N)$ | $P_{\text{wrong\_val}}^{(0)}(N)$ | $V_{\text{intact}}^{(0)}(N)$ | 95% Bootstrap CI | $\Delta \alpha_{\text{value\_spec}}^{(0)}(N)$ | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $N=0$ | +92.20 | +59.35 | **+32.85** | **[+8.81, +56.90]** | **+0.3052** | Positive (Resolved) |
| $N=16$ | -29.17 | -24.32 | **-4.85** | **[-29.90, +12.46]** | **-0.0237** | Unresolved |
| $N=64$ | -13.68 | +17.32 | **-31.00** | **[-47.37, -14.63]** | **-0.2130** | Negative (Resolved) |
| $N=256$ | +4.55 | -0.32 | **+4.87** | **[-15.76, +28.69]** | **+0.0204** | Unresolved |
| $N=1024$ | -27.24 | -6.25 | **-20.99** | **[-45.26, +3.27]** | **-0.1588** | Unresolved |
| $N=2048$ | -6.48 | +16.94 | **-23.42** | **[-37.77, -5.82]** | **-0.1362** | Negative (Resolved) |

## 2. Causal Arm Comparison: Intact Recurrence vs. RG-LRU Carry Clamped

| Horizon $N$ | $V_{\text{intact}}^{(0)}(N)$ | $V_{\text{clamped}}^{(0)}(N)$ | $\Delta V_{\text{carry\_effect}}^{(0)}(N)$ | 95% Bootstrap CI | Dynamical Interpretation |
| :---: | :---: | :---: | :---: | :---: | :--- |
| $N=0$ | +32.85 | +32.85 | **+0.00** | **[+0.00, +0.00]** | No Resolved Arm Difference |
| $N=16$ | -4.85 | +36.08 | **-40.93** | **[-84.58, -4.59]** | Old-axis attenuation relative to carry-clamped control |
| $N=64$ | -31.00 | +31.74 | **-62.74** | **[-110.60, -19.57]** | Old-axis attenuation relative to carry-clamped control |
| $N=256$ | +4.87 | +4.49 | **+0.38** | **[-35.52, +28.52]** | No Resolved Arm Difference |
| $N=1024$ | -20.99 | -13.44 | **-7.55** | **[-37.40, +16.81]** | No Resolved Arm Difference |
| $N=2048$ | -23.42 | +15.56 | **-38.98** | **[-60.30, -12.13]** | Old-axis attenuation relative to carry-clamped control |

## 3. Regime-Specific Longitudinal Trajectories & Causal Carry Effects

### 3.1 Intact Specificity $V_{\text{intact}}^{(0)}(N)$ by Drive Regime

| Horizon $N$ | `constant` [95% CI] | `random` [95% CI] | `natural` [95% CI] | `interfering` [95% CI] |
| :---: | :---: | :---: | :---: | :---: |
| $N=0$ | **+32.85** [+8.81, +56.90] | **+32.85** [+8.81, +56.90] | **+32.85** [+8.81, +56.90] | **+32.85** [+8.81, +56.90] |
| $N=16$ | **+37.84** [-4.59, +80.27] | **-36.01** [-57.46, -7.55] | **-22.06** [-89.65, +20.08] | **+0.84** [-14.90, +13.60] |
| $N=64$ | **-44.30** [-117.32, +13.14] | **-12.29** [-43.07, +7.96] | **-14.15** [-116.34, +74.29] | **-53.26** [-107.60, +1.07] |
| $N=256$ | **-9.19** [-43.17, +24.04] | **+6.19** [-15.19, +27.57] | **+4.63** [-3.55, +14.97] | **+17.85** [-38.74, +103.11] |
| $N=1024$ | **-26.62** [-51.73, -4.73] | **-2.49** [-8.13, +3.96] | **+34.68** [-1.80, +99.27] | **-89.54** [-134.07, -45.02] |
| $N=2048$ | **-46.29** [-69.88, -11.99] | **+1.46** [-15.28, +24.79] | **-23.44** [-104.88, +39.48] | **-25.41** [-78.13, +27.31] |

### 3.2 Causal Carry Effect $\Delta V_{\text{carry\_effect}}^{(0)}(N)$ by Drive Regime

| Horizon $N$ | `constant` [95% CI] | `random` [95% CI] | `natural` [95% CI] | `interfering` [95% CI] |
| :---: | :---: | :---: | :---: | :---: |
| $N=0$ | **+0.00** [+0.00, +0.00] | **+0.00** [+0.00, +0.00] | **+0.00** [+0.00, +0.00] | **+0.00** [+0.00, +0.00] |
| $N=16$ | **+63.49** [-12.22, +139.21] | **-68.91** [-127.71, -18.67] | **-119.29** [-235.19, -3.40] | **-39.01** [-104.56, +27.76] |
| $N=64$ | **-111.82** [-229.20, -19.56] | **-63.34** [-168.44, +22.10] | **+7.59** [-139.28, +140.84] | **-83.39** [-134.44, -32.35] |
| $N=256$ | **-19.79** [-56.72, +16.50] | **-16.34** [-104.38, +50.74] | **-4.35** [-66.08, +56.94] | **+41.98** [-5.00, +92.81] |
| $N=1024$ | **-30.38** [-101.38, +8.96] | **-18.51** [-27.71, -9.15] | **+95.97** [+26.06, +172.33] | **-77.29** [-115.48, -39.10] |
| $N=2048$ | **-40.45** [-58.50, -11.95] | **-4.79** [-8.13, +0.72] | **-7.37** [-103.54, +61.37] | **-103.29** [-181.02, -28.06] |

### 3.3 Contemporaneous Steerability $V_{\text{intact}}^{(N)}(N)$ (Projection on $u_N$)

| Horizon $N$ | Pooled [95% CI] | `constant` [95% CI] | `random` [95% CI] | `natural` [95% CI] | `interfering` [95% CI] |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $N=0$ | **+32.85** [+8.81, +56.90] | **+32.85** [+8.81, +56.90] | **+32.85** [+8.81, +56.90] | **+32.85** [+8.81, +56.90] | **+32.85** [+8.81, +56.90] |
| $N=16$ | **+26.91** [+18.80, +38.88] | **+26.36** [+11.95, +43.04] | **+46.47** [+30.10, +62.24] | **+27.53** [-20.21, +91.27] | **+7.27** [-13.09, +31.10] |
| $N=64$ | **+3.89** [-12.80, +20.51] | **+58.63** [+8.74, +135.87] | **-16.83** [-54.24, +6.93] | **-66.46** [-126.12, -6.80] | **+40.23** [-36.44, +114.83] |
| $N=256$ | **+36.68** [+14.56, +58.81] | **+52.40** [+24.33, +80.46] | **-1.38** [-21.69, +23.83] | **+23.83** [-2.93, +56.78] | **+71.89** [+5.65, +138.13] |
| $N=1024$ | **+35.13** [+3.24, +69.67] | **+16.07** [-42.30, +100.80] | **+5.41** [+3.67, +7.26] | **+58.93** [+13.21, +111.64] | **+60.12** [-33.04, +153.28] |
| $N=2048$ | **-11.05** [-24.23, +4.23] | **-11.02** [-44.98, +22.94] | **+11.77** [-3.83, +27.37] | **-35.60** [-135.49, +24.49] | **-9.36** [-98.78, +56.51] |

## 5. Secondary Controls & Structural Contrast ($N=0$ and $N=2048$)

| Horizon $N$ | $P_{\text{wrong\_val}}^{(0)}$ | $P_{\text{noise}}^{(0)}$ | $\Delta P_{\text{struct\_vs\_noise}}^{(0)}$ | 95% Bootstrap CI | Status |
| :---: | :---: | :---: | :---: | :---: | :--- |
| $N=0$ | +59.35 | +10.01 | **+49.34** | **[-13.64, +142.82]** | Unresolved |
| $N=2048$ | +16.94 | +16.10 | **+0.83** | **[-18.91, +18.42]** | Unresolved |

## 6. Primary Endpoint ($N=2048$) Leave-One-Family-Out (LOFO) Robustness

| Left-Out Family | Remaining Pairs | $V_{\text{intact}}^{(0)}(2048)$ (LOFO) | 95% Bootstrap CI | Robustness |
| :--- | :---: | :---: | :---: | :--- |
| `archived_artifact` | 3 | **-19.43** | **[-40.16, +4.03]** | Unresolved |
| `marked_object` | 3 | **-23.83** | **[-40.16, +4.03]** | Unresolved |
| `monitored_signal` | 3 | **-17.84** | **[-35.37, +4.03]** | Unresolved |
| `sealed_container` | 3 | **-32.57** | **[-40.16, -22.18]** | Robustly Negative |
