# Sprint S13 Controlled Recurrent Dynamics Report

**Model Target:** `google/recurrentgemma-2b`  
**Phase:** `scout`  
**Run Path:** `results\e13_controlled_recurrent_dynamics\run_e13_scout_20260818_193353`  

**Standardized Origin:** Random 2W Baseline ($L_0 = 4096$)  
**Longitudinal Ruler:** Frozen Baseline Axis $u_0 = (z_D(0) - z_R(0)) / \|z_D(0) - z_R(0)\|$  
**Inference:** Longitudinal Pair-Cluster Bootstrap ($B=10,000$) across 4 value pairs.

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
| $N=0$ | +32.85 | +32.85 | **+0.00** | **[+0.00, +0.00]** | No Resolved Difference |
| $N=16$ | -4.85 | +36.08 | **-40.93** | **[-84.58, -4.59]** | Recurrent-Carry Suppression |
| $N=64$ | -31.00 | +31.74 | **-62.74** | **[-110.60, -19.57]** | Recurrent-Carry Suppression |
| $N=256$ | +4.87 | +4.49 | **+0.38** | **[-35.52, +28.52]** | No Resolved Difference |
| $N=1024$ | -20.99 | -13.44 | **-7.55** | **[-37.40, +16.81]** | No Resolved Difference |
| $N=2048$ | -23.42 | +15.56 | **-38.98** | **[-60.30, -12.13]** | Recurrent-Carry Suppression |

## 3. Secondary Controls & Structural Contrast ($N=0$ and $N=2048$)

| Horizon $N$ | $P_{\text{wrong\_val}}^{(0)}$ | $P_{\text{noise}}^{(0)}$ | $\Delta P_{\text{struct\_vs\_noise}}^{(0)}$ | 95% Bootstrap CI | Status |
| :---: | :---: | :---: | :---: | :---: | :--- |
| $N=0$ | +59.35 | +10.01 | **+49.34** | **[-13.64, +142.82]** | Unresolved |
| $N=2048$ | +16.94 | +16.10 | **+0.83** | **[-18.91, +18.42]** | Unresolved |

## 4. Primary Endpoint ($N=2048$) Leave-One-Family-Out (LOFO) Robustness

| Left-Out Family | Remaining Pairs | $V_{\text{intact}}^{(0)}(2048)$ (LOFO) | 95% Bootstrap CI | Robustness |
| :--- | :---: | :---: | :---: | :--- |
| `archived_artifact` | 3 | **-19.43** | **[-40.16, +4.03]** | Unresolved |
| `marked_object` | 3 | **-23.83** | **[-40.16, +4.03]** | Unresolved |
| `monitored_signal` | 3 | **-17.84** | **[-35.37, +4.03]** | Unresolved |
| `sealed_container` | 3 | **-32.57** | **[-40.16, -22.18]** | Robustly Negative |
