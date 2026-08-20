# Sprint S13.3 Methodological Sensitivity Report: Strict Same-Four Paired Analysis (B=1 vs B=5)

**Target Scope:** Sparse 4-Pair Numerical Sensitivity Panel (1 canonical pair per template family):
1. `archived_artifact_p01_marble_quartz` (Family: `archived_artifact`)
2. `marked_object_p01_amber_cobalt` (Family: `marked_object`)
3. `monitored_signal_p01_alpha_delta` (Family: `monitored_signal`)
4. `sealed_container_p01_copper_silver` (Family: `sealed_container`)

**Methodological Design:**  
Evaluated across all 4 drive regimes (`constant`, `random`, `natural`, `interfering`) and 2 causal arms (`intact_recurrence`, `rglru_carry_clamped`) through $N=2048$ under pure sequential $B=1$ execution. These results are compared directly against the exact same 4 pairs extracted from the 11,520-row $B=5$ confirmatory dataset ([`5396259`](https://github.com/admiralorbiter/recurrence/commit/5396259)).

> **Panel Characterization Note:**  
> This 4-pair panel begins at $V^{(0)}(0) = -30.84$ (matching bit-identically between $B=1$ and $B=5$), while the full 24-pair population begins at $V^{(0)}(0) = +39.58$. This panel is designed as a sparse numerical sensitivity test spanning all template families to evaluate computational batch-shape dependence, not as a population re-estimation.

---

## 1. Longitudinal Paired Comparison across Horizons

$$\Delta_{\text{batch}} E = E_{B=1} - E_{B=5}$$

| Horizon $N$ | $V_{\text{intact}}^{(0)}$ ($B=1$) | $V_{\text{intact}}^{(0)}$ ($B=5$) | $\Delta V_{\text{carry}}^{(0)}$ ($B=1$) | $\Delta V_{\text{carry}}^{(0)}$ ($B=5$) | $C_R$ ($B=1$) | $C_R$ ($B=5$) | $V^{(N)}$ ($B=1$) | $V^{(N)}$ ($B=5$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N=0$** | **-30.84** | **-30.84** | **+0.00** | **+0.00** | **1.0000** | **1.0000** | **-30.84** | **-30.84** |
| **$N=16$** | -5.38 | +10.20 | -3.44 | +20.70 | 0.6679 | 0.6830 | +51.37 | +32.18 |
| **$N=64$** | -4.96 | +25.22 | -19.99 | +26.66 | 0.5874 | 0.5778 | +37.86 | +39.21 |
| **$N=256$** | -2.75 | -35.37 | -19.11 | -59.00 | 0.3941 | 0.4059 | -3.08 | +12.44 |
| **$N=1024$** | -12.52 | -12.43 | +0.68 | -14.00 | 0.2038 | 0.2010 | -14.62 | +11.89 |
| **$N=2048$** | **+9.20** | **+1.64** | **+7.31** | **-5.42** | **0.1953** | **0.1444** | **+30.89** | **+24.60** |

---

## 2. Per-Pair Primary Endpoints at $N=2048$ (Averaged across 4 Regimes)

| Pair ID (Family) | $V^{(0)}$ ($B=1$) | $V^{(0)}$ ($B=5$) | $\Delta V_{\text{carry}}^{(0)}$ ($B=1$) | $\Delta V_{\text{carry}}^{(0)}$ ($B=5$) | $V^{(N)}$ ($B=1$) | $V^{(N)}$ ($B=5$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `archived_artifact_p01` | +59.78 | -2.08 | +37.76 | -39.77 | +0.60 | +54.89 |
| `marked_object_p01` | -22.18 | -10.53 | -51.74 | +12.40 | -8.22 | -14.80 |
| `monitored_signal_p01` | -7.93 | +20.07 | +5.13 | +27.77 | +43.23 | +44.29 |
| `sealed_container_p01` | +7.13 | -0.89 | +38.10 | -22.07 | +87.95 | +14.02 |
| **Mean across 4 Pairs** | **+9.20** | **+1.64** | **+7.31** | **-5.42** | **+30.89** | **+24.60** |

---

## 3. Methodological Assessment & Theoretical Takeaways

### A. Aggregate State-Space Reorientation is Batch-Robust
Physical state rotation $C_R(N) = \cos(r_0, r_N)$ exhibits nearly identical monotonic decay curves across both batch executions:
- $N=16: 0.6679 \leftrightarrow 0.6830$
- $N=64: 0.5874 \leftrightarrow 0.5778$
- $N=256: 0.3941 \leftrightarrow 0.4059$
- $N=1024: 0.2038 \leftrightarrow 0.2010$
- $N=2048: 0.1953 \leftrightarrow 0.1444$

The aggregate geometric trajectory of the recurrent state difference vector moving toward near-orthogonality relative to $r_0$ is a robust property of the model's recurrent dynamics.

### B. Trajectory-Level Causal Expression is Execution-Sensitive
Individual pair-level causal projections show substantial path dependence:
- At $N=2048$, 3 of the 4 pairs change sign on the old-axis contrast $V^{(0)}$ between $B=1$ and $B=5$, and the magnitude of contemporaneous steerability $V^{(N)}$ varies across batch paths.
- While the aggregate mean at $N=2048$ remains consistent with the confirmatory story ($V^{(0)}(2048)$ near zero; $V^{(N)}(2048)$ positive), individual causal trajectories are sensitive to the finite-precision numerical execution geometry.

### C. Mechanistic Assessment
Step-by-step diagnostic monitoring confirmed that $B=1$ and $B=5$ are bit-identical at tokens $N=1$ and $N=2$, with divergence appearing between $N=2$ and $N=4$. This coincides with the complete turnover of the 4-tap Conv1D rolling buffer (`conv1d_width=4`, state width 3). The precise mechanistic interaction between Conv state turnover, layer gating, and BF16 GEMM accumulation remains an open question for dedicated numerical methods work.
