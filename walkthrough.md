# Horizon 2 Core Walkthrough & Synthesis (Sprints S10–S13)

**Model:** `google/recurrentgemma-2b` (Griffin Hybrid Recurrent-Attention Architecture, CUDA BF16)  
**Scope:** Sprints S10, S11b, S12b, S12c, S13 Confirmatory & S13.3 Methodological Sensitivity  
**Status:** **HORIZON 2 CORE FROZEN**

---

## 1. Executive Summary & The Six Horizon 2 Guardrails

Horizon 2 transitions from prompt-level memory scaffolds (Level 1) to **continuous latent recurrence** (Level 2). Across Sprints S10–S13, we established the empirical bounds of recurrent continuity:

1. **Hidden $\ne$ Privileged (S10):**
   * Under deterministic execution, recurrent state $S_t = \mathcal{F}_\theta(x_{1:t})$ is hidden from prompt text but exactly reconstructible by an external observer from public tokens.
2. **Persisting $\ne$ Reportable (S11b):**
   * Branch-specific RG-LRU recurrent state differences persist physically at $2W=4096$ tokens ($R_{\text{constant}} \approx 0.338$, $R_{\text{random}} \approx 0.045$), long after direct KV attention residency has expired. However, zero-shot factual cloze recall decays within the attention window.
3. **Different $\ne$ Causal (S12b):**
   * Multi-store surgical swaps prove that surviving RG-LRU recurrent state causally steers downstream output generation along the donor trajectory ($P_{\text{RGLRU}} = +74.10$ $[+46.79, +106.72]$).
4. **Causal $\ne$ Specific (S12c):**
   * Holding sentence template identical, matching history adds $+38.49$ $[+25.82, +50.85]$ ($\Delta \alpha = +0.1744$) over same-template wrong values, establishing value-specific causal binding.
5. **Specific $\ne$ Coordinate-Stable (S13):**
   * Under 2,048 tokens of subsequent task-irrelevant drive, the recurrent state difference vector $r(t) = s_A(t) - s_B(t)$ reorients toward near-orthogonality ($C_R(2048) = 0.1238$ $[0.0953, 0.1545]$). Consequently, historical value-specific steering along the baseline ruler $u_0$ decays to zero ($V^{(0)}(2048) = +4.70$ [$-5.52, +15.85$]), while contemporaneous steerability remains active in the model's evolved output geometry ($V^{(N)}(2048) = +13.95$ [$+3.20, +24.72$]).
6. **Same Mathematical Model $\ne$ Identical Realized Trajectory (S13.3):**
   * In a sparse 4-pair numerical sensitivity panel, aggregate state-space reorientation ($C_R(N)$) is batch-robust across $B=1 \leftrightarrow B=5$ ($0.1953 \leftrightarrow 0.1444$), while individual trajectory-level causal expressions ($V^{(0)}, V^{(N)}$) are execution-sensitive under finite-precision BF16 accumulation.

---

## 2. Definitive Confirmatory Results Table (Sprint S13)

**Dataset:** Full 24-Pair Confirmatory Run ($N=11,520$ records, 4 regimes $\times 2$ arms $\times 6$ horizons, 3.16 hours execution).  
**Inference:** 10,000-draw Pair-Cluster Longitudinal Bootstrap resampled at the pair level.

| Preregistered Endpoint / Diagnostic | Mathematical Formulation | Estimate | 95% Bootstrap CI | Confirmatory Inference |
| :--- | :--- | :---: | :---: | :--- |
| **Primary Endpoint A** | $V_{\text{intact}}^{(0)}(2048) = P_{\text{match}}^{(0)}(2048) - P_{\text{wrong}}^{(0)}(2048)$ | **+4.70** | **[-5.52, +15.85]** | **Unresolved (Null on $u_0$)** |
| **Primary Endpoint B** | $\Delta V_{\text{carry}}^{(0)}(2048) = V_{\text{intact}}^{(0)}(2048) - V_{\text{clamped}}^{(0)}(2048)$ | **+4.41** | **[-8.54, +17.71]** | **Unresolved (No Carry Advantage)** |
| **Primary Geometric Diagnostic** | $C_R(2048) = \cos(r_0, r_{2048})$ | **+0.1238** | **[+0.0953, +0.1545]** | **Severe State Rotation ($\sim 83^\circ$)** |
| **State Quotient** | $Q_R(2048) = \|r_{2048}\| / \|r_0\|$ | **+4.8504** | **[+3.2012, +6.6837]** | **Substantial Vector Expansion ($4.85\times$)** |
| **Logit Alignment** | $C_{\text{logit}}(2048) = u_0^\top u_{2048}$ | **+0.0391** | **[-0.0504, +0.1280]** | **Near-Complete Orthogonalization** |
| **Contemporaneous Steerability** | $V_{\text{intact}}^{(N)}(2048)$ (Projection on $u_N$) | **+13.95** | **[+3.20, +24.72]** | **Resolved Positive (Contemporaneous)** |

### Longitudinal Trajectory Anatomy
| Horizon $N$ | $P_{\text{match}}^{(0)}(N)$ | $P_{\text{wrong\_val}}^{(0)}(N)$ | $V_{\text{intact}}^{(0)}(N)$ | 95% Bootstrap CI | $C_R(N)$ [95% CI] | $C_{\text{logit}}(N)$ [95% CI] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N=0$** | +127.22 | +87.64 | **+39.58** | **[+2.97, +77.03]** | **+1.0000** [+1.0000, +1.0000] | **+1.0000** [+1.0000, +1.0000] |
| **$N=16$** | -2.75 | -7.59 | **+4.85** | **[-6.67, +15.68]** | **+0.6092** [+0.5789, +0.6431] | **-0.1004** [-0.2030, +0.0020] |
| **$N=64$** | -5.43 | -2.47 | **-2.96** | **[-14.93, +9.44]** | **+0.4843** [+0.4462, +0.5283] | **-0.0301** [-0.1256, +0.0739] |
| **$N=256$** | +4.74 | +3.29 | **+1.45** | **[-13.09, +15.82]** | **+0.3306** [+0.2900, +0.3749] | **-0.0086** [-0.1068, +0.0891] |
| **$N=1024$** | -5.30 | +10.88 | **-16.19** | **[-29.26, -3.11]** | **+0.1660** [+0.1404, +0.1940] | **+0.1256** [+0.0117, +0.2355] |
| **$N=2048$** | -5.79 | -10.48 | **+4.70** | **[-5.52, +15.85]** | **+0.1238** [+0.0953, +0.1545] | **+0.0391** [-0.0504, +0.1280] |

---

## 3. Strict Same-Four Paired Sensitivity Results (S13.3)

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

## 4. Horizon 2 Core Status & Roadmap to S14+

* **Horizon 2 Core (S10–S13):** **FROZEN**.
* **Active Frontier (Sprint S14):** Latent Metacognition, Reality Monitoring & State Ownership.
  * *Question:* A history-conditioned latent distinction persists causally while its representational coordinates evolve. Does the model have any privileged access to that evolving latent distinction?
