# Horizon 2 Core Retrospective Memo: Causal Latent Continuity & Empirical Dissociations

**Date:** 2026-08-20  
**Scope:** Sprints S10–S13 (`google/recurrentgemma-2b`)  
**Status:** **FROZEN RESEARCH MEMO (Calibrated with S13 Confirmatory & S13.3 Methodological Sensitivity Results)**

---

## 1. Executive Synthesis: What Horizon 2 Core Established

Across Sprints S10, S11b, S12b, S12c, and S13, Horizon 2 Core established **causal latent memory carrying value-specific historical information that undergoes rapid geometric reorientation under continued processing, rather than static, autonomous, or privileged continuity**:

> *The recurrent state of `RecurrentGemma-2B` is hidden from prompt text but exactly reconstructible from public token history ($S_t = \mathcal{F}_\theta(x_{1:t})$); it persists physically long after local sliding-window attention has evicted direct access to historical tokens ($L=4096 = 2W$); it directly and causally steers the downstream logit distribution along the donor trajectory ($P_{\text{RGLRU}} = +74.10$); it carries value-specific historical information beyond syntactic sentence templates ($\Delta P_{\text{value\_spec}} = +38.49$, $\Delta \alpha_{\text{value\_spec}} = +0.1744$); but explicit factual retrieval of the original binding has largely disappeared from behavioral output. Under continued task-irrelevant drive (S13), historical value-specific steering rapidly loses stable alignment with its original output-space direction ($V^{(0)}(2048) = +4.70$ [$-5.52, +15.85$]), while value-specific causal structure remains detectable relative to the model's evolved output geometry ($V^{(N)}(2048) = +13.95$ [$+3.20, +24.72$]). Meanwhile, the recurrent A/B state difference remains physically present but becomes largely reoriented relative to its $N=0$ direction ($C_R(2048) = +0.1238$ [$+0.0953, +0.1545$]), with magnitude evolution strongly dependent on input statistics. The state has not yet demonstrated autonomous internal evolution, source ownership, metacognitive access, or informational privilege.*

---

## 2. The Six-Way Theoretical Taxonomy

Horizon 2 Core replaces the coarse binary distinction ("external prompt memory vs. native recurrence") with a six-way property taxonomy:

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│ THE HORIZON 2 CORE THEORETICAL TAXONOMY                                                           │
├──────────────────────┬───────────────────────────────┬──────────────────────┬─────────────────────┤
│ Dimension            │ Core Question                 │ Empirical Status     │ Empirical Grounding │
├──────────────────────┼───────────────────────────────┼──────────────────────┼─────────────────────┤
│ 1. Reconstructibility│ Can an external observer      │ YES                  │ S10 Replay Invariant│
│                      │ reconstruct state from text?  │ (Privacy != Priv)    │ (S_t = F_theta(x))  │
├──────────────────────┼───────────────────────────────┼──────────────────────┼─────────────────────┤
│ 2. Persistence       │ Does historical information   │ YES                  │ S11b RG-LRU Traces  │
│                      │ physically survive over time? │ (At 2W = 4096 tokens)│ (R_RGLRU ~ 0.34)    │
├──────────────────────┼───────────────────────────────┼──────────────────────┼─────────────────────┤
│ 3. Causal Leverage   │ Does changing state change    │ YES                  │ S12b Surgical Swaps │
│                      │ subsequent model computation? │ (P_RGLRU = +74.10)   │ (CI [46.79, 106.72])│
├──────────────────────┼───────────────────────────────┼──────────────────────┼─────────────────────┤
│ 4. Value Specificity │ Does state carry specific     │ YES                  │ S12c Specificity    │
│                      │ token value vs template info? │ (Delta P = +38.49)   │ (CI [25.82, 50.85]) │
├──────────────────────┼───────────────────────────────┼──────────────────────┼─────────────────────┤
│ 5. Dynamical Fate &  │ Is historical causal steering │ NO (Coordinate loss  │ S13 Confirmatory &  │
│    Coordinate Stability│ stable as experience goes on?│ on u0; reorients)    │ S13.3 Sensitivity   │
├──────────────────────┼───────────────────────────────┼──────────────────────┼─────────────────────┤
│ 6. Access / Ownership│ Can the model monitor its own │ UNTESTED / UNKNOWN   │ Topic of S14+       │
│                      │ state with unique privilege?  │ (Secret Injections)  │ (Base vs IT Models) │
└──────────────────────┴───────────────────────────────┴──────────────────────┴─────────────────────┘
```

### Critical Epistemic Guardrails
- **Hidden $\ne$ Privileged:** An internal state variable not exposed in prompt text can still be 100% determined by public tokens.
- **Recurrent $\ne$ Autonomous:** Gated linear recurrence is input-driven; absence of token input means absence of transition clock.
- **Causal $\ne$ Conscious / Metacognitive:** A physical state can steer logits without the system having introspective access to that state.
- **Representation $\ne$ Reportability:** Latent output dispositions can carry historical structure even when factual cloze recovery fails.
- **Specific $\ne$ Coordinate-Stable:** Value-specific historical structure can remain causally active while rapidly losing alignment with its original output-space coordinate system ($u_0$).
- **Same Mathematical Model $\ne$ Identical Realized Trajectory:** Under long recurrent trajectories in finite precision (BF16), execution batch shape can materially alter realized output trajectories while preserving aggregate state-space reorientation.

---

## 3. The Longitudinal Dissociation Ladder (S10–S13)

```
N=0 (Standardized 2W Origin):
  Physical State Differs (r_0 != 0) ──► Specific Causal Steering on u_0 (V(0) = +39.58) ──► Cloze Fact Recovery Fails (Cloze ~ 0)

Continued Processing (N = 16 -> 2048):
  State Rotation Occurs (C_R -> 0.12) ──► Historical u_0 Steering Lost (V^(0)(2048) ~ 0) ──► Contemporaneous Steering Active (V^(N)(2048) = +13.95)
```

1. **Physical Persistence & State Reorientation (S11b / S13):**
   Branch-specific RG-LRU separation remains resolved at $2W=4096$ tokens ($R \approx 0.045\text{--}0.338$). Under subsequent task-irrelevant drive ($N=2048$), the state difference vector $r(t)$ does not vanish, but rotates toward near-orthogonality ($C_R(2048) = +0.1238$ [$+0.0953, +0.1545$]), while state quotient $Q_R(2048)$ exhibits massive drive-dependent divergence (`constant`: $Q_R \approx 15.31$ vs `random`: $Q_R \approx 0.94$).
2. **Loss of Historical Coordinate Alignment (S13):**
   Holding the baseline measurement axis $u_0$ frozen, steering capacity drops from $V(0) = +39.58$ [$+2.97, +77.03$] to an unresolved $V^{(0)}(2048) = +4.70$ [$-5.52, +15.85$].
3. **Persistence of Contemporaneous Steerability (S13):**
   When measured along the contemporaneous logit axis $u_N$, steerability remains resolved positive at $N=2048$ ($V^{(N)}(2048) = +13.95$ [$+3.20, +24.72$]).
4. **Causal Carry Clamping (S13):**
   Repeatedly clamping RG-LRU carry to $S_0$ vs allowing free evolution yields no resolved pooled advantage on $u_0$ at $N=2048$ ($\Delta V_{\text{carry}}^{(0)}(2048) = +4.41$ [$-8.54, +17.71$]), as both arms dissipate on the historical coordinate frame. Regime-specific estimates suggest that the causal effect of recurrent-carry evolution may depend on the statistics of subsequent input.
5. **Methodological Invariance (S13.3):**
   In a sparse 4-pair numerical sensitivity panel, aggregate state-space reorientation ($C_R(N)$) is qualitatively batch-robust across $B=1$ vs $B=5$ ($C_R(2048) \approx 0.195 \leftrightarrow 0.144$), while individual trajectory-level causal expressions are execution-sensitive.

---

## 4. Horizon 2 Core Status: FROZEN (S10–S13)

```
+---------------------------------------------------------------------------------------------------+
| HORIZON 2 CORE SPRINT PROGRESSION (S10–S13: FROZEN)                                               |
+---------------------------------------------------------------------------------------------------+
| S10: Fail-Closed Model Bring-Up & Replay Invariants       | FROZEN (Replay Invariant Verified)    |
| S11b: Latent Impulse Retention & Temporal Anatomy         | FROZEN (Physical Persistence at 2W)   |
| S12b: Multi-Store Surgical Swaps & Causal Attribution     | FROZEN (Causal Steering P = +74.10)   |
| S12c: Specificity Microscope (Within-Template vs Value)   | FROZEN (Value Spec = +38.49 [25, 50]) |
| S13: Controlled Dynamics, Coordinate Loss & Geometry      | FROZEN (C_R -> 0.12, V^(N) > 0, V^(0)~0)|
+---------------------------------------------------------------------------------------------------+
| HORIZON 2 METALOGICAL FRONTIER (S14+)                                                             |
+---------------------------------------------------------------------------------------------------+
| S14: Latent Metacognition, Reality Monitoring & Ownership | Secret Injections into Evolving States|
| S15: Recurrent Adapter Prototype & Low-Rank Continuity    | Cross-Session Parameterized Memory    |
| S16: Monitor/Content Dissociation & Level 2 Synthesis     | Final H2 Go/No-Go Decision for H3     |
+---------------------------------------------------------------------------------------------------+
```
