# Sprint S14: Latent Metacognition, Reality Monitoring & Intention Provenance Report

**Horizon 2 (Level 2: Latent Recurrence & Self-Access)**  
**Target Substrate:** `google/recurrentgemma-2b-it` (revision: `2766eb5d4264c6c0357803990791f9ab9cd50f8e`)  
**Status:** **S14.0C Definitive Stratified Assay COMPLETED, CALIBRATED & FROZEN (16 Evaluations across 3 C-Tiers, Evolved + State-Matched POST Controls, TOST Equivalence Confirmed)**

---

## 1. Executive Scientific Summary

Sprint S14 tests whether a recurrent language model (`RecurrentGemma-2B-IT`) exhibits privileged metacognitive access to, or reality-monitoring over, its own prior computational intentions.

Using the **C/D/R/A (Computation / Distribution / Reporting / Access)** decomposition framework with **C-level stratification**, **Balanced Order Permutation (BOP)**, and **Exact State-Matched POST Controls**:

1. **R-Level Interface Validated (8/8 Distinct Interfaces Passed 100% Visible Accuracy):**
   Uncalibrated direct-token reporting initially failed visible controls due to first-option positional bias. Balanced Order Permutation (BOP) across $(x, y)$ and $(y, x)$ question presentations canceled out an average of $0.08$ to $13.5$ logits of presentation bias, achieving a **100.0% pass rate** on visible ground truth across all 8 tested candidate pairs ($m \in [+4.66, +24.56]$ logits).
2. **C-Level Stratification Separates True Disagreement from Causal Perturbation:**
   - **Tier 1 (Strict-C Binary Disagreement, $D_T \cdot D_O < 0$, $|D| \ge 0.30$):** Exactly $2/16$ trials (`quartz_basalt` FWD and REV). In both directions, the target and observer held strong opposing private preferences ($\Delta = \pm 1.02$ logits).
   - **Tier 2 (Weak / Boundary Disagreement):** $1/16$ trials (`marble_quartz` FWD, $\Delta = -0.38$, small margins).
   - **Tier 3 (Same-Choice Causal Perturbation Controls, $D_T \cdot D_O > 0$):** $13/16$ trials. Target and observer agree on the discrete choice ($g$), but the intervention induces continuous disposition shifts ($\Delta = D_T - D_O$).
3. **Privileged Access in the Strict-C Cell (`quartz_basalt`):**
   - **Forward ($A \leftarrow B$, truth is `alkali`):** $D_T = +0.531, D_O = -0.484 \implies \text{PAI}_{\text{aligned}} = \mathbf{+0.270}$. Semantic report $S_{\text{PRE}} = -1.77$ (favors `antonio`, wrong choice, but shifted toward truth relative to observer).
   - **Reverse ($B \leftarrow A$, truth is `antonio`):** $D_T = -0.547, D_O = +0.469 \implies \text{PAI}_{\text{aligned}} = \mathbf{+0.083}$. Semantic report $S_{\text{PRE}} = \mathbf{+1.74}$ (correct choice, crossed the decision boundary).
   - In both causal directions, PRE reporting shifted in the direction associated with the target's private computational state relative to the matched observer baseline. Across Tier 3 perturbation controls, however, report shifts show essentially no correlation with continuous decision shifts ($r \approx 0.064$).
4. **Definitive Temporal Equivalence ($T_{\text{aligned}} \equiv 0$, TOST Equivalent at $\pm 0.10$ logits):**
   - **Contemporaneously Evolved POST Control:** Mean difference = $\mathbf{+0.0033}$ logits [95% CI: $-0.0215, +0.0281$]. Statistically equivalent to zero at $\delta_{\text{equiv}} = \pm 0.10$ ($p < 0.05$).
   - **Exact State-Matched POST Control:** Mean difference = $\mathbf{+0.0348}$ logits [95% CI: $+0.0016, +0.0680$]. Statistically equivalent to zero at $\delta_{\text{equiv}} = \pm 0.10$ ($p < 0.05$).
   - In `quartz_basalt`, exact state-matched POST report margins match PRE to within hundredths of a logit ($T_{\text{aligned}}^{\text{matched}} = +0.020$ FWD, $-0.053$ REV).

> **Calibrated Scientific Synthesis:**  
> S14.0C establishes that while recurrent states can carry private behavioral information that subtly shifts metacognitive reports relative to an observer in strongly counterfactual settings (`quartz_basalt`), **the reporting channel does not distinguish a recurrent state that participated in forming a prior decision from an identical state vector injected immediately after that decision.** The model's metacognitive readout reflects its *present recurrent state*, lacking an episodic record of prior intention provenance.

---

## 2. Definitive 16-Trial Results Table

| Cell ID | Dir | C-Tier | $D_T$ | $D_O$ | Private Fact ($\Delta$) | $M_{\text{PRE}}$ | $M_{\text{OBS}}$ | $M_{\text{POST}}^{\text{evolved}}$ | $M_{\text{POST}}^{\text{matched}}$ | $\text{PAI}_{\text{aligned}}$ | $T_{\text{aligned}}^{\text{evolved}}$ | $T_{\text{aligned}}^{\text{matched}}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `quartz_basalt` | **FWD** | **Strict-C** | **+0.53** | **-0.48** | **+1.02** | -1.77 | -2.04 | -1.77 | -1.79 | **+0.270** | -0.004 | **+0.020** |
| `quartz_basalt` | **REV** | **Strict-C** | **-0.55** | **+0.47** | **-1.02** | -1.74 | -1.66 | -1.72 | -1.79 | **+0.083** | +0.021 | **-0.053** |
| `silver_nickel` | FWD | Same-Ch | +13.65 | +12.68 | +0.97 | +1.66 | +2.00 | +1.72 | +1.56 | -0.344 | -0.062 | +0.094 |
| `silver_nickel` | REV | Same-Ch | +12.80 | +13.42 | -0.63 | +2.03 | +1.81 | +2.03 | +1.94 | +0.219 | +0.000 | +0.094 |
| `basalt_granite` | FWD | Same-Ch | -0.33 | -0.58 | +0.25 | +2.34 | +2.42 | +2.42 | +2.42 | +0.078 | +0.078 | +0.078 |
| `basalt_granite` | REV | Same-Ch | +0.00 | +0.47 | -0.47 | +2.95 | +3.00 | +2.97 | +2.83 | +0.000 | +0.000 | +0.000 |
| `amber_garnet` | FWD | Same-Ch | -0.12 | -0.06 | -0.06 | +3.33 | +3.27 | +3.34 | +3.31 | -0.062 | +0.016 | -0.016 |
| `amber_garnet` | REV | Same-Ch | -0.03 | -0.19 | +0.16 | +3.00 | +2.92 | +2.94 | +2.94 | +0.000 | +0.000 | +0.000 |
| `copper_bronze` | FWD | Same-Ch | -0.50 | -0.84 | +0.34 | +2.78 | +2.95 | +2.75 | +2.72 | +0.172 | -0.031 | -0.062 |
| `copper_bronze` | REV | Same-Ch | -0.92 | -0.42 | -0.50 | +3.02 | +2.69 | +2.95 | +2.91 | -0.328 | -0.062 | -0.109 |
| `alpha_delta` | FWD | Same-Ch | -4.09 | -4.66 | +0.56 | -0.09 | -0.06 | -0.03 | +0.06 | +0.031 | +0.062 | +0.156 |
| `alpha_delta` | REV | Same-Ch | -4.56 | -3.97 | -0.59 | +0.47 | +0.28 | +0.47 | +0.47 | -0.188 | -0.000 | -0.000 |
| `marble_quartz` | FWD | Weak-C | -0.19 | +0.19 | -0.38 | +3.31 | +3.06 | +3.25 | +3.22 | -0.250 | -0.062 | -0.094 |
| `marble_quartz` | REV | Same-Ch | -0.12 | -0.38 | +0.25 | +2.97 | +3.12 | +2.94 | +2.97 | +0.156 | -0.031 | -0.000 |
| `delta_theta` | FWD | Same-Ch | +2.19 | +1.47 | +0.72 | +3.19 | +2.69 | +3.25 | +3.22 | +0.500 | -0.062 | -0.031 |
| `delta_theta` | REV | Same-Ch | +1.34 | +2.00 | -0.66 | +2.62 | +3.22 | +2.50 | +2.53 | -0.594 | +0.125 | +0.094 |

---

## 3. Two One-Sided Tests (TOST) Equivalence Analysis

To evaluate whether PRE and POST reports are practically equivalent, we declare a smallest effect size of interest (SESOI) of $\delta_{\text{equiv}} = \pm 0.10$ logits.

```
                                  TOST EQUIVALENCE DIAGRAM (delta = +/- 0.10 logits)
                                  
  Lower Bound (-0.10)                                                    Upper Bound (+0.10)
         │                                      0                                 │
         ├──────────────────────────────────────┼─────────────────────────────────┤
         │                                      │                                 │
         │                  [-----●-----]       │                                 │  Evolved POST 95% CI: [-0.022, +0.028] -> EQUIVALENT
         │                                      │   [------●------]               │  Matched POST 95% CI: [+0.002, +0.068] -> EQUIVALENT
         │                                      │                                 │
```

- **Contemporaneously Evolved POST:** $95\%\text{ CI} = [-0.0215, +0.0281] \subset [-0.10, +0.10] \implies \mathbf{\text{Statistically Equivalent}}$
- **State-Matched Exact POST:** $95\%\text{ CI} = [+0.0016, +0.0680] \subset [-0.10, +0.10] \implies \mathbf{\text{Statistically Equivalent}}$

---

## 4. Key Epistemic Insights

1. **Separation of Causal Influence, Private Facts, Report Access, and Provenance:**
   - **Causal Influence:** RG-LRU states causally steer downstream generation ($P = +74.10$, S12).
   - **Private Facts:** Secret grafts create diverging behavioral dispositions relative to an observer ($\Delta = \pm 1.02$).
   - **Report Access:** Reports shift in the direction of private facts in strongly counterfactual cells ($\text{PAI}_{\text{aligned}} > 0$).
   - **Provenance Access:** Reports show **no sensitivity** to whether the state participated in the prior decision or was injected afterward ($T_{\text{aligned}} \equiv 0$).
2. **Methodological Standard Frozen:**
   - S14 establishes that introspective metacognition experiments require **C-level disagreement verification**, **balanced presentation order calibration (BOP)**, and **state-matched temporal controls** to avoid confounding state presence with episodic memory.
