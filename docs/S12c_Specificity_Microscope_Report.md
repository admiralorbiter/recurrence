# Sprint S12c: "Specificity Microscope" Causal Attribution Report

**Horizon 2 (Level 2: Latent Recurrence)**  
**Target Substrate:** Upstream Hugging Face `google/recurrentgemma-2b` (26 Layers, Hidden Size 2560, LRU Width 2560, Conv Width 4, Attention Window 2048)  
**Status:** **S12c Confirmatory Run COMPLETED, CALIBRATED & FROZEN (1,344 Evaluations across 24 Value Pairs, 4 Template Families, 4 Regimes; 10,000-Draw Pair-Cluster Bootstrap)**

---

## 1. Executive Scientific Summary

Sprint S12c directly tests the granularity of latent recurrent memory: **At $2W = 4096$ tokens, does the surviving RG-LRU recurrent state carry value-specific historical information, or does it merely encode generic task/template geometry?**

Using a held-out factorial panel of 24 canonical value pairs distributed across 4 syntactic template families (6 combinatorial pairs per family) evaluated at $2W=4096$ tokens across 4 filler regimes:

1. **Value-Specific Retention Confirmed ($\Delta P_{\text{value\_spec}}$):**
   $$\Delta P_{\text{value\_spec}} = P_{\text{matching}} - P_{\text{same\_template\_wrong\_value}} = \mathbf{+38.49} \quad [\text{95\% Pair-Cluster Bootstrap CI: } +25.82, +50.85]$$
   $$\Delta \alpha_{\text{value\_spec}}^{\text{proj}} = \alpha_{\text{matching}}^{\text{proj}} - \alpha_{\text{same\_template\_wrong\_value}}^{\text{proj}} = \mathbf{+0.1744} \quad [\text{95\% CI: } +0.1001, +0.2536]$$
   Holding the syntactic sentence template fixed (e.g. `"The marked object was garnet."` vs `"The marked object was zircon."`), transplanting the matching historical state (`"The marked object was cobalt."`) provides a **statistically resolved directional advantage of $+38.49$** along the recipient's target output axis (interval strictly excludes zero).
2. **Robustness Across Families & Leave-One-Family-Out (LOFO):**
   All 4 template families exhibit positive point estimates (ranging from $+14.35$ to $+64.24$), and every Leave-One-Family-Out (LOFO) subset remains robustly positive with 95% CIs strictly excluding zero ($\text{LOFO } \Delta P \in [+29.91, +46.54]$).
3. **Template Alignment Contrast ($\Delta P_{\text{template\_align}}$):**
   $$\Delta P_{\text{template\_align}} = P_{\text{same\_template\_wrong\_value}} - P_{\text{cross\_template}} = \mathbf{+7.38} \quad [\text{95\% CI: } -8.26, +24.73; \text{ spans zero}]$$
   Because this interval spans zero, we do not resolve an additional template increment over the cross-template control used here. Structured nonmatching histories steer substantially more than noise, while matching history provides a sharp, selective advantage.

> **Calibrated Scientific Core:**  
> At $2W = 4096$ tokens, RG-LRU recurrent state contains **value-specific historical information** with selective causal consequences for downstream token generation, even when the sentence template is held fixed.

---

## 2. Confirmatory Run Manifest & Provenance

Metadata serialized directly from `summary.json`:
- **Run Directory:** `results/e12_specificity_microscope/run_e12_confirmatory_20260818_155145`
- **Model Target:** `google/recurrentgemma-2b` (bfloat16 on CUDA, NVIDIA GeForce RTX 3060)
- **Model Revision:** `3620f4ca9c5d16ee56c00180474a3201ec7f734a`
- **Git HEAD Commit at Execution:** `484b16b207bf1d267599766c51500a7a7b7e16c4` (Clean worktree)
- **Protocol Code SHA-256:** `4511c424d566ced17708a8194973f83ed428cee2dbcacda869466910ce32d987`
- **Panel Audit SHA-256:** `d6c5d00168478f67b6440b653a92462c6bb79d3f61ffbb44e949079f3c719b18`
- **Execution Time:** 5,269.89s (1.46 hours)
- **Total Evaluations:** 1,344 records (24 total value pairs distributed across 4 families $\times$ 4 filler regimes $\times$ 14 symmetric conditions at $L=4096$).

> **Known Schema Note (Forward Calibrated):**  
> In the raw `microscope_trace.jsonl` artifact of this run, the `donor` column defaulted to `"cross"` for non-matching control rows (`intact`, `whole_swap`, `noise`). The analyzer relies strictly on the unambiguous `condition` column, so numerical calculations are unaffected. The runner has been updated forward to explicitly log `"none"`, `"A"`, `"B"`, `"C"`, `"D"`, `"cross"`, and `"noise"`.

---

## 3. Primary Estimands & 10,000-Draw Pair-Cluster Bootstrap Panel

| Estimand | Description | Observed Estimate | 95% Bootstrap CI | Confirmatory Inference |
| :--- | :--- | :---: | :---: | :--- |
| `delta_p_value_spec` | **Value-Specific Retention Contrast:** $P_{\text{match}} - P_{\text{same\_template\_wrong\_val}}$ | **+38.4939** | **[+25.8180, +50.8524]** | **Positive; excludes zero** |
| `delta_proj_value_spec` | **Normalized Value-Specific Projection:** $\Delta \alpha_{\text{value\_spec}}$ | **+0.1744** | **[+0.1001, +0.2536]** | **Positive; excludes zero** |
| `delta_p_template_align` | **Template Alignment Contrast:** $P_{\text{same\_template\_wrong\_val}} - P_{\text{cross}}$ | **+7.3798** | **[-8.2553, +24.7325]** | **Unresolved; spans zero** |
| `delta_p_template_vs_noise` | **Template vs. Noise Contrast:** $P_{\text{same\_template\_wrong\_val}} - P_{\text{noise}}$ | **+34.8949** | **[+5.9699, +69.6188]** | **Positive; excludes zero** |
| `delta_p_match_vs_cross` | **Matching vs. Cross-Template Contrast:** $P_{\text{match}} - P_{\text{cross}}$ | **+45.8737** | **[+30.7178, +61.6130]** | **Positive; excludes zero** |
| `delta_p_match_vs_noise` | **Matching vs. Noise Contrast:** $P_{\text{match}} - P_{\text{noise}}$ | **+73.3888** | **[+39.7344, +111.8121]** | **Positive; excludes zero** |
| `p_match` | Matching Historical State: $P_{\text{match}}(2W)$ | **+121.6190** | **[+105.9816, +138.2551]** | Target Value Steering |
| `p_wrong_val` | Same-Template Wrong-Value State: $P_{\text{wrong\_val}}(2W)$ | **+83.1252** | **[+71.7718, +95.1869]** | Alternate Value Baseline |
| `p_cross` | Cross-Template Historical State: $P_{\text{cross}}(2W)$ | **+75.7454** | **[+58.9245, +92.0811]** | Cross-Syntax Baseline |
| `p_noise` | Matched-Norm Frobenius Noise Control: $P_{\text{noise}}(2W)$ | **+48.2302** | **[+15.5731, +74.7612]** | Noise Perturbation Control |
| `p_whole` | Whole-State Positive Reference: $P_{\text{whole}}(2W)$ | **+218.7596** | **[+197.1265, +241.8022]** | Whole-State Reference |

---

## 4. Sensitivity Analyses

### 4.1. Regime-Specific Sensitivity Breakdown

Retention differs substantially by filler stream dynamics, consistent with S11b:

| Regime | $P_{\text{match}}$ | $P_{\text{wrong\_val}}$ | $P_{\text{noise}}$ | $\Delta P_{\text{value\_spec}}$ | 95% Bootstrap CI | $\Delta \alpha_{\text{value\_spec}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `constant` | +265.94 | +152.34 | +20.56 | **+113.60** | **[+79.82, +148.81]** | **+0.3504** |
| `random` | +127.22 | +87.64 | +94.87 | **+39.58** | **[+4.19, +76.49]** | **+0.2933** |
| `interfering` | +17.93 | +15.30 | +12.12 | **+2.63** | **[-0.16, +5.53]** | **+0.0194** |
| `natural` | +75.39 | +77.23 | +65.37 | **-1.83** | **[-35.92, +28.48]** | **+0.0344** |

- Under low-entropy drive (`constant`) and diverse token drive (`random`), value specificity is strongly positive and statistically resolved.
- Under active prose (`natural`) and semantic distractors (`interfering`), raw displacement decays towards the background manifold, while unitless normalized projection $\Delta \alpha$ remains positive ($+0.0344$ and $+0.0194$).

### 4.2. Template Family Breakdown & Leave-One-Family-Out (LOFO) Robustness

#### Family-Specific Value Contrasts (6 Pairs Each)
| Family | $N_{\text{pairs}}$ | $P_{\text{match}}$ | $P_{\text{wrong\_val}}$ | $\Delta P_{\text{value\_spec}}$ | 95% Bootstrap CI | $\Delta \alpha_{\text{value\_spec}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `archived_artifact` | 6 | +94.44 | +80.10 | **+14.35** | **[-9.33, +42.21]** | **+0.0372** |
| `marked_object` | 6 | +95.50 | +62.25 | **+33.25** | **[+21.55, +45.60]** | **+0.2050** |
| `monitored_signal` | 6 | +143.82 | +79.57 | **+64.24** | **[+52.59, +76.28]** | **+0.2778** |
| `sealed_container` | 6 | +152.72 | +110.58 | **+42.14** | **[+18.94, +67.07]** | **+0.1774** |

#### Leave-One-Family-Out (LOFO) Analysis (18 Pairs Each)
| Left-Out Family | Remaining Pairs | $\Delta P_{\text{value\_spec}}$ (LOFO) | 95% Bootstrap CI | Status |
| :--- | :---: | :---: | :---: | :--- |
| `archived_artifact` | 18 | **+46.54** | **[+34.73, +58.24]** | Robustly Positive |
| `marked_object` | 18 | **+40.24** | **[+24.27, +55.83]** | Robustly Positive |
| `monitored_signal` | 18 | **+29.91** | **[+16.37, +44.11]** | Robustly Positive |
| `sealed_container` | 18 | **+37.28** | **[+23.35, +50.80]** | Robustly Positive |

All 4 LOFO point estimates are robustly positive and all 4 intervals strictly exclude zero, confirming that the primary result is not an artifact of any single template family.

---

## 5. Scientific Discussion & Theoretical Framing

### 5.1. Descriptive Contrast Ladder

Rather than asserting independent latent orthogonal components, the empirical results are best understood as a **descriptive contrast ladder**:

$$\begin{aligned}
P_{\text{noise}} &= +48.23 \quad \text{[Matched-norm Frobenius noise control]} \\
P_{\text{wrong\_val}} &= +83.13 \quad \text{[Same-template wrong-value history: } \Delta P = +34.89 \text{ over noise]} \\
P_{\text{match}} &= +121.62 \quad \text{[Matching historical value: } \Delta P = +38.49 \text{ over wrong-value]} \\
P_{\text{whole}} &= +218.76 \quad \text{[Whole-state reference / positive control]}
\end{aligned}$$

1. **Non-Matching Structured Histories Exceed Noise:**  
   Any structured historical recurrent state (same-template $+83.13$ or cross-template $+75.75$) steers significantly more along the target axis than matched-norm Gaussian noise ($+48.23$, contrast $+34.89$ $[+5.97, +69.62]$).
2. **Matching History Provides a Selective Advantage:**  
   Holding sentence syntax identical, the matching historical state provides an additional $+38.49$ $[+25.82, +50.85]$ increment over alternate values in the same template.
3. **Whole-State Reference is Not a Ceiling:**  
   In individual cells (e.g. constant regime amber/cobalt), chimeric state transplantation ($P_{\text{RGLRU}} = +288.10$) overshoots the whole-state reference ($P_{\text{whole}} = +255.38$). Chimeric states can displace logits farther along the donor direction than restoring the complete donor context.

### 5.2. Methodological & Theoretical Connections

- **Causal Abstraction & Interchange Interventions:**  
  This methodology implements rigorous interchange interventions (Geiger et al., 2021; 2024), transplanting internal sub-network states produced by source inputs into counterfactual recipient trajectories to test causal alignment.
- **Activity-Silent Latent Working Memory:**  
  The dissociation observed between latent causal steerability ($P_{\text{match}} > P_{\text{wrong\_val}}$) and lack of explicit factual cloze retrieval parallels the "activity-silent" working memory literature (Wolff et al., 2017; Rose et al., 2016; Stokes, 2015), where item-specific information remains dormant in synaptic/latent parameters and is revealed only upon causal probe or perturbation.

---

## 6. Horizon 2 Roadmap Status

With S10, S11b, S12b, and S12c complete:
- **S10:** Model Bring-Up & Invariants (**COMPLETE**)
- **S11b:** Latent Impulse Retention & Scale-Relative Persistence (**FROZEN**)
- **S12b:** Multi-Store Surgical Swaps & Causal Channel Attribution (**FROZEN**)
- **S12c:** Specificity Microscope (**CONFIRMED & FROZEN**)
- **S13 (Next):** **Null-Observation / Controlled Recurrent Dynamics**  
  *Phase S13.0 (Native Null-Transition Audit) $\to$ Phase S13.1 (Controlled Task-Irrelevant Drive $V(N)$ Sweeps)*
- **S14:** **Latent Metacognition, Reality Monitoring & State Ownership**
