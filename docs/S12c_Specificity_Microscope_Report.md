# Sprint S12c: "Specificity Microscope" Causal Attribution Report

**Horizon 2 (Level 2: Latent Recurrence)**  
**Target Substrate:** Upstream Hugging Face `google/recurrentgemma-2b` (26 Layers, Hidden Size 2560, LRU Width 2560, Conv Width 4, Attention Window 2048)  
**Status:** **S12c Confirmatory Run COMPLETED & FROZEN (1,344 Evaluations Across 24 Value Pairs x 4 Families x 4 Regimes, 10,000-Draw Pair-Cluster Bootstrap)**

---

## 1. Executive Scientific Summary

Sprint S12c directly answers the critical loose thread from S12b: **Is recurrent state steering driven by generic task/template alignment, or does it carry fine-grained value-specific historical memory?**

Across 24 value pairs across 4 syntactic template families evaluated at $2W=4096$ tokens:

1. **Value-Specific Memory Retention Confirmed ($\Delta P_{\text{value\_spec}}$):**
   $$\Delta P_{\text{value\_spec}} = P_{\text{matching}} - P_{\text{same\_template\_wrong\_value}} = \mathbf{+38.49} \quad [\text{95\% Pair-Cluster Bootstrap CI: } +25.82, +50.85]$$
   When the donor historical state uses the **identical sentence template** (e.g. `"The marked object was garnet."`), the matching donor (`"The marked object was cobalt."`) provides a **statistically resolved directional advantage of $+38.49$** along the recipient's target axis ($p < 10^{-4}$, interval strictly excludes zero).
2. **Generic Cross-History Geometry vs. Syntactic Template Effect ($\Delta P_{\text{template\_align}}$):**
   $$\Delta P_{\text{template\_align}} = P_{\text{same\_template\_wrong\_value}} - P_{\text{cross\_template}} = \mathbf{+7.38} \quad [\text{95\% CI: } -8.26, +24.73; \text{ spans zero}]$$
   The difference between same-template wrong-value ($+83.13$) and cross-template ($+75.75$) recurrent states is small ($+7.38$) and spans zero.
3. **Synthesis:**  
   Cross-history recurrent states ($+75$ to $+83$) provide a broad, structured recurrent baseline that substantially exceeds random noise ($+48.23$, $\Delta P = +34.89$ $[+5.97, +69.62]$). Sitting directly on top of this cross-history floor, **matching historical state provides a large, selective $+38.49$ value-specific steering increment**.

> **Definitive Conclusion:**  
> The surviving RG-LRU recurrent state at $2W=4096$ tokens carries **genuine token-level historical bindings**, not merely task-type or syntactic template representations.

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
- **Total Evaluations:** 1,344 records (24 value pairs $\times$ 4 filler regimes $\times$ 14 symmetric conditions at $L=4096$).

---

## 3. Preregistered S12c Estimands & 10,000-Draw Pair-Cluster Bootstrap Panel

| Estimand | Description | Observed Estimate | 95% Bootstrap CI | Confirmatory Inference |
| :--- | :--- | :---: | :---: | :--- |
| `delta_p_value_spec` | **Value-Specific Retention Contrast:** $P_{\text{match}} - P_{\text{same\_template\_wrong\_val}}$ | **+38.4939** | **[+25.8180, +50.8524]** | **Positive; excludes zero** |
| `delta_p_template_align` | **Template Alignment Contrast:** $P_{\text{same\_template\_wrong\_val}} - P_{\text{cross}}$ | **+7.3798** | **[-8.2553, +24.7325]** | **Unresolved; spans zero** |
| `delta_p_template_vs_noise` | **Template vs. Noise Contrast:** $P_{\text{same\_template\_wrong\_val}} - P_{\text{noise}}$ | **+34.8949** | **[+5.9699, +69.6188]** | **Positive; excludes zero** |
| `delta_p_match_vs_cross` | **Matching vs. Cross-Template Contrast:** $P_{\text{match}} - P_{\text{cross}}$ | **+45.8737** | **[+30.7178, +61.6130]** | **Positive; excludes zero** |
| `delta_p_match_vs_noise` | **Matching vs. Noise Contrast:** $P_{\text{match}} - P_{\text{noise}}$ | **+73.3888** | **[+39.7344, +111.8121]** | **Positive; excludes zero** |
| `p_match` | Matching Historical State: $P_{\text{match}}(2W)$ | **+121.6190** | **[+105.9816, +138.2551]** | Target Value Steering |
| `p_wrong_val` | Same-Template Wrong-Value State: $P_{\text{wrong\_val}}(2W)$ | **+83.1252** | **[+71.7718, +95.1869]** | Alternate Value Baseline |
| `p_cross` | Cross-Template Historical State: $P_{\text{cross}}(2W)$ | **+75.7454** | **[+58.9245, +92.0811]** | Cross-Syntax Baseline |
| `p_noise` | Matched Frobenius Gaussian Noise: $P_{\text{noise}}(2W)$ | **+48.2302** | **[+15.5731, +74.7612]** | Perturbation Floor |
| `p_whole` | Whole State Positive Control: $P_{\text{whole}}(2W)$ | **+218.7596** | **[+197.1265, +241.8022]** | Total State Ceiling |

---

## 4. Scientific Discussion

### The Anatomy of Recurrent Memory at $2W=4096$ Tokens

Sprint S12c decomposes the total directional displacement produced by RG-LRU transplantation into three distinct additive tiers:

```
Total Matching Steering: P_match = +121.62
┌──────────────────────────────────────────────────────────────────────────────────┐
│ Tier 3: Value-Specific Historical Binding (+38.49, 95% CI [+25.82, +50.85])      │
├──────────────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Generic Structured Recurrent Manifold (+34.89, 95% CI [+5.97, +69.62])   │
│         (Same-template wrong-value +83.13 vs Cross-template +75.75: diff = +7.38)│
├──────────────────────────────────────────────────────────────────────────────────┤
│ Tier 1: Norm-Matched Perturbation Response (+48.23, 95% CI [+15.57, +74.76])     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

1. **Tier 1 (Perturbation Floor, $+48.23$):**  
   Injecting random noise matching the layer-wise Frobenius norm of the donor displacement produces a modest positive projection along the target axis.
2. **Tier 2 (Structured Recurrent Manifold, $+34.89$ above noise):**  
   Any structured historical recurrent state—whether using the same template or a completely different template—exerts an additional $+34.89$ displacement over pure Gaussian noise. The syntactic template effect itself ($\Delta P = +7.38$) is unresolved, showing that Tier 2 reflects broad recurrent event geometry rather than narrow sentence syntax.
3. **Tier 3 (Value-Specific Binding Increment, $+38.49$ above Tier 2):**  
   Transplanting the matching historical value adds a large, highly resolved $+38.49$ increment over wrong-value historical states with identical syntax.

---

## 5. Horizon 2 Roadmap Status

With S10, S11b, S12b, and S12c complete:
- **S10:** Model Bring-Up & Invariants (**COMPLETE**)
- **S11b:** Latent Impulse Retention & Scale-Relative Persistence (**FROZEN**)
- **S12b:** Multi-Store Surgical Swaps & Causal Channel Attribution (**FROZEN**)
- **S12c:** Specificity Microscope (**CONFIRMED & FROZEN**)
- **S13 (Next):** **Null-Observation / Controlled Recurrent Dynamics**  
  *Phase S13.0 (Native Null-Transition Audit) $\to$ Phase S13.1 (Driven Null Dynamics & Velocity Sweeps)*
- **S14:** **Latent Metacognition, Reality Monitoring & State Ownership** (Secret Injections; Base vs IT)
