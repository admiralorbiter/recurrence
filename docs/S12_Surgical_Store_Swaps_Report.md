# Sprint S12: Multi-Store Surgical State Swaps & Causal Attribution Report

**Horizon 2 (Level 2: Latent Recurrence)**  
**Milestone:** H2 Causal Latent-Continuity Core Milestone (S10–S12)  
**Substrate:** Upstream Hugging Face `google/recurrentgemma-2b` / Griffin Hybrid Architecture (26 Layers, Hidden Size 2560, LRU Width 2560, Conv Width 4, Attention Window 2048)  
**Status:** **S12b Confirmatory Run COMPLETED & FROZEN (5,280 Swap Records + 160 Mediational Unrolls, 10,000-Draw Pair-Cluster Bootstrap)**

---

## 1. Executive Scientific Summary

Sprint S12 provides **preregistered confirmatory causal evidence that RG-LRU state retains history-dependent output leverage at twice the model's local attention window** ($2W=4096$ tokens) in `google/recurrentgemma-2b`:

1. **Primary Physical-Causal Steering ($P_{\text{RGLRU}}$ at $2W=4096$):**
   $$\text{Observed } P_{\text{RGLRU}}(2W) = \mathbf{+74.10} \quad [\text{95\% Pair-Cluster Bootstrap CI: } +46.79, +106.72]$$
   Surgically transplanting matching RG-LRU state into recipient branches at twice the attention window produces a clearly positive confirmatory interval under the preregistered pair-cluster bootstrap. Recurrent states do not merely maintain passive physical differences (S11b); they actively steer downstream logit outputs along the donor trajectory.
2. **Matching-History Enrichment vs. Cross-History Steering ($\Delta P_{\text{spec}}$):**
   $$\Delta P_{\text{spec\_unrel}} = P_{\text{matching}} - P_{\text{unrelated}} = \mathbf{+19.68} \quad [\text{95\% CI: } +1.84, +39.12; \text{ interval excludes zero}]$$
   $$\Delta P_{\text{spec\_perm}} = P_{\text{matching}} - P_{\text{permuted}} = \mathbf{+29.64} \quad [\text{95\% CI: } +11.82, +52.47; \text{ interval excludes zero}]$$
   $$\Delta P_{\text{spec\_noise}} = P_{\text{matching}} - P_{\text{noise}} = \mathbf{+56.46} \quad [\text{95\% CI: } +29.45, +89.47; \text{ interval excludes zero}]$$
   Under balanced cyclic derangements ($+1$ and $+7$ shifts across the 20 stimulus pairs), matching RG-LRU produces a confirmed matching-history enrichment. However, cross-pair recurrent states also produce substantial positive alignment ($P_{\text{unrel}} = +54.42$, $P_{\text{perm}} = +44.46$), indicating that surviving recurrent states carry both a generic/cross-history event manifold and a selective matching-history increment.
3. **Preregistered Null: Absolute Store Contrast ($\Delta P_{\text{kv\_minus\_rglru}}$):**
   $$\Delta P_{\text{kv\_minus\_rglru}}(2W) = P_{\text{KV}}(2W) - P_{\text{RGLRU}}(2W) = \mathbf{-11.65} \quad [\text{95\% CI: } -49.02, +20.11; \text{ spans zero}]$$
   At $2W$, there is no resolved absolute displacement difference between KV and RG-LRU. While the cellwise normalized share $\alpha_{\text{KV}}^{\text{logit}} = 0.632$ vs $\alpha_{\text{RGLRU}}^{\text{logit}} = 0.368$ reflects algebraic complement identities under varying total contrast, absolute steering power between the two physical channels remains statistically unresolved at $2W$.
4. **Temporal Causal Growth Beyond the Window ($\Delta P_{\text{growth}}$):**
   $$\text{Observed } \Delta P_{\text{growth}}(2W - [W+1]) = \mathbf{+52.66} \quad [\text{95\% CI: } +26.66, +83.78; \text{ interval excludes zero}]$$
   The absolute directional displacement of RG-LRU expands from $W+1=2049$ ($P_{\text{RGLRU}} = +21.44$) to $2W=4096$ ($P_{\text{RGLRU}} = +74.10$).
5. **Descriptive KV Mediation Across Full Window Turnover ($N=2048$):**
   Full window turnover substantially reduces recipient anchoring in the post-graft KV representation ($\text{mean } \mathcal{M}_{\text{post}}$ shifts from $-0.451$ at $N=512$ to $-0.230$ at $N=2048$). The combined mean remains recipient-side at $N=2048$, though several constant-regime items cross into donor-side geometry ($\mathcal{M}_{\text{post}} > 0$).

---

## 2. Confirmatory S12b Run Manifest & Provenance

The following metadata are serialized directly from the authoritative run manifest (`summary.json`):

- **Run Directory:** `results/e11_surgical_swaps/run_e11_confirmatory_20260818_152553`
- **Model Target:** `google/recurrentgemma-2b` (bfloat16 on CUDA, NVIDIA GeForce RTX 3060)
- **Model Revision:** `3620f4ca9c5d16ee56c00180474a3201ec7f734a`
- **Git HEAD Commit at Execution:** `870e0a9fabcef1778fcd2c06d55021fe9fb92363` (Clean worktree)
- **Protocol Code SHA-256:** `83ef4b6135aead5487a6bcad1a29464dfa308880c780b92c1af8428b5eb78a94`
- **Donor Mapping SHA-256:** `6bd47bddd1daba45e3fbac3ace4d505eaa397bddb9b337e94ede5a89e601437a`
- **Audited Vocabulary SHA-256:** `29f280e76627890b0b4468e72b684c5b9d0dee348b6691f1bac36b32952e03c1`
- **Execution Time:** 14,723.10s (~4.08 hours)
- **Dataset Scope:** 5,280 swap records (20 pairs $\times$ 4 regimes $\times$ 3 lags $\times$ 22 conditions) + 160 mediational forward unrolls (horizons $N=512$ and $N=2048$).

---

## 3. Preregistered S12b Estimands & 10,000-Draw Pair-Cluster Bootstrap Panel

| Estimand | Description | Observed Estimate | 95% Bootstrap CI | Confirmatory Inference |
| :--- | :--- | :---: | :---: | :--- |
| `p_match_2w` | **Primary Physical-Causal Endpoint:** $P_{\text{RGLRU}}(2W)$ | **+74.0994** | **[+46.7899, +106.7161]** | **Positive; excludes zero** |
| `delta_p_spec_unrel_2w` | **Primary Paired Specificity:** $P_{\text{match}}(2W) - P_{\text{unrel}}(2W)$ | **+19.6759** | **[+1.8384, +39.1219]** | **Positive; excludes zero** |
| `delta_p_spec_perm_2w` | **Secondary Paired Specificity:** $P_{\text{match}}(2W) - P_{\text{perm}}(2W)$ | **+29.6404** | **[+11.8234, +52.4697]** | **Positive; excludes zero** |
| `delta_p_spec_noise_2w` | **Matched Frobenius Noise Contrast:** $P_{\text{match}}(2W) - P_{\text{noise}}(2W)$ | **+56.4601** | **[+29.4490, +89.4735]** | **Positive; excludes zero** |
| `delta_p_growth_2w_minus_w1` | **Temporal Causal Growth:** $P_{\text{match}}(2W) - P_{\text{match}}(W+1)$ | **+52.6587** | **[+26.6603, +83.7810]** | **Positive; excludes zero** |
| `delta_p_kv_minus_rglru_2w` | **Absolute Store Contrast:** $P_{\text{KV}}(2W) - P_{\text{match}}(2W)$ | **-11.6512** | **[-49.0193, +20.1108]** | **Unresolved; spans zero** |
| `alpha_match_2w` | **RG-LRU Relative Directional Share:** $\alpha_{\text{RGLRU}}^{\text{logit}}(2W)$ | **+0.3680** | **[+0.3330, +0.4034]** | Normalized cellwise fraction |
| `alpha_kv_2w` | **KV Relative Directional Share:** $\alpha_{\text{KV}}^{\text{logit}}(2W)$ | **+0.6320** | **[+0.5966, +0.6670]** | Normalized cellwise fraction |
| `p_match_w1` | RG-LRU Displacement at $W+1$: $P_{\text{RGLRU}}(W+1)$ | +21.4408 | [+17.3425, +25.8959] | Baseline checkpoint |
| `p_match_l8` | RG-LRU Displacement at $L=8$: $P_{\text{RGLRU}}(L=8)$ | +18.2964 | [+13.5355, +23.1668] | Baseline checkpoint |
| `p_unrel_2w` | Unrelated Donor Displacement at $2W$: $P_{\text{unrel}}(2W)$ | +54.4236 | [+32.2609, +77.1805] | Cross-pair baseline |
| `p_perm_2w` | Permuted Donor Displacement at $2W$: $P_{\text{perm}}(2W)$ | +44.4590 | [+32.0241, +57.5830] | Cross-pair baseline |
| `p_noise_2w` | Matched Noise Displacement at $2W$: $P_{\text{noise}}(2W)$ | +17.6393 | [+10.7672, +25.4122] | Perturbation floor |
| `p_kv_2w` | KV Displacement at $2W$: $P_{\text{KV}}(2W)$ | +62.4483 | [+54.7684, +69.7399] | Attention cache store |
| `p_whole_2w` | Whole State Displacement at $2W$: $P_{\text{whole}}(2W)$ | +136.5477 | [+111.7998, +165.1752] | Total state transfer |

---

## 4. Scientific Discussion & Methodological Calibration

### A. Matching-History Specificity vs. Generic Recurrent Alignment
At $2W=4096$:
- Matching RG-LRU: $P_{\text{match}} = +74.10$
- Unrelated Derangement (+1): $P_{\text{unrel}} = +54.42$
- Permuted Derangement (+7): $P_{\text{perm}} = +44.46$
- Frobenius Noise: $P_{\text{noise}} = +17.64$

The matching increments ($\Delta P_{\text{spec\_unrel}} = +19.68$, $\Delta P_{\text{spec\_perm}} = +29.64$) prove a selective matching-history component. However, the substantial baseline displacement of cross-pair recurrent donors ($+54.42$) indicates that recurrent states also carry shared event-manifold or task-template structure across items. Because the canonical stimulus panel uses distinct syntactic templates ("marked object," "container," "artifact," "signal"), cross-pair donors shift both historical value and task template. Isolating pure value-specific memory from same-template alignment will be addressed in future within-template controls.

### B. Store Share vs. Absolute Displacement
While RG-LRU's normalized share rises from $\alpha = 1.8\%$ at $L=8$ to $\alpha = 36.8\%$ at $2W$, the preregistered absolute displacement contrast ($P_{\text{KV}} - P_{\text{RGLRU}} = -11.65$, 95% CI $[-49.02, +20.11]$) spans zero. The rise in $\alpha$ reflects the collapse of the dominant initial KV contrast alongside positive absolute RG-LRU displacement, not a proven superiority of one store over another.

### C. Descriptive KV Mediation Dynamics
Unrolling future filler tokens to full window turnover ($N=2048$) reduces recipient anchoring ($\mathcal{M}_{\text{post}}$ moves $+0.221$ toward the donor), but the grand mean remains recipient-side ($\mathcal{M}_{\text{post}} = -0.230$). While several low-entropy items demonstrate donor-side migration, formal causal inference on representation reshaping requires further dedicated modeling.

---

## 5. Horizon 2 Roadmap Integration & Next Steps

With S10, S11b, and S12b complete, the **H2 Causal Latent-Continuity Core** is frozen. The program continues within Horizon 2 across S13–S16:

- **S10:** Recurrent Model Bring-Up & Invariants (**COMPLETE**)
- **S11b:** Latent Impulse Retention & Scale-Relative Persistence (**FROZEN**)
- **S12b:** Multi-Store Surgical Swaps & Causal Attribution (**FROZEN**)
- **S13 (Next Sprint):** **Null-Observation Recurrence & Spontaneous State Evolution**  
  *Question:* Does latent recurrent state undergo selective, stable causal evolution when no new external semantic input enters (null tokens, identity, decay)?
- **S14:** **Latent Metacognition, Reality Monitoring & State Ownership**  
  *Question:* Can the model distinguish internal interventions on its own recurrent state from external narrative events?
- **S15:** **Recurrent Adapter Prototype & Low-Rank State Continuity**
- **S16:** **Monitor/Content Dissociation & Level 2 Synthesis**
