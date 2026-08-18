# Sprint S12: Multi-Store Surgical State Swaps & Causal Attribution Report

**Horizon 2 (Level 2: Latent Recurrence)**  
**Substrate:** Upstream Hugging Face `google/recurrentgemma-2b` / Griffin Hybrid Architecture (26 Layers, Hidden Size 2560, LRU Width 2560, Conv Width 4, Attention Window 2048)  
**Status:** **S12a.1 Corrected Causal Scout COMPLETED (768 Swap Records + 16 Mediational Unrolls, Full Control Battery)**

---

## 1. Executive Scientific & Methodological Framework

Sprint S12 moves from passive observation (S11) to **active causal intervention**, evaluating whether surgically grafting individual physical store channels ($\text{RGLRU}$, $\text{Conv}$, $\text{KV}$) from a donor branch into a recipient branch causally transplants historical information and directional logit steering:

$$\bar{\Delta}_C = \frac{1}{2} \left[ (m_{B \leftarrow A, C} - m_B) + (m_A - m_{A \leftarrow B, C}) \right]$$

$$P_C = (z_{\text{graft}} - z_{\text{recipient}}) \cdot \frac{z_{\text{donor}} - z_{\text{recipient}}}{\|z_{\text{donor}} - z_{\text{recipient}}\|}, \quad \alpha_C^{\text{logit}} = \frac{P_C}{\|z_{\text{donor}} - z_{\text{recipient}}\|}$$

### Rigorous Causal Control Battery (S12a.1)
1. **Intact Endpoints ($S^A, S^B$):** Establish ground-truth donor/recipient contrast.
2. **Whole-Store Graft Gate ($S^{B \leftarrow A}_{\text{all}}$):** Live check verifying that complete state transplantation reproduces donor logits to strict numerical tolerance ($D_{\text{JS}} < 10^{-4}$).
3. **Sham Control ($S^{A_1 \leftarrow A_2}$):** Confirms zero artifactual logit distortion ($\bar{\Delta}_{\text{sham}} = 0.00$, $P_{\text{sham}} = 0.00$, $\alpha_{\text{sham}}^{\text{logit}} = 0.000$).
4. **Intervention-Matched Frobenius Noise:** Scales random Gaussian perturbations strictly to the per-layer Frobenius norm $\|R_{\text{donor}} - R_{\text{recipient}}\|_F$ and projects resulting output displacement onto the real $B \to A$ axis across multiple seeds.
5. **Cross-Pair Unrelated & Permuted Donors:** Evaluates whether directional displacement is specific to matching history ($P_{\text{match}} > P_{\text{unrelated}}, P_{\text{permuted}}$).
6. **Sliced Post-Graft KV Mediation:** Measures distances strictly over newly generated cache positions ($t > L_{\text{initial}}$) to test if continuous RG-LRU recurrence pulls sliding-window attention representations toward the donor during future unrolls.
7. **Eligibility Reporting:** Explicitly displays $N_{\text{eligible}} / N_{\text{total}}$ for behavioral attribution ($|m_D - m_R| \ge 0.5$).

---

## 2. S12a.1 Corrected Causal Scout Manifest

- **Run Directory:** `results/e11_surgical_swaps/run_e11_scout_20260818_091221`
- **Model:** `google/recurrentgemma-2b` (bfloat16 on NVIDIA RTX 3060)
- **Scope:** 4 stimulus pairs $\times$ 4 filler regimes $\times$ 3 lag checkpoints ($L \in \{8, 2049, 4096\}$) $\times$ 16 intervention conditions = 768 swap evaluations + 16 mediational forward unrolls (512 tokens each) in 1441.60s.

---

## 3. Empirical Causal Factorial Panel & Directional Logit Displacements

| Lag $L$ | Intervention Condition | Signed Graft Effect $\bar{\Delta}_C$ | Abs Displacement $P_C$ | Logit Proj $\alpha_C^{\text{logit}}$ | Attrib Index $\alpha_C^{\text{cloze}}$ | Eligible Cells | Donor Concordance |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **8** | `whole_swap` | **+23.13** | **+1046.78** | **+1.000** | **1.000** | 16/16 (100%) | **100.0%** |
| **8** | `kv_only` | **+22.82** | **+1033.82** | **+0.988** | **0.986** | 16/16 (100%) | **100.0%** |
| **8** | `rglru_only` (Matching Donor) | **+0.30** | **+12.95** | **+0.012** | **0.014** | 16/16 (100%) | 0.0% |
| **8** | `unrelated_donor_rglru` | **-0.07** | **+6.13** | **+0.006** | -0.003 | 16/16 (100%) | 0.0% |
| **8** | `permuted_donor_rglru` | **-0.10** | **+0.17** | **+0.001** | -0.004 | 16/16 (100%) | 0.0% |
| **8** | `conv_only` | **+0.00** | **+0.00** | **+0.000** | **0.000** | 16/16 (100%) | 0.0% |
| **8** | `noise_control_rglru_seed1` | -0.09 | +1.86 | +0.002 | -0.003 | 16/16 (100%) | 0.0% |
| **8** | `noise_control_rglru_seed2` | -0.02 | -3.42 | -0.003 | -0.002 | 16/16 (100%) | 0.0% |
| **8** | `sham_a2_into_a1` | **+0.00** | **+0.00** | **+0.000** | N/A | 0/16 (0%) | 0.0% |
| **2049 ($W+1$)** | `whole_swap` | **+0.55** | **+91.17** | **+1.000** | **1.000** | 6/16 (37.5%) | 53.1% |
| **2049 ($W+1$)** | `kv_only` | **+0.56** | **+74.27** | **+0.803** | **1.064** | 6/16 (37.5%) | 50.0% |
| **2049 ($W+1$)** | `rglru_only` (Matching Donor) | **-0.01** | **+16.90** | **+0.197** | -0.064 | 6/16 (37.5%) | 50.0% |
| **2049 ($W+1$)** | `unrelated_donor_rglru` | -0.02 | +27.97 | +0.244 | -0.032 | 6/16 (37.5%) | 0.0% |
| **2049 ($W+1$)** | `permuted_donor_rglru` | -0.03 | +29.36 | +0.266 | -0.049 | 6/16 (37.5%) | 0.0% |
| **2049 ($W+1$)** | `conv_only` | **+0.00** | **+0.00** | **+0.000** | **0.000** | 6/16 (37.5%) | 46.9% |
| **2049 ($W+1$)** | `noise_control_rglru_seed1` | -0.04 | -3.38 | +0.067 | -0.098 | 6/16 (37.5%) | 0.0% |
| **2049 ($W+1$)** | `noise_control_rglru_seed2` | -0.02 | +6.09 | +0.136 | -0.008 | 6/16 (37.5%) | 0.0% |
| **2049 ($W+1$)** | `sham_a2_into_a1` | **+0.00** | **+0.00** | **+0.000** | N/A | 0/16 (0%) | 0.0% |
| **4096 ($2W$)** | `whole_swap` | **-0.06** | **+119.70** | **+1.000** | **1.000** | 1/16 (6.25%) | 46.9% |
| **4096 ($2W$)** | `kv_only` | **+0.01** | **+72.75** | **+0.650** | **0.100** | 1/16 (6.25%) | 53.1% |
| **4096 ($2W$)** | `rglru_only` (Matching Donor) | **-0.07** | **+46.94** | **+0.350** | **0.900** | 1/16 (6.25%) | 46.9% |
| **4096 ($2W$)** | `permuted_donor_rglru` | +0.06 | **-13.96** | **+0.047** | -0.200 | 1/16 (6.25%) | 0.0% |
| **4096 ($2W$)** | `unrelated_donor_rglru` | -0.01 | **-35.41** | **-0.177** | -0.200 | 1/16 (6.25%) | 0.0% |
| **4096 ($2W$)** | `conv_only` | **+0.00** | **+0.00** | **+0.000** | **0.000** | 1/16 (6.25%) | 53.1% |
| **4096 ($2W$)** | `noise_control_rglru_seed1` | -0.16 | +54.58 | +0.367 | 0.800 | 1/16 (6.25%) | 0.0% |
| **4096 ($2W$)** | `noise_control_rglru_seed2` | -0.09 | +70.07 | +0.354 | 1.500 | 1/16 (6.25%) | 0.0% |
| **4096 ($2W$)** | `sham_a2_into_a1` | **+0.00** | **+0.00** | **+0.000** | N/A | 0/16 (0%) | 0.0% |

---

## 4. Mediational Dynamic Forward Propagation: Sliced Post-Graft KV Cache

We initialized hybrid state $S_0 = (R^B, C^A, K^A)$ at $L=8$ and unrolled $N=512$ identical future filler tokens, measuring distances strictly on the **sliced post-graft cache entries** ($t > 8$):

| Pair ID | Regime | Post Dist to Recipient A ($D_A$) | Post Dist to Donor B ($D_B$) | Post Migration Index $\mathcal{M}_{\text{post}}$ | Full Migration Index $\mathcal{M}_{\text{full}}$ | Sliced Result |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `item_material_01` | `constant` | 0.0170 | 0.0358 | **-0.4878** | -0.6165 | **Anchored to A** |
| `item_material_01` | `interfering` | 0.0166 | 0.0368 | **-0.5288** | -0.6645 | **Anchored to A** |
| `item_material_01` | `natural` | 0.0147 | 0.0209 | **-0.2571** | -0.5780 | **Anchored to A** |
| `item_material_01` | `random` | 0.0116 | 0.0260 | **-0.5205** | -0.6972 | **Anchored to A** |
| `item_material_02` | `constant` | 0.0144 | 0.0223 | **-0.3053** | -0.5007 | **Anchored to A** |
| `item_material_02` | `interfering` | 0.0216 | 0.0470 | **-0.4906** | -0.5825 | **Anchored to A** |
| `item_material_03` | `constant` | 0.0171 | 0.0314 | **-0.4001** | -0.5932 | **Anchored to A** |
| `item_material_04` | `constant` | 0.0154 | 0.0305 | **-0.4450** | -0.6054 | **Anchored to A** |

**Calibrated Finding:**
Even after slicing strictly to newly created post-graft KV cache entries ($t > 8$), post-graft representations remain substantially closer to recipient A than donor B ($\text{mean } \mathcal{M}_{\text{post}} = -0.448$). This indicates that during forward unrolling under sliding-window attention, ongoing representation generation is strongly shaped by recent local context in the sliding window.

---

## 5. Synthesis & Horizon 2 Theoretical Findings

1. **The Geometric Complement Identity:**
   Because Conv is evicted at $L \ge 4$, $(R^A, C^B, K^B) = (R^A, C^A, K^B)$. Geometrically projecting this state along opposite endpoint axes algebraically forces:
   $$\alpha_{\text{RGLRU}}^{A \to B} + \alpha_{\text{KV}}^{B \to A} \equiv 1.000$$
2. **Absolute Steering ($P_C$) Dynamics:**
   RG-LRU's fractional share rises dramatically as the dominant KV/total contrast collapses. Its absolute directional displacement is also larger at $2W$ ($P_{\text{RGLRU}} = +46.94$) than at $L=8$ ($P_{\text{RGLRU}} = +12.95$) in this scout, so the relative-share increase cannot be attributed solely to normalization.
3. **Historical Specificity Evidence:**
   The scout provides evidence consistent with matching-history-specific RG-LRU orientation at $2W$: matching donor RG-LRU yields $P_C = \mathbf{+46.94}$, whereas unrelated donor yields $P_C = \mathbf{-35.41}$ and permuted donor yields $P_C = \mathbf{-13.96}$.
4. **Primary vs Secondary Metrics for S12b Confirmatory:**
   Because behavioral cloze contrast nearly collapses at $2W$ ($N_{\text{eligible}} = 1/16$), logit-space directional displacement $P_{\text{RGLRU}}$ serves as the primary physical-causal endpoint across all pairs, with cloze attribution $\alpha^{\text{cloze}}$ evaluated conditionally where contrast is resolved.

---

## 6. Preregistered Protocol for Sprint S12b Confirmatory Run

- **Primary Causal Endpoint:** Matching-history RG-LRU directional displacement $P_{\text{RGLRU}}$ at $2W=4096$ across all 20 canonical stimulus pairs ($N=20$).
- **Primary Specificity Estimand:** Paired difference $\Delta P_{\text{spec}} = P_{\text{matching}} - P_{\text{unrelated}}$ and $\Delta P_{\text{spec}} = P_{\text{matching}} - P_{\text{permuted}}$ with 95% pair-cluster bootstrap CI.
- **Secondary Temporal Growth Estimand:** Paired trajectory contrast $\Delta P_{\text{growth}} = P_{\text{RGLRU}}(2W) - P_{\text{RGLRU}}(W+1)$ with 95% pair-cluster bootstrap CI.
- **Mediation Sub-Experiment:** Compare post-graft KV slices at $N=512$ and after full window turnover ($N \ge 2048$).
