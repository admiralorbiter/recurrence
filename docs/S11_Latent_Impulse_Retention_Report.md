# Sprint S11: Latent Impulse Response, Retention & Store Localization Report

**Horizon 2 (Level 2: Latent Recurrence)**  
**Substrate:** Upstream Hugging Face `google/recurrentgemma-2b` / Griffin Hybrid Architecture (26 Layers, Hidden Size 2560, LRU Width 2560, Conv Width 4, Attention Window 2048)  
**Status:** **S11b Confirmatory Run COMPLETED & FROZEN (All 20 Pairs x 4 Regimes, 10,000-Draw Pair-Cluster Bootstrap)**

---

## 1. Executive Scientific & Methodological Summary

Sprint S11 constructs the empirical **temporal anatomy** of `RecurrentGemma` across its three physical stores without off-manifold vector injections:
1. **1D Temporal Convolution Buffers (`conv[layer]`):** Local sliding token history of width $K=4$. Direct residency ends at $L \ge 3$.
2. **Sliding Window Attention KV Cache (`kv[layer]`):** Local attention window of width $W=2048$. Direct residency ends at $L \ge 2047$.
3. **Real-Gated Linear Recurrent Units (`rglru[layer]`):** Input-gated continuous recurrent state carrying long-range history via Griffin Eq. 4:
   $$h_t = a_t \odot h_{t-1} + \sqrt{1 - a_t^2} \odot (i_t \odot x_t), \quad a_t = \exp(-8 \sigma(W_a x_t + b_a) \cdot \text{softplus}(\Lambda))$$

### Confirmatory S11b Methodological Standards
- **Audited 1-Token Target Length Parity:** All 20 canonical stimulus pairs audited under `google/recurrentgemma-2b` tokenizer with exact 1-token target equality ($\text{len}(\text{target}_A) = \text{len}(\text{target}_B) = 1$) and exact event token parity ($\text{len}(\text{event}_A) = \text{len}(\text{event}_B) = 6$).
- **Length-Normalized Continuation Log-Likelihood Scorer:** $\frac{1}{T} \sum_t \log P(\text{tok}_t \mid \text{context}_{<t})$ evaluated via detached state snapshots.
- **Non-Repeating Long-Horizon Fillers:** Deterministic non-repeating natural prose and semantic interference stream generators ($\ge 4096$ tokens).
- **Pair-Cluster Bootstrap ($B=10,000$):** Resamples stimulus pairs conditional on the frozen filler panel and deterministic seed assignment to construct rigorous 95% confidence intervals.
- **Complete Temporal-State Parity Invariant:** Proved chunked-vs-step mathematical equivalence across all 3 stores and cache bookkeeping.

---

## 2. Confirmatory S11b Run Manifest & Provenance

- **Run Directory:** `results/e10_latent_impulse/run_e10_confirmatory_20260818_033015`
- **Model:** `google/recurrentgemma-2b` (26 Layers, Hidden Size 2560, Attention Window $W=2048$, Conv Width $K=4$)
- **Scope:** 20 stimulus pairs $\times$ 4 filler regimes $\times$ 18 architectural lag checkpoints out to $L=4096$ ($2W$)
- **Summary Rows:** 1,440 | **Layer Trace Rows:** 63,360 | **Total Walltime:** 3355.25s (55.9m on NVIDIA RTX 3060)
- **Environment:** `torch==2.5.1+cu121`, `transformers==5.15.0`, CUDA bfloat16

---

## 3. Confirmatory Multi-Store Empirical Retention & 50% Thresholds

| Filler Regime | RGLRU First <50% | RGLRU Sustained <50% | Conv First <50% | KV First <50% | RGLRU Log-AUC | KV Log-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Constant Token Filler** | $L=8$ | $L=8$ | $L=2$ | $L=32$ | 3.369 | 3.963 |
| **Semantic Active Interference** | $L=16$ | $L=16$ | $L=8$ | $L=128$ | 3.241 | 4.538 |
| **Natural Prose Narrative** | $L=8$ | $L=8$ | $L=2$ | $L=64$ | 2.779 | 4.148 |
| **Diverse Random Tokens** | $L=8$ | $L=8$ | $L=3$ | $L=64$ | 2.825 | 4.099 |

---

## 4. Primary Confirmatory S11b Estimands & 95% Pair-Cluster Bootstrap CIs ($B=10,000$)

| Primary Estimand | Point Estimate / Mean | 95% Pair-Cluster Bootstrap CI |
| :--- | :---: | :---: |
| **$R_{\text{RGLRU}}(W+1)$ [Constant]** | **0.2845** | **[0.2204, 0.3582]** |
| **$R_{\text{RGLRU}}(2W)$ [Constant]** | **0.3384** | **[0.2484, 0.4401]** |
| **$R_{\text{RGLRU}}(W+1)$ [Interfering]** | **0.0983** | **[0.0896, 0.1075]** |
| **$R_{\text{RGLRU}}(2W)$ [Interfering]** | **0.0798** | **[0.0734, 0.0864]** |
| **$R_{\text{RGLRU}}(W+1)$ [Natural]** | **0.0636** | **[0.0584, 0.0694]** |
| **$R_{\text{RGLRU}}(2W)$ [Natural]** | **0.0514** | **[0.0461, 0.0571]** |
| **$R_{\text{RGLRU}}(W+1)$ [Random]** | **0.0551** | **[0.0489, 0.0613]** |
| **$R_{\text{RGLRU}}(2W)$ [Random]** | **0.0453** | **[0.0402, 0.0501]** |
| **$\Delta R_{\text{interf - const}}(2W)$** | **-0.2586** | **[-0.3566, -0.1697]** |
| **$\Delta R_{\text{reexpand}}(2W - [W+1])$ [Constant]** | **+0.0539** | **[-0.0103, +0.1379]** |
| **Cloze Margin at $W+1$ ($L=2049$) [Constant]** | **+0.5187** | **[+0.3812, +0.6719]** |
| **Cloze Margin at $W+1$ ($L=2049$) [Random]** | **+0.0667** | **[+0.0219, +0.1094]** |
| **Cloze Margin at $W+1$ ($L=2049$) [Interfering]** | **+0.0307** | **[-0.0281, +0.0953]** |
| **Cloze Margin at $W+1$ ($L=2049$) [Natural]** | **-0.0182** | **[-0.0703, +0.0344]** |
| **Cloze Margin at $2W$ ($L=4096$) [Constant]** | **-0.0265** | **[-0.1109, +0.0656]** |
| **Cloze Margin at $2W$ ($L=4096$) [Natural]** | **+0.0123** | **[-0.0125, +0.0375]** |
| **Cloze Margin at $2W$ ($L=4096$) [Interfering]** | **+0.0044** | **[-0.0750, +0.0844]** |
| **Cloze Margin at $2W$ ($L=4096$) [Random]** | **-0.0030** | **[-0.0203, +0.0125]** |

---

## 5. Confirmatory Trajectory Evolution Across Architectural Boundaries ($W=2048$)

| Lag $L$ | Conv Res? | KV Res? | RGLRU Ret (Const) | KV Ret (Const) | RGLRU Ret (Interf) | KV Ret (Interf) | Cloze Margin (Const) | Cloze Acc (Const) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **Yes** | **Yes** | 1.000 | 1.000 | 1.000 | 1.000 | **+10.78** | **1.00** |
| **1** | **Yes** | **Yes** | 0.842 | 0.962 | 0.833 | 0.968 | **+11.03** | **1.00** |
| **2** | **Yes** | **Yes** | 0.767 | 0.925 | 0.776 | 0.940 | **+11.14** | **0.97** |
| **3** | **No** | **Yes** | 0.692 | 0.885 | 0.797 | 0.924 | **+10.86** | **1.00** |
| **4** | **No** | **Yes** | 0.633 | 0.849 | 0.701 | 0.901 | **+10.47** | **1.00** |
| **8** | **No** | **Yes** | 0.460 | 0.748 | 0.602 | 0.824 | **+12.26** | **1.00** |
| **16** | **No** | **Yes** | 0.386 | 0.623 | 0.426 | 0.712 | **+11.11** | **0.95** |
| **32** | **No** | **Yes** | 0.307 | 0.494 | 0.366 | 0.601 | **+7.85** | **0.95** |
| **64** | **No** | **Yes** | 0.292 | 0.399 | 0.270 | 0.501 | **+3.73** | **0.82** |
| **128** | **No** | **Yes** | 0.254 | 0.309 | 0.234 | 0.407 | **+4.96** | **0.90** |
| **256** | **No** | **Yes** | 0.234 | 0.244 | 0.195 | 0.339 | **+0.54** | **0.55** |
| **512** | **No** | **Yes** | 0.196 | 0.196 | 0.184 | 0.280 | **+1.85** | **0.72** |
| **1024** | **No** | **Yes** | 0.194 | 0.153 | 0.121 | 0.228 | **+0.62** | **0.55** |
| **2040** | **No** | **Yes** | 0.291 | 0.124 | 0.105 | 0.186 | **+0.50** | **0.57** |
| **2047** | **No** | **No** | 0.286 | 0.106 | 0.101 | 0.168 | **+0.47** | **0.50** |
| **2048** | **No** | **No** | 0.285 | 0.105 | 0.101 | 0.167 | **+0.52** | **0.50** |
| **2049** | **No** | **No** | 0.285 | 0.105 | 0.098 | 0.167 | **+0.52** | **0.50** |
| **4096** | **No** | **No** | 0.340 | 0.086 | 0.080 | 0.097 | **-0.03** | **0.50** |

---

## 6. Core Scientific Discoveries & Horizon 2 Theoretical Synthesis

1. **Physical Persistence vs Bounded Behavioral Accessibility:**
   - **Physical Trace:** Branch-specific RG-LRU states remain robustly and significantly separated at $2W=4096$ across all regimes ($R_{\text{RGLRU}} \in [0.045, 0.340]$, 95% CIs strictly excluding zero).
   - **Behavioral Retrieval:** Strict paired retrieval approaches chance around the attention-window boundary; graded cloze evidence remains detectable for constant ($+0.5187$ [0.3812, 0.6719]) and random ($+0.0667$ [0.0219, 0.1094]) regimes at $W+1$, but no tested regime resolves behavioral retrieval at $2W=4096$.
   - **Theoretical Implication:** *Historical state physically encoded* $\neq$ *zero-shot task-usable retrieval*.
2. **Replicated Constant-Input Re-Expansion Point Estimate:**
   - In the constant filler regime, mean RG-LRU retention increases from $0.2845$ at $W+1$ to $0.3384$ at $2W$ ($\Delta R_{\text{reexpand}} = +0.0539$, 95% paired CI [-0.0103, +0.1379]). While the point estimate reflects positive re-expansion consistent with input-gated RG-LRU attractor dynamics under stationary input, the paired confidence interval spans zero.
3. **Transition to Sprint S12 (Surgical Causal Swaps):**
   - S11b provides the exact cell-level eligibility map ($|m_D - m_R| \ge 0.5$) for Sprint S12 to test whether grafting this physical recurrent trace causally transfers historical knowledge.
