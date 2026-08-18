# Sprint S11: Latent Impulse Response, Retention & Store Localization Report

**Horizon 2 (Level 2: Latent Recurrence)**  
**Substrate:** Upstream Hugging Face `RecurrentGemma` / Griffin Hybrid Architecture  
**Status:** **Implemented, Invariant-Verified & Phase S11a Scout Completed**

---

## 1. Executive Scientific & Architectural Synthesis

Sprint S11 constructs the first empirical **temporal anatomy** of `RecurrentGemma` across its three physical stores without off-manifold vector injections:
1. **1D Temporal Convolution Buffers (`conv[layer]`):** Local sliding token history of width $K=4$.
2. **Sliding Window Attention KV Cache (`kv[layer]`):** Local attention window of width $W$.
3. **Real-Gated Linear Recurrent Units (`rglru[layer]`):** Input-gated continuous recurrent state carrying long-range history via Griffin Eq. 4:
   $$h_t = a_t \odot h_{t-1} + \sqrt{1 - a_t^2} \odot (i_t \odot x_t), \quad a_t = \exp(-8 \sigma(W_a x_t + b_a) \cdot \text{softplus}(\Lambda))$$

### Core Methodological Innovations
- **Separation of Physical Residency vs Branch Divergence:** Distinguishes whether the original token physically resides within the buffer/cache ($L < K-1$ or $L < W-1$) from whether the store state remains divergent ($D_C(S_L^A, S_L^B) > 0$).
- **Normalized Cross-Channel Distance:** Replaced raw Frobenius comparisons with Scale-Relative Distance ($D_{\text{rel}} \in [0, \sqrt{2}]$), Root Mean Square Difference ($\text{RMSDiff}$), Cosine Similarity, and within-layer Retention Ratio $R_C(L) = D_{\text{rel}}(L) / D_{\text{rel}}(0)$.
- **Symmetric Behavioral Divergence:** Measured via bounded Jensen-Shannon Divergence $D_{\text{JS}}(P_A \parallel P_B)$ and top-1 prediction disagreement.
- **Detached 2AFC Usability Probing:** Factual retrieval margin $m = \ell_{\text{target\_correct}} - \ell_{\text{target\_incorrect}}$ evaluated from cloned branch snapshots without mutating the measurement trajectory.
- **A/A Sham Noise Baseline:** Evaluated identical $A_1 / A_2$ trajectories establishing the empirical numerical noise floor at exact zero ($D_{\text{rel}} = 0.00000000$, $D_{\text{JS}} = 0.00000000$).

---

## 2. Invariant Verification Results (11/11 Passed)

All harness and mathematical invariants were verified in [`tests/test_s11_latent_impulse.py`](file:///c:/Users/admir/Github/recurrence/tests/test_s11_latent_impulse.py):

| Test Case | Invariant Property Verified | Result |
| :--- | :--- | :---: |
| **`test_dynamic_lag_grid_hits_architectural_boundaries`** | Grid includes Conv buffer ($0..K$), sliding window ($W/2, W-1, W, W+1$), and post-eviction ($2W$) | **PASSED** |
| **`test_matched_branches_receive_identical_filler`** | Filler token IDs are strictly identical across Branch A and Branch B for all 4 regimes | **PASSED** |
| **`test_event_variants_have_equal_token_length`** | Stimulus pairs have exact token count parity between Event A and Event B | **PASSED** |
| **`test_impulse_creates_nonzero_initial_separation`** | Matched impulse creates non-zero initial separation ($D_{\text{rel}}(0) > 0$) across all stores | **PASSED** |
| **`test_sham_pair_stays_at_numerical_floor`** | Sham $A_1 / A_2$ stays at numerical zero floor ($D_{\text{rel}} < 10^{-5}, D_{\text{JS}} < 10^{-5}$) | **PASSED** |
| **`test_checkpoint_capture_does_not_mutate_trajectory`** | Interrupted stepping with checkpoint captures reproduces continuous unroll exactly | **PASSED** |
| **`test_probe_uses_detached_branch_snapshot`** | 2AFC retrieval probes clone snapshot without advancing main trajectory cache position | **PASSED** |
| **`test_direct_conv_residency_boundary`** | Conv residency flag switches off exactly at $L = \text{conv1d\_width} - 1$ | **PASSED** |
| **`test_direct_kv_residency_boundary`** | KV residency flag switches off exactly at $L = \text{sliding\_window} - 1$ | **PASSED** |
| **`test_distance_metrics_are_finite_and_normalized`** | $D_{\text{rel}} \in [0, \sqrt{2}]$, $\text{CosSim} \in [-1, 1]$, $D_{\text{JS}} \in [0, \ln 2]$, finite values | **PASSED** |
| **`test_end_to_end_all_regimes_reference_model`** | Complete end-to-end execution across Constant, Random, Natural, and Interfering regimes | **PASSED** |

---

## 3. Empirical Results: Phase S11a Scout Run

**Scout Configuration:** 4 Length-Equated Pairs $\times$ 4 Regimes $\times$ 14 Dynamic Lags ($N=224$ checkpoint records).

### Table 1: Empirical 50%-Retention Crossings ($L_{50\%}$) & Area Under Curve (AUC)

| Filler Regime | RGLRU 50% Crossing ($L_{50\%}$) | Conv1D 50% Crossing ($L_{50\%}$) | KV 50% Crossing ($L_{50\%}$) | RGLRU AUC | KV AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Constant Token Filler** | **$L=8$** | $L=3$ | $L=15$ | 14.37 | 8.58 |
| **Natural Prose Narrative** | **$L=15$** | $L=3$ | $L=15$ | 26.06 | 8.48 |
| **Diverse Random Tokens** | **$L=15$** | $L=3$ | $L=15$ | 27.36 | 8.73 |
| **Semantic Active Interference** | **$L=32$** | $L=3$ | $L=15$ | **27.15** | 8.67 |

### Table 2: Multi-Store Retention Dynamics Across Lag Steps

| Lag $L$ | Conv Res? | KV Res? | RGLRU Ret (Const) | KV Ret (Const) | RGLRU Ret (Interf) | KV Ret (Interf) | $D_{\text{JS}}$ (Const) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **Yes** | **Yes** | 1.000 | 1.000 | 1.000 | 1.000 | 0.0309 |
| **1** | **Yes** | **Yes** | **1.103** | 0.900 | **1.087** | 0.896 | 0.0051 |
| **2** | **Yes** | **Yes** | **1.079** | 0.825 | **1.116** | 0.824 | 0.0037 |
| **3** | **No** | **Yes** | 0.859 | 0.764 | 0.975 | 0.758 | 0.0026 |
| **4** | **No** | **Yes** | 0.724 | 0.714 | 0.860 | 0.727 | 0.0018 |
| **8** | **No** | **Yes** | 0.464 | 0.576 | 0.680 | 0.582 | 0.0004 |
| **15** | **No** | **No** | 0.219 | **0.027** | **0.524** | **0.027** | 0.0001 |
| **16** | **No** | **No** | 0.198 | **0.026** | **0.552** | **0.023** | 0.0000 |
| **17** | **No** | **No** | 0.178 | **0.021** | **0.534** | **0.021** | 0.0000 |
| **32** | **No** | **No** | 0.067 | 0.004 | 0.167 | 0.007 | 0.0000 |
| **64** | **No** | **No** | 0.027 | 0.002 | 0.082 | 0.002 | 0.0000 |
| **128** | **No** | **No** | 0.008 | 0.001 | 0.021 | 0.001 | 0.0000 |
| **256** | **No** | **No** | 0.002 | 0.000 | 0.004 | 0.000 | 0.0000 |
| **512** | **No** | **No** | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 |

---

## 4. Key Scientific Findings from the Scout Data

1. **Empirical Non-Monotonicity in RG-LRU Recurrence:**
   At lags $L=1$ and $L=2$, RGLRU retention ratio rises above 1.0 ($R=1.103$ and $R=1.079$), confirming that input-gated recurrence can amplify branch separation under subsequent inputs rather than decaying purely exponentially.
2. **Post-Eviction Physical Trace Persistence:**
   When the perturbing token exits the Conv buffer ($L=3$) or sliding KV cache ($L=15$), the downstream Conv and KV store representations do **not** become identical:
   - Conv retains $R=0.190$ at $L=3$.
   - KV retains $R=0.027$ at $L=15$.
   This confirms that upstream recurrent differences continually propagate into subsequent local attention and convolution representations.
3. **Input-Dependent Retention Timescales:**
   The effective 50%-retention crossing of RG-LRU varies by a factor of 4 across input regimes: from $L=8$ under constant-token filler to **$L=32$** under semantic active interference (AUC jumps from 14.37 to 27.15).
