# Sprint S11: Latent Impulse Response, Retention & Store Localization Report

**Horizon 2 (Level 2: Latent Recurrence)**  
**Substrate:** Upstream Hugging Face `RecurrentGemma` / Griffin Hybrid Architecture  
**Status:** **Harness Hardened, Invariant-Verified (149/149 Green) & Reference-Model Scout Completed**

---

## 1. Executive Scientific & Methodological Summary

Sprint S11 constructs the empirical **temporal anatomy** of `RecurrentGemma` across its three physical stores without off-manifold vector injections:
1. **1D Temporal Convolution Buffers (`conv[layer]`):** Local sliding token history of width $K=4$.
2. **Sliding Window Attention KV Cache (`kv[layer]`):** Local attention window of width $W$ (queried via `config.attention_window_size`).
3. **Real-Gated Linear Recurrent Units (`rglru[layer]`):** Input-gated continuous recurrent state carrying long-range history via Griffin Eq. 4:
   $$h_t = a_t \odot h_{t-1} + \sqrt{1 - a_t^2} \odot (i_t \odot x_t), \quad a_t = \exp(-8 \sigma(W_a x_t + b_a) \cdot \text{softplus}(\Lambda))$$

### Core Hardened Innovations (S11.1 Gate)
- **Fail-Closed Execution:** `--model_id` and `--phase confirmatory` abort immediately if model loading fails, rather than silently falling back to a random model. Manifest explicitly tracks `is_reference_model: bool`.
- **Per-Layer Persistence (`layer_trace.jsonl`):** Persists all layer-level physical metrics ($\text{RMSDiff}$, $D_{\text{rel}}$, $\text{CosSim}$, Frobenius, $R(L)$) across every layer index and store to build granular layer $\times$ lag temporal maps.
- **Separation of Physical Residency vs Branch Divergence:** Distinguishes whether the original token physically resides within the buffer/cache ($L < K-1$ or $L < W-1$) from whether the store state remains divergent ($D_C(S_L^A, S_L^B) > 0$).
- **Normalized Cross-Channel Distance:** Scale-Relative Distance ($D_{\text{rel}} \in [0, \sqrt{2}]$), Root Mean Square Difference ($\text{RMSDiff}$), Cosine Similarity, and within-layer Retention Ratio $R_C(L) = D_{\text{rel}}(L) / D_{\text{rel}}(0)$.
- **Audited Vocabulary Pooling & Length-Equated Stimuli:** 20 length-equated stimulus pairs (`CANONICAL_STIMULI_PAIRS`), audited vocabulary pool with recorded SHA256 digest, and multi-passage/multi-seed filler generators.
- **Cloze Retrieval Probing:** Evaluates logit margin $m = \ell_{\text{target\_correct}} - \ell_{\text{target\_incorrect}}$ from detached branch clones without mutating the measurement trajectory.
- **A/A Sham Noise Baseline:** Evaluates identical $A_1 / A_2$ trajectories establishing the empirical numerical noise floor at exact zero ($D_{\text{rel}} = 0.00000000$, $D_{\text{JS}} = 0.00000000$).

---

## 2. Invariant Verification Results (11/11 Passed)

All harness and mathematical invariants were verified in [`tests/test_s11_latent_impulse.py`](file:///c:/Users/admir/Github/recurrence/tests/test_s11_latent_impulse.py):

| Test Case | Invariant Property Verified | Result |
| :--- | :--- | :---: |
| **`test_dynamic_lag_grid_hits_architectural_boundaries`** | Grid includes Conv buffer ($0..K$), sliding window ($W/2, W-1, W, W+1$), and post-eviction ($2W$) | **PASSED** |
| **`test_matched_branches_receive_identical_filler`** | Filler token IDs are strictly identical across Branch A and Branch B for all 4 regimes | **PASSED** |
| **`test_event_variants_have_equal_token_length`** | All 20 stimulus pairs have exact token count parity between Event A and Event B | **PASSED** |
| **`test_impulse_creates_nonzero_initial_separation`** | Matched impulse creates non-zero initial separation ($D_{\text{rel}}(0) > 0$) across all stores | **PASSED** |
| **`test_sham_pair_stays_at_numerical_floor`** | Sham $A_1 / A_2$ stays at numerical zero floor ($D_{\text{rel}} < 10^{-5}, D_{\text{JS}} < 10^{-5}$) | **PASSED** |
| **`test_checkpoint_capture_does_not_mutate_trajectory`** | Interrupted stepping with checkpoint captures reproduces continuous unroll exactly | **PASSED** |
| **`test_probe_uses_detached_branch_snapshot`** | Cloze retrieval probes clone snapshot without advancing main trajectory cache position | **PASSED** |
| **`test_direct_conv_residency_boundary`** | Conv residency flag switches off exactly at $L = \text{conv1d\_width} - 1$ | **PASSED** |
| **`test_direct_kv_residency_boundary`** | KV residency flag switches off exactly at $L = \text{attention\_window\_size} - 1$ | **PASSED** |
| **`test_distance_metrics_are_finite_and_normalized`** | $D_{\text{rel}} \in [0, \sqrt{2}]$, $\text{CosSim} \in [-1, 1]$, $D_{\text{JS}} \in [0, \ln 2]$, finite values | **PASSED** |
| **`test_end_to_end_all_regimes_with_layer_traces`** | Complete end-to-end execution across Constant, Random, Natural, and Interfering regimes with layer traces | **PASSED** |

---

## 3. Empirical Results: Reference-Model Engineering Scout

> [!NOTE]
> **Engineering Scout Epistemic Scope:** This dataset evaluates the lightweight reference model architecture (`reference_random_recurrentgemma`) to verify instrumentation sensitivity, residency boundary transitions, non-monotonic dynamics, and sham noise floor. Pretrained model parameter values (`google/recurrentgemma-2b`) are evaluated in the live capability scout.

**Run Summary (`results/e10_latent_impulse/run_e10_scout_20260818_011053`):**
- 4 Length-Equated Pairs $\times$ 4 Regimes $\times$ 14 Architectural Lags
- Total Summary Records: 224
- Total Layer Trace Records: 1,568
- Device: CUDA (bfloat16)
- Audited Vocab Pool SHA256: `cdbaa85adaf13b19`

### Table 1: Empirical 50%-Retention Thresholds & Log-Lag AUC

| Filler Regime | RGLRU First <50% | RGLRU Sustained <50% | Conv First <50% | KV First <50% | RGLRU Log-AUC | KV Log-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Constant Token Filler** | $L=16$ | $L=16$ | $L=3$ | $L=15$ | 2.892 | 1.976 |
| **Natural Prose Narrative** | $L=16$ | $L=16$ | $L=3$ | $L=15$ | 2.912 | 1.956 |
| **Diverse Random Tokens** | $L=17$ | $L=17$ | $L=3$ | $L=15$ | 2.824 | 1.959 |
| **Semantic Active Interference** | $L=15$ | $L=15$ | $L=3$ | $L=15$ | 2.947 | 1.974 |

### Table 2: Multi-Store Retention Dynamics Across Architectural Lags

| Lag $L$ | Conv Res? | KV Res? | RGLRU Ret (Const) | KV Ret (Const) | RGLRU Ret (Interf) | KV Ret (Interf) | $D_{\text{JS}}$ (Const) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **Yes** | **Yes** | 1.000 | 1.000 | 1.000 | 1.000 | 0.0309 |
| **1** | **Yes** | **Yes** | **1.162** | 0.900 | **1.148** | 0.906 | 0.0121 |
| **2** | **Yes** | **Yes** | **1.189** | 0.827 | **1.218** | 0.830 | 0.0101 |
| **3** | **No** | **Yes** | 0.976 | 0.769 | 1.052 | 0.763 | 0.0095 |
| **4** | **No** | **Yes** | 0.889 | 0.723 | 0.931 | 0.709 | 0.0091 |
| **8** | **No** | **Yes** | 0.726 | 0.596 | 0.702 | 0.583 | 0.0043 |
| **15** | **No** | **No** | 0.533 | **0.014** | 0.464 | **0.025** | 0.0044 |
| **16** | **No** | **No** | 0.490 | **0.013** | 0.445 | **0.024** | 0.0000 |
| **17** | **No** | **No** | 0.433 | **0.009** | 0.416 | **0.020** | 0.0000 |
| **32** | **No** | **No** | 0.128 | 0.004 | 0.169 | 0.005 | 0.0000 |
| **64** | **No** | **No** | 0.043 | 0.001 | 0.081 | 0.002 | 0.0000 |
| **128** | **No** | **No** | 0.011 | 0.000 | 0.022 | 0.001 | 0.0000 |
| **256** | **No** | **No** | 0.002 | 0.000 | 0.005 | 0.000 | 0.0000 |
| **512** | **No** | **No** | 0.000 | 0.000 | 0.001 | 0.000 | 0.0000 |

---

## 4. Key Epistemic & Methodological Findings

1. **Direct Residency vs Downstream Divergence:**
   After direct event residency ends, branch-specific differences persist in later Conv and KV representations, consistent with historical information being propagated through the hybrid recurrent system. (Causal decomposition between specific channels is formally evaluated in Sprint S12).
2. **Recurrent Non-Monotonicity:**
   At lags $L=1$ and $L=2$, the RGLRU retention ratio rises above 1.0 ($R=1.162$ and $R=1.189$), validating that input-gated recurrence can amplify branch differences under subsequent inputs.
3. **Zero Sham Floor:**
   Identical $A_1 / A_2$ controls confirmed an empirical measurement floor of $0.00000000$ for scale-relative distance and Jensen-Shannon divergence across all tested lags.
