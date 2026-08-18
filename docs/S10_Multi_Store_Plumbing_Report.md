# Sprint S10 / S10.1: Multi-Store Plumbing, Upstream Adapter & Replay Invariants Report

**Horizon 2 (Level 2: Latent Recurrence)**  
**Substrate:** Upstream Hugging Face `RecurrentGemma` (`google/recurrentgemma-2b` / Griffin architecture)  
**Status:** **Scientifically Hardened & Frozen (S10.1 Gate Cleared)**

---

## 1. Architectural Overview & Design Decisions

In accordance with the approved S10 / S10.1 specification, we instrumented the official upstream `RecurrentGemmaForCausalLM` via `RecurrentGemmaAdapter` with strict 3-channel state coverage.

### The Layer-Indexed Temporal State Model
State is represented via [`RecurrentStateSnapshot`](file:///c:/Users/admir/Github/recurrence/src/recurrence/state/temporal_inventory.py#L18-L190), mapping layer index to physical channel tensors:

1. **Linear Recurrence Channel (`rglru[layer_idx] -> Tensor`):** Captures `layer.temporal_block.rg_lru.recurrent_states` (shape `[B, lru_width]`, `float32`).
2. **1D Temporal Convolution Channel (`conv[layer_idx] -> Tensor`):** Captures `layer.temporal_block.conv1d_state` (shape `[B, hidden_size, conv1d_width - 1]`).
3. **Sliding Window Attention Channel (`kv[layer_idx] -> {key, value, cumulative_length, sliding_window}`):** Captures key/value states, cumulative token offset, and sliding-window configuration from `DynamicCache` layers.

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                 RecurrentGemmaAdapter                   │
                    │         step(token_id, snapshot) -> (logits, snapshot)  │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
                  ┌──────────────────────────────┼──────────────────────────────┐
                  ▼                              ▼                              ▼
     ┌────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
     │   RGLRU Recurrence     │     │   1D Temporal Conv     │     │   Sliding Window KV    │
     │  rglru[layer_idx]      │     │  conv[layer_idx]       │     │  kv[layer_idx]         │
     │  (long-range carry)    │     │  (local short n-grams) │     │  (attention cache +    │
     │                        │     │                        │     │   cumulative_length)   │
     └────────────────────────┘     └────────────────────────┘     └────────────────────────┘
```

---

## 2. Invariant Verification Results (S10.1 Hardened Battery)

All 11 core mathematical, architectural, and replay invariants were verified in [`tests/test_s10_recurrent_invariants.py`](file:///c:/Users/admir/Github/recurrence/tests/test_s10_recurrent_invariants.py):

| Invariant / Test Case | Invariant Property Verified | Empirical Result |
| :--- | :--- | :---: |
| **`test_cloned_branch_storage_independence`** | Deep cloning creates isolated tensors across all 3 stores; zeroing clone leaves original untouched | **PASSED** |
| **`test_store_isolation_zeroing`** | Surgical 3-way isolation: zeroing any store leaves the other two stores bitwise identical (`torch.equal`) | **PASSED** |
| **`test_snapshot_restore_determinism`** | Restoring snapshot $S_t$ reproduces exact downstream logits and passes `assert_strict_equal` | **PASSED** ($\text{atol} < 10^{-5}$) |
| **`test_one_step_rglru_equation_parity`** | Live RG-LRU module matches independent mathematical evaluation of Griffin Equation 4 | **PASSED** ($\text{atol} < 10^{-5}$) |
| **`test_public_history_replay_reconstruction`** | Replaying public tokens $x_{1:t}$ from canonical zero state reconstructs all 3 stores ($S_t \equiv S_t'$) | **PASSED** ($\text{atol} < 10^{-5}$) |
| **`test_sliding_window_boundary_restore`** | Snapshot $\to$ restore correctly handles KV eviction across sliding window boundaries without logit drift | **PASSED** ($\text{atol} < 10^{-5}$) |
| **`test_explicit_step_vs_upstream_prefill_parity`** | Explicit single-token stepping loop matches upstream multi-token forward prefill pass | **PASSED** ($\text{atol} < 10^{-5}$) |
| **`test_adapter_device_dtype_preservation`** | Passing existing model preserves caller's device/dtype without unintentional CPU/FP32 downcasting | **PASSED** |
| **`test_cross_branch_injection_smoke_test`** | Injecting snapshot $S^A$ into sequence B executes with finite logits and valid cache position | **PASSED** |
| **`test_state_serialization_roundtrip`** | Lossless CPU dictionary and binary byte roundtrip reconstruction with `assert_strict_equal` | **PASSED** ($\text{atol} < 10^{-6}$) |
| **`test_environment_provenance_manifest`** | Recorded hardware, library versions (`transformers>=5.15.0`, `torch==2.5.1+cu121`), and model metadata | **PASSED** |

---

## 3. Environment & Provenance Record

The environment is explicitly recorded in [`docs/s10_environment_manifest.json`](file:///c:/Users/admir/Github/recurrence/docs/s10_environment_manifest.json) and pinned in [`pyproject.toml`](file:///c:/Users/admir/Github/recurrence/pyproject.toml):
- **Framework:** PyTorch `2.5.1+cu121`
- **Transformers:** `5.15.0` (pinned in `pyproject.toml`)
- **Target Model Architecture:** `google/recurrentgemma-2b` (`RecurrentGemmaForCausalLM`)
- **Block Configuration:** Recurrent blocks (RG-LRU + Conv1D width=4) interleaved with local sliding window attention (window=2048).
