# Sprint S10: Multi-Store Plumbing, Upstream Adapter & Replay Invariants Report

**Horizon 2 (Level 2: Latent Recurrence)**  
**Substrate:** Upstream Hugging Face `RecurrentGemma` (`google/recurrentgemma-2b` / Griffin architecture)  
**Status:** **Completed & Verified**

---

## 1. Architectural Overview & Design Decisions

In accordance with the approved S10 specification, we avoided reimplementing Griffin layer equations from scratch and instead instrumented the official upstream `RecurrentGemmaForCausalLM` via `RecurrentGemmaAdapter`.

### The Layer-Indexed Temporal State Model
Rather than flattening recurrence into a monolithic tensor, the `RecurrentStateSnapshot` models state as layer-indexed mappings across the three temporal channels:

1. **Linear Recurrence Channel (`rglru[layer_idx] -> Tensor`):** Captures `layer.temporal_block.rg_lru.recurrent_states` (shape `[B, lru_width]`, `float32`).
2. **1D Temporal Convolution Channel (`conv[layer_idx] -> Tensor`):** Captures `layer.temporal_block.conv1d_state` (shape `[B, hidden_size, conv1d_width - 1]`).
3. **Sliding Window KV Attention Channel (`kv[layer_idx] -> {key, value}`):** Captures key and value tensors from `DynamicCache` layers.

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
     │  (long-range carry)    │     │  (local short n-grams) │     │  (attention cache)     │
     └────────────────────────┘     └────────────────────────┘     └────────────────────────┘
```

---

## 2. Invariant Verification Results

All 7 core mathematical, architectural, and replay invariants were verified in [`tests/test_s10_recurrent_invariants.py`](file:///c:/Users/admir/Github/recurrence/tests/test_s10_recurrent_invariants.py):

| Invariant / Test Case | Mathematical / Methodological Property | Empirical Result |
| :--- | :--- | :---: |
| **`test_cloned_branch_storage_independence`** | Deep cloning creates isolated tensors; zeroing branch B leaves branch A untouched | **PASSED** |
| **`test_store_isolation_zeroing`** | Zeroing RGLRU leaves Conv/KV unchanged; zeroing Conv leaves RGLRU/KV unchanged | **PASSED** |
| **`test_snapshot_restore_determinism`** | Restoring snapshot $S_t$ reproduces exact downstream logits on identical token | **PASSED** ($\text{atol} < 10^{-5}$) |
| **`test_one_step_rglru_equation_parity`** | Live module matches independent mathematical evaluation of Griffin Equation 4 | **PASSED** ($\text{atol} < 10^{-5}$) |
| **`test_public_history_replay_reconstruction`** | Replaying public tokens $x_{1:t}$ from canonical zero state reconstructs exact state $S_t \equiv S_t'$ | **PASSED** ($\text{Frobenius} < 10^{-5}$) |
| **`test_cross_branch_injection_smoke_test`** | Injecting snapshot $S^A$ into sequence B executes with finite logits and correct cache position | **PASSED** |
| **`test_state_serialization_roundtrip`** | Lossless CPU dictionary and binary byte roundtrip reconstruction | **PASSED** ($\text{Frobenius} < 10^{-6}$) |

---

## 3. The Foundational S10 Scientific Insight: Operational vs Informational Privacy

The success of `test_public_history_replay_reconstruction` provides the core intellectual anchor for Horizon 2:
> **Core Empirical Principle:**  
> *Under deterministic inference, native recurrent state is completely reconstructible from public token history ($x_{1:t} \to S_t$). Therefore, persistent hidden state is **operationally private** (inaccessible to a prompt-level reader without running the model) but **not inherently informationally privileged** over a matched observer who replays the public transcript.*

This principle directly shapes the downstream design of **Sprint S13 (Reality Monitoring & Introspection)**:
To demonstrate true privileged access ($PAI > 0$), the target model must undergo causal events that an external replay observer cannot recover from prompt tokens alone (such as pre-output activation snapshots, latent interventions, or private endogenous variables).
