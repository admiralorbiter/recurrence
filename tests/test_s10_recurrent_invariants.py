"""Sprint S10 / S10.1: Multi-Store Plumbing, Isolation & Invariant Verification Suite.

Validates the complete architectural and mathematical invariants of upstream RecurrentGemma
under the layer-indexed temporal state inventory with strict 3-channel coverage:
1. Cloned branch memory independence across RGLRU, Conv1D, and KV cache.
2. Surgical 3-way store isolation zeroing (RGLRU vs Conv1D vs KV cache).
3. Snapshot -> Restore determinism and next-token logit reproduction (with strict structural equality).
4. One-step RG-LRU mathematical equation parity (Griffin Equation 4).
5. Public-history replay reconstruction across all 3 stores.
6. Sliding-window boundary restore across window evictions.
7. Explicit single-token step vs upstream multi-token prefill parity.
8. Caller device and dtype preservation.
9. Cross-branch state injection plumbing smoke test.
10. Lossless CPU dictionary and binary serialization roundtrips.
"""

import json
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RecurrentGemmaConfig, RecurrentGemmaForCausalLM, DynamicCache

from recurrence.state.temporal_inventory import RecurrentStateSnapshot
from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter


@pytest.fixture
def test_adapter() -> RecurrentGemmaAdapter:
    """Create a lightweight RecurrentGemma adapter for fast, deterministic unit testing."""
    config = RecurrentGemmaConfig(
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        lru_width=64,
        conv1d_width=4,
        sliding_window=4,
        block_types=["recurrent", "recurrent", "attention", "recurrent"],
        vocab_size=200,
    )
    torch.manual_seed(42)
    return RecurrentGemmaAdapter(config=config, device="cpu", dtype=torch.float32)


def test_cloned_branch_storage_independence(test_adapter: RecurrentGemmaAdapter):
    """Verify that cloning a snapshot creates storage-independent tensors across all 3 channels."""
    init_state = test_adapter.create_canonical_initial_state()
    # Populate with non-zero dummy values
    for l in init_state.rglru:
        init_state.rglru[l] = torch.randn_like(init_state.rglru[l]) + 1.0
    for l in init_state.conv:
        init_state.conv[l] = torch.randn_like(init_state.conv[l]) + 2.0
    for l in init_state.kv:
        init_state.kv[l]["key"] = torch.randn(1, 1, 3, 32) + 3.0
        init_state.kv[l]["value"] = torch.randn(1, 1, 3, 32) + 4.0

    clone_state = init_state.clone()

    # Mutate cloned branch
    clone_state.zero_store("all")

    # Assert original state was NOT mutated in any channel
    for l in init_state.rglru:
        assert torch.all(init_state.rglru[l] != 0.0), f"Original RGLRU layer {l} was mutated by clone modification!"
    for l in init_state.conv:
        assert torch.all(init_state.conv[l] != 0.0), f"Original Conv layer {l} was mutated by clone modification!"
    for l in init_state.kv:
        assert torch.all(init_state.kv[l]["key"] != 0.0), f"Original KV key layer {l} was mutated by clone modification!"
        assert torch.all(init_state.kv[l]["value"] != 0.0), f"Original KV value layer {l} was mutated by clone modification!"


def test_store_isolation_zeroing(test_adapter: RecurrentGemmaAdapter):
    """Verify surgical 3-way isolation: zeroing any store leaves the other two stores bitwise unchanged."""
    snapshot = test_adapter.create_canonical_initial_state()
    for l in snapshot.rglru:
        snapshot.rglru[l] = torch.randn_like(snapshot.rglru[l]) + 0.5
    for l in snapshot.conv:
        snapshot.conv[l] = torch.randn_like(snapshot.conv[l]) + 0.7
    for l in snapshot.kv:
        snapshot.kv[l]["key"] = torch.randn(1, 1, 2, 32) + 0.9
        snapshot.kv[l]["value"] = torch.randn(1, 1, 2, 32) + 1.1

    baseline = snapshot.clone()

    # 1. Zero RGLRU only
    test_rglru = snapshot.clone()
    test_rglru.zero_store("rglru")
    for l in test_rglru.rglru:
        assert torch.all(test_rglru.rglru[l] == 0.0)
    for l in test_rglru.conv:
        assert torch.equal(test_rglru.conv[l], baseline.conv[l]), f"Conv layer {l} changed when zeroing RGLRU!"
    for l in test_rglru.kv:
        assert torch.equal(test_rglru.kv[l]["key"], baseline.kv[l]["key"]), f"KV key layer {l} changed when zeroing RGLRU!"
        assert torch.equal(test_rglru.kv[l]["value"], baseline.kv[l]["value"]), f"KV value layer {l} changed when zeroing RGLRU!"

    # 2. Zero Conv only
    test_conv = snapshot.clone()
    test_conv.zero_store("conv")
    for l in test_conv.conv:
        assert torch.all(test_conv.conv[l] == 0.0)
    for l in test_conv.rglru:
        assert torch.equal(test_conv.rglru[l], baseline.rglru[l]), f"RGLRU layer {l} changed when zeroing Conv!"
    for l in test_conv.kv:
        assert torch.equal(test_conv.kv[l]["key"], baseline.kv[l]["key"]), f"KV key layer {l} changed when zeroing Conv!"
        assert torch.equal(test_conv.kv[l]["value"], baseline.kv[l]["value"]), f"KV value layer {l} changed when zeroing Conv!"

    # 3. Zero KV only
    test_kv = snapshot.clone()
    test_kv.zero_store("kv")
    for l in test_kv.kv:
        assert torch.all(test_kv.kv[l]["key"] == 0.0)
        assert torch.all(test_kv.kv[l]["value"] == 0.0)
    for l in test_kv.rglru:
        assert torch.equal(test_kv.rglru[l], baseline.rglru[l]), f"RGLRU layer {l} changed when zeroing KV!"
    for l in test_kv.conv:
        assert torch.equal(test_kv.conv[l], baseline.conv[l]), f"Conv layer {l} changed when zeroing KV!"


def test_snapshot_restore_determinism(test_adapter: RecurrentGemmaAdapter):
    """Verify that snapshot -> restore reproduces next-step logits and strict structural equality."""
    prompt_tokens = [10, 25, 42, 88, 105]
    _, state_snapshot = test_adapter.encode_sequence(prompt_tokens)

    # Branch A: Step with token 50
    logits_a, state_a = test_adapter.step(50, state_snapshot.clone())

    # Branch B: Reload saved snapshot and step with identical token 50
    logits_b, state_b = test_adapter.step(50, state_snapshot.clone())

    # Verify logit equality
    assert torch.allclose(logits_a, logits_b, atol=1e-5), "Restored snapshot produced divergent logits on identical token!"
    
    # Strict structural & numerical equality across all 3 stores
    state_a.assert_strict_equal(state_b, atol=1e-5)


def test_one_step_rglru_equation_parity(test_adapter: RecurrentGemmaAdapter):
    """Verify Griffin RG-LRU Equation 4 mathematically against the live module forward step."""
    recurrent_layer = None
    for layer in test_adapter.model.model.layers:
        if hasattr(layer.temporal_block, "rg_lru"):
            recurrent_layer = layer.temporal_block.rg_lru
            break

    assert recurrent_layer is not None, "No RG-LRU layer found in model!"

    batch_size = 1
    seq_len = 1
    lru_width = recurrent_layer.recurrent_param.shape[0]
    num_heads = recurrent_layer.num_attention_heads
    block_width = recurrent_layer.block_width

    # Prepare inputs
    torch.manual_seed(1337)
    x = torch.randn(batch_size, seq_len, lru_width)
    pos_ids = torch.tensor([[1]])  # Non-reset tick

    # Set prior hidden state
    h_prev = torch.randn(batch_size, lru_width, dtype=torch.float32)
    recurrent_layer.recurrent_states = h_prev.clone()

    # Step live module
    y_module = recurrent_layer(x, pos_ids)
    h_module = recurrent_layer.recurrent_states

    # Independent mathematical computation of Griffin Equation 4
    reshape_act = x.reshape(batch_size * seq_len, num_heads, block_width).permute(1, 0, 2)
    
    # Input gate i_t = sigmoid(W_x x_t + b_x)
    res_i = torch.baddbmm(recurrent_layer.input_gate_bias[:, None, :], reshape_act, recurrent_layer.input_gate_weight)
    i_gate = torch.sigmoid(res_i.transpose(0, 1).reshape(batch_size, seq_len, lru_width))

    # Recurrent gate r_t = sigmoid(W_a x_t + b_a)
    res_r = torch.baddbmm(recurrent_layer.recurrent_gate_bias[:, None, :], reshape_act, recurrent_layer.recurrent_gate_weight)
    r_gate = torch.sigmoid(res_r.transpose(0, 1).reshape(batch_size, seq_len, lru_width))

    # Recurrence coefficient a_t = exp(-8.0 * r_gate * softplus(param))
    log_a = -8.0 * r_gate * F.softplus(recurrent_layer.recurrent_param)
    a_t = torch.exp(log_a)
    a_square = torch.exp(2.0 * log_a)

    # Input multiplier = sqrt(1 - a_t^2)
    multiplier = torch.sqrt(torch.clamp(1.0 - a_square, min=0.0))
    gated_input = x * i_gate * multiplier

    # Recurrence step h_t = a_t * h_{t-1} + gated_input
    h_expected = (a_t[:, 0, :] * h_prev) + gated_input[:, 0, :]

    # Verify parity
    assert torch.allclose(h_module, h_expected, atol=1e-5), "Live RG-LRU module diverged from mathematical Equation 4!"


def test_public_history_replay_reconstruction(test_adapter: RecurrentGemmaAdapter):
    """Mandatory Invariant: Deterministic replay of public token history reconstructs all 3 stores exactly."""
    tokens = [5, 18, 33, 77, 92, 114, 150, 180]

    # Pass 1: Continuous execution
    logits_1, state_1 = test_adapter.encode_sequence(tokens)

    # Pass 2: Fresh re-initialization and identical replay
    logits_2, state_2 = test_adapter.encode_sequence(tokens, initial_snapshot=test_adapter.create_canonical_initial_state())

    # Assert exact reconstruction parity across all stores and logits
    assert torch.allclose(logits_1, logits_2, atol=1e-5), "Replay reconstruction produced divergent logits!"
    state_1.assert_strict_equal(state_2, atol=1e-5)


def test_sliding_window_boundary_restore(test_adapter: RecurrentGemmaAdapter):
    """S10.1 Invariant: Verify snapshot -> restore accurately handles sliding-window KV eviction."""
    # Config has sliding_window=4. Execute 10 tokens to cross sliding-window boundary multiple times.
    tokens = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
    
    # Path 1: Uninterrupted continuous stepping
    logits_cont, state_cont = test_adapter.encode_sequence(tokens)

    # Path 2: Encode first 7 tokens -> snapshot -> restore -> step remaining 3 tokens
    _, state_snap = test_adapter.encode_sequence(tokens[:7])
    
    logits_restored, state_restored = test_adapter.encode_sequence(tokens[7:], initial_snapshot=state_snap)

    # Verify logit parity and strict state equality across the sliding window boundary
    assert torch.allclose(logits_cont, logits_restored, atol=1e-5), "Sliding window boundary restore diverged in logits!"
    state_cont.assert_strict_equal(state_restored, atol=1e-5)


def test_explicit_step_vs_upstream_prefill_parity(test_adapter: RecurrentGemmaAdapter):
    """S10.1 Invariant: Verify explicit step() loop matches upstream multi-token prefill forward pass."""
    tokens = [15, 30, 45, 60, 75, 90]
    input_ids = torch.tensor([tokens], device=test_adapter.device)

    # Path A: Normal upstream prefill
    cache = DynamicCache(config=test_adapter.config)
    with torch.no_grad():
        out_prefill = test_adapter.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    prefill_logits = out_prefill.logits[:, -1, :]

    # Path B: Explicit step-by-step adapter loop
    step_logits, _ = test_adapter.encode_sequence(tokens)

    assert torch.allclose(prefill_logits, step_logits, atol=1e-5), "Explicit single-token stepping diverged from upstream prefill!"


def test_adapter_device_dtype_preservation():
    """S10.1 Invariant: Verify adapter preserves caller's existing model device and dtype."""
    config = RecurrentGemmaConfig(
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        block_types=["recurrent", "attention"],
        vocab_size=100,
    )
    raw_model = RecurrentGemmaForCausalLM(config).to(dtype=torch.float32)
    adapter = RecurrentGemmaAdapter(model=raw_model)

    assert adapter.device.type == "cpu"
    assert adapter.dtype == torch.float32
    assert adapter.model is raw_model


def test_cross_branch_injection_smoke_test(test_adapter: RecurrentGemmaAdapter):
    """Plumbing smoke test: Verify injecting snapshot S^A into sequence B runs without runtime or shape faults."""
    seq_a = [10, 20, 30, 40]
    seq_b = [55, 65, 75]

    _, state_a = test_adapter.encode_sequence(seq_a)

    # Inject state_a as prior context for sequence B
    logits_b, state_b = test_adapter.encode_sequence(seq_b, initial_snapshot=state_a)

    assert logits_b.shape[-1] == test_adapter.config.vocab_size
    assert not torch.isnan(logits_b).any(), "Injected state execution produced NaNs in logits!"
    assert not torch.isinf(logits_b).any(), "Injected state execution produced Infs in logits!"
    assert state_b.cache_position == len(seq_a) + len(seq_b)


def test_state_serialization_roundtrip(test_adapter: RecurrentGemmaAdapter):
    """Verify bitwise tensor and metadata equality after serialization and deserialization across all stores."""
    _, state = test_adapter.encode_sequence([12, 34, 56, 78])

    # 1. CPU Dict roundtrip
    cpu_dict = state.to_cpu_dict()
    reconstructed_dict = RecurrentStateSnapshot.from_cpu_dict(cpu_dict, device="cpu")
    state.assert_strict_equal(reconstructed_dict, atol=1e-6)

    # 2. Binary bytes roundtrip
    raw_bytes = state.serialize()
    reconstructed_bytes = RecurrentStateSnapshot.deserialize(raw_bytes, device="cpu")
    state.assert_strict_equal(reconstructed_bytes, atol=1e-6)


def test_environment_provenance_manifest():
    """Verify that the S10 environment provenance manifest exists and records valid metadata."""
    manifest_path = Path("docs/s10_environment_manifest.json")
    assert manifest_path.exists(), "Environment provenance manifest docs/s10_environment_manifest.json missing!"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "provenance" in data
    assert "transformers_version" in data["provenance"]
    assert "torch_version" in data["provenance"]
    assert data["model_target"]["model_id"] == "google/recurrentgemma-2b"
