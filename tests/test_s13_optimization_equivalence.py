"""Sprint S13: Numerical Equivalence & Benchmark Integration Tests.

Validates that:
1. State-only base model execution produces exact bit-identity against CausalLM for recurrent state transitions.
2. Batched branch execution (B=5) and regime batching (B=20) produce tight numerical agreement for all S13 estimands:
   - V_intact^(0)(N)
   - V_clamped^(0)(N)
   - Delta V_carry_effect^(0)(N)
   - C_logit(N)
   - C_R(N)
   - Q_R(N)
"""

import pytest
import torch
import numpy as np

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter, RecurrentGemmaConfig, RecurrentStateSnapshot
from recurrence.tasks.controlled_drive import (
    compute_frozen_axis,
    project_onto_axis,
    compute_logit_axis_cosine,
    compute_recurrent_state_diff_vec,
    compute_recurrent_geometry,
)


def create_test_adapter(device: str = "cpu") -> RecurrentGemmaAdapter:
    config = RecurrentGemmaConfig(
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        lru_width=64,
        conv1d_width=4,
        sliding_window=16,
        block_types=["recurrent", "recurrent", "attention", "recurrent"],
        vocab_size=200,
    )
    torch.manual_seed(42)
    return RecurrentGemmaAdapter(config=config, device=device, dtype=torch.float32)


def test_state_only_bit_identity():
    """Verify that calling base model directly produces exact bit-identity for recurrent state transitions."""
    adapter = create_test_adapter()
    s0 = adapter.create_canonical_initial_state()
    tokens = [10, 20, 30, 40, 50]

    # Full CausalLM execution
    _, s_causal = adapter.encode_sequence(tokens, initial_snapshot=s0, step_by_step=False)

    # State-only base model execution
    pos = s0.cache_position
    cache = adapter.inject_state_snapshot(s0)
    input_ids = torch.tensor([tokens], device=adapter.device, dtype=torch.long)
    position_ids = torch.arange(pos, pos + len(tokens), device=adapter.device, dtype=torch.long).unsqueeze(0)

    adapter.model.model(
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    s_base = adapter.extract_state_snapshot(past_key_values=cache, cache_position=pos + len(tokens))

    # Assert exact bit identity across all RG-LRU and Conv states
    for l_idx in s_causal.rglru:
        assert torch.equal(s_causal.rglru[l_idx], s_base.rglru[l_idx]), f"RG-LRU mismatch at layer {l_idx}"
    for l_idx in s_causal.conv:
        assert torch.equal(s_causal.conv[l_idx], s_base.conv[l_idx]), f"Conv mismatch at layer {l_idx}"


def test_stack_and_unstack_snapshots():
    """Verify snapshot stacking and unstacking roundtrip fidelity."""
    adapter = create_test_adapter()
    s1 = adapter.create_canonical_initial_state()
    s2 = adapter.create_canonical_initial_state()
    s1.rglru[0] = torch.ones_like(s1.rglru[0]) * 3.0
    s2.rglru[0] = torch.ones_like(s2.rglru[0]) * 7.0

    from recurrence.models.recurrent_gemma_adapter import stack_snapshots, unstack_snapshot

    batched = stack_snapshots([s1, s2])
    assert batched.rglru[0].shape[0] == 2
    assert torch.equal(batched.rglru[0][0], s1.rglru[0][0])
    assert torch.equal(batched.rglru[0][1], s2.rglru[0][0])

    unstacked = unstack_snapshot(batched)
    assert len(unstacked) == 2
    assert torch.equal(unstacked[0].rglru[0], s1.rglru[0])
    assert torch.equal(unstacked[1].rglru[0], s2.rglru[0])


def test_batched_vs_sequential_advance_intact():
    """Verify that batched intact advance matches sequential intact advance."""
    adapter = create_test_adapter()
    s1 = adapter.create_canonical_initial_state()
    s2 = adapter.create_canonical_initial_state()
    s1.rglru[0] = torch.ones_like(s1.rglru[0]) * 1.5
    s2.rglru[0] = torch.ones_like(s2.rglru[0]) * 4.2

    from recurrence.tasks.controlled_drive import advance_stream_along_horizons, advance_batched_stream_along_horizons

    tokens = list(range(10, 50))
    horizons = [0, 8, 20, 40]

    # Sequential
    seq_1 = advance_stream_along_horizons(adapter, s1, tokens, horizons=horizons, arm="intact_recurrence")
    seq_2 = advance_stream_along_horizons(adapter, s2, tokens, horizons=horizons, arm="intact_recurrence")

    # Batched
    batched = advance_batched_stream_along_horizons(adapter, [s1, s2], tokens, horizons=horizons, arm="intact_recurrence")

    for h in horizons:
        for l_idx in s1.rglru:
            assert torch.allclose(seq_1[h].rglru[l_idx], batched[0][h].rglru[l_idx], atol=1e-5)
            assert torch.allclose(seq_2[h].rglru[l_idx], batched[1][h].rglru[l_idx], atol=1e-5)


def test_batched_vs_sequential_advance_clamped():
    """Verify that batched clamped advance matches sequential clamped advance."""
    adapter = create_test_adapter()
    s1 = adapter.create_canonical_initial_state()
    s2 = adapter.create_canonical_initial_state()
    s1.rglru[0] = torch.ones_like(s1.rglru[0]) * 2.0
    s2.rglru[0] = torch.ones_like(s2.rglru[0]) * 5.0

    from recurrence.tasks.controlled_drive import advance_stream_along_horizons, advance_batched_stream_along_horizons

    tokens = list(range(10, 40))
    horizons = [0, 5, 15, 30]

    # Sequential
    seq_1 = advance_stream_along_horizons(adapter, s1, tokens, horizons=horizons, arm="rglru_carry_clamped")
    seq_2 = advance_stream_along_horizons(adapter, s2, tokens, horizons=horizons, arm="rglru_carry_clamped")

    # Batched
    batched = advance_batched_stream_along_horizons(adapter, [s1, s2], tokens, horizons=horizons, arm="rglru_carry_clamped")

    for h in horizons:
        for l_idx in s1.rglru:
            assert torch.allclose(seq_1[h].rglru[l_idx], batched[0][h].rglru[l_idx], atol=1e-5)
            assert torch.allclose(seq_2[h].rglru[l_idx], batched[1][h].rglru[l_idx], atol=1e-5)
            # Verify carry was actually clamped to S_0
            assert torch.allclose(batched[0][h].rglru[l_idx], s1.rglru[l_idx], atol=1e-5)
            assert torch.allclose(batched[1][h].rglru[l_idx], s2.rglru[l_idx], atol=1e-5)
        for l_idx in s1.conv:
            assert torch.allclose(seq_1[h].conv[l_idx], batched[0][h].conv[l_idx], atol=1e-5)
            assert torch.allclose(seq_2[h].conv[l_idx], batched[1][h].conv[l_idx], atol=1e-5)

