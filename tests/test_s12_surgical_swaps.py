"""Sprint S12: Multi-Store Surgical State Swaps Invariant Test Suite.

Validates the surgical channel transplantation operations, non-mutation of base trajectories,
isolated channel independence, and causal attribution harness:
1. RGLRU-only swap modifies only RGLRU while Conv and KV remain strictly untouched.
2. Conv-only swap modifies only Conv while RGLRU and KV remain strictly untouched.
3. KV-only swap modifies only Key/Value tensors while RGLRU and Conv remain strictly untouched.
4. Surgical swaps do not mutate donor or recipient snapshots.
5. Whole-store swap matches donor state strictly across all 3 channels.
6. Sham swap produces numerical zero difference.
7. End-to-end causal swap harness executes all 12 intervention conditions cleanly.
"""

import math
import pytest
import torch
from transformers import RecurrentGemmaConfig

from recurrence.interventions.surgical_swaps import swap_stores
from recurrence.loop.surgical_swap_harness import evaluate_surgical_swaps, SwapCondition
from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import CANONICAL_STIMULI_PAIRS


@pytest.fixture
def test_adapter() -> RecurrentGemmaAdapter:
    """Create a lightweight RecurrentGemma adapter for fast deterministic tests."""
    config = RecurrentGemmaConfig(
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        lru_width=64,
        conv1d_width=4,
        sliding_window=8,
        block_types=["recurrent", "recurrent", "attention", "recurrent"],
        vocab_size=200,
    )
    torch.manual_seed(42)
    return RecurrentGemmaAdapter(config=config, device="cpu", dtype=torch.float32)


def test_swap_stores_rglru_only_mutates_only_rglru(test_adapter: RecurrentGemmaAdapter):
    """Verify that RGLRU-only swap transplants RGLRU state without modifying Conv or KV."""
    tokens_a = [10, 20, 30]
    tokens_b = [10, 50, 60]

    _, state_a = test_adapter.encode_sequence(tokens_a)
    _, state_b = test_adapter.encode_sequence(tokens_b)

    grafted = swap_stores(recipient=state_b, donor=state_a, channels="rglru")

    # RGLRU must match donor (state_a)
    for l in state_a.rglru:
        assert torch.allclose(grafted.rglru[l], state_a.rglru[l])

    # Conv and KV must match recipient (state_b)
    for l in state_b.conv:
        assert torch.allclose(grafted.conv[l], state_b.conv[l])
    for l in state_b.kv:
        for k in ["key", "value"]:
            if k in state_b.kv[l]:
                assert torch.allclose(grafted.kv[l][k], state_b.kv[l][k])


def test_swap_stores_conv_only_mutates_only_conv(test_adapter: RecurrentGemmaAdapter):
    """Verify that Conv-only swap transplants Conv state without modifying RGLRU or KV."""
    tokens_a = [10, 20, 30]
    tokens_b = [10, 50, 60]

    _, state_a = test_adapter.encode_sequence(tokens_a)
    _, state_b = test_adapter.encode_sequence(tokens_b)

    grafted = swap_stores(recipient=state_b, donor=state_a, channels="conv")

    # Conv must match donor (state_a)
    for l in state_a.conv:
        assert torch.allclose(grafted.conv[l], state_a.conv[l])

    # RGLRU and KV must match recipient (state_b)
    for l in state_b.rglru:
        assert torch.allclose(grafted.rglru[l], state_b.rglru[l])
    for l in state_b.kv:
        for k in ["key", "value"]:
            if k in state_b.kv[l]:
                assert torch.allclose(grafted.kv[l][k], state_b.kv[l][k])


def test_swap_stores_kv_only_mutates_only_kv(test_adapter: RecurrentGemmaAdapter):
    """Verify that KV-only swap transplants Key/Value state without modifying RGLRU or Conv."""
    tokens_a = [10, 20, 30]
    tokens_b = [10, 50, 60]

    _, state_a = test_adapter.encode_sequence(tokens_a)
    _, state_b = test_adapter.encode_sequence(tokens_b)

    grafted = swap_stores(recipient=state_b, donor=state_a, channels="kv")

    # KV must match donor (state_a)
    for l in state_a.kv:
        for k in ["key", "value"]:
            if k in state_a.kv[l]:
                assert torch.allclose(grafted.kv[l][k], state_a.kv[l][k])

    # RGLRU and Conv must match recipient (state_b)
    for l in state_b.rglru:
        assert torch.allclose(grafted.rglru[l], state_b.rglru[l])
    for l in state_b.conv:
        assert torch.allclose(grafted.conv[l], state_b.conv[l])


def test_swap_does_not_mutate_donor_or_recipient(test_adapter: RecurrentGemmaAdapter):
    """Verify that swap_stores leaves original donor and recipient snapshots intact."""
    tokens_a = [10, 20, 30]
    tokens_b = [10, 50, 60]

    _, state_a = test_adapter.encode_sequence(tokens_a)
    _, state_b = test_adapter.encode_sequence(tokens_b)

    clone_a = state_a.clone()
    clone_b = state_b.clone()

    _ = swap_stores(recipient=state_b, donor=state_a, channels="all")

    state_a.assert_strict_equal(clone_a)
    state_b.assert_strict_equal(clone_b)


def test_whole_swap_matches_donor_state(test_adapter: RecurrentGemmaAdapter):
    """Verify that whole-store swap produces a snapshot strictly equal to donor."""
    tokens_a = [10, 20, 30]
    tokens_b = [10, 50, 60]

    _, state_a = test_adapter.encode_sequence(tokens_a)
    _, state_b = test_adapter.encode_sequence(tokens_b)

    grafted = swap_stores(recipient=state_b, donor=state_a, channels="all")
    grafted.assert_strict_equal(state_a)


def test_sham_swap_produces_zero_difference(test_adapter: RecurrentGemmaAdapter):
    """Verify that sham swap A2 -> A1 reproduces exact original state."""
    tokens = [10, 20, 30]
    _, state_a1 = test_adapter.encode_sequence(tokens)
    _, state_a2 = test_adapter.encode_sequence(tokens)

    grafted = swap_stores(recipient=state_a1, donor=state_a2, channels="all")
    grafted.assert_strict_equal(state_a1)


def test_end_to_end_surgical_swap_harness(test_adapter: RecurrentGemmaAdapter):
    """End-to-end dry run verifying all 14 causal swap conditions execute cleanly with raw and logit metrics."""
    pair = CANONICAL_STIMULI_PAIRS[0]
    target_lags = [0, 2, 8]

    records = evaluate_surgical_swaps(
        adapter=test_adapter,
        pair=pair,
        regime="constant",
        target_lags=target_lags,
        seed=42,
    )

    # 3 lags x 14 conditions = 42 records
    assert len(records) == 42
    for r in records:
        assert not math.isnan(r.cloze_margin)
        assert not math.isnan(r.raw_graft_effect)
        assert not math.isnan(r.absolute_displacement)
        assert not math.isnan(r.donor_recipient_norm)
        assert not math.isnan(r.logit_directional_projection)
        assert r.target_choice in ("A", "B")
        if r.causal_attribution_index is not None:
            assert not math.isnan(r.causal_attribution_index)


def test_mediational_dynamic_forward_propagation(test_adapter: RecurrentGemmaAdapter):
    """Verify that mediational forward unroll evaluates KV migration toward donor cleanly."""
    from recurrence.loop.surgical_swap_harness import evaluate_mediational_propagation
    pair = CANONICAL_STIMULI_PAIRS[0]
    res = evaluate_mediational_propagation(
        adapter=test_adapter,
        pair=pair,
        regime="constant",
        initial_lag=2,
        future_tokens=8,
        seed=42,
    )

    assert "kv_migration_index" in res
    assert not math.isnan(res["kv_migration_index"])
    assert "d_med_to_a" in res
    assert "d_med_to_b" in res
    assert "d_a_to_b" in res


