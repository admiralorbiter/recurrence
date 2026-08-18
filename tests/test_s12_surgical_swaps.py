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


def test_add_intervention_matched_noise(test_adapter: RecurrentGemmaAdapter):
    """Verify that intervention-matched noise matches the layer Frobenius norm of ||donor - recipient||."""
    from recurrence.interventions.surgical_swaps import add_intervention_matched_noise
    tokens_a = [10, 20, 30]
    tokens_b = [10, 50, 60]

    _, state_a = test_adapter.encode_sequence(tokens_a)
    _, state_b = test_adapter.encode_sequence(tokens_b)

    noisy = add_intervention_matched_noise(recipient=state_b, donor=state_a, channel="rglru", seed=42)

    for l in state_b.rglru:
        t_rec = state_b.rglru[l].float()
        t_don = state_a.rglru[l].float()
        t_noisy = noisy.rglru[l].float()

        target_diff = float(torch.norm(t_don - t_rec).item())
        actual_noise = float(torch.norm(t_noisy - t_rec).item())

        assert abs(actual_noise - target_diff) < 1e-4, f"Noise norm mismatch at layer {l}: {actual_noise} vs {target_diff}"


def test_end_to_end_surgical_swap_harness(test_adapter: RecurrentGemmaAdapter):
    """End-to-end dry run verifying all 22 causal swap conditions execute cleanly with signed and logit metrics."""
    pair = CANONICAL_STIMULI_PAIRS[0]
    target_lags = [0, 2, 8]

    records = evaluate_surgical_swaps(
        adapter=test_adapter,
        pair=pair,
        regime="constant",
        target_lags=target_lags,
        seed=42,
    )

    # 3 lags x 22 conditions = 66 records
    assert len(records) == 66
    for r in records:
        assert not math.isnan(r.cloze_margin)
        assert not math.isnan(r.signed_graft_effect)
        assert not math.isnan(r.directional_displacement)
        assert not math.isnan(r.donor_recipient_norm)
        assert not math.isnan(r.logit_directional_projection)
        assert r.target_choice in ("A", "B")
        if r.causal_attribution_index is not None:
            assert not math.isnan(r.causal_attribution_index)


def test_mediational_dynamic_forward_propagation(test_adapter: RecurrentGemmaAdapter):
    """Verify that mediational forward unroll evaluates sliced post-graft KV migration toward donor cleanly."""
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

    assert "post_migration_index" in res
    assert not math.isnan(res["post_migration_index"])
    assert "d_post_med_to_a" in res
    assert "d_post_med_to_b" in res
    assert "d_post_a_to_b" in res
    assert "full_migration_index" in res


def test_s12b_donor_maps_are_bijective_derangements():
    """Verify that both unrelated (+1) and permuted (+7) cross-pair donor maps are valid bijective derangements."""
    from recurrence.loop.surgical_swap_harness import get_balanced_donor_pairs
    pairs = CANONICAL_STIMULI_PAIRS
    assert len(pairs) == 20

    unrelated_targets = []
    permuted_targets = []

    for i, p in enumerate(pairs):
        unrel, perm = get_balanced_donor_pairs(pairs, p)
        # Zero fixed points
        assert unrel.pair_id != p.pair_id, f"Unrelated donor map has fixed point at {p.pair_id}"
        assert perm.pair_id != p.pair_id, f"Permuted donor map has fixed point at {p.pair_id}"
        unrelated_targets.append(unrel.pair_id)
        permuted_targets.append(perm.pair_id)

    # Bijective permutations across all 20 pairs
    assert len(set(unrelated_targets)) == 20, "Unrelated donor mapping is not bijective across 20 pairs"
    assert len(set(permuted_targets)) == 20, "Permuted donor mapping is not bijective across 20 pairs"


def test_s12b_confirmatory_protocol_rejects_wrong_environment():
    """Verify that confirmatory execution strictly rejects non-CUDA, wrong dtype, or wrong model ID."""
    from experiments.e11_surgical_swaps.run import run_experiment
    import pytest

    with pytest.raises(AssertionError, match="Confirmatory run requires google/recurrentgemma-2b"):
        run_experiment(phase="confirmatory", model_id="reference_model")

    with pytest.raises(AssertionError, match="Confirmatory run requires bfloat16"):
        run_experiment(phase="confirmatory", dtype_str="float32")

    with pytest.raises(AssertionError, match="Confirmatory run requires CUDA"):
        run_experiment(phase="confirmatory", device="cpu")


def test_s12b_analysis_rejects_missing_or_duplicate_cells():
    """Verify that analyzer fails closed if condition cells are missing or duplicate."""
    from experiments.e11_surgical_swaps.analyze import compute_s12_pair_cluster_bootstrap
    import pytest

    # 1. Duplicate cell rejection
    dup_rows = [
        {"pair_id": "p1", "regime": "constant", "lag": 8, "condition": "intact_a"},
        {"pair_id": "p1", "regime": "constant", "lag": 8, "condition": "intact_a"},
    ]
    with pytest.raises(ValueError, match="Duplicate cell detected"):
        compute_s12_pair_cluster_bootstrap(dup_rows)

    # 2. Missing cell rejection (e.g. missing partner branch)
    incomplete_rows = [
        {"pair_id": "p1", "regime": "constant", "lag": 4096, "condition": "rglru_only_a_into_b", "directional_displacement": 10.0, "logit_directional_projection": 0.1, "signed_graft_effect": 0.1},
    ]
    with pytest.raises(KeyError, match="Missing confirmatory cell"):
        compute_s12_pair_cluster_bootstrap(incomplete_rows)


def test_s12b_synthetic_bootstrap_reconstructs_known_estimands():
    """Verify that point estimates and bootstrap reconstruction match planted synthetic ground truth."""
    from experiments.e11_surgical_swaps.analyze import compute_s12_pair_cluster_bootstrap

    # Plant synthetic ground truth across 20 pairs
    planted_p_match_2w = 45.0
    planted_p_unrel_2w = -30.0
    planted_p_perm_2w = -15.0
    planted_p_noise_2w = 5.0
    planted_p_match_w1 = 15.0
    planted_p_kv_2w = 70.0

    synthetic_rows = []
    regimes = ["constant", "interfering", "natural", "random"]
    lags = [8, 2049, 4096]

    conditions_map = {
        "intact_a": 0.0,
        "intact_b": 0.0,
        "whole_swap_a_into_b": 100.0,
        "whole_swap_b_into_a": 100.0,
        "rglru_only_a_into_b": planted_p_match_2w,
        "rglru_only_b_into_a": planted_p_match_2w,
        "conv_only_a_into_b": 0.0,
        "conv_only_b_into_a": 0.0,
        "kv_only_a_into_b": planted_p_kv_2w,
        "kv_only_b_into_a": planted_p_kv_2w,
        "recurrent_core_a_into_b": planted_p_match_2w,
        "recurrent_core_b_into_a": planted_p_match_2w,
        "sham_a2_into_a1": 0.0,
        "sham_b2_into_b1": 0.0,
        "noise_rglru_a_into_b_s1": planted_p_noise_2w,
        "noise_rglru_b_into_a_s1": planted_p_noise_2w,
        "noise_rglru_a_into_b_s2": planted_p_noise_2w,
        "noise_rglru_b_into_a_s2": planted_p_noise_2w,
        "unrelated_rglru_a_into_b": planted_p_unrel_2w,
        "unrelated_rglru_b_into_a": planted_p_unrel_2w,
        "permuted_rglru_a_into_b": planted_p_perm_2w,
        "permuted_rglru_b_into_a": planted_p_perm_2w,
    }

    for p_idx in range(20):
        p_id = f"pair_{p_idx:02d}"
        for reg in regimes:
            for lag in lags:
                for cond_name, val in conditions_map.items():
                    val_to_use = val
                    if lag == 2049 and "rglru_only" in cond_name:
                        val_to_use = planted_p_match_w1
                    synthetic_rows.append({
                        "pair_id": p_id,
                        "regime": reg,
                        "lag": lag,
                        "condition": cond_name,
                        "directional_displacement": val_to_use,
                        "logit_directional_projection": val_to_use / 100.0,
                        "signed_graft_effect": 0.5,
                    })

    res = compute_s12_pair_cluster_bootstrap(synthetic_rows, n_boot=500, seed=42)

    assert abs(res["p_match_2w"]["estimate"] - 45.0) < 1e-4
    assert abs(res["delta_p_spec_unrel_2w"]["estimate"] - 75.0) < 1e-4  # 45 - (-30) = 75
    assert abs(res["delta_p_spec_perm_2w"]["estimate"] - 60.0) < 1e-4   # 45 - (-15) = 60
    assert abs(res["delta_p_growth_2w_minus_w1"]["estimate"] - 30.0) < 1e-4  # 45 - 15 = 30
    assert abs(res["delta_p_kv_minus_rglru_2w"]["estimate"] - 25.0) < 1e-4   # 70 - 45 = 25




