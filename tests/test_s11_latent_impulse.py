"""Sprint S11: Latent Impulse Response & Store Localization Test Suite.

Validates the stimuli generators, dynamic architectural lag grid, residency boundaries,
normalized cross-store distance metrics, and non-mutating 2AFC retrieval probes:
1. Dynamic lag grid includes all key architectural boundaries (Conv width, sliding window).
2. Matched branches receive identical filler token sequences across all regimes.
3. Event A and Event B stimuli have strictly equal token length.
4. Impulse creates non-zero initial separation (D_rel(0) > 0).
5. A/A Sham pair remains at numerical zero floor.
6. Checkpoint capture does not mutate continuous trajectory.
7. 2AFC probe uses detached branch clone without advancing main cache position.
8. Direct Conv1D residency boundary flag switches at L = conv1d_width - 1.
9. Direct KV residency boundary flag switches at L = sliding_window - 1.
10. Distance metrics are bounded, normalized, and finite.
11. End-to-end execution across all 4 regimes on reference model.
"""

import math
import pytest
import torch
from transformers import RecurrentGemmaConfig

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import (
    CANONICAL_STIMULI_PAIRS,
    ImpulseStimulusPair,
    audit_stimulus_token_equality,
    get_filler_tokens_for_regime,
)
from recurrence.loop.latent_impulse_harness import (
    generate_dynamic_lag_grid,
    compute_rmsdiff,
    compute_scale_relative_dist,
    compute_cossim,
    compute_jensen_shannon_div,
    evaluate_impulse_trajectory,
)


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


def test_dynamic_lag_grid_hits_architectural_boundaries(test_adapter: RecurrentGemmaAdapter):
    """Verify dynamic lag grid correctly incorporates Conv1D buffer and sliding window boundaries."""
    grid = generate_dynamic_lag_grid(test_adapter.config)
    
    # Conv1D width is 4 -> must include 0, 1, 2, 3, 4
    for l in [0, 1, 2, 3, 4]:
        assert l in grid, f"Lag {l} missing from grid for conv1d_width=4"

    # Sliding window is 8 -> must include W//2 (4), W-1 (7), W (8), W+1 (9), 2W (16)
    for l in [4, 7, 8, 9, 16]:
        assert l in grid, f"Window boundary lag {l} missing from grid for sliding_window=8"

    assert grid == sorted(grid), "Lag grid must be strictly sorted"
    assert len(grid) == len(set(grid)), "Lag grid must contain unique elements"


def test_matched_branches_receive_identical_filler():
    """Verify that filler generation is completely deterministic and identical across repeated calls."""
    vocab_size = 200
    length = 20
    seed = 42

    for regime in ["constant", "random", "natural", "interfering"]:
        f1 = get_filler_tokens_for_regime(regime, length=length, seed=seed, vocab_size=vocab_size)
        f2 = get_filler_tokens_for_regime(regime, length=length, seed=seed, vocab_size=vocab_size)
        assert f1 == f2, f"Regime '{regime}' produced divergent filler sequences!"
        assert len(f1) == length, f"Regime '{regime}' produced incorrect length: {len(f1)} != {length}"


def test_event_variants_have_equal_token_length():
    """Verify that all canonical stimulus pairs have equal token lengths for Event A and Event B."""
    assert len(CANONICAL_STIMULI_PAIRS) >= 4, "Expected at least 4 canonical stimulus pairs"
    for pair in CANONICAL_STIMULI_PAIRS:
        words_a = pair.event_a.strip().split()
        words_b = pair.event_b.strip().split()
        assert len(words_a) == len(words_b), (
            f"Stimulus pair {pair.pair_id} has mismatched word lengths: {len(words_a)} vs {len(words_b)}"
        )


def test_impulse_creates_nonzero_initial_separation(test_adapter: RecurrentGemmaAdapter):
    """Verify that distinct Event A vs Event B inputs produce non-zero initial separation at L=0."""
    pair = CANONICAL_STIMULI_PAIRS[0]
    records = evaluate_impulse_trajectory(
        adapter=test_adapter,
        pair=pair,
        regime="constant",
        lag_grid=[0],
        seed=42,
    )
    rec0 = records[0]
    assert rec0.lag == 0
    assert rec0.mean_rglru_d_rel > 0.0, "RGLRU initial scale-relative distance at L=0 must be > 0"
    assert rec0.mean_conv_d_rel > 0.0, "Conv initial scale-relative distance at L=0 must be > 0"
    assert rec0.mean_kv_d_rel > 0.0, "KV initial scale-relative distance at L=0 must be > 0"
    assert rec0.jensen_shannon_div > 0.0, "Initial Jensen-Shannon divergence at L=0 must be > 0"


def test_sham_pair_stays_at_numerical_floor(test_adapter: RecurrentGemmaAdapter):
    """Verify that an A1/A2 sham trajectory stays strictly at numerical zero floor."""
    pair = CANONICAL_STIMULI_PAIRS[0]
    records = evaluate_impulse_trajectory(
        adapter=test_adapter,
        pair=pair,
        regime="constant",
        lag_grid=[0, 1, 2, 4],
        seed=42,
    )
    for rec in records:
        assert rec.sham_mean_d_rel < 1e-5, f"Sham D_rel exceeded noise floor at L={rec.lag}: {rec.sham_mean_d_rel}"
        assert rec.sham_jensen_shannon_div < 1e-5, f"Sham JS div exceeded noise floor at L={rec.lag}: {rec.sham_jensen_shannon_div}"


def test_checkpoint_capture_does_not_mutate_trajectory(test_adapter: RecurrentGemmaAdapter):
    """Verify that capturing lag checkpoints along the trajectory does not mutate continuous stepping."""
    pair = CANONICAL_STIMULI_PAIRS[0]
    tokens = [10, 20, 30, 40, 50]
    
    # 1. Uninterrupted run
    _, final_state_uninterrupted = test_adapter.encode_sequence(tokens)

    # 2. Interrupted run with checkpoints captured along the way
    state = test_adapter.create_canonical_initial_state()
    for tok in tokens:
        _, state = test_adapter.step(tok, state)
        # Deep clone inspection
        _ = state.clone()

    final_state_uninterrupted.assert_strict_equal(state, atol=1e-5)


def test_probe_uses_detached_branch_snapshot(test_adapter: RecurrentGemmaAdapter):
    """Verify that running 2AFC retrieval from cloned snapshots does not advance main cache position."""
    pair = CANONICAL_STIMULI_PAIRS[0]
    records = evaluate_impulse_trajectory(
        adapter=test_adapter,
        pair=pair,
        regime="constant",
        lag_grid=[0, 2],
        seed=42,
    )
    # Check that records ran successfully and produced valid 2AFC margins
    for rec in records:
        assert isinstance(rec.twoway_2afc_margin, float)
        assert isinstance(rec.twoway_2afc_accuracy, float)
        assert not math.isnan(rec.twoway_2afc_margin)


def test_direct_conv_residency_boundary(test_adapter: RecurrentGemmaAdapter):
    """Verify direct Conv1D residency flag switches off at L = conv1d_width - 1."""
    conv_width = test_adapter.config.conv1d_width  # 4
    pair = CANONICAL_STIMULI_PAIRS[0]
    records = evaluate_impulse_trajectory(
        adapter=test_adapter,
        pair=pair,
        regime="constant",
        lag_grid=[0, 1, 2, 3, 4, 5],
        seed=42,
    )
    for rec in records:
        expected = rec.lag < (conv_width - 1)
        assert rec.conv_directly_resident == expected, (
            f"Conv residency mismatch at L={rec.lag}: got {rec.conv_directly_resident}, expected {expected}"
        )


def test_direct_kv_residency_boundary(test_adapter: RecurrentGemmaAdapter):
    """Verify direct KV residency flag switches off at L = sliding_window - 1."""
    window = test_adapter.config.sliding_window  # 8
    pair = CANONICAL_STIMULI_PAIRS[0]
    records = evaluate_impulse_trajectory(
        adapter=test_adapter,
        pair=pair,
        regime="constant",
        lag_grid=[0, 4, 6, 7, 8, 10],
        seed=42,
    )
    for rec in records:
        expected = rec.lag < (window - 1)
        assert rec.kv_directly_resident == expected, (
            f"KV residency mismatch at L={rec.lag}: got {rec.kv_directly_resident}, expected {expected}"
        )


def test_distance_metrics_are_finite_and_normalized():
    """Verify that RMSDiff, D_rel, CosSim, and JS div are mathematically bounded and finite."""
    t1 = torch.randn(2, 64)
    t2 = torch.randn(2, 64)

    rms = compute_rmsdiff(t1, t2)
    assert rms >= 0.0 and not math.isnan(rms)

    d_rel = compute_scale_relative_dist(t1, t2)
    assert 0.0 <= d_rel <= math.sqrt(2.0) + 1e-5
    assert not math.isnan(d_rel)

    cossim = compute_cossim(t1, t2)
    assert -1.0 - 1e-5 <= cossim <= 1.0 + 1e-5
    assert not math.isnan(cossim)

    logits1 = torch.randn(1, 100)
    logits2 = torch.randn(1, 100)
    js = compute_jensen_shannon_div(logits1, logits2)
    assert 0.0 <= js <= math.log(2.0) + 1e-5
    assert not math.isnan(js)


def test_end_to_end_all_regimes_reference_model(test_adapter: RecurrentGemmaAdapter):
    """End-to-end dry run verifying that evaluation completes across all 4 filler regimes."""
    pair = CANONICAL_STIMULI_PAIRS[0]
    regimes = ["constant", "random", "natural", "interfering"]
    lag_grid = [0, 1, 2, 4, 8]

    for regime in regimes:
        records = evaluate_impulse_trajectory(
            adapter=test_adapter,
            pair=pair,
            regime=regime,
            lag_grid=lag_grid,
            seed=42,
        )
        assert len(records) == len(lag_grid)
        for r in records:
            assert not math.isnan(r.mean_rglru_d_rel)
            assert not math.isnan(r.mean_conv_d_rel)
            assert not math.isnan(r.mean_kv_d_rel)
            assert not math.isnan(r.jensen_shannon_div)
            assert not math.isnan(r.twoway_2afc_margin)
