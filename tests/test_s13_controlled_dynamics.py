"""Sprint S13: Controlled Task-Irrelevant Recurrent Dynamics Tests."""

import pytest
import torch
import numpy as np

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter, RecurrentGemmaConfig
from recurrence.tasks.specificity_microscope import build_microscope_pairs
from recurrence.tasks.controlled_drive import (
    verify_token_clock_invariance,
    generate_single_drive_stream,
    compute_frozen_axis,
    project_onto_axis,
    advance_stream,
    advance_stream_along_horizons,
)
from experiments.e13_controlled_recurrent_dynamics.run import select_scout_pairs
from experiments.e13_controlled_recurrent_dynamics.analyze import compute_s13_pair_cluster_bootstrap


def test_s13_token_clock_invariance():
    """Verify Phase S13.0: T_theta^(0)(S) = S for empty token sequences."""
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
    adapter = RecurrentGemmaAdapter(config=config, device="cpu", dtype=torch.float32)
    assert verify_token_clock_invariance(adapter) is True


def test_s13_scout_pair_selection_covers_four_families():
    """Verify that scout selects exactly 4 pairs, 1 from each template family."""
    all_pairs = build_microscope_pairs()
    scout_pairs = select_scout_pairs(all_pairs)
    assert len(scout_pairs) == 4
    families = {p.family_id for p in scout_pairs}
    assert families == {"marked_object", "sealed_container", "monitored_signal", "archived_artifact"}


def test_s13_single_stream_prefix_consistency():
    """Verify that horizons are exact substrings of the 2048-token stream."""
    stream_2048 = generate_single_drive_stream(2048, regime="random", seed=42)
    assert len(stream_2048) == 2048

    for h in [0, 16, 64, 256, 1024, 2048]:
        prefix = stream_2048[:h]
        assert len(prefix) == h
        if h > 0:
            assert prefix == stream_2048[:h]


def test_s13_frozen_axis_and_projection_invariants():
    """Verify u_0 computation and projection arithmetic."""
    z_don = torch.tensor([10.0, 0.0, 0.0])
    z_rec = torch.tensor([0.0, 0.0, 0.0])
    u_0, norm_0 = compute_frozen_axis(z_don, z_rec)

    assert np.isclose(norm_0, 10.0)
    assert torch.allclose(u_0, torch.tensor([1.0, 0.0, 0.0]))

    # Test projection
    z_int = torch.tensor([6.0, 2.0, 0.0])
    disp, proj = project_onto_axis(z_int, z_rec, u_0, norm_0)
    assert np.isclose(disp, 6.0)
    assert np.isclose(proj, 0.60)


def test_s13_advance_stream_arm_mechanics():
    """Verify advance_stream for intact vs rglru_carry_clamped arms."""
    config = RecurrentGemmaConfig(
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        lru_width=32,
        conv1d_width=4,
        sliding_window=16,
        block_types=["recurrent", "attention"],
        vocab_size=100,
    )
    adapter = RecurrentGemmaAdapter(config=config, device="cpu", dtype=torch.float32)
    s0 = adapter.create_canonical_initial_state()
    # Set non-zero initial state
    s0.rglru[0] = torch.ones_like(s0.rglru[0]) * 5.0

    token_ids = [10, 11, 12, 13]

    # Arm 1: Intact
    s_intact = advance_stream(adapter, s0, token_ids, arm="intact_recurrence")
    assert not torch.equal(s_intact.rglru[0], s0.rglru[0]), "Intact RG-LRU should have evolved"

    # Arm 2: Clamped
    s_clamped = advance_stream(adapter, s0, token_ids, arm="rglru_carry_clamped")
    assert torch.equal(s_clamped.rglru[0], s0.rglru[0]), "Clamped RG-LRU carry must equal S_0"
    assert not torch.equal(s_clamped.conv[0], s0.conv[0]), "Conv should have advanced in clamped arm"

    # Test sequential advance along horizons
    snaps_intact = advance_stream_along_horizons(adapter, s0, token_ids, horizons=[0, 2, 4], arm="intact_recurrence")
    assert torch.equal(snaps_intact[0].rglru[0], s0.rglru[0])
    assert not torch.equal(snaps_intact[4].rglru[0], s0.rglru[0])

    snaps_clamped = advance_stream_along_horizons(adapter, s0, token_ids, horizons=[0, 2, 4], arm="rglru_carry_clamped")
    assert torch.equal(snaps_clamped[0].rglru[0], s0.rglru[0])
    assert torch.equal(snaps_clamped[2].rglru[0], s0.rglru[0])
    assert torch.equal(snaps_clamped[4].rglru[0], s0.rglru[0])


def test_s13_synthetic_longitudinal_bootstrap():
    """Verify synthetic reconstruction of known longitudinal V(N) trajectory."""
    pairs = select_scout_pairs(build_microscope_pairs())
    regimes = ["constant", "random", "natural", "interfering"]
    arms = ["intact_recurrence", "rglru_carry_clamped"]
    horizons = [0, 16, 64, 256, 1024, 2048]

    rows = []
    for p in pairs:
        for reg in regimes:
            for arm in arms:
                for h in horizons:
                    # Synthetic values: planted match=100 - h*0.01, planted wrong=60 - h*0.01
                    v_match = 100.0 - h * 0.01 if arm == "intact_recurrence" else 80.0
                    v_wrong = 60.0 - h * 0.01 if arm == "intact_recurrence" else 60.0
                    v_noise = 20.0
                    v_cross = 50.0

                    for cond_name, val in [
                        ("matching_rglru_a_into_b", v_match),
                        ("matching_rglru_b_into_a", v_match),
                        ("same_template_wrong_c_into_b", v_wrong),
                        ("same_template_wrong_d_into_b", v_wrong),
                        ("same_template_wrong_c_into_a", v_wrong),
                        ("same_template_wrong_d_into_a", v_wrong),
                    ]:
                        rows.append({
                            "pair_id": p.pair_id,
                            "family_id": p.family_id,
                            "regime": reg,
                            "arm": arm,
                            "horizon": h,
                            "condition": cond_name,
                            "directional_displacement_u0": val,
                            "normalized_projection_u0": val / 100.0,
                            "directional_displacement_uN": val,
                            "normalized_projection_uN": val / 100.0,
                            "cloze_margin": 0.5,
                            "donor_is_top1": True,
                        })

                    if h in (0, 2048):
                        for cond_name, val in [
                            ("noise_rglru_a_into_b", v_noise),
                            ("noise_rglru_b_into_a", v_noise),
                            ("cross_template_e_into_b", v_cross),
                            ("cross_template_e_into_a", v_cross),
                        ]:
                            rows.append({
                                "pair_id": p.pair_id,
                                "family_id": p.family_id,
                                "regime": reg,
                                "arm": arm,
                                "horizon": h,
                                "condition": cond_name,
                                "directional_displacement_u0": val,
                                "normalized_projection_u0": val / 100.0,
                                "directional_displacement_uN": val,
                                "normalized_projection_uN": val / 100.0,
                                "cloze_margin": 0.5,
                                "donor_is_top1": True,
                            })

    results, meta = compute_s13_pair_cluster_bootstrap(rows, n_boot=200, seed=42)

    # Verify N=0 planted V(0) = 40.0
    assert np.isclose(results["intact_recurrence_N0_v0_spec"]["estimate"], 40.0, atol=1e-4)
    # Verify N=2048 intact vs clamped carry effect: 40.0 - 20.0 = 20.0
    assert np.isclose(results["delta_v0_carry_effect_N2048"]["estimate"], 20.0, atol=1e-4)
    # Verify struct vs noise at N=0: 60.0 - 20.0 = 40.0
    assert np.isclose(results["intact_recurrence_N0_v0_struct_vs_noise"]["estimate"], 40.0, atol=1e-4)
