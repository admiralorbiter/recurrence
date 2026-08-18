"""Sprint S12c: Specificity Microscope Invariant & Unit Tests."""

import pytest
import numpy as np
from recurrence.tasks.specificity_microscope import (
    build_microscope_pairs,
    audit_microscope_panel,
    MICROSCOPE_FAMILIES,
)
from experiments.e12_specificity_microscope.analyze import compute_s12c_pair_cluster_bootstrap


def test_s12c_panel_construction_and_pair_count():
    """Verify that 4 template families generate exactly 24 canonical pairs with correct mappings."""
    pairs = build_microscope_pairs()
    assert len(pairs) == 24

    for p in pairs:
        assert p.val_a != p.val_b
        assert p.val_c != p.val_d
        assert len({p.val_a, p.val_b, p.val_c, p.val_d}) == 4
        assert p.cross_val not in (p.val_a, p.val_b, p.val_c, p.val_d)
        assert p.family_id != p.cross_family_id
        assert p.target_a == f" {p.val_a}"
        assert p.target_b == f" {p.val_b}"


def test_s12c_mock_audit_microscope_panel():
    """Verify panel audit on mock tokenizer."""
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [hash(text) % 1000]

    tok = MockTokenizer()
    is_valid, panel_hash, audit_info = audit_microscope_panel(tok)
    assert is_valid is True
    assert len(panel_hash) == 64
    assert len(audit_info["token_info"]) == 16


def test_s12c_confirmatory_protocol_rejects_wrong_environment():
    """Verify fail-closed environment assertions for confirmatory phase."""
    from experiments.e12_specificity_microscope.run import run_experiment

    with pytest.raises(AssertionError, match="Confirmatory requires google/recurrentgemma-2b"):
        run_experiment(phase="confirmatory", model_id="some-other-model", dtype_str="bfloat16", device="cuda")

    with pytest.raises(AssertionError, match="Confirmatory requires bfloat16"):
        run_experiment(phase="confirmatory", model_id="google/recurrentgemma-2b", dtype_str="float32", device="cuda")


def test_s12c_synthetic_bootstrap_reconstructs_known_estimands():
    """Verify that point estimates and bootstrap reconstruction match planted synthetic ground truth."""
    planted_p_match = 80.0
    planted_p_wrong = 50.0
    planted_p_cross = 30.0
    planted_p_noise = 10.0
    planted_p_whole = 100.0

    synthetic_rows = []
    pairs = build_microscope_pairs()
    regimes = ["constant", "interfering", "natural", "random"]

    conditions_map = {
        "intact_a": 0.0,
        "intact_b": 0.0,
        "whole_swap_a_into_b": planted_p_whole,
        "whole_swap_b_into_a": planted_p_whole,
        "matching_rglru_a_into_b": planted_p_match,
        "matching_rglru_b_into_a": planted_p_match,
        "same_template_wrong_c_into_b": planted_p_wrong,
        "same_template_wrong_d_into_b": planted_p_wrong,
        "same_template_wrong_c_into_a": planted_p_wrong,
        "same_template_wrong_d_into_a": planted_p_wrong,
        "cross_template_e_into_b": planted_p_cross,
        "cross_template_e_into_a": planted_p_cross,
        "noise_rglru_a_into_b": planted_p_noise,
        "noise_rglru_b_into_a": planted_p_noise,
    }

    for p in pairs:
        for reg in regimes:
            for cond_name, val in conditions_map.items():
                synthetic_rows.append({
                    "pair_id": p.pair_id,
                    "family_id": p.family_id,
                    "val_a": p.val_a,
                    "val_b": p.val_b,
                    "regime": reg,
                    "lag": 4096,
                    "condition": cond_name,
                    "directional_displacement": val,
                    "logit_directional_projection": val / 100.0,
                    "donor_cloze_margin": 0.5,
                    "donor_is_top1": True,
                })

    results, meta = compute_s12c_pair_cluster_bootstrap(synthetic_rows, n_boot=500, seed=42)

    assert meta["n_pairs"] == 24
    assert np.isclose(results["p_match"]["estimate"], planted_p_match, atol=1e-5)
    assert np.isclose(results["p_wrong_val"]["estimate"], planted_p_wrong, atol=1e-5)
    assert np.isclose(results["p_cross"]["estimate"], planted_p_cross, atol=1e-5)
    assert np.isclose(results["p_noise"]["estimate"], planted_p_noise, atol=1e-5)

    expected_value_spec = planted_p_match - planted_p_wrong  # 30.0
    expected_template_align = planted_p_wrong - planted_p_cross  # 20.0

    assert np.isclose(results["delta_p_value_spec"]["estimate"], expected_value_spec, atol=1e-5)
    assert np.isclose(results["delta_p_template_align"]["estimate"], expected_template_align, atol=1e-5)
    assert results["delta_p_value_spec"]["ci_low"] > 25.0
    assert results["delta_p_value_spec"]["ci_high"] < 35.0
