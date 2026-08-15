"""Unit and integration tests for Horizon 0 v2 2AFC psychophysics, task generators, and difficulty mapping."""

import shutil
from pathlib import Path
import pytest
import numpy as np

from recurrence.tasks.adaptive_metacognition import (
    AdaptiveMetacognition2AFCTask,
    DifficultyConfig,
)
from recurrence.analysis.psychophysics import (
    compute_wilson_score_interval,
    compute_psychometric_curve,
    compute_monotonicity_diagnostics,
    compute_elicitation_reactivity,
)
from experiments.e02b_difficulty_map.run import run_e02b_difficulty_mapping


@pytest.fixture
def tmp_artifact_dir(tmp_path):
    """Temporary artifact directory for E02b tests."""
    d = tmp_path / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_2afc_counterbalancing():
    """Verify exact 50/50 A/B counterbalancing across all generated sweeps."""
    task = AdaptiveMetacognition2AFCTask()
    
    # 1. Distractor sweep
    d_items = task.generate_distractor_sweep(levels=[4, 8], count_per_level=10, base_seed=42)
    assert len(d_items) == 20
    a_count_d = sum(1 for item in d_items if item.ground_truth == "A")
    b_count_d = sum(1 for item in d_items if item.ground_truth == "B")
    assert a_count_d == 10
    assert b_count_d == 10

    # 2. Multi-hop sweep
    h_items = task.generate_multi_hop_sweep(levels=[1, 2], count_per_level=8, base_seed=42)
    assert len(h_items) == 16
    a_count_h = sum(1 for item in h_items if item.ground_truth == "A")
    b_count_h = sum(1 for item in h_items if item.ground_truth == "B")
    assert a_count_h == 8
    assert b_count_h == 8

    # 3. Overwrite sweep
    u_items = task.generate_overwrite_sweep(levels=[0, 1], count_per_level=6, base_seed=42)
    assert len(u_items) == 12
    a_count_u = sum(1 for item in u_items if item.ground_truth == "A")
    b_count_u = sum(1 for item in u_items if item.ground_truth == "B")
    assert a_count_u == 6
    assert b_count_u == 6


def test_fresh_item_determinism():
    """Verify exact seed determinism: same seed yields bitwise identical item; different seed yields different item."""
    task = AdaptiveMetacognition2AFCTask()
    item1 = task.generate_distractor_item(distractor_count=16, seed=123, target_option_letter="A")
    item2 = task.generate_distractor_item(distractor_count=16, seed=123, target_option_letter="A")
    item3 = task.generate_distractor_item(distractor_count=16, seed=999, target_option_letter="A")

    assert item1.prompt == item2.prompt
    assert item1.ground_truth == item2.ground_truth
    assert item1.metadata == item2.metadata

    assert item1.prompt != item3.prompt
    assert item1.metadata["target_key"] != item3.metadata["target_key"]


def test_foil_properties():
    """Verify target and foil values are correctly matched and distinct."""
    task = AdaptiveMetacognition2AFCTask()
    item = task.generate_distractor_item(distractor_count=8, seed=42, target_option_letter="A")
    
    assert item.ground_truth == "A"
    opt_map = item.metadata["option_map"]
    assert len(opt_map) == 2
    assert "A" in opt_map and "B" in opt_map
    assert opt_map["A"] == item.metadata["target_val"]
    assert opt_map["B"] == item.metadata["foil_val"]
    assert opt_map["A"] != opt_map["B"]
    assert "(A)" in item.prompt and "(B)" in item.prompt


def test_multi_hop_chain_integrity():
    """Verify multi-hop relational chain resolves accurately from start key to target terminal value."""
    task = AdaptiveMetacognition2AFCTask()
    hop_depth = 4
    item = task.generate_multi_hop_item(hop_depth=hop_depth, distractor_count=8, seed=42, target_option_letter="B")

    chain_keys = item.metadata["chain_keys"]
    assert len(chain_keys) == hop_depth
    target_val = item.metadata["target_terminal_val"]

    # Verify chain statements appear in prompt
    for h in range(hop_depth - 1):
        expected_ptr = f"{chain_keys[h]} points to {chain_keys[h+1]}"
        assert expected_ptr in item.prompt

    expected_terminal = f"{chain_keys[-1]} maps to {target_val}"
    assert expected_terminal in item.prompt
    assert item.ground_truth == "B"
    assert item.metadata["option_map"]["B"] == target_val


def test_overwrite_timeline_integrity():
    """Verify overwrite timeline updates in chronological order with correct current and stale foil values."""
    task = AdaptiveMetacognition2AFCTask()
    u_count = 3
    item = task.generate_overwrite_item(overwrite_count=u_count, distractor_count=6, seed=42, target_option_letter="A")

    target_key = item.metadata["target_key"]
    val_seq = item.metadata["target_val_sequence"]
    assert len(val_seq) == u_count + 1

    current_val = item.metadata["current_target_val"]
    stale_val = item.metadata["stale_foil_val"]

    assert current_val == val_seq[-1]
    assert stale_val == val_seq[-2]
    assert item.metadata["option_map"]["A"] == current_val
    assert item.metadata["option_map"]["B"] == stale_val


def test_schema_scoring_and_fallback():
    """Verify strict JSON schema compliance and robust fallback scoring."""
    task = AdaptiveMetacognition2AFCTask()
    item = task.generate_distractor_item(distractor_count=4, seed=42, target_option_letter="A")

    # 1. Exact valid JSON answer+confidence
    resp1 = '{"answer": "A", "probability": 85}'
    s1 = task.score_response(item, resp1)
    assert s1["correct"] is True
    assert s1["schema_valid"] is True
    assert s1["parsed_answer"] == "A"
    assert s1["probability"] == pytest.approx(0.85)

    # 2. Wrong answer letter (valid schema)
    resp2 = '{"answer": "B", "probability": 40}'
    s2 = task.score_response(item, resp2)
    assert s2["correct"] is False
    assert s2["schema_valid"] is True
    assert s2["parsed_answer"] == "B"
    assert s2["failure_type"] == "foil_selection_error"

    # 3. Malformed JSON with natural language (schema_valid False, fallback parsed)
    resp3 = 'Answer: A\nProbability: 90%'
    s3 = task.score_response(item, resp3)
    assert s3["correct"] is True
    assert s3["schema_valid"] is False
    assert s3["parsed_answer"] == "A"
    assert s3["probability"] == pytest.approx(0.90)

    # 4. Out of range probability rejected
    resp4 = '{"answer": "A", "probability": 150}'
    s4 = task.score_response(item, resp4)
    assert s4["correct"] is True
    assert s4["schema_valid"] is False
    assert s4["probability"] is None

    # 5. Invalid option letter (e.g. C or D in 2AFC)
    resp5 = '{"answer": "C", "probability": 50}'
    s5 = task.score_response(item, resp5)
    assert s5["correct"] is False
    assert s5["schema_valid"] is False
    assert s5["parsed_answer"] is None
    assert s5["failure_type"] == "response_format_noncompliance"


def test_reactivity_parity():
    """Verify answer-only and answer+confidence items share identical underlying context and options."""
    task_conf = AdaptiveMetacognition2AFCTask(ask_confidence=True)
    task_only = AdaptiveMetacognition2AFCTask(ask_confidence=False)

    item_conf = task_conf.generate_distractor_item(distractor_count=8, seed=42, ask_confidence=True)
    item_only = task_only.generate_distractor_item(distractor_count=8, seed=42, ask_confidence=False)

    assert item_conf.ground_truth == item_only.ground_truth
    assert item_conf.metadata["target_key"] == item_only.metadata["target_key"]
    assert item_conf.metadata["option_map"] == item_only.metadata["option_map"]
    assert "probability" in item_conf.prompt
    assert "probability" not in item_only.prompt


def test_wilson_score_interval_properties():
    """Verify Wilson interval properties: proper bounds, non-degeneracy, and coverage."""
    # 0 out of 10
    l0, u0 = compute_wilson_score_interval(0, 10, confidence=0.95)
    assert l0 == 0.0
    assert 0.20 < u0 < 0.35

    # 10 out of 10
    l10, u10 = compute_wilson_score_interval(10, 10, confidence=0.95)
    assert u10 == 1.0
    assert 0.65 < l10 < 0.80

    # 5 out of 10
    l5, u5 = compute_wilson_score_interval(5, 10, confidence=0.95)
    assert 0.20 < l5 < 0.30
    assert 0.70 < u5 < 0.80


def test_psychometric_curve_and_monotonicity():
    """Verify monotonicity diagnostics on synthetic declining and flat curves."""
    # Monotonically declining records: D=4 (90%), D=8 (80%), D=16 (70%), D=32 (50%)
    declining_records = []
    for d, acc in [(4, 0.9), (8, 0.8), (16, 0.7), (32, 0.5)]:
        for i in range(20):
            declining_records.append({
                "difficulty_level": d,
                "correct": (i < int(acc * 20)),
                "parsed_answer": "A" if i % 2 == 0 else "B",
                "schema_valid": True,
                "answer_parse_valid": True,
                "probability": 0.8 if (i < int(acc * 20)) else 0.4,
                "prompt": "x" * (d * 50),
            })

    curve = compute_psychometric_curve(declining_records)
    mono = curve["monotonicity_diagnostics"]
    assert mono["spearman_rho"] == pytest.approx(-1.0)
    assert mono["kendall_tau"] == pytest.approx(-1.0)
    assert mono["staircase_readiness"] == "staircase_ready"
    assert mono["max_accuracy_drop"] == pytest.approx(0.40)


def test_elicitation_reactivity_analysis():
    """Verify paired reactivity statistics calculation."""
    paired = []
    for i in range(20):
        rec_only = {"parsed_answer": "A" if i % 2 == 0 else "B", "correct": (i < 15)}
        rec_conf = {"parsed_answer": "A" if i % 2 == 0 else "B", "correct": (i < 15)}
        paired.append((rec_only, rec_conf))

    react = compute_elicitation_reactivity(paired)
    assert react["paired_trials_count"] == 20
    assert react["exact_answer_concordance_rate"] == 1.0
    assert react["delta_accuracy_conf_minus_only"] == 0.0
    assert react["mcnemar_p_value"] == 1.0
    assert react["reactivity_status"] == "negligible_reactivity"


def test_e02b_difficulty_map_toy_execution(tmp_artifact_dir):
    """End-to-end dry run of E02b experiment runner using Mock2AFCBackend."""
    res = run_e02b_difficulty_mapping(
        model_name="mock-qwen2.5:3b",
        sweeps="all",
        trials_per_level=2,
        paired_reactivity=True,
        seed=42,
        dry_run=True,
        output_dir=tmp_artifact_dir / "e02b_toy",
    )

    out_dir = tmp_artifact_dir / "e02b_toy"
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "trials.jsonl").exists()
    assert (out_dir / "trials.parquet").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "report.md").exists()

    assert "distractor_load" in res["sweep_results"]
    assert "multi_hop" in res["sweep_results"]
    assert "overwrite_load" in res["sweep_results"]
    assert res["reactivity_results"] is not None
