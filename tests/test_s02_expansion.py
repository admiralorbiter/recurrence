"""Tests for Sprint S02: Deterministic Replay, Counterbalancing, Paired Calibration, and Strict Scoring."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
from recurrence.tasks.kv_retrieval import KVRetrievalTask
from recurrence.tasks.context_tracking import ContextTrackingTask
from recurrence.analysis.calibration import (
    compute_post_decision_discrimination_from_pairs,
    compute_calibration_metrics,
    compute_auroc2,
)
from recurrence.core.logging import ExperimentLogger, TrialEvent
from experiments.e01_expansion.run import run_e01_expansion


@pytest.fixture
def tmp_artifact_dir(tmp_path):
    d = tmp_path / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_logger_collision_protection(tmp_artifact_dir):
    """Verify ExperimentLogger raises FileExistsError on run collision when overwrite=False."""
    run_dir = tmp_artifact_dir / "test_run_01"
    logger = ExperimentLogger(output_dir=run_dir, run_id="test_run_01", overwrite=False)
    event = TrialEvent(run_id="test_run_01", step=1, event_type="test")
    logger.log_event(event)

    # Re-instantiating with overwrite=False must raise FileExistsError because run_dir exists
    with pytest.raises(FileExistsError):
        ExperimentLogger(output_dir=run_dir, run_id="test_run_01", overwrite=False)

    # Re-instantiating with overwrite=True succeeds and clears
    logger_overwrite = ExperimentLogger(output_dir=run_dir, run_id="test_run_01", overwrite=True)
    assert len(logger_overwrite.events) == 0


def test_kv_paired_matrix_generation_and_strict_scoring():
    """Verify paired matrix generation and strict exact equality scoring."""
    raw_pairs = KVRetrievalTask.generate_raw_pairs(
        count=4, distractor_count=3, identifier_type="opaque", seed=42
    )
    task_fc = KVRetrievalTask(identifier_type="opaque", mode="forced_choice")
    task_fg = KVRetrievalTask(identifier_type="opaque", mode="free_generation")

    items_fc = task_fc.generate_items_from_raw(raw_pairs, seed=42)
    items_fg = task_fg.generate_items_from_raw(raw_pairs, seed=42)

    # Assert exact same target keys across conditions
    assert items_fc[0].metadata["target_key"] == items_fg[0].metadata["target_key"]

    # Test strict exact scoring: exact letter succeeds
    score_fc = task_fc.score_response(items_fc[0], f"Answer: {items_fc[0].ground_truth}\nConfidence: 5")
    assert score_fc["correct"] is True

    # Forced choice prefix like "A because..." must FAIL under strict exact matching
    score_fc_prefix = task_fc.score_response(items_fc[0], f"Answer: {items_fc[0].ground_truth} because it is the first option\nConfidence: 5")
    assert score_fc_prefix["correct"] is False

    # Free gen with exact match succeeds
    score_fg_exact = task_fg.score_response(items_fg[0], f"Answer: {items_fg[0].ground_truth}\nConfidence: 5")
    assert score_fg_exact["correct"] is True

    # Free gen with substring in non-answer preamble fails
    score_fg_fail = task_fg.score_response(
        items_fg[0],
        f"Answer: wrong_value_string but maybe {items_fg[0].ground_truth}\nConfidence: 1"
    )
    assert score_fg_fail["correct"] is False


def test_counterbalanced_forced_choice_target_positions():
    """Verify forced choice exactly counterbalances target positions across A, B, C, D."""
    raw_pairs = KVRetrievalTask.generate_raw_pairs(
        count=20, distractor_count=5, identifier_type="opaque", seed=42
    )
    task_fc = KVRetrievalTask(identifier_type="opaque", mode="forced_choice")
    items = task_fc.generate_items_from_raw(raw_pairs, seed=42)

    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for item in items:
        counts[item.ground_truth] += 1

    # In 20 items, exactly 5 targets must be at A, 5 at B, 5 at C, 5 at D
    assert counts == {"A": 5, "B": 5, "C": 5, "D": 5}


def test_cross_process_deterministic_replay():
    """Verify that item generation produces identical output across two fresh Python processes."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import json, sys; "
            "from recurrence.tasks.kv_retrieval import KVRetrievalTask; "
            "raw = KVRetrievalTask.generate_raw_pairs(10, 5, 'opaque', 42); "
            "print(json.dumps(raw))"
        ),
    ]

    out1 = subprocess.check_output(cmd, text=True).strip()
    out2 = subprocess.check_output(cmd, text=True).strip()

    assert out1 == out2
    data1 = json.loads(out1)
    data2 = json.loads(out2)
    assert len(data1) == 10
    assert data1 == data2


def test_calibration_pairing_with_no_confidence_blocks():
    """Verify that compute_post_decision_discrimination_from_pairs maintains alignment even with interleaved None."""
    # Sequence: (5, True), (None, False), (None, False), (1, False)
    paired_data = [
        (5, True),
        (None, False),
        (None, False),
        (1, False),
    ]

    metrics = compute_post_decision_discrimination_from_pairs(paired_data)
    assert metrics["valid_confidence_count"] == 2
    assert metrics["mean_confidence_correct"] == 5.0
    assert metrics["mean_confidence_incorrect"] == 1.0
    assert metrics["confidence_separation"] == 4.0
    assert metrics["auroc2"] == 1.0


def test_interleaved_context_tracking_lag_and_scoring():
    """Verify interleaved context tracking creates valid items and identifies error categories."""
    task = ContextTrackingTask(num_objects=3, total_transitions=6, lag_k=2)
    items = task.generate_items(count=2, seed=42)
    item = items[0]

    assert item.metadata["lag_k"] == 2
    assert "target_object" in item.metadata

    # Correct exact answer
    res_corr = task.score_response(item, f"Answer: {item.ground_truth}\nConfidence: 4")
    assert res_corr["correct"] is True

    # Recency bias test (answering with the location in the final sentence)
    last_loc = item.metadata["last_event_location"]
    if last_loc.lower() != item.ground_truth.lower():
        res_recency = task.score_response(item, f"Answer: {last_loc}\nConfidence: 2")
        assert res_recency["correct"] is False
        assert res_recency["substitution_category"] == "terminal_sentence_recency_bias"


def test_e01_expansion_toy_runner_end_to_end(tmp_artifact_dir):
    """Verify the 6-condition expansion runner executes end-to-end on ToyBackend and writes files."""
    results_dir = tmp_artifact_dir / "results"
    summary = run_e01_expansion(
        use_ollama=False,
        items_per_condition=2,
        base_output_dir=str(tmp_artifact_dir),
        results_base_dir=str(results_dir),
        run_id="test_exp_toy_001",
        overwrite=True,
    )

    assert summary["total_items"] == 12
    assert "paired_factorial_2x2_matrix" in summary
    assert "confidence_elicitation_paired_contingency" in summary
    assert "paired_response_mode_contingency" in summary
    assert Path(summary["parquet_path"]).exists()
    assert (results_dir / "test_exp_toy_001" / "trials.jsonl").exists()
    assert (results_dir / "test_exp_toy_001" / "summary.json").exists()
    assert (results_dir / "latest.json").exists()
