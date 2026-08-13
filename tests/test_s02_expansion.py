"""Tests for Sprint S02: Paired Factorial Matrix, Interleaved Context Tracking, Logger Collision Guards, and Strict Scoring."""

import shutil
from pathlib import Path
import pytest
from recurrence.tasks.kv_retrieval import KVRetrievalTask
from recurrence.tasks.context_tracking import ContextTrackingTask
from recurrence.analysis.calibration import compute_calibration_metrics, compute_auroc2
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
    logger = ExperimentLogger(output_dir=tmp_artifact_dir, run_id="test_run_01", overwrite=False)
    event = TrialEvent(run_id="test_run_01", step=1, event_type="test")
    logger.log_event(event)

    # Re-instantiating with overwrite=False must raise FileExistsError
    with pytest.raises(FileExistsError):
        ExperimentLogger(output_dir=tmp_artifact_dir, run_id="test_run_01", overwrite=False)

    # Re-instantiating with overwrite=True succeeds and clears
    logger_overwrite = ExperimentLogger(output_dir=tmp_artifact_dir, run_id="test_run_01", overwrite=True)
    assert len(logger_overwrite.events) == 0


def test_kv_paired_matrix_generation_and_strict_scoring():
    """Verify paired matrix generation and strict exact equality scoring."""
    raw_pairs = KVRetrievalTask.generate_raw_pairs(
        count=2, distractor_count=3, identifier_type="opaque", seed=42
    )
    task_fc = KVRetrievalTask(identifier_type="opaque", mode="forced_choice")
    task_fg = KVRetrievalTask(identifier_type="opaque", mode="free_generation")

    items_fc = task_fc.generate_items_from_raw(raw_pairs, seed=42)
    items_fg = task_fg.generate_items_from_raw(raw_pairs, seed=42)

    # Assert exact same target keys across conditions
    assert items_fc[0].metadata["target_key"] == items_fg[0].metadata["target_key"]

    # Test strict exact scoring
    score_fc = task_fc.score_response(items_fc[0], f"Answer: {items_fc[0].ground_truth}\nConfidence: 5")
    assert score_fc["correct"] is True

    # Free gen with exact match succeeds
    score_fg_exact = task_fg.score_response(items_fg[0], f"Answer: {items_fg[0].ground_truth}\nConfidence: 5")
    assert score_fg_exact["correct"] is True

    # Free gen with substring in non-answer preamble fails
    score_fg_fail = task_fg.score_response(
        items_fg[0],
        f"Answer: wrong_value_string but maybe {items_fg[0].ground_truth}\nConfidence: 1"
    )
    assert score_fg_fail["correct"] is False


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


def test_calibration_discrimination_math():
    """Test calibration metrics on known separation."""
    confs = [5, 4, 2, 1]
    labels = [True, True, False, False]
    metrics = compute_calibration_metrics(confs, labels)

    assert metrics["mean_confidence_correct"] == 4.5
    assert metrics["mean_confidence_incorrect"] == 1.5
    assert metrics["confidence_separation"] == 3.0
    assert metrics["auroc2"] == 1.0


def test_e01_expansion_hardened_pipeline_toy(tmp_artifact_dir):
    """Verify full hardened E01 expansion pipeline runs cleanly with toy backend."""
    results = run_e01_expansion(
        use_ollama=False,
        items_per_condition=2,
        seed=42,
        output_dir=tmp_artifact_dir / "e01_hardened_toy",
        run_id="run_hardened_toy",
        overwrite=True,
    )

    assert results["total_items"] == 12  # 6 conditions * 2 items
    assert "factorial_2x2_matrix" in results
    assert "confidence_elicitation_intervention_check" in results
    assert "context_tracking" in results
    assert Path(results["manifest_path"]).exists()
    assert Path(results["jsonl_path"]).exists()
    assert Path(results["parquet_path"]).exists()
