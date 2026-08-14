"""Unit and regression tests for Sprint S03 Observers, Structured JSON, Distribution Reconstruction, and Stratified Contrasts."""

import math
import shutil
import numpy as np
import pytest
from pathlib import Path
from recurrence.backends.toy import ToyBackend
from recurrence.observers.visible import (
    VisibleAnswerOnlyObserver,
    VisibleFullTranscriptObserver,
    _parse_probability_from_text,
)
from recurrence.observers.reconstruction import ReconstructionObserver, _extract_target_letter
from recurrence.observers.ablated import (
    EqualComputeReviewObserver,
    InputOnlyObserver,
    OutputFullResponseOnlyObserver,
)
from recurrence.analysis.privileged_access import (
    compute_continuous_brier_score,
    compute_item_paired_contrasts,
    compute_direct_pairwise_contrast,
    _stratified_paired_bootstrap_indices,
)
from recurrence.tasks.kv_retrieval import KVRetrievalTask


@pytest.fixture
def tmp_artifact_dir(tmp_path):
    d = tmp_path / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_visible_observers_probability_parsing():
    """Verify Visible observers extract probabilities on strict 0-100 scale."""
    backend = ToyBackend(seed=42)
    obs_ans = VisibleAnswerOnlyObserver(backend=backend)
    obs_ans.backend.step = lambda prompt, *args, **kwargs: ('{"probability": 85}', "hash", {})
    eval_ans = obs_ans.evaluate(
        task_prompt="Key: alpha, Value: 12345. What is the value? Options: (A) 12345 (B) 67890",
        target_answer='{"answer": "A", "probability": 85}',
        seed=42,
    )
    assert eval_ans.observer_name == "observer_visible_answer_only"
    assert eval_ans.predicted_probability == pytest.approx(0.85)
    assert eval_ans.predicted_correct is True

    obs_full = VisibleFullTranscriptObserver(backend=backend)
    obs_full.backend.step = lambda prompt, *args, **kwargs: ('{"probability": 90}', "hash", {})
    eval_full = obs_full.evaluate(
        task_prompt="Key: alpha, Value: 12345. What is the value?",
        target_answer="Answer: A\nProbability correct: 90",
        seed=42,
    )
    assert eval_full.observer_name == "observer_visible_full_transcript"
    assert eval_full.predicted_probability == pytest.approx(0.90)


def test_reconstruction_observer_4way_distribution_lookup():
    """Verify ReconstructionObserver performs 4-option distribution lookup for target choice."""
    backend = ToyBackend(seed=42)
    obs = ReconstructionObserver(backend=backend)
    
    task_prompt = "Retrieve key 'cat'. Options: (A) feline (B) canine (C) avian (D) reptile."
    
    # Evaluate target choosing B
    eval_b = obs.evaluate(
        task_prompt=task_prompt,
        target_answer='{"answer": "B"}',
        seed=42,
    )
    # Toy backend returns default dist {"A": 10, "B": 70, "C": 10, "D": 10} -> B has 70/100 = 0.70
    assert eval_b.predicted_probability == pytest.approx(0.70)
    assert eval_b.reconstructed_answer == "B"
    assert eval_b.predicted_correct is True


def test_reconstruction_rejects_incomplete_distribution():
    """Verify ReconstructionObserver strictly rejects incomplete distributions without manufacturing 0s."""
    backend = ToyBackend(seed=42)
    obs = ReconstructionObserver(backend=backend)
    
    # Incomplete raw response with only A and B
    obs.backend.step = lambda prompt, *args, **kwargs: ('{"A": 60, "B": 40}', "hash", {})
    eval_res = obs.evaluate(
        task_prompt="Prompt",
        target_answer='{"answer": "A"}',
        seed=42,
    )
    assert eval_res.predicted_probability is None
    assert eval_res.predicted_correct is None
    assert eval_res.reconstructed_answer is None
    assert eval_res.metadata["distribution_complete"] is False


def test_reconstruction_no_ground_truth_leakage_fallback():
    """Verify ReconstructionObserver does NOT fall back to item_metadata ground-truth option when target answer is unparseable."""
    backend = ToyBackend(seed=42)
    obs = ReconstructionObserver(backend=backend)
    
    # Target answer is unparseable/garbled
    target_answer = '{"garbled_field": "some_random_text"}'
    item_metadata = {
        "target_key": "key_gold",
        "target_value": "val_falcon",
        "target_option_letter": "B",  # Ground-truth letter that must NOT be leaked
    }
    
    eval_res = obs.evaluate(
        task_prompt="Prompt",
        target_answer=target_answer,
        item_metadata=item_metadata,
        seed=42,
    )
    assert eval_res.predicted_probability is None
    assert eval_res.predicted_correct is None
    assert eval_res.metadata["target_answer_parsed"] is None


def test_reconstruction_rejects_out_of_bounds_distribution():
    """Verify ReconstructionObserver strictly rejects distributions containing numbers outside [0, 100]."""
    backend = ToyBackend(seed=42)
    obs = ReconstructionObserver(backend=backend)
    
    # Distribution with value > 100
    obs.backend.step = lambda prompt, *args, **kwargs: ('{"A": 150, "B": 20, "C": 10, "D": 10}', "hash", {})
    eval_res = obs.evaluate(task_prompt="Prompt", target_answer='{"answer": "A"}', seed=42)
    assert eval_res.predicted_probability is None
    assert eval_res.metadata["distribution_complete"] is False

    # Distribution with negative value
    obs.backend.step = lambda prompt, *args, **kwargs: ('{"A": -10, "B": 60, "C": 30, "D": 20}', "hash", {})
    eval_res2 = obs.evaluate(task_prompt="Prompt", target_answer='{"answer": "A"}', seed=42)
    assert eval_res2.predicted_probability is None
    assert eval_res2.metadata["distribution_complete"] is False


def test_reject_out_of_range_and_non_finite_probabilities():
    """Verify that values <0, >100, NaN, and Inf are strictly rejected (None) rather than clamped."""
    task = KVRetrievalTask(mode="forced_choice", ask_confidence=True, confidence_format="probability")
    raw = task.generate_raw_pairs(count=1, seed=42)
    item = task.generate_items_from_raw(raw, seed=42)[0]

    # Negative values
    assert task.score_response(item, '{"answer": "A", "probability": -5}')["probability"] is None
    assert task.score_response(item, 'Answer: A\nProbability: -10')["probability"] is None

    # Values > 100
    assert task.score_response(item, '{"answer": "A", "probability": 101}')["probability"] is None
    assert task.score_response(item, '{"answer": "A", "probability": 150}')["probability"] is None
    assert task.score_response(item, 'Answer: A\nProbability: 500')["probability"] is None

    # Non-finite values
    assert task.score_response(item, '{"answer": "A", "probability": "nan"}')["probability"] is None
    assert task.score_response(item, '{"answer": "A", "probability": "inf"}')["probability"] is None

    # Observer parser directly
    assert _parse_probability_from_text('{"probability": -5}') is None
    assert _parse_probability_from_text('{"probability": 105}') is None
    assert _parse_probability_from_text('{"probability": 150}') is None
    assert _parse_probability_from_text('{"probability": "nan"}') is None
    assert _parse_probability_from_text('{"probability": "infinity"}') is None


def test_nested_dict_answer_resolution_and_probability():
    """Regression test for Trial 6 and Trial 11 failure shapes: nested dictionary answer structures."""
    task = KVRetrievalTask(mode="forced_choice", ask_confidence=True, confidence_format="probability")
    raw = task.generate_raw_pairs(count=1, seed=42)
    item = task.generate_items_from_raw(raw, seed=42)[0]

    # Shape 1: Trial 6 {"answer": {"option": "B", "probability": 30}}
    resp_trial_6 = '{\n  "answer": {\n    "option": "B",\n    " probability": 30\n  }\n}'
    res6 = task.score_response(item, resp_trial_6)
    assert res6["parsed_answer"] == "B"
    assert res6["probability"] == pytest.approx(0.30)

    # Shape 2: Trial 11 {"answer": {"letter": "C", "probability": 75}}
    resp_trial_11 = '{\n  "answer": {\n    "letter": "C",\n    "probability": 75\n  }\n}'
    res11 = task.score_response(item, resp_trial_11)
    assert res11["parsed_answer"] == "C"
    assert res11["probability"] == pytest.approx(0.75)


def test_unified_0_to_100_probability_scale():
    """Verify that probability is strictly parsed on a 0-100 percentage scale."""
    task = KVRetrievalTask(mode="forced_choice", ask_confidence=True, confidence_format="probability")
    raw = task.generate_raw_pairs(count=1, seed=42)
    item = task.generate_items_from_raw(raw, seed=42)[0]

    # 100 -> 1.00
    res100 = task.score_response(item, '{"answer": "A", "probability": 100}')
    assert res100["probability"] == pytest.approx(1.00)

    # 85 -> 0.85
    res85 = task.score_response(item, '{"answer": "A", "probability": 85}')
    assert res85["probability"] == pytest.approx(0.85)

    # 1.0 -> 0.01 (1% under 0-100 scale contract)
    res1 = task.score_response(item, '{"answer": "A", "probability": 1.0}')
    assert res1["probability"] == pytest.approx(0.01)

    # 0.5 -> 0.005 (0.5% under 0-100 scale contract)
    res_half = task.score_response(item, '{"answer": "A", "probability": 0.5}')
    assert res_half["probability"] == pytest.approx(0.005)


def test_extract_target_letter_nested():
    """Verify _extract_target_letter resolves nested dicts cleanly."""
    assert _extract_target_letter('{"answer": {"option": "B"}}') == "B"
    assert _extract_target_letter('{"answer": {"letter": "C"}}') == "C"
    assert _extract_target_letter('{"answer": "D"}') == "D"
    assert _extract_target_letter('Answer: A') == "A"
    assert _extract_target_letter('{"random": "junk"}') is None


def test_equal_compute_review_observer():
    """Verify EqualComputeReviewObserver formats prompts for self and other framing."""
    backend = ToyBackend(seed=42)
    obs_self = EqualComputeReviewObserver(backend=backend, framing="self")
    eval_self = obs_self.evaluate(
        task_prompt="Task prompt",
        target_answer='{"answer": "A", "probability": 90}',
        seed=42,
    )
    assert eval_self.observer_name == "self_review_equal_compute"
    assert eval_self.predicted_probability is not None

    obs_other = EqualComputeReviewObserver(backend=backend, framing="other")
    eval_other = obs_other.evaluate(
        task_prompt="Task prompt",
        target_answer="Answer: A",
        seed=42,
    )
    assert eval_other.observer_name == "observer_review_other"
    assert eval_other.predicted_probability is not None


def test_continuous_brier_score():
    """Verify continuous Brier score (mean squared error between prob and actual {0, 1})."""
    preds = [(0.9, True), (0.1, False), (0.8, False), (0.2, True)]
    brier = compute_continuous_brier_score(preds)
    assert brier == pytest.approx(0.325)


def test_stratified_paired_bootstrap():
    """Verify stratified paired bootstrap preserves positive and negative label counts."""
    labels = [True, True, True, False, False]
    rng = np.random.RandomState(42)
    for _ in range(20):
        idx = _stratified_paired_bootstrap_indices(labels, rng)
        sampled_labels = [labels[i] for i in idx]
        assert sum(sampled_labels) == 3
        assert sum(not y for y in sampled_labels) == 2


def test_direct_pairwise_contrast():
    """Verify direct pairwise contrast between two evaluators on shared items."""
    map_self_rev = {
        "item_1": (0.8, True),
        "item_2": (0.2, False),
        "item_3": (0.7, True),
    }
    map_other_rev = {
        "item_1": (0.75, True),
        "item_2": (0.30, False),
        "item_3": (0.65, True),
    }

    contrast = compute_direct_pairwise_contrast(
        map_a=map_self_rev,
        map_b=map_other_rev,
        name_a="self_review",
        name_b="other_review",
        sesoi=0.10,
        n_bootstraps=100,
        seed=42,
    )

    assert contrast["shared_items_count"] == 3
    assert "delta_auroc2" in contrast
    assert "ci_95_lower" in contrast
    assert "ci_95_upper" in contrast
    assert "delta_brier_score" in contrast


def test_decoupled_validity_flags_and_trial_21_shape():
    """Verify that answer parse validity, probability parse validity, and schema validity are decoupled."""
    task = KVRetrievalTask(mode="forced_choice", ask_confidence=True, confidence_format="probability")
    raw = task.generate_raw_pairs(count=1, seed=42)
    item = task.generate_items_from_raw(raw, seed=42)[0]
    # Set ground truth to 'A' for test
    item.ground_truth = "A"

    # Shape 1: Trial 21 malformed envelope with clean "answer": "A"
    resp_trial_21 = '{\n  "answer": "A",\n  " ": "probability"'
    score21 = task.score_response(item, resp_trial_21)
    assert score21["parsed_answer"] == "A"
    assert score21["correct"] is True
    assert score21["answer_parse_valid"] is True
    assert score21["probability"] is None
    assert score21["probability_parse_valid"] is False
    assert score21["schema_valid"] is False

    # Shape 2: Clean valid schema output
    resp_clean = '{"answer": "A", "probability": 85}'
    score_clean = task.score_response(item, resp_clean)
    assert score_clean["parsed_answer"] == "A"
    assert score_clean["correct"] is True
    assert score_clean["answer_parse_valid"] is True
    assert score_clean["probability"] == pytest.approx(0.85)
    assert score_clean["probability_parse_valid"] is True
    assert score_clean["schema_valid"] is True

    # Shape 3: Strict rejection of trailing text / invalid option strings
    resp_trailing = '{"answer": "C_bronze_tiger", "probability": 50}'
    score_trailing = task.score_response(item, resp_trailing)
    assert score_trailing["parsed_answer"] == "C_bronze_tiger"
    assert score_trailing["correct"] is False
    assert score_trailing["schema_valid"] is False


def test_failed_compliance_gate_suppresses_inferential_claims():
    """Verify that generate_markdown_report suppresses confirmatory claims when compliance gate fails."""
    from experiments.e02_observer.run import generate_markdown_report

    mock_failed_summary = {
        "sprint": "S03.3",
        "run_id": "run_e02_obs_test",
        "model_name": "qwen2.5:3b",
        "total_items": 40,
        "compliance_rates": {
            "primary_compliance_rates": {"self_immediate": 0.70, "observer_reconstruction": 0.35},
            "min_primary_compliance": 0.35,
            "compliance_gate_passed": False,
        },
        "target_task_performance": {"overall_accuracy": 0.45},
        "joint_pai_summary": {"joint_shared_items_count": 4, "point_pai": -0.375, "ci_95_lower": -0.75, "ci_95_upper": 0.0},
        "paired_intersection_contrasts": {},
        "direct_pairwise_contrasts": {},
    }

    report = generate_markdown_report(mock_failed_summary)
    assert "Measurement Validity Gate Failed" in report
    assert "do NOT support a Level-0 privileged-access conclusion" in report
    assert "Scientific Interpretation (Measurement Gate Passed)" not in report
