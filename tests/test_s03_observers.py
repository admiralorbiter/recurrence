"""Unit and regression tests for Sprint S03.2 Observers, Structured JSON, Distribution Reconstruction, and Stratified Contrasts."""

import pytest
import shutil
import numpy as np
from pathlib import Path
from recurrence.backends.toy import ToyBackend
from recurrence.observers.visible import VisibleAnswerOnlyObserver, VisibleFullTranscriptObserver
from recurrence.observers.reconstruction import ReconstructionObserver
from recurrence.observers.ablated import EqualComputeReviewObserver, InputOnlyObserver, OutputOnlyObserver
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
    """Verify Visible observers extract probabilities in [0.0, 1.0]."""
    backend = ToyBackend(seed=42)
    obs_ans = VisibleAnswerOnlyObserver(backend=backend)
    eval_ans = obs_ans.evaluate(
        task_prompt="Key: alpha, Value: 12345. What is the value? Options: (A) 12345 (B) 67890",
        target_answer='{"answer": "A", "probability": 85}',
        seed=42,
    )
    assert eval_ans.observer_name == "observer_visible_answer_only"
    assert eval_ans.predicted_probability is not None
    assert 0.0 <= eval_ans.predicted_probability <= 1.0
    assert eval_ans.predicted_correct is not None

    obs_full = VisibleFullTranscriptObserver(backend=backend)
    eval_full = obs_full.evaluate(
        task_prompt="Key: alpha, Value: 12345. What is the value?",
        target_answer="Answer: A\nProbability correct: 90",
        seed=42,
    )
    assert eval_full.observer_name == "observer_visible_full_transcript"
    assert eval_full.predicted_probability is not None
    assert 0.0 <= eval_full.predicted_probability <= 1.0


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

    # Evaluate target choosing A (low probability in reconstruction distribution)
    eval_a = obs.evaluate(
        task_prompt=task_prompt,
        target_answer="Answer: A",
        seed=42,
    )
    # A has 10/100 = 0.10
    assert eval_a.predicted_probability == pytest.approx(0.10)
    assert eval_a.reconstructed_answer == "B"
    assert eval_a.predicted_correct is False


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


def test_strict_scoring_no_likert_fallback_in_probability_mode():
    """Verify that score_response does not convert Likert 1-5 when confidence_format='probability'."""
    task = KVRetrievalTask(
        mode="forced_choice",
        identifier_type="semantic",
        ask_confidence=True,
        confidence_format="probability",
    )
    raw = task.generate_raw_pairs(count=1, seed=42)
    items = task.generate_items_from_raw(raw, seed=42)
    item = items[0]

    # Malformed text containing ONLY Likert confidence (Confidence: 4)
    resp_likert_only = "Answer: A\nConfidence: 4"
    res = task.score_response(item, resp_likert_only)
    assert res["probability"] is None  # Must NOT convert 4/5 into 0.8 in probability mode!

    # Valid probability format
    resp_prob = "Answer: A\nProbability correct: 80%"
    res2 = task.score_response(item, resp_prob)
    assert res2["probability"] == pytest.approx(0.80)

    # Valid JSON format
    resp_json = '{"answer": "A", "probability": 85}'
    res3 = task.score_response(item, resp_json)
    assert res3["probability"] == pytest.approx(0.85)
    assert res3["parsed_answer"] == "A"
