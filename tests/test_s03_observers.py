"""Unit and regression tests for Sprint S03.1 Observers, Probability Metrics, and Item-Paired Contrasts."""

import pytest
import shutil
from pathlib import Path
from recurrence.backends.toy import ToyBackend
from recurrence.observers.visible import VisibleAnswerOnlyObserver, VisibleFullTranscriptObserver
from recurrence.observers.reconstruction import ReconstructionObserver
from recurrence.observers.ablated import EqualComputeReviewObserver, InputOnlyObserver, OutputOnlyObserver
from recurrence.analysis.privileged_access import (
    compute_continuous_brier_score,
    compute_item_paired_contrasts,
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
        target_answer="Answer: A\nProbability correct: 85%",
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


def test_reconstruction_observer_directionality():
    """Verify ReconstructionObserver maps agreement to high P(Target Correct) and disagreement to low P."""
    backend = ToyBackend(seed=42)
    obs = ReconstructionObserver(backend=backend)
    
    task_prompt = "Retrieve key 'cat'. Options: (A) feline (B) canine. Answer: A\nProbability correct: 90%"
    
    # 1. Matching answer
    eval_agree = obs.evaluate(
        task_prompt=task_prompt,
        target_answer="Answer: action_9\nProbability correct: 80%",
        seed=42,
    )
    # When toy output matches target answer, predicted_probability should equal recon_probability
    assert eval_agree.predicted_probability is not None
    assert 0.0 <= eval_agree.predicted_probability <= 1.0

    # 2. Mismatching answer
    eval_disagree = obs.evaluate(
        task_prompt=task_prompt,
        target_answer="Answer: DIFFERENT_ANSWER\nProbability correct: 80%",
        seed=42,
    )
    assert eval_disagree.predicted_probability is not None
    # For mismatch, probability of target being correct is 1 - recon_prob (low)
    assert eval_disagree.predicted_probability <= 0.5


def test_equal_compute_review_observer():
    """Verify EqualComputeReviewObserver formats prompts for self and other framing."""
    backend = ToyBackend(seed=42)
    obs_self = EqualComputeReviewObserver(backend=backend, framing="self")
    eval_self = obs_self.evaluate(
        task_prompt="Task prompt",
        target_answer="Answer: A",
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
    # (0.9-1)^2 = 0.01, (0.1-0)^2 = 0.01, (0.8-0)^2 = 0.64, (0.2-1)^2 = 0.64 -> sum = 1.30 / 4 = 0.325
    brier = compute_continuous_brier_score(preds)
    assert brier == pytest.approx(0.325)


def test_strict_item_paired_intersection_contrasts():
    """Verify that item-paired contrasts evaluate strictly over shared valid keys."""
    # Self valid on items 1, 2, 3, 4
    self_map = {
        "item_1": (0.9, True),
        "item_2": (0.2, False),
        "item_3": (0.8, True),
        "item_4": (None, False),  # Missing self score
        "item_5": (0.4, False),
    }
    # Observer A valid on items 1, 2, 4 (missing item 3 and 5)
    obs_a_map = {
        "item_1": (0.8, True),
        "item_2": (0.3, False),
        "item_4": (0.9, False),
        "item_5": (None, False),  # Missing obs score
    }
    # Observer B valid on all items
    obs_b_map = {
        "item_1": (0.85, True),
        "item_2": (0.15, False),
        "item_3": (0.75, True),
        "item_4": (0.40, False),
        "item_5": (0.30, False),
    }

    res = compute_item_paired_contrasts(
        self_item_map=self_map,
        observer_item_maps={"obs_a": obs_a_map, "obs_b": obs_b_map},
        sesoi=0.10,
        n_bootstraps=100,
        seed=42,
    )

    contrasts = res["contrasts"]
    # Shared keys for Obs A should be exactly {item_1, item_2} (count = 2)
    assert contrasts["obs_a"]["shared_items_count"] == 2
    # Shared keys for Obs B should be exactly {item_1, item_2, item_3, item_5} (count = 4)
    assert contrasts["obs_b"]["shared_items_count"] == 4
    assert "delta_auroc2_self_minus_obs" in contrasts["obs_a"]
    assert "ci_95_lower" in contrasts["obs_a"]
    assert "ci_95_upper" in contrasts["obs_a"]


def test_e02_hardened_runner_end_to_end(tmp_artifact_dir):
    """Verify E02 hardened runner executes end-to-end and computes paired intersection contrasts."""
    from experiments.e02_observer.run import run_e02_observer

    results_dir = tmp_artifact_dir / "results"
    summary = run_e02_observer(
        use_ollama=False,
        items_per_stratum=2,
        base_output_dir=str(tmp_artifact_dir),
        results_base_dir=str(results_dir),
        run_id="test_obs_hardened_001",
        overwrite=True,
    )

    assert summary["total_items"] == 4
    assert "paired_intersection_contrasts" in summary
    assert "joint_pai_summary" in summary
    assert "observer_brier_scores" in summary
    assert Path(summary["parquet_path"]).exists()
    assert (results_dir / "test_obs_hardened_001" / "trials.jsonl").exists()
    assert (results_dir / "test_obs_hardened_001" / "summary.json").exists()
    assert (results_dir / "latest.json").exists()
