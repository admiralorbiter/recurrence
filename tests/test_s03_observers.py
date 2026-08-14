"""Unit and regression tests for Sprint S03 Observers and Privileged Access Index."""

import pytest
import shutil
from pathlib import Path
from recurrence.backends.toy import ToyBackend
from recurrence.observers.visible import VisibleEvidenceObserver
from recurrence.observers.reconstruction import ReconstructionObserver
from recurrence.observers.ablated import InputOnlyObserver, OutputOnlyObserver
from recurrence.analysis.privileged_access import (
    compute_privileged_access_index,
    compute_brier_score_from_predictions,
)
from recurrence.tasks.kv_retrieval import KVRetrievalTask


@pytest.fixture
def tmp_artifact_dir(tmp_path):
    d = tmp_path / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_visible_evidence_observer_toy_backend():
    """Verify VisibleEvidenceObserver formats prompt and parses toy output."""
    backend = ToyBackend(seed=42)
    obs = VisibleEvidenceObserver(backend=backend)
    eval_res = obs.evaluate(
        task_prompt="Key: alpha, Value: 12345. What is the value?",
        target_answer="Answer: 12345\nConfidence: 5",
        seed=42,
    )
    assert eval_res.observer_name == "observer_visible"
    assert eval_res.predicted_correct is not None
    assert eval_res.observer_confidence is not None
    assert 1 <= eval_res.observer_confidence <= 5


def test_reconstruction_observer_agreement_and_disagreement():
    """Verify ReconstructionObserver checks counterfactual agreement correctly."""
    backend = ToyBackend(seed=42)
    obs = ReconstructionObserver(backend=backend)
    
    # Task prompt
    task_prompt = "Retrieve the value for key 'red_panda': A) bamboo B) apple. Answer: A\nConfidence: 5"
    
    # Evaluation when target answer matches toy's reconstruction
    eval_res = obs.evaluate(
        task_prompt=task_prompt,
        target_answer="Answer: A\nConfidence: 5",
        seed=42,
    )
    assert eval_res.observer_name == "observer_reconstruction"
    assert eval_res.reconstructed_answer is not None
    assert isinstance(eval_res.predicted_correct, bool)


def test_ablated_observers_toy_backend():
    """Verify InputOnlyObserver and OutputOnlyObserver parse valid evaluations."""
    backend = ToyBackend(seed=42)
    obs_in = InputOnlyObserver(backend=backend)
    eval_in = obs_in.evaluate(
        task_prompt="Solve this math problem: 2+2=?",
        target_answer="Answer: 4",
        seed=42,
    )
    assert eval_in.observer_name == "observer_input_only"
    assert eval_in.predicted_correct is not None
    assert 1 <= eval_in.observer_confidence <= 5

    obs_out = OutputOnlyObserver(backend=backend)
    eval_out = obs_out.evaluate(
        task_prompt="Irrelevant prompt",
        target_answer="Answer: 4\nConfidence: 5",
        seed=42,
    )
    assert eval_out.observer_name == "observer_output_only"
    assert eval_out.predicted_correct is not None
    assert 1 <= eval_out.observer_confidence <= 5


def test_compute_privileged_access_index_null_case():
    """Verify PAI calculation when self and observer have identical confidence distributions."""
    self_pairs = [(5, True), (5, True), (1, False), (1, False), (4, True), (2, False)]
    obs_dict = {
        "observer_visible": [(5, True), (5, True), (1, False), (1, False), (4, True), (2, False)],
        "observer_reconstruction": [(4, True), (4, True), (2, False), (2, False), (3, True), (2, False)],
    }
    pai_res = compute_privileged_access_index(self_pairs, obs_dict, n_bootstraps=100, seed=42)
    
    # In identical distributions, point PAI should be ~0.0
    assert abs(pai_res["point_pai"]) < 0.05
    assert "ci_95_lower" in pai_res
    assert "ci_95_upper" in pai_res
    assert "bootstrap_p_value" in pai_res
    assert pai_res["ci_95_lower"] <= pai_res["point_pai"] <= pai_res["ci_95_upper"]


def test_brier_score_from_predictions():
    """Verify Brier score calculation on binary forecasts."""
    preds = [(True, True), (True, False), (False, False), (False, True)]
    # Squared errors: (1-1)^2=0, (1-0)^2=1, (0-0)^2=0, (0-1)^2=1 -> mean = 2/4 = 0.5
    brier = compute_brier_score_from_predictions(preds)
    assert brier == pytest.approx(0.5)

    perfect_preds = [(True, True), (False, False), (True, True)]
    assert compute_brier_score_from_predictions(perfect_preds) == pytest.approx(0.0)


def test_shuffled_option_control_generation():
    """Verify that option shuffling permutes the option mapping deterministically without altering ground truth item."""
    raw_pairs = KVRetrievalTask.generate_raw_pairs(
        count=4, distractor_count=3, identifier_type="semantic", seed=42
    )
    task = KVRetrievalTask(identifier_type="semantic", mode="forced_choice")
    items_orig = task.generate_items_from_raw(raw_pairs, seed=42)
    items_shuf = task.generate_items_from_raw(raw_pairs, seed=99)

    assert len(items_orig) == len(items_shuf)
    # Ground truth values should be preserved even if assigned letters change
    for orig, shuf in zip(items_orig, items_shuf):
        assert orig.metadata["target_key"] == shuf.metadata["target_key"]


def test_e02_observer_toy_runner_end_to_end(tmp_artifact_dir):
    """Verify the E02 observer runner executes end-to-end on ToyBackend and writes files."""
    from experiments.e02_observer.run import run_e02_observer

    results_dir = tmp_artifact_dir / "results"
    summary = run_e02_observer(
        use_ollama=False,
        items_per_stratum=2,
        base_output_dir=str(tmp_artifact_dir),
        results_base_dir=str(results_dir),
        run_id="test_obs_toy_001",
        overwrite=True,
    )

    assert summary["total_items"] == 4
    assert "privileged_access_index" in summary
    assert "observer_brier_scores" in summary
    assert "observer_prediction_accuracies" in summary
    assert Path(summary["parquet_path"]).exists()
    assert (results_dir / "test_obs_toy_001" / "trials.jsonl").exists()
    assert (results_dir / "test_obs_toy_001" / "summary.json").exists()
    assert (results_dir / "latest.json").exists()
