"""Unit tests for Sprint S01 tasks (KVRetrievalTask and ContextTrackingTask)."""

from recurrence.tasks.kv_retrieval import KVRetrievalTask
from recurrence.tasks.context_tracking import ContextTrackingTask


def test_kv_retrieval_reproducibility():
    """Verify that KV retrieval generates identical items for a given seed."""
    task = KVRetrievalTask(distractor_count=5)
    items1 = task.generate_items(count=5, seed=123)
    items2 = task.generate_items(count=5, seed=123)

    assert len(items1) == 5
    for item1, item2 in zip(items1, items2):
        assert item1.prompt == item2.prompt
        assert item1.ground_truth == item2.ground_truth


def test_kv_retrieval_scoring():
    """Verify scoring logic for KV retrieval."""
    task = KVRetrievalTask()
    items = task.generate_items(count=1, seed=42)
    target_item = items[0]

    # Correct response
    res_correct = task.score_response(target_item, f"The answer is {target_item.ground_truth}.")
    assert res_correct["correct"] is True
    assert res_correct["score"] == 1.0

    # Incorrect response
    res_wrong = task.score_response(target_item, "val_wrong123")
    assert res_wrong["correct"] is False
    assert res_wrong["score"] == 0.0


def test_context_tracking_reproducibility():
    """Verify context tracking item generation and scoring."""
    task = ContextTrackingTask(num_events=4)
    items = task.generate_items(count=3, seed=99)
    assert len(items) == 3

    target = items[0]
    res = task.score_response(target, target.ground_truth)
    assert res["correct"] is True
