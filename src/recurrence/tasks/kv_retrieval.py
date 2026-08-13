"""Key-Value Retrieval task testing exact memory lookup under distractor load."""

import random
import string
from typing import Any, Dict, List
from recurrence.tasks.base import BaseTask, TaskItem


class KVRetrievalTask(BaseTask):
    """Generates Key-Value pair retrieval items with controllable distractor load."""

    def __init__(self, distractor_count: int = 10):
        super().__init__(
            name="kv_retrieval",
            description="Exact Key-Value pair retrieval under distractor load.",
        )
        self.distractor_count = distractor_count

    def _random_key(self, rng: random.Random) -> str:
        return "key_" + "".join(rng.choices(string.ascii_lowercase + string.digits, k=6))

    def _random_value(self, rng: random.Random) -> str:
        return "val_" + "".join(rng.choices(string.ascii_lowercase + string.digits, k=6))

    def generate_items(self, count: int = 10, seed: int = 42) -> List[TaskItem]:
        """Generate reproducible KV retrieval items."""
        rng = random.Random(seed)
        items: List[TaskItem] = []

        for i in range(count):
            keys = [self._random_key(rng) for _ in range(self.distractor_count + 1)]
            values = [self._random_value(rng) for _ in range(self.distractor_count + 1)]
            pairs = list(zip(keys, values))

            # Target key is chosen randomly
            target_key, target_value = rng.choice(pairs)
            rng.shuffle(pairs)

            formatted_pairs = "\n".join([f"- {k}: {v}" for k, v in pairs])
            prompt = (
                f"Below is a list of key-value pairs:\n\n{formatted_pairs}\n\n"
                f"Question: What is the value associated with '{target_key}'? Reply with ONLY the exact value string."
            )

            item = TaskItem(
                item_id=f"kv_item_{i:03d}",
                prompt=prompt,
                ground_truth=target_value,
                distractors=[v for k, v in pairs if v != target_value],
                metadata={"target_key": target_key, "distractor_count": self.distractor_count},
            )
            items.append(item)

        return items

    def score_response(self, item: TaskItem, response: str) -> Dict[str, Any]:
        """Score model response against ground truth value."""
        cleaned_resp = response.strip()
        exact_match = (item.ground_truth.lower() in cleaned_resp.lower())
        return {
            "correct": exact_match,
            "score": 1.0 if exact_match else 0.0,
            "ground_truth": item.ground_truth,
            "response": cleaned_resp,
        }
