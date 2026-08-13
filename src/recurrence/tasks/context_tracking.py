"""Context Tracking task testing multi-step state tracking across event streams."""

import random
from typing import Any, Dict, List
from recurrence.tasks.base import BaseTask, TaskItem


class ContextTrackingTask(BaseTask):
    """Generates multi-step event tracking items (e.g. object movements across locations)."""

    def __init__(self, num_events: int = 6):
        super().__init__(
            name="context_tracking",
            description="Multi-step object movement and location state tracking.",
        )
        self.num_events = num_events

    def generate_items(self, count: int = 10, seed: int = 42) -> List[TaskItem]:
        """Generate reproducible multi-step context tracking items."""
        rng = random.Random(seed)
        objects = ["key", "wallet", "book", "laptop", "phone"]
        locations = ["kitchen", "bedroom", "office", "living room", "garden"]
        items: List[TaskItem] = []

        for i in range(count):
            obj = rng.choice(objects)
            locs = rng.sample(locations, k=min(self.num_events, len(locations)))
            
            event_lines = []
            current_loc = locs[0]
            event_lines.append(f"Initially, the {obj} is in the {current_loc}.")
            
            for next_loc in locs[1:]:
                actor = rng.choice(["Alice", "Bob", "Charlie"])
                event_lines.append(f"{actor} moved the {obj} to the {next_loc}.")
                current_loc = next_loc

            formatted_events = "\n".join(event_lines)
            prompt = (
                f"Read the following sequence of events:\n\n{formatted_events}\n\n"
                f"Question: Where is the {obj} at the end? Reply with ONLY the location name."
            )

            item = TaskItem(
                item_id=f"ctx_item_{i:03d}",
                prompt=prompt,
                ground_truth=current_loc,
                distractors=[l for l in locations if l != current_loc],
                metadata={"object": obj, "final_location": current_loc, "num_events": self.num_events},
            )
            items.append(item)

        return items

    def score_response(self, item: TaskItem, response: str) -> Dict[str, Any]:
        """Score model response against ground truth location."""
        cleaned_resp = response.strip().lower()
        exact_match = (item.ground_truth.lower() in cleaned_resp)
        return {
            "correct": exact_match,
            "score": 1.0 if exact_match else 0.0,
            "ground_truth": item.ground_truth,
            "response": response.strip(),
        }
