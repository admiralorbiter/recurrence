"""Interleaved Multi-Object Context Tracking with parametric interfering lag and strict scoring."""

import random
import re
from typing import Any, Dict, List, Optional
from recurrence.tasks.base import BaseTask, TaskItem


_OBJECTS = ["key", "book", "phone", "wallet", "compass", "lantern"]
_LOCATIONS = ["kitchen", "bedroom", "office", "living room", "garden", "library", "attic", "basement"]
_ACTORS = ["Alice", "Bob", "Charlie", "David", "Emma"]


def _normalize_location(text: str) -> str:
    """Strict normalization: lowercase, strip punctuation and leading articles."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
    cleaned = re.sub(r"[^\w\s]", "", cleaned).strip()
    return cleaned


class ContextTrackingTask(BaseTask):
    """Interleaved multi-object location tracking.

    Interleaves movements across multiple objects so that the queried target object
    is followed by k intervening updates to other objects.
    """

    def __init__(
        self,
        num_objects: int = 3,
        total_transitions: int = 6,
        lag_k: Optional[int] = None,
        num_events: Optional[int] = None,
    ):
        if num_events is not None:
            total_transitions = num_events
        name = f"context_tracking_obj{num_objects}_trans{total_transitions}"
        desc = (
            f"Interleaved multi-object tracking ({num_objects} objects, {total_transitions} transitions, "
            f"lag_k={lag_k if lag_k is not None else 'varied'})."
        )
        super().__init__(name=name, description=desc)
        self.num_objects = num_objects
        self.total_transitions = total_transitions
        self.lag_k = lag_k

    def generate_items(self, count: int = 10, seed: int = 42) -> List[TaskItem]:
        """Generate reproducible interleaved tracking items."""
        rng = random.Random(seed)
        items: List[TaskItem] = []

        for i in range(count):
            # 1. Sample objects and initial distinct locations
            active_objects = rng.sample(_OBJECTS, k=self.num_objects)
            target_obj = active_objects[0]
            other_objects = active_objects[1:]

            loc_pool = list(_LOCATIONS)
            rng.shuffle(loc_pool)
            object_states: Dict[str, str] = {
                obj: loc_pool[idx] for idx, obj in enumerate(active_objects)
            }
            object_histories: Dict[str, List[str]] = {
                obj: [object_states[obj]] for obj in active_objects
            }

            # 2. Determine target object final move position
            # lag_k is how many moves of other objects occur AFTER target's final move
            k = self.lag_k if self.lag_k is not None else rng.choice([0, 1, 2, 3])
            k = min(k, self.total_transitions - 1)
            target_last_step = self.total_transitions - k  # 1-indexed step where target moves last

            events: List[Dict[str, Any]] = []
            
            # Initial placement sentences
            initial_sentences = [
                f"Initially, the {obj} is in the {object_states[obj]}."
                for obj in active_objects
            ]

            # Generate sequence of transitions
            for step_idx in range(1, self.total_transitions + 1):
                actor = rng.choice(_ACTORS)
                if step_idx == target_last_step:
                    moving_obj = target_obj
                elif step_idx > target_last_step:
                    moving_obj = rng.choice(other_objects)
                else:
                    # Before target last step, choose randomly
                    moving_obj = rng.choice(active_objects)

                curr_loc = object_states[moving_obj]
                avail_locs = [loc for loc in _LOCATIONS if loc != curr_loc]
                next_loc = rng.choice(avail_locs)

                object_states[moving_obj] = next_loc
                object_histories[moving_obj].append(next_loc)

                sentence = f"{actor} moved the {moving_obj} to the {next_loc}."
                events.append({
                    "step": step_idx,
                    "actor": actor,
                    "object": moving_obj,
                    "to_location": next_loc,
                    "sentence": sentence,
                })

            story_text = "\n".join(initial_sentences + [e["sentence"] for e in events])
            last_event_loc = events[-1]["to_location"]
            last_event_obj = events[-1]["object"]

            target_terminal = object_states[target_obj]
            target_prev = (
                object_histories[target_obj][-2]
                if len(object_histories[target_obj]) >= 2
                else object_histories[target_obj][0]
            )
            target_initial = object_histories[target_obj][0]

            prompt = (
                f"Read the following sequence of events carefully:\n\n{story_text}\n\n"
                f"Question: Where is the {target_obj} at the very end?\n\n"
                f"Format your response strictly as:\n"
                f"Answer: <location name>\n"
                f"Confidence: <1 to 5, where 1 is total guess and 5 is absolutely certain>"
            )

            all_visited = set(loc for hist in object_histories.values() for loc in hist)
            unvisited = [loc for loc in _LOCATIONS if loc not in all_visited]

            metadata = {
                "target_object": target_obj,
                "all_objects": active_objects,
                "target_terminal_location": target_terminal,
                "target_previous_location": target_prev,
                "target_initial_location": target_initial,
                "lag_k": k,
                "last_event_location": last_event_loc,
                "last_event_object": last_event_obj,
                "unvisited_locations": unvisited,
                "object_histories": object_histories,
            }

            item = TaskItem(
                item_id=f"ctx_interleave_k{k}_{i:03d}",
                prompt=prompt,
                ground_truth=target_terminal,
                distractors=[loc for loc in _LOCATIONS if loc != target_terminal],
                metadata=metadata,
            )
            items.append(item)

        return items

    def score_response(self, item: TaskItem, response: str) -> Dict[str, Any]:
        """Strict answer parsing, exact normalized comparison, and substitution classification."""
        cleaned_resp = response.strip()

        # Robust Answer extraction: stops before inline Confidence or newline
        ans_match = re.search(
            r"Answer:\s*(?:<[^>]+>:\s*|[a-zA-Z]+:\s*)?<?([a-zA-Z0-9_\s]+?)>?(?:\s*Confidence|\s*Confident|;|\n|\r|$)",
            cleaned_resp,
            re.IGNORECASE,
        )
        raw_answer = ans_match.group(1).strip() if ans_match else cleaned_resp

        # Parse Confidence: ...
        conf_match = re.search(r"(?:Confidence|Confident):\s*<?([1-5])>?", cleaned_resp, re.IGNORECASE)
        confidence = int(conf_match.group(1)) if conf_match else None

        norm_answer = _normalize_location(raw_answer)
        norm_ground_truth = _normalize_location(item.ground_truth)

        # STRICT EXACT EQUALITY (not substring containment)
        correct = (norm_answer == norm_ground_truth)

        substitution_category = None
        if not correct:
            target_prev = _normalize_location(item.metadata.get("target_previous_location", ""))
            target_init = _normalize_location(item.metadata.get("target_initial_location", ""))
            last_event_loc = _normalize_location(item.metadata.get("last_event_location", ""))
            unvisited = [_normalize_location(u) for u in item.metadata.get("unvisited_locations", [])]

            if not norm_answer or norm_answer in ["none", "unknown", "null"]:
                substitution_category = "response_format_noncompliance"
            elif norm_answer == last_event_loc and item.metadata.get("lag_k", 0) > 0:
                # Model picked the location mentioned in the final sentence (recency heuristic)
                substitution_category = "terminal_sentence_recency_bias"
            elif norm_answer == target_prev:
                substitution_category = "previous_state_of_target"
            elif norm_answer == target_init:
                substitution_category = "initial_state_of_target"
            elif norm_answer in unvisited:
                substitution_category = "unvisited_distractor"
            elif any(norm_answer == _normalize_location(loc) for loc in _LOCATIONS):
                substitution_category = "other_interfering_object_location"
            else:
                substitution_category = "unresolved_hallucination"

        return {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "parsed_answer": raw_answer,
            "normalized_answer": norm_answer,
            "confidence": confidence,
            "ground_truth": item.ground_truth,
            "normalized_ground_truth": norm_ground_truth,
            "substitution_category": substitution_category,
            "lag_k": item.metadata.get("lag_k"),
            "raw_response": cleaned_resp,
        }
