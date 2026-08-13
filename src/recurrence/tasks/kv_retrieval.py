"""Key-Value Retrieval task with paired item matrix, strict normalized scoring, and semantic randomization."""

import random
import re
import string
from typing import Any, Dict, List, Literal, Optional, Tuple
from recurrence.tasks.base import BaseTask, TaskItem


_ADJECTIVES = [
    "golden", "silver", "crimson", "azure", "emerald", "amber", "obsidian",
    "crystal", "velvet", "iron", "bronze", "shadow", "solar", "lunar", "frost"
]
_NOUNS = [
    "falcon", "tiger", "shield", "compass", "anchor", "lantern", "beacon",
    "castle", "temple", "canyon", "meadow", "river", "glacier", "forest", "tower"
]


def _normalize_string(text: str) -> str:
    """Strict normalization: lowercase, strip outer punctuation and whitespace."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", cleaned)
    return cleaned


class KVRetrievalTask(BaseTask):
    """Generates Key-Value items across a 2x2 matrix with strict exact scoring."""

    def __init__(
        self,
        identifier_type: Literal["opaque", "semantic"] = "opaque",
        mode: Literal["free_generation", "forced_choice"] = "free_generation",
        distractor_count: int = 5,
        ask_confidence: bool = True,
    ):
        conf_tag = "_conf" if ask_confidence else "_noconf"
        name = f"kv_{identifier_type}_{mode}{conf_tag}"
        desc = (
            f"Key-Value Retrieval ({identifier_type} identifiers, {mode} mode, "
            f"{distractor_count} distractors, ask_confidence={ask_confidence})."
        )
        super().__init__(name=name, description=desc)
        self.identifier_type = identifier_type
        self.mode = mode
        self.distractor_count = distractor_count
        self.ask_confidence = ask_confidence

    @staticmethod
    def generate_raw_pairs(
        count: int,
        distractor_count: int,
        identifier_type: str,
        seed: int
    ) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
        """Generate canonical underlying (target_key, target_value, all_pairs) instances.

        Uses ordered lists to guarantee cross-process deterministic reproducibility.
        """
        rng = random.Random(seed)
        instances = []

        for _ in range(count):
            keys: List[str] = []
            seen_keys = set()
            while len(keys) < distractor_count + 1:
                if identifier_type == "opaque":
                    cand = "key_" + "".join(rng.choices(string.ascii_lowercase + string.digits, k=6))
                else:
                    cand = "key_" + rng.choice(_ADJECTIVES) + "_" + rng.choice(_NOUNS)
                if cand not in seen_keys:
                    seen_keys.add(cand)
                    keys.append(cand)

            values: List[str] = []
            seen_values = set()
            while len(values) < distractor_count + 1:
                if identifier_type == "opaque":
                    cand = "val_" + "".join(rng.choices(string.ascii_lowercase + string.digits, k=6))
                else:
                    # Random independent pairing to avoid semantic leakage
                    cand = "val_" + rng.choice(_ADJECTIVES) + "_" + rng.choice(_NOUNS)
                if cand not in seen_values:
                    seen_values.add(cand)
                    values.append(cand)

            pairs = list(zip(keys, values))
            target_key, target_value = rng.choice(pairs)
            rng.shuffle(pairs)
            instances.append((target_key, target_value, pairs))

        return instances

    def generate_items_from_raw(
        self,
        raw_instances: List[Tuple[str, str, List[Tuple[str, str]]]],
        seed: int = 42
    ) -> List[TaskItem]:
        """Format raw paired instances into TaskItems with exact option-letter counterbalancing."""
        rng = random.Random(seed)
        items: List[TaskItem] = []
        option_labels = ["A", "B", "C", "D"]

        for i, (target_key, target_value, pairs) in enumerate(raw_instances):
            formatted_pairs = "\n".join([f"- {k}: {v}" for k, v in pairs])
            distractor_values = [v for k, v in pairs if v != target_value]

            if self.mode == "forced_choice":
                # Exact 4-way counterbalancing: cycles A -> B -> C -> D -> A ...
                target_label = option_labels[i % len(option_labels)]
                sampled_distractors = rng.sample(distractor_values, k=3)
                
                # Assign distractors to the remaining 3 labels
                option_map: Dict[str, str] = {}
                dist_idx = 0
                for lbl in option_labels:
                    if lbl == target_label:
                        option_map[lbl] = target_value
                    else:
                        option_map[lbl] = sampled_distractors[dist_idx]
                        dist_idx += 1

                options_text = "\n".join([f"({lbl}) {option_map[lbl]}" for lbl in option_labels])
                
                if self.ask_confidence:
                    prompt = (
                        f"Below is a list of key-value pairs:\n\n{formatted_pairs}\n\n"
                        f"Question: What value is associated with '{target_key}'?\n\n"
                        f"Options:\n{options_text}\n\n"
                        f"Format your response strictly as:\n"
                        f"Answer: <Option letter, e.g. A>\n"
                        f"Confidence: <1 to 5, where 1 is total guess and 5 is absolutely certain>"
                    )
                else:
                    prompt = (
                        f"Below is a list of key-value pairs:\n\n{formatted_pairs}\n\n"
                        f"Question: What value is associated with '{target_key}'?\n\n"
                        f"Options:\n{options_text}\n\n"
                        f"Format your response strictly as:\n"
                        f"Answer: <Option letter, e.g. A>"
                    )

                ground_truth = target_label
                metadata = {
                    "target_key": target_key,
                    "target_value": target_value,
                    "target_option_letter": target_label,
                    "option_map": option_map,
                    "mode": self.mode,
                    "identifier_type": self.identifier_type,
                    "ask_confidence": self.ask_confidence,
                }
            else:
                if self.ask_confidence:
                    prompt = (
                        f"Below is a list of key-value pairs:\n\n{formatted_pairs}\n\n"
                        f"Question: What is the exact value string associated with '{target_key}'?\n\n"
                        f"Format your response strictly as:\n"
                        f"Answer: <exact value string>\n"
                        f"Confidence: <1 to 5, where 1 is total guess and 5 is absolutely certain>"
                    )
                else:
                    prompt = (
                        f"Below is a list of key-value pairs:\n\n{formatted_pairs}\n\n"
                        f"Question: What is the exact value string associated with '{target_key}'?\n\n"
                        f"Format your response strictly as:\n"
                        f"Answer: <exact value string>"
                    )

                ground_truth = target_value
                metadata = {
                    "target_key": target_key,
                    "mode": self.mode,
                    "identifier_type": self.identifier_type,
                    "ask_confidence": self.ask_confidence,
                }

            item = TaskItem(
                item_id=f"kv_{self.identifier_type}_{self.mode}_{i:03d}",
                prompt=prompt,
                ground_truth=ground_truth,
                distractors=distractor_values,
                metadata=metadata,
            )
            items.append(item)

        return items

    def generate_items(self, count: int = 10, seed: int = 42) -> List[TaskItem]:
        """Generate standalone items for this condition."""
        raw = self.generate_raw_pairs(
            count=count,
            distractor_count=self.distractor_count,
            identifier_type=self.identifier_type,
            seed=seed,
        )
        return self.generate_items_from_raw(raw, seed=seed)

    def score_response(self, item: TaskItem, response: str) -> Dict[str, Any]:
        """Parse structured answer and apply strict exact normalized comparison."""
        cleaned_resp = response.strip()

        # Robust Answer extraction: stops before inline Confidence or newline
        ans_match = re.search(
            r"Answer:\s*(?:<[^>]+>:\s*)?<?([a-zA-Z0-9_\s]+?)>?(?:\s*Confidence|\s*Confident|;|\n|\r|$)",
            cleaned_resp,
            re.IGNORECASE,
        )
        raw_answer = ans_match.group(1).strip() if ans_match else cleaned_resp

        # Parse Confidence: ...
        conf_match = re.search(r"(?:Confidence|Confident):\s*<?([1-5])>?", cleaned_resp, re.IGNORECASE)
        confidence = int(conf_match.group(1)) if conf_match else None

        norm_answer = _normalize_string(raw_answer)
        norm_ground_truth = _normalize_string(item.ground_truth)

        # STRICT SCORING
        if self.mode == "forced_choice":
            # Exact option letter match OR exact option value string match
            target_val = _normalize_string(item.metadata.get("target_value", ""))
            correct = (
                norm_answer.upper() == item.ground_truth.upper() or
                norm_answer == target_val
            )
        else:
            # STRICT EXACT EQUALITY for free generation
            correct = (norm_answer == norm_ground_truth)

        # Classify failure type if incorrect
        failure_type = None
        if not correct:
            if not norm_answer or norm_answer in ["none", "null", "unknown"]:
                failure_type = "response_format_noncompliance"
            elif self.mode == "free_generation" and any(_normalize_string(d) == norm_answer for d in item.distractors):
                failure_type = "target_process_distractor_confusion"
            elif self.mode == "free_generation" and len(norm_answer) > 0:
                failure_type = "partial_string_corruption"
            else:
                failure_type = "unresolved_generation_or_retrieval"

        return {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "parsed_answer": raw_answer,
            "normalized_answer": norm_answer,
            "confidence": confidence,
            "ground_truth": item.ground_truth,
            "normalized_ground_truth": norm_ground_truth,
            "failure_type": failure_type,
            "raw_response": cleaned_resp,
        }
