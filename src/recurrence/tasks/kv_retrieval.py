"""Key-Value Retrieval task with paired item matrix, strict normalized scoring, and semantic randomization."""

import json
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


def _extract_answer_from_value(ans_val: Any, mode: str = "forced_choice") -> str:
    """Extract clean answer string from string, number, or nested dict structure."""
    if isinstance(ans_val, dict):
        # 1. Look for known answer keys inside the nested dict
        inner_clean = {re.sub(r"[^a-zA-Z0-9_]", "", str(k)).lower(): v for k, v in ans_val.items()}
        for k in ["option", "letter", "choice", "ans", "answer", "value", "val", "selected"]:
            if k in inner_clean:
                v = inner_clean[k]
                if isinstance(v, (str, int, float)):
                    return str(v).strip()
                if isinstance(v, dict):
                    return _extract_answer_from_value(v, mode=mode)
        # 2. Look for single option letter in values
        for v in ans_val.values():
            if isinstance(v, str):
                m = re.search(r"\b([A-D])\b", v, re.IGNORECASE) if mode == "forced_choice" else None
                if m:
                    return m.group(1).upper()
                if mode != "forced_choice" and len(v.strip()) > 0:
                    return v.strip()
        # 3. Fallback to first non-probability value
        for k, v in ans_val.items():
            if "prob" not in str(k).lower():
                if isinstance(v, dict):
                    return _extract_answer_from_value(v, mode=mode)
                return str(v).strip()
        return str(list(ans_val.values())[0]) if ans_val else ""
    return str(ans_val).strip()


def _extract_probability_from_dict(data: dict) -> Optional[float]:
    """Extract probability from dict or nested dict on strict 0-100 percentage scale."""
    clean_dict = {re.sub(r"[^a-zA-Z0-9_]", "", str(k)).lower(): v for k, v in data.items()}
    
    # Check top-level probability keys
    raw_val = None
    for k in ["probability", "prob", "probabilityprobability", "probabilitycorrect", "p"]:
        if k in clean_dict:
            raw_val = clean_dict[k]
            break
            
    # If not found at top level, check inside nested dicts (e.g. data["answer"])
    if raw_val is None:
        for v in data.values():
            if isinstance(v, dict):
                inner = {re.sub(r"[^a-zA-Z0-9_]", "", str(ik)).lower(): iv for ik, iv in v.items()}
                for k in ["probability", "prob", "probabilityprobability", "probabilitycorrect", "p"]:
                    if k in inner:
                        raw_val = inner[k]
                        break
                if raw_val is not None:
                    break

    if raw_val is not None:
        if isinstance(raw_val, dict) and "probability" in raw_val:
            raw_val = raw_val["probability"]
        if isinstance(raw_val, (int, float, str)):
            try:
                if isinstance(raw_val, str):
                    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw_val)
                    if m:
                        val = float(m.group(1))
                    else:
                        return None
                else:
                    val = float(raw_val)
                # STRICT 0-100 scale contract: Always divide by 100.0
                return max(0.0, min(1.0, float(val / 100.0)))
            except (ValueError, TypeError):
                pass
    return None


class KVRetrievalTask(BaseTask):
    """Key-Value Retrieval Task with item-level distractor tracking, strict exact scoring,
    and paired item generation support.
    """

    def __init__(
        self,
        identifier_type: Literal["semantic", "opaque"] = "opaque",
        mode: Literal["forced_choice", "free_generation"] = "forced_choice",
        distractor_count: int = 5,
        ask_confidence: bool = True,
        confidence_format: Literal["likert", "probability"] = "probability",
    ):
        conf_tag = "_conf" if ask_confidence else "_noconf"
        name = f"kv_{identifier_type}_{mode}{conf_tag}"
        desc = (
            f"Key-Value Retrieval ({identifier_type} identifiers, {mode} mode, "
            f"{distractor_count} distractors, ask_confidence={ask_confidence}, format={confidence_format})."
        )
        super().__init__(name=name, description=desc)
        self.identifier_type = identifier_type
        self.mode = mode
        self.distractor_count = distractor_count
        self.ask_confidence = ask_confidence
        self.confidence_format = confidence_format

    @staticmethod
    def generate_raw_pairs(
        count: int = 10,
        distractor_count: int = 5,
        identifier_type: str = "opaque",
        seed: int = 42,
    ) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
        """Generate structured key-value raw sets with fixed seed determinism."""
        rng = random.Random(seed)
        instances = []

        for _ in range(count):
            keys: List[str] = []
            values: List[str] = []
            seen_keys = set()
            seen_values = set()

            total_needed = distractor_count + 1
            while len(keys) < total_needed:
                if identifier_type == "opaque":
                    cand = "key_" + "".join(rng.choices(string.ascii_lowercase + string.digits, k=5))
                else:
                    cand = "key_" + rng.choice(_ADJECTIVES) + "_" + rng.choice(_NOUNS)
                if cand not in seen_keys:
                    seen_keys.add(cand)
                    keys.append(cand)

            while len(values) < total_needed:
                if identifier_type == "opaque":
                    cand = "val_" + "".join(rng.choices(string.ascii_lowercase + string.digits, k=5))
                else:
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
                    if self.confidence_format == "probability":
                        prompt = (
                            f"Below is a list of key-value pairs:\n\n{formatted_pairs}\n\n"
                            f"Question: What value is associated with '{target_key}'?\n\n"
                            f"Options:\n{options_text}\n\n"
                            f"Respond strictly with a JSON object in this format with no other text:\n"
                            f'{{\n  "answer": "<Option letter, e.g. A, B, C, or D>",\n  "probability": <0 to 100, where 0 is definitely incorrect and 100 is definitely correct>\n}}'
                        )
                    else:
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
                    "confidence_format": self.confidence_format,
                }
            else:
                if self.ask_confidence:
                    if self.confidence_format == "probability":
                        prompt = (
                            f"Below is a list of key-value pairs:\n\n{formatted_pairs}\n\n"
                            f"Question: What is the exact value string associated with '{target_key}'?\n\n"
                            f"Respond strictly with a JSON object in this format with no other text:\n"
                            f'{{\n  "answer": "<exact value string>",\n  "probability": <0 to 100, where 0 is definitely incorrect and 100 is definitely correct>\n}}'
                        )
                    else:
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
                    "confidence_format": self.confidence_format,
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
        raw_answer: str = cleaned_resp
        probability: Optional[float] = None
        confidence: Optional[int] = None

        # 1. Try structured JSON extraction first
        json_match = re.search(r"\{.*\}", cleaned_resp, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict):
                    clean_dict = {re.sub(r"[^a-zA-Z0-9_]", "", str(k)).lower(): v for k, v in data.items()}
                    
                    # Look for answer key
                    for k in ["answer", "ans", "choice", "option", "in", "selected", "answeranswer"]:
                        if k in clean_dict:
                            raw_answer = _extract_answer_from_value(clean_dict[k], mode=self.mode)
                            break
                    
                    # Look for probability
                    probability = _extract_probability_from_dict(data)
            except Exception:
                pass

        # 2. Robust Regex Fallbacks for Answer if still unresolved
        if raw_answer == cleaned_resp:
            ans_match = re.search(
                r"Answer:\s*(?:<[^>]+>:\s*)?<?([a-zA-Z0-9_\s]+?)>?(?:\s*(?:Probability|Confidence|Confident)|\%|;|\n|\r|$)",
                cleaned_resp,
                re.IGNORECASE,
            )
            if ans_match:
                raw_answer = ans_match.group(1).strip()

        # 3. Probability parsing from text (if not already parsed from JSON)
        if probability is None:
            prob_match = re.search(r"(?:Probability\s*(?:correct)?|Prob|p):\s*<?([0-9]+(?:\.[0-9]+)?)\s*\%?>?", cleaned_resp, re.IGNORECASE)
            if prob_match:
                try:
                    val = float(prob_match.group(1))
                    # Strict 0-100 scale contract
                    probability = max(0.0, min(1.0, float(val / 100.0)))
                except ValueError:
                    probability = None

        # 4. Confidence Likert parsing ONLY if task confidence_format is 'likert'
        if self.confidence_format != "probability":
            conf_match = re.search(r"(?:Confidence|Confident):\s*<?([1-5])>?", cleaned_resp, re.IGNORECASE)
            if conf_match:
                try:
                    confidence = int(conf_match.group(1))
                    if probability is None:
                        probability = float(confidence / 5.0)
                except ValueError:
                    confidence = None

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
            "probability": probability,
            "confidence": confidence,
            "ground_truth": item.ground_truth,
            "normalized_ground_truth": norm_ground_truth,
            "failure_type": failure_type,
            "raw_response": cleaned_resp,
        }
