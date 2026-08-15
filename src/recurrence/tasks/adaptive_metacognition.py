"""Horizon 0 v2 Standalone 2AFC Adaptive Metacognition Task Generator & Scoring Engine."""

import json
import math
import random
import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from recurrence.tasks.base import BaseTask, TaskItem


_ADJECTIVES = [
    "golden", "silver", "crimson", "azure", "emerald", "amber", "obsidian",
    "crystal", "velvet", "iron", "bronze", "shadow", "solar", "lunar", "frost",
    "cobalt", "scarlet", "amethyst", "topaz", "granite", "quartz", "opal", "ruby"
]
_NOUNS = [
    "falcon", "tiger", "shield", "compass", "anchor", "lantern", "beacon",
    "castle", "temple", "canyon", "meadow", "river", "glacier", "forest", "tower",
    "cipher", "vortex", "citadel", "haven", "monolith", "spire", "prism", "tempest"
]


@dataclass(frozen=True)
class DifficultyConfig:
    """Configuration defining difficulty parameters for 2AFC procedural tasks."""
    task_family: Literal["distractor_load", "multi_hop", "overwrite_load"]
    distractor_count: int = 16
    hop_depth: int = 1
    overwrite_count: int = 0
    target_position: Literal["middle", "early", "late"] = "middle"
    foil_mode: Literal["matched_category", "stale_overwrite", "orthogonal"] = "matched_category"
    confidence_format: Literal["probability", "likert_1_4", "none"] = "probability"
    ask_confidence: bool = True


def _normalize_string(text: str) -> str:
    """Strict normalization: lowercase, strip outer punctuation and whitespace."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", cleaned)
    return cleaned


def _extract_answer_from_json_dict(data: dict) -> Optional[str]:
    """Extract 2AFC option letter (A or B) from JSON dictionary."""
    clean_dict = {re.sub(r"[^a-zA-Z0-9_]", "", str(k)).lower(): v for k, v in data.items()}
    for k in ["answer", "ans", "choice", "option", "selected", "letter", "response"]:
        if k in clean_dict:
            v = clean_dict[k]
            if isinstance(v, str):
                m = re.search(r"\b([AB])\b", v, re.IGNORECASE)
                if m:
                    return m.group(1).upper()
            elif isinstance(v, (int, float)):
                return str(v).strip()
    return None


def _extract_probability_from_json_dict(data: dict) -> Optional[float]:
    """Extract probability from dictionary on strict 0-100 percentage scale."""
    clean_dict = {re.sub(r"[^a-zA-Z0-9_]", "", str(k)).lower(): v for k, v in data.items()}
    raw_val = None
    for k in ["probability", "prob", "probabilitycorrect", "p", "cert"]:
        if k in clean_dict:
            raw_val = clean_dict[k]
            break
    if raw_val is not None:
        try:
            if isinstance(raw_val, str):
                cleaned_str = raw_val.strip()
                if cleaned_str.lower() in ["nan", "inf", "-inf", "infinity", "-infinity"]:
                    return None
                m = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)", cleaned_str)
                if m:
                    val = float(m.group(1))
                else:
                    return None
            else:
                val = float(raw_val)
            if not math.isfinite(val) or val < 0.0 or val > 100.0:
                return None
            return float(val / 100.0)
        except (ValueError, TypeError):
            pass
    return None


def _extract_likert_from_json_dict(data: dict) -> Optional[int]:
    """Extract Likert confidence (1..4) from dictionary."""
    clean_dict = {re.sub(r"[^a-zA-Z0-9_]", "", str(k)).lower(): v for k, v in data.items()}
    for k in ["confidence", "conf", "rating"]:
        if k in clean_dict:
            try:
                val = int(clean_dict[k])
                if 1 <= val <= 4:
                    return val
            except (ValueError, TypeError):
                pass
    return None


class AdaptiveMetacognition2AFCTask(BaseTask):
    """Standalone 2-Alternative Forced Choice (2AFC) task generator and scoring engine
    for Horizon 0 v2 psychophysical calibration and metacognitive evaluation.
    """

    def __init__(
        self,
        task_family: Literal["distractor_load", "multi_hop", "overwrite_load"] = "distractor_load",
        ask_confidence: bool = True,
        confidence_format: Literal["probability", "likert_1_4", "none"] = "probability",
        response_mode: Literal["direct_value", "symbolic_letter"] = "direct_value",
    ):
        name = f"h0v2_2afc_{task_family}_{'conf' if ask_confidence else 'noconf'}_{response_mode}"
        desc = (
            f"H0-v2 Matched 2AFC Task ({task_family}, ask_confidence={ask_confidence}, "
            f"confidence_format={confidence_format}, response_mode={response_mode})"
        )
        super().__init__(name=name, description=desc)
        self.task_family = task_family
        self.ask_confidence = ask_confidence
        self.confidence_format = confidence_format
        self.response_mode = response_mode

    @staticmethod
    def _generate_semantic_identifiers(rng: random.Random, count: int) -> Tuple[List[str], List[str]]:
        """Generate unique semantic key and value identifier pairs."""
        keys: List[str] = []
        values: List[str] = []
        seen_keys = set()
        seen_values = set()

        while len(keys) < count:
            cand = f"key_{rng.choice(_ADJECTIVES)}_{rng.choice(_NOUNS)}"
            if cand not in seen_keys:
                seen_keys.add(cand)
                keys.append(cand)

        while len(values) < count:
            cand = f"val_{rng.choice(_ADJECTIVES)}_{rng.choice(_NOUNS)}"
            if cand not in seen_values:
                seen_values.add(cand)
                values.append(cand)

        return keys, values

    def generate_distractor_item(
        self,
        distractor_count: int,
        seed: int,
        ask_confidence: bool = True,
        target_option_letter: str = "A",
        target_position: Literal["middle", "early", "late"] = "middle",
        pre_generated_distractors: Optional[List[Tuple[str, str]]] = None,
        pre_generated_target: Optional[Tuple[str, str]] = None,
        pre_generated_foil: Optional[Tuple[str, str]] = None,
    ) -> TaskItem:
        """Generate a 1-hop 2AFC needle-in-a-haystack item with controlled distractor volume.
        
        ANTI-SHORTCUT DESIGN:
        Both target_val and foil_val appear explicitly in the context.
        foil_val is the ground-truth value belonging to an actual distractor key present in the evidence.
        """
        rng = random.Random(seed)

        if pre_generated_distractors is not None and pre_generated_target is not None and pre_generated_foil is not None:
            target_key, target_val = pre_generated_target
            foil_key, foil_val = pre_generated_foil
            # Truncate distractors to distractor_count without reshuffling (preserves exact prefix order)
            distractor_pairs = list(pre_generated_distractors[:distractor_count])
            # Ensure foil_key pair is present in distractor_pairs
            if (foil_key, foil_val) not in distractor_pairs:
                if distractor_pairs:
                    distractor_pairs[0] = (foil_key, foil_val)
                else:
                    distractor_pairs.append((foil_key, foil_val))
            shuffled_distractors = list(distractor_pairs)
            # Deterministic normalized middle insertion: D // 2
            insert_idx = len(shuffled_distractors) // 2
        else:
            total_pairs = max(distractor_count + 5, 10)
            keys, values = self._generate_semantic_identifiers(rng, total_pairs + 5)
            target_key = keys[0]
            target_val = values[0]
            # Distractor pairs: from index 1..distractor_count
            distractor_pairs = list(zip(keys[1:1 + distractor_count], values[1:1 + distractor_count]))
            # Foil is the value belonging to the first distractor key
            foil_key, foil_val = distractor_pairs[0]
            # Shuffle distractor pairs
            shuffled_distractors = list(distractor_pairs)
            rng.shuffle(shuffled_distractors)

            # Place target needle according to position stratum
            if target_position == "early":
                insert_idx = rng.randint(0, max(0, len(shuffled_distractors) // 4))
            elif target_position == "late":
                insert_idx = rng.randint(max(0, 3 * len(shuffled_distractors) // 4), len(shuffled_distractors))
            else:  # middle 40-60%
                mid_start = int(0.4 * len(shuffled_distractors))
                mid_end = max(mid_start, int(0.6 * len(shuffled_distractors)))
                insert_idx = rng.randint(mid_start, mid_end) if mid_end >= mid_start else 0

        context_pairs = list(shuffled_distractors)
        context_pairs.insert(insert_idx, (target_key, target_val))

        formatted_context = "\n".join([f"- {k} = {v}" for k, v in context_pairs])

        # Form 2AFC options
        foil_letter = "B" if target_option_letter == "A" else "A"
        option_map = {
            target_option_letter: target_val,
            foil_letter: foil_val,
        }

        prompt = self._build_2afc_prompt(
            context_text=formatted_context,
            query_text=f"What value is associated with '{target_key}'?",
            option_a=option_map["A"],
            option_b=option_map["B"],
            ask_confidence=ask_confidence,
        )

        metadata = {
            "task_family": "distractor_load",
            "distractor_count": distractor_count,
            "hop_depth": 1,
            "overwrite_count": 0,
            "target_key": target_key,
            "target_val": target_val,
            "foil_key": foil_key,
            "foil_val": foil_val,
            "target_option_letter": target_option_letter,
            "option_map": option_map,
            "target_position": target_position,
            "needle_index": insert_idx,
            "total_context_items": len(context_pairs),
            "ask_confidence": ask_confidence,
            "confidence_format": self.confidence_format,
            "seed": seed,
        }

        return TaskItem(
            item_id=f"h0v2_dist_{distractor_count:04d}_s{seed}_{target_option_letter}",
            prompt=prompt,
            ground_truth=target_option_letter,
            distractors=[foil_val],
            metadata=metadata,
        )

    def generate_multi_hop_item(
        self,
        hop_depth: int,
        distractor_count: int = 16,
        seed: int = 42,
        ask_confidence: bool = True,
        target_option_letter: str = "A",
    ) -> TaskItem:
        """Generate a multi-hop pointer-chasing item with MATCHED DUAL CHAINS.
        
        ANTI-SHORTCUT DESIGN:
        The context contains TWO parallel relational pointer chains of identical length H:
        - Target Chain: K_0 -> K_1 -> ... -> K_{H-1} -> V_target
        - Foil Chain:   K'_0 -> K'_1 -> ... -> K'_{H-1} -> V_foil
        Query asks only for the terminal value starting from K_0.
        Both V_target and V_foil appear in the evidence as terminal values of relational chains.
        Candidate presence alone achieves exactly chance (50%).
        """
        rng = random.Random(seed)
        total_keys_needed = (hop_depth * 2) + (distractor_count * 4) + 50
        total_vals_needed = (distractor_count * 2) + 50
        keys, values = self._generate_semantic_identifiers(rng, total_keys_needed)

        # Target chain keys & terminal value
        target_chain_keys = keys[:hop_depth]
        target_terminal_val = values[0]

        # Foil chain keys & terminal value (matched depth H)
        foil_chain_keys = keys[hop_depth:2 * hop_depth]
        foil_terminal_val = values[1]

        # Build target chain statements
        target_statements: List[str] = []
        for h in range(hop_depth - 1):
            target_statements.append(f"{target_chain_keys[h]} points to {target_chain_keys[h+1]}")
        target_statements.append(f"{target_chain_keys[-1]} maps to {target_terminal_val}")

        # Build foil chain statements (matched depth H)
        foil_statements: List[str] = []
        for h in range(hop_depth - 1):
            foil_statements.append(f"{foil_chain_keys[h]} points to {foil_chain_keys[h+1]}")
        foil_statements.append(f"{foil_chain_keys[-1]} maps to {foil_terminal_val}")

        # Build distractor statements
        dist_key_idx = 2 * hop_depth
        dist_val_idx = 2
        distractor_statements: List[str] = []
        for _ in range(distractor_count):
            if rng.random() < 0.5 and dist_key_idx + 1 < len(keys):
                distractor_statements.append(f"{keys[dist_key_idx]} points to {keys[dist_key_idx+1]}")
                dist_key_idx += 2
            else:
                distractor_statements.append(f"{keys[dist_key_idx]} maps to {values[dist_val_idx]}")
                dist_key_idx += 1
                dist_val_idx += 1

        all_statements = distractor_statements + target_statements + foil_statements
        rng.shuffle(all_statements)
        formatted_context = "\n".join([f"- {s}" for s in all_statements])

        foil_letter = "B" if target_option_letter == "A" else "A"
        option_map = {
            target_option_letter: target_terminal_val,
            foil_letter: foil_terminal_val,
        }

        prompt = self._build_2afc_prompt(
            context_text=formatted_context,
            query_text=f"Tracing all pointers starting from '{target_chain_keys[0]}', what is the terminal value reached?",
            option_a=option_map["A"],
            option_b=option_map["B"],
            ask_confidence=ask_confidence,
        )

        metadata = {
            "task_family": "multi_hop",
            "distractor_count": distractor_count,
            "hop_depth": hop_depth,
            "overwrite_count": 0,
            "start_key": target_chain_keys[0],
            "target_chain_keys": target_chain_keys,
            "target_terminal_val": target_terminal_val,
            "foil_chain_keys": foil_chain_keys,
            "foil_terminal_val": foil_terminal_val,
            "target_option_letter": target_option_letter,
            "option_map": option_map,
            "ask_confidence": ask_confidence,
            "confidence_format": self.confidence_format,
            "seed": seed,
        }

        return TaskItem(
            item_id=f"h0v2_hop_{hop_depth:02d}_s{seed}_{target_option_letter}",
            prompt=prompt,
            ground_truth=target_option_letter,
            distractors=[foil_terminal_val],
            metadata=metadata,
        )

    def generate_overwrite_item(
        self,
        overwrite_count: int,
        distractor_count: int = 16,
        seed: int = 42,
        ask_confidence: bool = True,
        target_option_letter: str = "A",
    ) -> TaskItem:
        """Generate a sequential timeline item with U target overwrites.
        
        ANTI-SHORTCUT DESIGN:
        - For U >= 1: Foil is the immediately preceding stale value V_{U-1}. Both target and foil appear in timeline.
        - For U == 0: Foil is the value of another distractor key explicitly inserted into the timeline.
        Both candidate values always appear in the context.
        """
        rng = random.Random(seed)
        total_keys_needed = distractor_count + 10
        total_vals_needed = overwrite_count + distractor_count + 15
        keys, values = self._generate_semantic_identifiers(rng, total_keys_needed)

        target_key = keys[0]
        # Target updates: V_0, V_1, ..., V_U
        target_val_sequence = values[:overwrite_count + 1]
        current_target_val = target_val_sequence[-1]
        
        # Determine foil value (must appear in timeline)
        if overwrite_count > 0:
            stale_foil_val = target_val_sequence[-2]  # Immediately previous stale value
            foil_key = target_key
        else:
            # U=0 baseline: Foil is a real distractor key-value pair that appears in timeline
            foil_key = keys[1]
            stale_foil_val = values[overwrite_count + 2]

        timeline_events: List[Tuple[int, str]] = []
        step = 1

        # Interleave target overwrites across distractor events
        dist_key_idx = 2 if overwrite_count == 0 else 1
        dist_val_idx = overwrite_count + 3 if overwrite_count == 0 else overwrite_count + 2

        # Distribute target updates across timeline
        target_step_positions = sorted(
            rng.sample(range(1, distractor_count + overwrite_count + 2), k=overwrite_count + 1)
        )

        # For U=0, reserve one step specifically for the foil key-value pair
        foil_step_position = None
        if overwrite_count == 0:
            remaining_steps = [s for s in range(1, distractor_count + 2) if s not in target_step_positions]
            foil_step_position = rng.choice(remaining_steps)

        target_update_idx = 0
        total_steps = distractor_count + overwrite_count + 1
        for s in range(1, total_steps + 1):
            if target_update_idx < len(target_step_positions) and s == target_step_positions[target_update_idx]:
                val = target_val_sequence[target_update_idx]
                timeline_events.append((s, f"Step {s:02d}: Update {target_key} = {val}"))
                target_update_idx += 1
            elif foil_step_position is not None and s == foil_step_position:
                timeline_events.append((s, f"Step {s:02d}: Set {foil_key} = {stale_foil_val}"))
            else:
                d_key = keys[dist_key_idx % (len(keys) - 2) + 2]
                d_val = values[dist_val_idx % len(values)]
                timeline_events.append((s, f"Step {s:02d}: Set {d_key} = {d_val}"))
                dist_key_idx += 1
                dist_val_idx += 1

        formatted_context = "\n".join([event[1] for event in timeline_events])

        foil_letter = "B" if target_option_letter == "A" else "A"
        option_map = {
            target_option_letter: current_target_val,
            foil_letter: stale_foil_val,
        }

        prompt = self._build_2afc_prompt(
            context_text=formatted_context,
            query_text=f"According to the chronological sequence of events, what is the CURRENT (latest) value of '{target_key}'?",
            option_a=option_map["A"],
            option_b=option_map["B"],
            ask_confidence=ask_confidence,
        )

        metadata = {
            "task_family": "overwrite_load",
            "distractor_count": distractor_count,
            "hop_depth": 1,
            "overwrite_count": overwrite_count,
            "target_key": target_key,
            "current_target_val": current_target_val,
            "stale_foil_val": stale_foil_val,
            "target_val_sequence": target_val_sequence,
            "target_option_letter": target_option_letter,
            "option_map": option_map,
            "ask_confidence": ask_confidence,
            "confidence_format": self.confidence_format,
            "seed": seed,
        }

        return TaskItem(
            item_id=f"h0v2_over_{overwrite_count:02d}_s{seed}_{target_option_letter}",
            prompt=prompt,
            ground_truth=target_option_letter,
            distractors=[stale_foil_val],
            metadata=metadata,
        )

    def _build_2afc_prompt(
        self,
        context_text: str,
        query_text: str,
        option_a: str,
        option_b: str,
        ask_confidence: bool,
    ) -> str:
        """Construct strict 2AFC prompt with JSON schema format constraints."""
        if getattr(self, "response_mode", "direct_value") == "direct_value":
            options_block = f"Candidate Values:\n- \"{option_a}\"\n- \"{option_b}\""
            if not ask_confidence:
                schema_block = (
                    'Respond strictly with a JSON object in this format with no other text:\n'
                    '{\n'
                    f'  "answer": "<exact candidate value string: \\"{option_a}\\" or \\"{option_b}\\">"\n'
                    '}'
                )
            elif self.confidence_format == "likert_1_4":
                schema_block = (
                    'Respond strictly with a JSON object in this format with no other text:\n'
                    '{\n'
                    f'  "answer": "<exact candidate value string: \\"{option_a}\\" or \\"{option_b}\\">",\n'
                    '  "confidence": <1 to 4, where 1 is definitely guessing and 4 is definitely certain>\n'
                    '}'
                )
            else:  # probability 0-100
                schema_block = (
                    'Respond strictly with a JSON object in this format with no other text:\n'
                    '{\n'
                    f'  "answer": "<exact candidate value string: \\"{option_a}\\" or \\"{option_b}\\">",\n'
                    '  "probability": <0 to 100, where 0 is definitely incorrect, 50 is even odds / guess, and 100 is definitely correct>\n'
                    '}'
                )
        else:  # symbolic_letter (A / B)
            options_block = f"(A) {option_a}\n(B) {option_b}"
            if not ask_confidence:
                schema_block = (
                    'Respond strictly with a JSON object in this format with no other text:\n'
                    '{\n'
                    '  "answer": "<A or B>"\n'
                    '}'
                )
            elif self.confidence_format == "likert_1_4":
                schema_block = (
                    'Respond strictly with a JSON object in this format with no other text:\n'
                    '{\n'
                    '  "answer": "<A or B>",\n'
                    '  "confidence": <1 to 4, where 1 is definitely guessing and 4 is definitely certain>\n'
                    '}'
                )
            else:  # probability 0-100
                schema_block = (
                    'Respond strictly with a JSON object in this format with no other text:\n'
                    '{\n'
                    '  "answer": "<A or B>",\n'
                    '  "probability": <0 to 100, where 0 is definitely incorrect, 50 is even odds / guess, and 100 is definitely correct>\n'
                    '}'
                )

        return (
            f"Context Information:\n{context_text}\n\n"
            f"Question: {query_text}\n\n"
            f"Options:\n{options_block}\n\n"
            f"{schema_block}"
        )

    def generate_nested_distractor_sweep(
        self,
        levels: List[int] = [4, 8, 16, 32, 64, 128, 256],
        count_per_level: int = 16,
        base_seed: int = 42,
        ask_confidence: Optional[bool] = None,
    ) -> List[TaskItem]:
        """Generate a distractor sweep where difficulty levels for item i are NESTED subsets of a maximal context.
        
        For each base item i in [0..count_per_level-1]:
          - A maximal distractor pool (D_max = max(levels)) and fixed (target_key, target_val, foil_val) are created.
          - Each level D_k slices the identical prefix of D_k distractors.
          - Thus across D_k, problem identity is held constant while only context load expands.
        """
        conf = self.ask_confidence if ask_confidence is None else ask_confidence
        items: List[TaskItem] = []
        option_cycle = ["A", "B"]
        max_d = max(levels)

        # Pre-generate base episodes for each item index i
        base_episodes = []
        for i in range(count_per_level):
            item_seed = base_seed + i
            rng = random.Random(item_seed)
            keys, values = self._generate_semantic_identifiers(rng, max_d + 20)
            target = (keys[0], values[0])
            foil = (keys[1], values[1])
            # Foil is at index 0; additional distractors are fixed in order
            distractor_pool = [(keys[1], values[1])] + list(zip(keys[2:2 + max_d], values[2:2 + max_d]))
            base_episodes.append((item_seed, target, foil, distractor_pool))

        for lvl_idx, d_count in enumerate(levels):
            for i in range(count_per_level):
                target_letter = option_cycle[i % 2]
                item_seed, target, foil, distractor_pool = base_episodes[i]
                item = self.generate_distractor_item(
                    distractor_count=d_count,
                    seed=item_seed,
                    ask_confidence=conf,
                    target_option_letter=target_letter,
                    pre_generated_distractors=distractor_pool,
                    pre_generated_target=target,
                    pre_generated_foil=foil,
                )
                items.append(item)
        return items

    def generate_distractor_sweep(
        self,
        levels: List[int] = [4, 8, 16, 32, 64, 128, 256],
        count_per_level: int = 16,
        base_seed: int = 42,
        ask_confidence: Optional[bool] = None,
        nested: bool = True,
    ) -> List[TaskItem]:
        """Generate a complete distractor sweep with exact 50/50 A/B counterbalancing per level."""
        if nested:
            return self.generate_nested_distractor_sweep(
                levels=levels,
                count_per_level=count_per_level,
                base_seed=base_seed,
                ask_confidence=ask_confidence,
            )
        conf = self.ask_confidence if ask_confidence is None else ask_confidence
        items: List[TaskItem] = []
        option_cycle = ["A", "B"]

        for lvl_idx, d_count in enumerate(levels):
            for i in range(count_per_level):
                target_letter = option_cycle[i % 2]
                item_seed = base_seed + (lvl_idx * 1000) + i
                item = self.generate_distractor_item(
                    distractor_count=d_count,
                    seed=item_seed,
                    ask_confidence=conf,
                    target_option_letter=target_letter,
                )
                items.append(item)
        return items

    def generate_multi_hop_sweep(
        self,
        levels: List[int] = [1, 2, 3, 4, 5],
        count_per_level: int = 16,
        distractor_count: int = 16,
        base_seed: int = 42,
        ask_confidence: Optional[bool] = None,
    ) -> List[TaskItem]:
        """Generate a complete multi-hop pointer depth sweep with exact 50/50 A/B counterbalancing."""
        conf = self.ask_confidence if ask_confidence is None else ask_confidence
        items: List[TaskItem] = []
        option_cycle = ["A", "B"]

        for lvl_idx, hop in enumerate(levels):
            for i in range(count_per_level):
                target_letter = option_cycle[i % 2]
                item_seed = base_seed + (lvl_idx * 1000) + i
                item = self.generate_multi_hop_item(
                    hop_depth=hop,
                    distractor_count=distractor_count,
                    seed=item_seed,
                    ask_confidence=conf,
                    target_option_letter=target_letter,
                )
                items.append(item)
        return items

    def generate_overwrite_sweep(
        self,
        levels: List[int] = [0, 1, 2, 3, 4],
        count_per_level: int = 16,
        distractor_count: int = 16,
        base_seed: int = 42,
        ask_confidence: Optional[bool] = None,
    ) -> List[TaskItem]:
        """Generate a complete overwrite load sweep with exact 50/50 A/B counterbalancing."""
        conf = self.ask_confidence if ask_confidence is None else ask_confidence
        items: List[TaskItem] = []
        option_cycle = ["A", "B"]

        for lvl_idx, overwrites in enumerate(levels):
            for i in range(count_per_level):
                target_letter = option_cycle[i % 2]
                item_seed = base_seed + (lvl_idx * 1000) + i
                item = self.generate_overwrite_item(
                    overwrite_count=overwrites,
                    distractor_count=distractor_count,
                    seed=item_seed,
                    ask_confidence=conf,
                    target_option_letter=target_letter,
                )
                items.append(item)
        return items

    def generate_items(self, count: int = 10, seed: int = 42) -> List[TaskItem]:
        """Default generator producing items for the configured task family."""
        if self.task_family == "distractor_load":
            return self.generate_distractor_sweep(levels=[16], count_per_level=count, base_seed=seed)
        elif self.task_family == "multi_hop":
            return self.generate_multi_hop_sweep(levels=[2], count_per_level=count, base_seed=seed)
        elif self.task_family == "overwrite_load":
            return self.generate_overwrite_sweep(levels=[2], count_per_level=count, base_seed=seed)
        return []

    def score_response(self, item: TaskItem, response: str) -> Dict[str, Any]:
        """Strictly parse 2AFC answer and confidence rating, evaluating correctness and schema validity."""
        cleaned_resp = response.strip()
        raw_answer: Optional[str] = None
        probability: Optional[float] = None
        confidence: Optional[int] = None
        schema_valid: bool = False
        answer_parse_valid: bool = False

        opt_map = item.metadata.get("option_map", {})
        val_a = opt_map.get("A")
        val_b = opt_map.get("B")

        # 1. JSON parsing
        json_match = re.search(r"\{.*\}", cleaned_resp, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict):
                    raw_val = data.get("answer")
                    if raw_val is not None:
                        raw_str = str(raw_val).strip()
                        if raw_str.upper() in ["A", "B"]:
                            raw_answer = raw_str.upper()
                            answer_parse_valid = True
                        elif val_a and raw_str == str(val_a):
                            raw_answer = "A"
                            answer_parse_valid = True
                        elif val_b and raw_str == str(val_b):
                            raw_answer = "B"
                            answer_parse_valid = True
                        elif val_a and raw_str.lower() == str(val_a).lower():
                            raw_answer = "A"
                            answer_parse_valid = True
                        elif val_b and raw_str.lower() == str(val_b).lower():
                            raw_answer = "B"
                            answer_parse_valid = True
                        else:
                            ans_extracted = _extract_answer_from_json_dict(data)
                            if ans_extracted in ["A", "B"]:
                                raw_answer = ans_extracted
                                answer_parse_valid = True

                    if self.ask_confidence:
                        if self.confidence_format == "likert_1_4":
                            confidence = _extract_likert_from_json_dict(data)
                            if confidence is not None:
                                probability = float((confidence - 1) / 3.0)  # Map 1..4 -> 0.0..1.0
                                if answer_parse_valid and len(data) == 2:
                                    schema_valid = True
                        else:  # probability
                            probability = _extract_probability_from_json_dict(data)
                            if probability is not None:
                                if answer_parse_valid and len(data) == 2:
                                    schema_valid = True
                    else:  # answer-only
                        if answer_parse_valid and len(data) == 1:
                            schema_valid = True
            except Exception:
                pass

        # 2. JSON-like unclosed regex fallback for answer
        if not answer_parse_valid:
            if val_a and str(val_a) in cleaned_resp and not (val_b and str(val_b) in cleaned_resp):
                raw_answer = "A"
                answer_parse_valid = True
            elif val_b and str(val_b) in cleaned_resp and not (val_a and str(val_a) in cleaned_resp):
                raw_answer = "B"
                answer_parse_valid = True
            else:
                ans_match = re.search(r'["\']answer["\']\s*:\s*["\']?([AB])["\']?', cleaned_resp, re.IGNORECASE)
                if ans_match:
                    raw_answer = ans_match.group(1).upper()
                    answer_parse_valid = True

        # 3. Text fallback for answer
        if not answer_parse_valid:
            nl_match = re.search(r"(?:Answer|Option|Choice):\s*<?\b([AB])\b>?", cleaned_resp, re.IGNORECASE)
            if nl_match:
                raw_answer = nl_match.group(1).upper()
                answer_parse_valid = True

        # 4. Text fallback for probability
        if self.ask_confidence and probability is None:
            prob_match = re.search(r"(?:Probability|Prob|p|Confidence|Conf):\s*<?(-?[0-9]+(?:\.[0-9]+)?)\s*\%?>?", cleaned_resp, re.IGNORECASE)
            if prob_match:
                try:
                    val = float(prob_match.group(1))
                    if 0.0 <= val <= 100.0:
                        probability = float(val / 100.0)
                    elif 1 <= val <= 4 and self.confidence_format == "likert_1_4":
                        confidence = int(val)
                        probability = float((confidence - 1) / 3.0)
                except ValueError:
                    pass

        norm_answer = _normalize_string(raw_answer) if raw_answer else ""
        norm_ground_truth = _normalize_string(item.ground_truth)

        # STRICT SCORING: Must be valid option letter and match ground truth
        correct = bool(answer_parse_valid and (norm_answer.upper() == norm_ground_truth.upper()))
        probability_parse_valid = (probability is not None)

        failure_type = None
        if not correct:
            if not answer_parse_valid:
                failure_type = "response_format_noncompliance"
            else:
                failure_type = "foil_selection_error"

        return {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "parsed_answer": raw_answer,
            "normalized_answer": norm_answer,
            "answer_parse_valid": answer_parse_valid,
            "probability": probability,
            "probability_parse_valid": probability_parse_valid,
            "confidence": confidence,
            "schema_valid": schema_valid,
            "ground_truth": item.ground_truth,
            "normalized_ground_truth": norm_ground_truth,
            "failure_type": failure_type,
            "raw_response": cleaned_resp,
            "option_map": item.metadata.get("option_map", {}),
        }
