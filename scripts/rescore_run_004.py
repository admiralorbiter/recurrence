"""Offline Reanalysis of Run 004 with Decoupled Answer Validity and Suppression of Inferential Language."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from recurrence.tasks.kv_retrieval import KVRetrievalTask
from recurrence.tasks.base import TaskItem
from recurrence.observers.visible import _parse_probability_from_text
from recurrence.observers.reconstruction import _extract_target_letter
from recurrence.analysis.privileged_access import (
    compute_continuous_brier_score,
    compute_item_paired_contrasts,
    compute_direct_pairwise_contrast,
)


def rescore_run_004(events_path: str = "artifacts/e02_observer/run_e02_obs_004/run_e02_obs_004_events.jsonl") -> Dict[str, Any]:
    events_file = Path(events_path)
    if not events_file.exists():
        raise FileNotFoundError(f"Events file not found: {events_file}")

    events = []
    with open(events_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    print(f"Loaded {len(events)} events from {events_file}")
    task_fc = KVRetrievalTask(mode="forced_choice", ask_confidence=True, confidence_format="probability")

    self_item_map: Dict[str, Tuple[Optional[float], bool]] = {}
    observer_item_maps: Dict[str, Dict[str, Tuple[Optional[float], bool]]] = {
        "self_review_equal_compute": {},
        "observer_review_other": {},
        "observer_visible_answer_only": {},
        "observer_visible_full_transcript": {},
        "observer_reconstruction": {},
        "observer_input_only": {},
        "observer_output_full_response_only": {},
    }

    semantic_correct = 0
    opaque_correct = 0
    total_items = len(events)
    answer_valid_count = 0
    prob_valid_count = 0
    schema_valid_count = 0

    for ev in events:
        item_id = ev["metadata"]["item_id"]
        task_name = ev["metadata"]["task_name"]
        ground_truth = ev["metadata"]["ground_truth"]
        target_raw_response = ev["action"]
        metadata = ev["metadata"]

        dummy_item = TaskItem(
            item_id=item_id,
            prompt=ev["observation"],
            ground_truth=ground_truth,
            distractors=[],
            metadata=metadata,
        )

        score_res = task_fc.score_response(dummy_item, target_raw_response)
        is_corr = score_res["correct"]
        self_prob = score_res.get("probability")

        if score_res.get("answer_parse_valid"):
            answer_valid_count += 1
        if score_res.get("probability_parse_valid"):
            prob_valid_count += 1
        if score_res.get("schema_valid"):
            schema_valid_count += 1

        if is_corr:
            if "semantic" in task_name:
                semantic_correct += 1
            else:
                opaque_correct += 1

        self_item_map[item_id] = (self_prob, is_corr)

        # Observers
        self_rev_prob = _parse_probability_from_text(ev["metadata"]["eval_self_review"]["raw_response"])
        observer_item_maps["self_review_equal_compute"][item_id] = (self_rev_prob, is_corr)

        oth_rev_prob = _parse_probability_from_text(ev["metadata"]["eval_other_review"]["raw_response"])
        observer_item_maps["observer_review_other"][item_id] = (oth_rev_prob, is_corr)

        vis_ans_prob = _parse_probability_from_text(ev["metadata"]["eval_visible_answer_only"]["raw_response"])
        observer_item_maps["observer_visible_answer_only"][item_id] = (vis_ans_prob, is_corr)

        vis_full_prob = _parse_probability_from_text(ev["metadata"]["eval_visible_full_transcript"]["raw_response"])
        observer_item_maps["observer_visible_full_transcript"][item_id] = (vis_full_prob, is_corr)

        # Reconstruction: strict bounds and completeness
        recon_raw = ev["metadata"]["eval_reconstruction"]["raw_response"]
        recon_dist: Dict[str, float] = {}
        import re, math
        json_m = re.search(r"\{.*\}", recon_raw, re.DOTALL)
        if json_m:
            try:
                data = json.loads(json_m.group(0))
                if isinstance(data, dict):
                    for k, v in data.items():
                        m = re.search(r"\b([A-D])\b", str(k), re.IGNORECASE) or re.search(r"([A-D])", str(k), re.IGNORECASE)
                        if m:
                            opt = m.group(1).upper()
                            try:
                                if isinstance(v, (int, float)):
                                    recon_dist[opt] = float(v)
                                elif isinstance(v, str):
                                    num_m = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)", v.strip())
                                    if num_m:
                                        recon_dist[opt] = float(num_m.group(1))
                            except Exception:
                                pass
            except Exception:
                pass

        for opt in ["A", "B", "C", "D"]:
            if opt not in recon_dist:
                m = re.search(rf'["\']?{opt}["\']?\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)', recon_raw, re.IGNORECASE)
                if m:
                    try:
                        recon_dist[opt] = float(m.group(1))
                    except ValueError:
                        pass

        has_all_4 = all(
            opt in recon_dist and math.isfinite(recon_dist[opt]) and 0.0 <= recon_dist[opt] <= 100.0
            for opt in ["A", "B", "C", "D"]
        )
        total_mass = sum(recon_dist[opt] for opt in ["A", "B", "C", "D"]) if has_all_4 else 0.0
        recon_prob = None
        if has_all_4 and total_mass > 0.0:
            norm_dist = {opt: float(recon_dist[opt] / total_mass) for opt in ["A", "B", "C", "D"]}
            target_letter = _extract_target_letter(target_raw_response)
            if target_letter is not None and target_letter in norm_dist:
                recon_prob = norm_dist[target_letter]

        observer_item_maps["observer_reconstruction"][item_id] = (recon_prob, is_corr)

        inp_prob = _parse_probability_from_text(ev["metadata"]["eval_input_only"]["raw_response"])
        observer_item_maps["observer_input_only"][item_id] = (inp_prob, is_corr)

        out_prob = _parse_probability_from_text(ev["metadata"]["eval_output_full_response_only"]["raw_response"])
        observer_item_maps["observer_output_full_response_only"][item_id] = (out_prob, is_corr)

    overall_accuracy = (semantic_correct + opaque_correct) / total_items
    print(f"\n--- RUN 004 RESCORE ---")
    print(f"First-Order Accuracy: {overall_accuracy:.1%} ({semantic_correct + opaque_correct}/{total_items})")
    print(f"Answer Parse Valid: {answer_valid_count}/{total_items} ({answer_valid_count/total_items:.1%})")
    print(f"Probability Parse Valid: {prob_valid_count}/{total_items} ({prob_valid_count/total_items:.1%})")
    print(f"Schema Valid: {schema_valid_count}/{total_items} ({schema_valid_count/total_items:.1%})")

    return {
        "overall_accuracy": overall_accuracy,
        "answer_valid_count": answer_valid_count,
        "prob_valid_count": prob_valid_count,
        "schema_valid_count": schema_valid_count,
    }


if __name__ == "__main__":
    rescore_run_004()
