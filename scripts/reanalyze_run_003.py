"""Offline Reanalysis of Run 003 Events with Hardened Scoring and Unified Probability Scale."""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from recurrence.tasks.kv_retrieval import KVRetrievalTask
from recurrence.tasks.base import TaskItem
from recurrence.observers.visible import _parse_probability_from_text
from recurrence.observers.reconstruction import _extract_target_letter
from recurrence.analysis.privileged_access import (
    compute_continuous_brier_score,
    compute_item_paired_contrasts,
    compute_direct_pairwise_contrast,
)


def reanalyze_run_003(events_path: str = "artifacts/e02_observer/run_e02_obs_003/run_e02_obs_003_events.jsonl") -> Dict[str, Any]:
    events_file = Path(events_path)
    if not events_file.exists():
        raise FileNotFoundError(f"Events file not found: {events_file}")

    events = []
    with open(events_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    print(f"Loaded {len(events)} raw events from {events_file}.")

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

    trial_records = []
    semantic_correct = 0
    opaque_correct = 0
    total_items = len(events)

    for ev in events:
        step = ev["step"]
        item_id = ev["metadata"]["item_id"]
        task_name = ev["metadata"]["task_name"]
        ground_truth = ev["metadata"]["ground_truth"]
        target_raw_response = ev["action"]
        metadata = ev["metadata"]

        # Create dummy TaskItem for scoring
        dummy_item = TaskItem(
            item_id=item_id,
            prompt=ev["observation"],
            ground_truth=ground_truth,
            distractors=[],
            metadata=metadata,
        )

        # 1. Rescore target model response
        target_score = task_fc.score_response(dummy_item, target_raw_response)
        is_corr = target_score["correct"]
        self_prob = target_score.get("probability")

        if is_corr:
            if "semantic" in task_name:
                semantic_correct += 1
            else:
                opaque_correct += 1

        self_item_map[item_id] = (self_prob, is_corr)

        # 2. Rescore observers
        # Self-review
        self_rev_raw = ev["metadata"]["eval_self_review"]["raw_response"]
        self_rev_prob = _parse_probability_from_text(self_rev_raw)
        observer_item_maps["self_review_equal_compute"][item_id] = (self_rev_prob, is_corr)

        # Other-review
        oth_rev_raw = ev["metadata"]["eval_other_review"]["raw_response"]
        oth_rev_prob = _parse_probability_from_text(oth_rev_raw)
        observer_item_maps["observer_review_other"][item_id] = (oth_rev_prob, is_corr)

        # Visible answer only
        vis_ans_raw = ev["metadata"]["eval_visible_answer_only"]["raw_response"]
        vis_ans_prob = _parse_probability_from_text(vis_ans_raw)
        observer_item_maps["observer_visible_answer_only"][item_id] = (vis_ans_prob, is_corr)

        # Visible full transcript
        vis_full_raw = ev["metadata"]["eval_visible_full_transcript"]["raw_response"]
        vis_full_prob = _parse_probability_from_text(vis_full_raw)
        observer_item_maps["observer_visible_full_transcript"][item_id] = (vis_full_prob, is_corr)

        # Reconstruction: Parse 4-way distribution with strict completeness (no zero-fill)
        recon_raw = ev["metadata"]["eval_reconstruction"]["raw_response"]
        recon_dist: Dict[str, float] = {}
        import re
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
                                if isinstance(v, (int, float, str)):
                                    num_m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(v))
                                    if num_m:
                                        recon_dist[opt] = float(num_m.group(1))
                            except Exception:
                                pass
            except Exception:
                pass

        for opt in ["A", "B", "C", "D"]:
            if opt not in recon_dist:
                m = re.search(rf'["\']?{opt}["\']?\s*:\s*([0-9]+(?:\.[0-9]+)?)', recon_raw, re.IGNORECASE)
                if m:
                    try:
                        recon_dist[opt] = float(m.group(1))
                    except ValueError:
                        pass

        has_all_4 = all(opt in recon_dist and recon_dist[opt] >= 0.0 for opt in ["A", "B", "C", "D"])
        total_mass = sum(recon_dist[opt] for opt in ["A", "B", "C", "D"]) if has_all_4 else 0.0
        recon_prob = None
        recon_ans = None

        if has_all_4 and total_mass > 0.0:
            norm_dist = {opt: float(recon_dist[opt] / total_mass) for opt in ["A", "B", "C", "D"]}
            recon_ans = max(norm_dist, key=lambda k: norm_dist[k])
            target_letter = _extract_target_letter(target_raw_response)
            if target_letter in norm_dist:
                recon_prob = norm_dist[target_letter]

        observer_item_maps["observer_reconstruction"][item_id] = (recon_prob, is_corr)

        # Input only
        inp_raw = ev["metadata"]["eval_input_only"]["raw_response"]
        inp_prob = _parse_probability_from_text(inp_raw)
        observer_item_maps["observer_input_only"][item_id] = (inp_prob, is_corr)

        # Output full response only
        out_raw = ev["metadata"]["eval_output_only"]["raw_response"]
        out_prob = _parse_probability_from_text(out_raw)
        observer_item_maps["observer_output_full_response_only"][item_id] = (out_prob, is_corr)

        trial_records.append({
            "run_id": ev["run_id"],
            "step": step,
            "task_name": task_name,
            "item_id": item_id,
            "ground_truth": ground_truth,
            "target_raw_response": target_raw_response,
            "target_parsed_answer": target_score.get("parsed_answer"),
            "target_correct": is_corr,
            "self_probability": self_prob,
            "self_review_prob": self_rev_prob,
            "observer_review_other_prob": oth_rev_prob,
            "obs_vis_ans_prob": vis_ans_prob,
            "obs_vis_full_prob": vis_full_prob,
            "obs_recon_prob": recon_prob,
            "obs_recon_answer": recon_ans,
            "obs_input_prob": inp_prob,
            "obs_output_prob": out_prob,
        })

    # Paired intersection contrasts
    contrast_analysis = compute_item_paired_contrasts(
        self_item_map=self_item_map,
        observer_item_maps=observer_item_maps,
        sesoi=0.10,
        n_bootstraps=1000,
        seed=42,
    )

    # Direct pairwise contrasts
    framing_contrast = compute_direct_pairwise_contrast(
        map_a=observer_item_maps["self_review_equal_compute"],
        map_b=observer_item_maps["observer_review_other"],
        name_a="self_review_equal_compute",
        name_b="observer_review_other",
        sesoi=0.10,
        n_bootstraps=1000,
        seed=42,
    )

    channel_contrast = compute_direct_pairwise_contrast(
        map_a=observer_item_maps["observer_visible_answer_only"],
        map_b=observer_item_maps["observer_visible_full_transcript"],
        name_a="observer_visible_answer_only",
        name_b="observer_visible_full_transcript",
        sesoi=0.10,
        n_bootstraps=1000,
        seed=42,
    )

    self_valid_count = sum(1 for p, y in self_item_map.values() if p is not None)
    observer_valid_counts = {
        name: sum(1 for p, y in mp.values() if p is not None)
        for name, mp in observer_item_maps.items()
    }

    primary_conditions = [
        "self_immediate",
        "self_review_equal_compute",
        "observer_review_other",
        "observer_visible_answer_only",
        "observer_visible_full_transcript",
        "observer_reconstruction",
    ]
    primary_rates = {
        "self_immediate": self_valid_count / total_items if total_items else 0.0,
    }
    for c in primary_conditions:
        if c != "self_immediate":
            primary_rates[c] = observer_valid_counts[c] / total_items if total_items else 0.0

    min_primary_compliance = min(primary_rates.values())
    compliance_gate_passed = bool(min_primary_compliance >= 0.90)

    brier_scores = {
        "self_immediate": compute_continuous_brier_score(list(self_item_map.values())),
    }
    for name, mp in observer_item_maps.items():
        brier_scores[name] = compute_continuous_brier_score(list(mp.values()))

    res = {
        "reanalysis": "S03.3_Offline_Rescoring",
        "total_items": total_items,
        "first_order_accuracy": {
            "semantic_fc_accuracy": semantic_correct / 20.0,
            "opaque_fc_accuracy": opaque_correct / 20.0,
            "overall_accuracy": (semantic_correct + opaque_correct) / total_items,
            "correct_count": semantic_correct + opaque_correct,
        },
        "compliance_rates": {
            "self_valid_count": self_valid_count,
            "primary_compliance_rates": primary_rates,
            "min_primary_compliance": min_primary_compliance,
            "observer_valid_counts": observer_valid_counts,
            "compliance_gate_passed": compliance_gate_passed,
        },
        "observer_brier_scores": brier_scores,
        "paired_intersection_contrasts": contrast_analysis["contrasts"],
        "direct_pairwise_contrasts": {
            "framing_self_vs_other_review": framing_contrast,
            "channel_answer_only_vs_full_transcript": channel_contrast,
        },
        "joint_pai_summary": contrast_analysis["joint_pai_summary"],
        "trial_records": trial_records,
    }

    print("\n--- REANALYSIS SUMMARY ---")
    print(f"Overall Accuracy: {res['first_order_accuracy']['overall_accuracy']:.1%} ({res['first_order_accuracy']['correct_count']}/40)")
    print(f"Primary Compliance Rates: {json.dumps(primary_rates, indent=2)}")
    print(f"Min Primary Compliance: {min_primary_compliance:.1%}")
    print(f"Compliance Gate Passed (>=90% min): {compliance_gate_passed}")
    print(f"Joint PAI: {res['joint_pai_summary'].get('point_pai')}, CI: [{res['joint_pai_summary'].get('ci_95_lower')}, {res['joint_pai_summary'].get('ci_95_upper')}]")
    return res


if __name__ == "__main__":
    reanalyze_run_003()
