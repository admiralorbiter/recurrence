import sys
sys.path.insert(0, ".")

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from recurrence.analysis.privileged_access import (
    compute_continuous_brier_score,
    compute_item_paired_contrasts,
    compute_direct_pairwise_contrast,
)
from experiments.e02_observer.run import generate_markdown_report


def recompute_run_005():
    run_dir = Path("results/e02_observer/run_e02_obs_005")
    trials_path = run_dir / "trials.jsonl"

    trials = []
    with open(trials_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trials.append(json.loads(line))

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
    total_items = len(trials)
    semantic_count = 0
    opaque_count = 0

    for tr in trials:
        item_id = tr["item_id"]
        task_name = tr["task_name"]
        is_corr = tr["target_correct"]
        self_prob = tr["self_probability"]

        if "semantic" in task_name:
            semantic_count += 1
            if is_corr:
                semantic_correct += 1
        else:
            opaque_count += 1
            if is_corr:
                opaque_correct += 1

        self_item_map[item_id] = (self_prob, is_corr)
        observer_item_maps["self_review_equal_compute"][item_id] = (tr["self_review_prob"], is_corr)
        observer_item_maps["observer_review_other"][item_id] = (tr["observer_review_other_prob"], is_corr)
        observer_item_maps["observer_visible_answer_only"][item_id] = (tr["obs_vis_ans_prob"], is_corr)
        observer_item_maps["observer_visible_full_transcript"][item_id] = (tr["obs_vis_full_prob"], is_corr)
        observer_item_maps["observer_reconstruction"][item_id] = (tr["obs_recon_prob"], is_corr)
        observer_item_maps["observer_input_only"][item_id] = (tr["obs_input_prob"], is_corr)
        observer_item_maps["observer_output_full_response_only"][item_id] = (tr["obs_output_prob"], is_corr)

    # 1. Item-paired intersection contrasts
    paired_contrasts_res = compute_item_paired_contrasts(
        self_item_map=self_item_map,
        observer_item_maps=observer_item_maps,
        sesoi=0.10,
        n_bootstraps=1000,
        seed=42,
    )
    contrasts = paired_contrasts_res["contrasts"]
    joint_pai_summary = paired_contrasts_res["joint_pai_summary"]

    # 2. Direct Pre-specified pairwise contrasts
    direct_contrasts = {
        "framing_self_vs_other_review": compute_direct_pairwise_contrast(
            map_a=observer_item_maps["self_review_equal_compute"],
            map_b=observer_item_maps["observer_review_other"],
            name_a="self_review_equal_compute",
            name_b="observer_review_other",
            sesoi=0.10,
            n_bootstraps=1000,
            seed=42,
        ),
        "channel_answer_only_vs_full_transcript": compute_direct_pairwise_contrast(
            map_a=observer_item_maps["observer_visible_answer_only"],
            map_b=observer_item_maps["observer_visible_full_transcript"],
            name_a="observer_visible_answer_only",
            name_b="observer_visible_full_transcript",
            sesoi=0.10,
            n_bootstraps=1000,
            seed=42,
        ),
    }

    # Brier scores
    brier_scores = {
        "self_immediate": compute_continuous_brier_score(list(self_item_map.values()))
    }
    for name, mp in observer_item_maps.items():
        brier_scores[name] = compute_continuous_brier_score(list(mp.values()))

    # Compliance
    self_valid = sum(1 for p, _ in self_item_map.values() if p is not None)
    obs_valid_counts = {name: sum(1 for p, _ in mp.values() if p is not None) for name, mp in observer_item_maps.items()}
    primary_rates = {
        "self_immediate": self_valid / total_items,
        "self_review_equal_compute": obs_valid_counts["self_review_equal_compute"] / total_items,
        "observer_review_other": obs_valid_counts["observer_review_other"] / total_items,
        "observer_visible_answer_only": obs_valid_counts["observer_visible_answer_only"] / total_items,
        "observer_visible_full_transcript": obs_valid_counts["observer_visible_full_transcript"] / total_items,
        "observer_reconstruction": obs_valid_counts["observer_reconstruction"] / total_items,
    }
    min_primary = min(primary_rates.values())
    gate_passed = min_primary >= 0.90

    results_summary = {
        "experiment_id": "E02_Observer_Hardened",
        "sprint": "S03.4",
        "run_id": "run_e02_obs_005",
        "model_name": "qwen2.5:3b",
        "model_digest": "304499d63f972de98436573c09b8de218c5e62c129e92ae446e5071190c1db35",
        "seed": 42,
        "total_items": total_items,
        "compliance_rates": {
            "self_valid_count": self_valid,
            "observer_valid_counts": obs_valid_counts,
            "primary_compliance_rates": primary_rates,
            "min_primary_compliance": min_primary,
            "compliance_gate_passed": gate_passed,
        },
        "target_task_performance": {
            "semantic_fc_accuracy": semantic_correct / semantic_count if semantic_count else 0.0,
            "opaque_fc_accuracy": opaque_correct / opaque_count if opaque_count else 0.0,
            "overall_accuracy": (semantic_correct + opaque_correct) / total_items,
        },
        "brier_scores": brier_scores,
        "paired_intersection_contrasts": contrasts,
        "direct_pairwise_contrasts": direct_contrasts,
        "joint_pai_summary": joint_pai_summary,
        "manifest_path": "artifacts/e02_observer/run_e02_obs_005/run_e02_obs_005_manifest.json",
        "jsonl_path": "artifacts/e02_observer/run_e02_obs_005/run_e02_obs_005_events.jsonl",
        "parquet_path": "artifacts/e02_observer/run_e02_obs_005/run_e02_obs_005_events.parquet",
        "checksum": "f9aa1553ef8d69011e0ecbed36e39e2e15417f0782e9398e2e4282418b64eb0f",
        "environment_hash": "ee73e35d02efcd5248ec0ccc21b4ecff5e532ba92bbfb893fa6580f7b734e11b",
    }

    # Write summary.json
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    # Write report.md
    report_md = generate_markdown_report(results_summary)
    with open(run_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("Recomputed summary and report successfully:")
    print(f"Sprint: {results_summary['sprint']}")
    print(f"Joint PAI: {joint_pai_summary['point_pai']:+.3f}")
    print(f"Joint 95% CI: [{joint_pai_summary['ci_95_lower']:.3f}, {joint_pai_summary['ci_95_upper']:.3f}]")


if __name__ == "__main__":
    recompute_run_005()
