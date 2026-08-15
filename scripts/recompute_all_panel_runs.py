"""Offline recomputation for all comparative model panel runs.

Recomputes paired contrasts, direct contrasts, joint PAI, and markdown reports
from frozen trials.jsonl while preserving all original execution provenance.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from recurrence.analysis.privileged_access import (
    compute_item_paired_contrasts,
    compute_direct_pairwise_contrast,
)
from recurrence.analysis.calibration import compute_post_decision_discrimination_from_pairs
from experiments.e02_observer.run import generate_markdown_report

PANEL_RUNS = [
    "run_e02_obs_qwen1_5b_001",
    "run_e02_obs_qwen7b_001",
    "run_e02_obs_qwen14b_001",
    "run_e02_obs_llama3_2_3b_001",
    "run_e02_obs_mistral7b_001",
]

COND_KEYS = [
    ("self_review_equal_compute", "self_review_prob"),
    ("observer_review_other", "observer_review_other_prob"),
    ("observer_visible_answer_only", "obs_vis_ans_prob"),
    ("observer_visible_full_transcript", "obs_vis_full_prob"),
    ("observer_reconstruction", "obs_recon_prob"),
    ("observer_input_only", "obs_input_prob"),
    ("observer_output_full_response_only", "obs_output_prob"),
]


def recompute_run(run_id: str):
    res_dir = Path("results/e02_observer") / run_id
    summary_file = res_dir / "summary.json"
    trials_file = res_dir / "trials.jsonl"
    report_file = res_dir / "report.md"

    if not summary_file.exists() or not trials_file.exists():
        print(f"Skipping {run_id}: files not found.")
        return

    with open(summary_file, "r", encoding="utf-8") as f:
        orig_summary = json.load(f)

    # Load trials
    trials = []
    with open(trials_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trials.append(json.loads(line))

    # Build maps
    self_item_map = {}
    observer_item_maps = {k: {} for k, _ in COND_KEYS}

    for t in trials:
        item_id = t.get("item_id")
        is_corr = bool(t.get("target_correct", False))
        
        self_p = t.get("self_probability")
        self_item_map[item_id] = (float(self_p) if self_p is not None else None, is_corr)

        for cond_name, col_name in COND_KEYS:
            obs_p = t.get(col_name)
            observer_item_maps[cond_name][item_id] = (float(obs_p) if obs_p is not None else None, is_corr)

    # Recompute strict paired contrasts and joint PAI
    contrasts_res = compute_item_paired_contrasts(
        self_item_map=self_item_map,
        observer_item_maps=observer_item_maps,
        sesoi=0.10,
        n_bootstraps=1000,
        seed=42,
    )

    # Recompute direct pairwise contrasts
    direct_contrasts = {}
    direct_contrasts["framing_self_vs_other_review"] = compute_direct_pairwise_contrast(
        map_a=observer_item_maps["self_review_equal_compute"],
        map_b=observer_item_maps["observer_review_other"],
        name_a="self_review_equal_compute",
        name_b="observer_review_other",
        sesoi=0.10,
        n_bootstraps=1000,
        seed=42,
    )
    direct_contrasts["channel_answer_only_vs_full_transcript"] = compute_direct_pairwise_contrast(
        map_a=observer_item_maps["observer_visible_answer_only"],
        map_b=observer_item_maps["observer_visible_full_transcript"],
        name_a="observer_visible_answer_only",
        name_b="observer_visible_full_transcript",
        sesoi=0.10,
        n_bootstraps=1000,
        seed=42,
    )

    # Build updated summary preserving original provenance
    updated_summary = dict(orig_summary)
    updated_summary["paired_intersection_contrasts"] = contrasts_res["contrasts"]
    updated_summary["joint_pai_summary"] = contrasts_res["joint_pai_summary"]
    updated_summary["direct_pairwise_contrasts"] = direct_contrasts

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(updated_summary, f, indent=2)

    # Generate updated markdown report
    report_md = generate_markdown_report(updated_summary)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Successfully recomputed {run_id}")


if __name__ == "__main__":
    for r in PANEL_RUNS:
        recompute_run(r)
