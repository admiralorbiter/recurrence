"""Synthesize and generate the multi-model comparative panel markdown and JSON tables directly from canonical run summaries."""

import json
from pathlib import Path

MODELS = [
    {"name": "qwen2.5:1.5b", "family": "Qwen 2.5", "scale": "1.5B", "run_id": "run_e02_obs_qwen1_5b_001", "role": "Scale Diagnostic"},
    {"name": "qwen2.5:3b", "family": "Qwen 2.5", "scale": "3B", "run_id": "run_e02_obs_005", "role": "Canonical Baseline (H0 Ref)"},
    {"name": "qwen2.5:7b", "family": "Qwen 2.5", "scale": "7B", "run_id": "run_e02_obs_qwen7b_001", "role": "Scale Diagnostic"},
    {"name": "qwen2.5:14b", "family": "Qwen 2.5", "scale": "14B", "run_id": "run_e02_obs_qwen14b_001", "role": "Scale Ceiling"},
    {"name": "llama3.2:3b", "family": "Llama 3.2", "scale": "3B", "run_id": "run_e02_obs_llama3_2_3b_001", "role": "Family Ceiling"},
    {"name": "mistral:latest", "family": "Mistral", "scale": "7B", "run_id": "run_e02_obs_mistral7b_001", "role": "Family Ceiling"},
]


def generate_panel_data():
    results_base = Path("results/e02_observer")
    panel_data = []

    for m in MODELS:
        summary_file = results_base / m["run_id"] / "summary.json"
        if not summary_file.exists():
            print(f"Missing summary for {m['run_id']}")
            continue
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        perf = summary.get("target_task_performance", {})
        contrasts = summary.get("paired_intersection_contrasts", {})
        vis_ans = contrasts.get("observer_visible_answer_only", {})
        vis_full = contrasts.get("observer_visible_full_transcript", {})
        recon = contrasts.get("observer_reconstruction", {})
        other_rev = contrasts.get("observer_review_other", {})
        self_rev = contrasts.get("self_review_equal_compute", {})
        joint = summary.get("joint_pai_summary", {})
        framing = summary.get("direct_pairwise_contrasts", {}).get("framing_self_vs_other_review", {})
        comp = summary.get("compliance_rates", {})

        entry = {
            "model": m["name"],
            "family": m["family"],
            "scale": m["scale"],
            "role": m["role"],
            "run_id": m["run_id"],
            "accuracy_overall": perf.get("overall_accuracy"),
            "accuracy_semantic": perf.get("semantic_fc_accuracy"),
            "accuracy_opaque": perf.get("opaque_fc_accuracy"),
            "self_auroc2": vis_ans.get("self_auroc2"),
            "vis_ans_auroc2": vis_ans.get("observer_auroc2"),
            "vis_full_auroc2": vis_full.get("observer_auroc2"),
            "other_rev_auroc2": other_rev.get("observer_auroc2"),
            "self_rev_auroc2": self_rev.get("observer_auroc2"),
            "recon_auroc2": recon.get("observer_auroc2"),
            "joint_pai": joint.get("point_pai"),
            "joint_pai_ci": [joint.get("ci_95_lower"), joint.get("ci_95_upper")],
            "framing_delta": framing.get("delta_auroc2"),
            "framing_ci": [framing.get("ci_95_lower"), framing.get("ci_95_upper")],
            "self_brier": vis_ans.get("self_brier_score"),
            "self_rev_brier": self_rev.get("observer_brier_score"),
            "other_rev_brier": other_rev.get("observer_brier_score"),
            "compliance_min": comp.get("min_primary_compliance"),
            "gate_passed": comp.get("compliance_gate_passed"),
        }
        panel_data.append(entry)

    return panel_data


def format_markdown_table(panel_data):
    lines = []
    lines.append("| Model Checkpoint | Model Family | Scale | 1st-Order Accuracy | Self AUROC2 | Vis-Ans AUROC2 | Vis-Full AUROC2 | Other Review AUROC2 | Joint PAI (95% CI) | Framing Delta (Self - Other) | Gate & Status |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for row in panel_data:
        acc_str = f"{row['accuracy_overall']*100:.1f}%" if row['accuracy_overall'] is not None else "N/A"
        self_str = f"{row['self_auroc2']:.3f}" if row['self_auroc2'] is not None else "N/A (no errors)"
        vis_str = f"{row['vis_ans_auroc2']:.3f}" if row['vis_ans_auroc2'] is not None else "N/A (no errors)"
        full_str = f"{row['vis_full_auroc2']:.3f}" if row['vis_full_auroc2'] is not None else "N/A (no errors)"
        other_str = f"{row['other_rev_auroc2']:.3f}" if row['other_rev_auroc2'] is not None else "N/A (no errors)"

        if row['joint_pai'] is not None and row['joint_pai_ci'][0] is not None:
            pai_str = f"{row['joint_pai']:+.3f} [{row['joint_pai_ci'][0]:+.3f}, {row['joint_pai_ci'][1]:+.3f}]"
        else:
            pai_str = "N/A (no errors)" if row['accuracy_overall'] == 1.0 else "N/A"

        if row['framing_delta'] is not None and row['framing_ci'][0] is not None:
            framing_str = f"{row['framing_delta']:+.3f} [{row['framing_ci'][0]:+.3f}, {row['framing_ci'][1]:+.3f}]"
        else:
            framing_str = "N/A (no errors)" if row['accuracy_overall'] == 1.0 else "N/A"

        if row['accuracy_overall'] == 1.0:
            gate_str = "**PASSED** (Ceiling: Type-2 N/A)"
        elif row['gate_passed']:
            gate_str = "**PASSED** (Confirmatory Ref)"
        else:
            gate_str = f"Diagnostic Only ({row['compliance_min']*100:.1f}% min comp)"

        lines.append(f"| **`{row['model']}`** | {row['family']} | {row['scale']} | **{acc_str}** | {self_str} | {vis_str} | {full_str} | {other_str} | {pai_str} | {framing_str} | {gate_str} |")

    return "\n".join(lines)


if __name__ == "__main__":
    data = generate_panel_data()
    table = format_markdown_table(data)
    print(table)
