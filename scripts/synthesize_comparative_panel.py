import json
from pathlib import Path
from recurrence.analysis.privileged_access import compute_sdt_metacognition

MODELS = [
    {"name": "qwen2.5:1.5b", "family": "Qwen 2.5", "scale": "1.5B", "run_id": "run_e02_obs_qwen1_5b_001"},
    {"name": "qwen2.5:3b", "family": "Qwen 2.5", "scale": "3B", "run_id": "run_e02_obs_005"},
    {"name": "qwen2.5:7b", "family": "Qwen 2.5", "scale": "7B", "run_id": "run_e02_obs_qwen7b_001"},
    {"name": "qwen2.5:14b", "family": "Qwen 2.5", "scale": "14B", "run_id": "run_e02_obs_qwen14b_001"},
    {"name": "llama3.2:3b", "family": "Llama 3.2", "scale": "3B", "run_id": "run_e02_obs_llama3_2_3b_001"},
    {"name": "mistral:latest", "family": "Mistral", "scale": "7B", "run_id": "run_e02_obs_mistral7b_001"},
]

def synthesize():
    results_base = Path("results/e02_observer")
    panel_data = []

    for m in MODELS:
        summary_file = results_base / m["run_id"] / "summary.json"
        if not summary_file.exists():
            print(f"Missing summary for {m['run_id']}")
            continue
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        trials_file = results_base / m["run_id"] / "trials.jsonl"
        trials = []
        if trials_file.exists():
            with open(trials_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        trials.append(json.loads(line))

        self_pairs = []
        for t in trials:
            p = t.get("self_probability")
            corr = t.get("target_correct")
            if p is not None and corr is not None:
                self_pairs.append((float(p), bool(corr)))

        sdt_stats = {}
        if len(self_pairs) >= 10:
            sdt_stats = compute_sdt_metacognition(self_pairs)

        perf = summary.get("target_task_performance", {})
        contrasts = summary.get("paired_intersection_contrasts", {})
        vis_ans = contrasts.get("observer_visible_answer_only", {})
        vis_full = contrasts.get("observer_visible_full_transcript", {})
        recon = contrasts.get("observer_reconstruction", {})
        joint = summary.get("joint_pai_summary", {})
        framing = summary.get("direct_pairwise_contrasts", {}).get("framing_self_vs_other_review", {})
        comp = summary.get("compliance_rates", {})

        entry = {
            "model": m["name"],
            "family": m["family"],
            "scale": m["scale"],
            "run_id": m["run_id"],
            "accuracy_overall": perf.get("overall_accuracy"),
            "accuracy_semantic": perf.get("semantic_fc_accuracy"),
            "accuracy_opaque": perf.get("opaque_fc_accuracy"),
            "self_auroc2": vis_ans.get("self_auroc2"),
            "vis_ans_auroc2": vis_ans.get("observer_auroc2"),
            "vis_full_auroc2": vis_full.get("observer_auroc2"),
            "recon_auroc2": recon.get("observer_auroc2"),
            "joint_pai": joint.get("point_pai"),
            "joint_pai_ci": [joint.get("ci_95_lower"), joint.get("ci_95_upper")],
            "framing_delta": framing.get("delta_auroc2"),
            "framing_ci": [framing.get("ci_95_lower"), framing.get("ci_95_upper")],
            "self_brier": vis_ans.get("self_brier_score"),
            "sdt_d_prime": sdt_stats.get("d_prime"),
            "sdt_meta_d": sdt_stats.get("meta_d"),
            "sdt_m_ratio": sdt_stats.get("m_ratio"),
            "compliance_min": comp.get("min_primary_compliance"),
            "gate_passed": comp.get("compliance_gate_passed"),
        }
        panel_data.append(entry)

    # Format Markdown Table
    print("\n### Multi-Model Comparative Table\n")
    headers = ["Model", "Family", "Scale", "1st-Order Acc", "Self AUROC2", "Vis-Ans AUROC2", "Vis-Full AUROC2", "Joint PAI (95% CI)", "Framing Delta", "Compliance Gate"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |")
    for row in panel_data:
        acc_str = f"{row['accuracy_overall']*100:.1f}%" if row['accuracy_overall'] is not None else "N/A"
        self_str = f"{row['self_auroc2']:.3f}" if row['self_auroc2'] is not None else "N/A"
        vis_str = f"{row['vis_ans_auroc2']:.3f}" if row['vis_ans_auroc2'] is not None else "N/A"
        full_str = f"{row['vis_full_auroc2']:.3f}" if row['vis_full_auroc2'] is not None else "N/A"
        pai_str = f"{row['joint_pai']:+.3f} [{row['joint_pai_ci'][0]:.3f}, {row['joint_pai_ci'][1]:.3f}]" if row['joint_pai'] is not None else "N/A"
        framing_str = f"{row['framing_delta']:+.3f}" if row['framing_delta'] is not None else "N/A"
        gate_str = "PASSED" if row['gate_passed'] else f"FAILED ({row['compliance_min']*100:.1f}%)"
        print(f"| **{row['model']}** | {row['family']} | {row['scale']} | {acc_str} | {self_str} | {vis_str} | {full_str} | {pai_str} | {framing_str} | {gate_str} |")

    return panel_data

if __name__ == "__main__":
    synthesize()
