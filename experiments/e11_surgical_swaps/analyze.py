"""Experiment E11 Analysis Script: Multi-Store Surgical State Swaps & Mediational Propagation (Sprint S12).

Aggregates causal channel attribution indices, absolute directional displacements,
mediational forward propagation, and cloze margin trajectories.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List


def analyze_run(run_dir: str) -> Dict[str, Any]:
    run_path = Path(run_dir)
    trace_file = run_path / "swap_trace.jsonl"
    med_file = run_path / "mediational_propagation.jsonl"
    summary_file = run_path / "summary.json"

    if not trace_file.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_file}")

    run_meta = {}
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            run_meta = json.load(f)

    rows = []
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # Group by (lag, condition)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["lag"], r["condition"])].append(r)

    lags = sorted(list({r["lag"] for r in rows}))
    conditions = sorted(list({r["condition"] for r in rows}))

    causal_table: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for lag in lags:
        for cond in conditions:
            items = grouped.get((lag, cond), [])
            if not items:
                continue
            mean_margin = sum(x["cloze_margin"] for x in items) / len(items)
            mean_signed_delta = sum(x.get("signed_graft_effect", x.get("raw_graft_effect", 0.0)) for x in items) / len(items)
            mean_abs_disp = sum(x.get("absolute_displacement", 0.0) for x in items) / len(items)
            mean_donor_norm = sum(x.get("donor_recipient_norm", 0.0) for x in items) / len(items)
            mean_logit_proj = sum(x.get("logit_directional_projection", 0.0) for x in items) / len(items)

            eligible_items = [x for x in items if x.get("is_eligible_for_attribution") and x.get("causal_attribution_index") is not None]
            if eligible_items:
                mean_alpha = sum(x["causal_attribution_index"] for x in eligible_items) / len(eligible_items)
            else:
                mean_alpha = None

            donor_target = items[0]["target_donor"]
            if donor_target in ("A", "B"):
                flip_rate = sum(1.0 for x in items if x["target_choice"] == donor_target) / len(items)
            else:
                flip_rate = 0.0

            causal_table[lag][cond] = {
                "mean_cloze_margin": round(mean_margin, 4),
                "mean_signed_graft_effect": round(mean_signed_delta, 4),
                "mean_absolute_displacement": round(mean_abs_disp, 4),
                "mean_donor_recipient_norm": round(mean_donor_norm, 4),
                "mean_logit_projection": round(mean_logit_proj, 4),
                "mean_attribution_index": round(mean_alpha, 4) if mean_alpha is not None else None,
                "donor_concordance_rate": round(flip_rate, 4),
                "n_samples": len(items),
                "n_eligible": len(eligible_items),
            }

    # Mediational propagation analysis
    med_results = []
    if med_file.exists():
        with open(med_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    med_results.append(json.loads(line))

    analysis_result = {
        "run_dir": str(run_path),
        "model_provenance": run_meta.get("model_provenance", {}),
        "lags": lags,
        "conditions": conditions,
        "causal_table": dict(causal_table),
        "mediational_propagation": med_results,
    }

    with open(run_path / "analysis_summary.json", "w", encoding="utf-8") as f_out:
        json.dump(analysis_result, f_out, indent=2)

    # Generate Markdown Report
    report_file = run_path / "report.md"
    model_name = run_meta.get("model_provenance", {}).get("model_id", "recurrentgemma")

    with open(report_file, "w", encoding="utf-8") as f_rep:
        f_rep.write(f"# E11 Multi-Store Surgical State Swaps Causal Attribution Report\n\n")
        f_rep.write(f"**Model Target:** `{model_name}`\n")
        f_rep.write(f"**Run Path:** `{run_path}`\n\n")
        f_rep.write("## 1. Causal Transfer & Directional Logit Displacement Across Lags\n\n")
        f_rep.write("| Lag $L$ | Condition | Signed Graft $\\bar{\\Delta}_C$ | Abs Displacement $P_C$ | Logit Proj $\\alpha_C^{\\text{logit}}$ | Attrib Index $\\alpha_C^{\\text{cloze}}$ | Eligible N | Donor Concord |\n")
        f_rep.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for lag in lags:
            for cond in sorted(causal_table[lag].keys()):
                d = causal_table[lag][cond]
                alpha_str = f"{d['mean_attribution_index']:.3f}" if d['mean_attribution_index'] is not None else "N/A"
                elig_str = f"{d['n_eligible']}/{d['n_samples']}"
                f_rep.write(f"| {lag} | `{cond}` | {d['mean_signed_graft_effect']:+.2f} | {d['mean_absolute_displacement']:+.2f} | {d['mean_logit_projection']:+.3f} | {alpha_str} | {elig_str} | {d['donor_concordance_rate']*100:.1f}% |\n")

        if med_results:
            f_rep.write("\n## 2. Mediational Forward Dynamic Propagation ($R^B \\to K_{\\text{future}}^B$)\n\n")
            f_rep.write("| Pair ID | Regime | Init Lag | Future Tokens | Turnover? | Post Dist Rec A | Post Dist Don B | Post Migr $\\mathcal{M}_{\\text{post}}$ | Full Migr $\\mathcal{M}_{\\text{full}}$ |\n")
            f_rep.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for m in med_results:
                turn_str = "Yes" if m.get("is_full_window_turnover") else "No"
                post_migr = m.get("post_migration_index", m.get("kv_migration_index", 0.0))
                full_migr = m.get("full_migration_index", 0.0)
                d_post_a = m.get("d_post_med_to_a", m.get("d_med_to_a", 0.0))
                d_post_b = m.get("d_post_med_to_b", m.get("d_med_to_b", 0.0))
                f_rep.write(f"| `{m['pair_id']}` | `{m['regime']}` | {m['initial_lag']} | {m['future_tokens']} | {turn_str} | {d_post_a:.4f} | {d_post_b:.4f} | {post_migr:+.4f} | {full_migr:+.4f} |\n")

        f_rep.write("\n## 3. Causal Interpretation & Control Framework\n\n")
        f_rep.write("1. **Absolute Displacement ($P_C$) as Primary Causal Metric:** Directional displacement $P_C = (z_G - z_R) \\cdot \\frac{z_D - z_R}{\\|z_D - z_R\\|}$ distinguishes true causal steering magnitude from normalized share $\\alpha_C^{\\text{logit}}$ when the total donor-recipient contrast $\\|z_D - z_R\\|$ collapses at deep lags.\n")
        f_rep.write("2. **Historical Specificity vs Matched Perturbations:** Matching donor RG-LRU is compared against unrelated-donor, permuted-donor, and Frobenius-matched Gaussian noise controls projected along the real donor direction ($P_{\\text{match}} > P_{\\text{control}}$).\n")
        f_rep.write("3. **Dynamic Post-Graft KV Mediation:** Measures distances strictly over newly generated post-graft cache entries to determine whether continuous recurrent state propagates historical steering into downstream attention representations.\n")

    print(f"[E11] Analysis complete! Wrote report to {report_file}")
    return analysis_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze E11 Surgical Swaps Run")
    parser.add_argument("run_dir", type=str, help="Path to run output directory")
    args = parser.parse_args()
    analyze_run(args.run_dir)
