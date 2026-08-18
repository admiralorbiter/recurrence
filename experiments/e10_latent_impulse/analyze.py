"""Experiment E10 Analysis Script (Sprint S11).

Aggregates empirical retention trajectories, Area Under the Retention Curve (AUC),
effective 50%-retention crossings, and behavioral readout across physical channels and filler regimes.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def analyze_run(run_dir: str) -> Dict[str, Any]:
    run_path = Path(run_dir)
    trace_file = run_path / "state_trace.jsonl"
    
    if not trace_file.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_file}")

    rows = []
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # Group by (regime, lag)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["regime"], r["lag"])].append(r)

    regimes = sorted(list({r["regime"] for r in rows}))
    lags = sorted(list({r["lag"] for r in rows}))

    regime_curves: Dict[str, Dict[str, Any]] = {}

    for regime in regimes:
        lag_records = []
        for lag in lags:
            items = grouped.get((regime, lag), [])
            if not items:
                continue
            n = len(items)
            agg = {
                "lag": lag,
                "conv_directly_resident": items[0]["conv_directly_resident"],
                "kv_directly_resident": items[0]["kv_directly_resident"],
                "rglru_d_rel": sum(x["mean_rglru_d_rel"] for x in items) / n,
                "conv_d_rel": sum(x["mean_conv_d_rel"] for x in items) / n,
                "kv_d_rel": sum(x["mean_kv_d_rel"] for x in items) / n,
                "rglru_retention": sum(x["mean_rglru_retention"] for x in items) / n,
                "conv_retention": sum(x["mean_conv_retention"] for x in items) / n,
                "kv_retention": sum(x["mean_kv_retention"] for x in items) / n,
                "jensen_shannon_div": sum(x["jensen_shannon_div"] for x in items) / n,
                "twoway_2afc_margin": sum(x["twoway_2afc_margin"] for x in items) / n,
                "twoway_2afc_accuracy": sum(x["twoway_2afc_accuracy"] for x in items) / n,
            }
            lag_records.append(agg)

        # Compute effective 50%-retention crossing (L_50%)
        def find_50_crossing(key: str) -> Optional[int]:
            for item in lag_records:
                if item[key] < 0.50:
                    return item["lag"]
            return None  # Right-censored

        # Compute simple trapezoidal AUC over tested lags
        def compute_auc(key: str) -> float:
            auc = 0.0
            for i in range(len(lag_records) - 1):
                l1, l2 = lag_records[i]["lag"], lag_records[i + 1]["lag"]
                v1, v2 = lag_records[i][key], lag_records[i + 1][key]
                auc += 0.5 * (v1 + v2) * (l2 - l1)
            return round(auc, 2)

        regime_curves[regime] = {
            "lag_records": lag_records,
            "rglru_50pct_crossing": find_50_crossing("rglru_retention"),
            "conv_50pct_crossing": find_50_crossing("conv_retention"),
            "kv_50pct_crossing": find_50_crossing("kv_retention"),
            "rglru_auc": compute_auc("rglru_retention"),
            "conv_auc": compute_auc("conv_retention"),
            "kv_auc": compute_auc("kv_retention"),
        }

    analysis_result = {
        "run_dir": str(run_path),
        "regimes": regimes,
        "lags": lags,
        "regime_curves": regime_curves,
    }

    with open(run_path / "analysis_summary.json", "w", encoding="utf-8") as f_out:
        json.dump(analysis_result, f_out, indent=2)

    # Generate Markdown Report
    report_file = run_path / "report.md"
    with open(report_file, "w", encoding="utf-8") as f_rep:
        f_rep.write(f"# E10 Latent Impulse Response & Store Localization Report\n\n")
        f_rep.write(f"**Run Path:** `{run_path}`\n\n")
        f_rep.write("## 1. Summary of Empirical Retention Trajectories\n\n")
        f_rep.write("| Filler Regime | RGLRU 50% Crossing ($L_{50\\%}$) | Conv1D 50% Crossing ($L_{50\\%}$) | KV 50% Crossing ($L_{50\\%}$) | RGLRU AUC | KV AUC |\n")
        f_rep.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for reg, c in regime_curves.items():
            rglru_cross = f"L={c['rglru_50pct_crossing']}" if c['rglru_50pct_crossing'] is not None else "> max lag (censored)"
            conv_cross = f"L={c['conv_50pct_crossing']}" if c['conv_50pct_crossing'] is not None else "> max lag (censored)"
            kv_cross = f"L={c['kv_50pct_crossing']}" if c['kv_50pct_crossing'] is not None else "> max lag (censored)"
            f_rep.write(f"| **{reg}** | {rglru_cross} | {conv_cross} | {kv_cross} | {c['rglru_auc']} | {c['kv_auc']} |\n")

        f_rep.write("\n## 2. Retention Trajectories Table (Constant vs Natural vs Interfering)\n\n")
        f_rep.write("| Lag $L$ | Conv Res? | KV Res? | RGLRU Ret (Const) | KV Ret (Const) | RGLRU Ret (Interf) | KV Ret (Interf) | $D_{\\text{JS}}$ (Const) |\n")
        f_rep.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        const_recs = {r["lag"]: r for r in regime_curves.get("constant", {}).get("lag_records", [])}
        interf_recs = {r["lag"]: r for r in regime_curves.get("interfering", {}).get("lag_records", [])}
        
        for lag in lags:
            c_r = const_recs.get(lag, {})
            i_r = interf_recs.get(lag, {})
            conv_res = "Yes" if c_r.get("conv_directly_resident") else "No"
            kv_res = "Yes" if c_r.get("kv_directly_resident") else "No"
            f_rep.write(
                f"| {lag} | {conv_res} | {kv_res} | "
                f"{c_r.get('rglru_retention', 0.0):.3f} | {c_r.get('kv_retention', 0.0):.3f} | "
                f"{i_r.get('rglru_retention', 0.0):.3f} | {i_r.get('kv_retention', 0.0):.3f} | "
                f"{c_r.get('jensen_shannon_div', 0.0):.4f} |\n"
            )

    print(f"[E10] Analysis complete! Wrote report to {report_file}")
    return analysis_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze E10 Latent Impulse Run")
    parser.add_argument("run_dir", type=str, help="Path to run output directory")
    args = parser.parse_args()
    analyze_run(args.run_dir)
