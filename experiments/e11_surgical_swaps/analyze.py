"""Experiment E11 Analysis Script: Multi-Store Surgical State Swaps & Pair-Cluster Bootstrap (Sprint S12).

Preregistered statistical engine computing:
1. Primary physical-causal endpoint: P_RGLRU at 2W (Directional Logit Displacement).
2. Primary paired specificity contrast: Delta P_spec_unrel = P_match(2W) - P_unrel(2W).
3. Secondary paired specificity contrast: Delta P_spec_perm = P_match(2W) - P_perm(2W).
4. Secondary paired growth contrast: Delta P_growth = P_match(2W) - P_match(W+1).
5. Secondary KV vs RG-LRU contrast: Delta P_kv_minus_rglru at 2W.
6. Fail-closed dataset completeness validation (no silent zeros).
7. Separation of observed sample point estimates from 10,000-draw Pair-Cluster Bootstrap distributions.
8. Mediational sliced post-graft and full-window turnover migration indices.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple


def compute_s12_pair_cluster_bootstrap(
    rows: List[Dict[str, Any]],
    n_boot: int = 10000,
    seed: int = 42,
    is_confirmatory: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Compute 10,000-replicate Pair-Cluster Bootstrap 95% CIs across stimulus pairs.

    Aggregates the 4 fixed regimes within each stimulus pair cluster, computes pair-level
    causal estimands and specificity contrasts, then bootstraps the 20 pair-level summaries.
    Fails closed if any expected condition cell is missing.
    """
    # Index rows by (pair_id, regime, lag, condition)
    cell_map = {}
    for r in rows:
        cell_key = (r["pair_id"], r["regime"], r["lag"], r["condition"])
        if cell_key in cell_map:
            raise ValueError(f"[Fail-Closed Gate] Duplicate cell detected in dataset: {cell_key}")
        cell_map[cell_key] = r

    pairs = sorted(list({r["pair_id"] for r in rows}))
    regimes = sorted(list({r["regime"] for r in rows}))
    lags = sorted(list({r["lag"] for r in rows}))

    # Fail-closed checks for confirmatory mode
    if is_confirmatory or len(pairs) == 20:
        assert len(pairs) == 20, f"[Fail-Closed Gate] Expected 20 stimulus pairs, found {len(pairs)}"
        assert set(regimes) == {"constant", "interfering", "natural", "random"}, (
            f"[Fail-Closed Gate] Regimes mismatch: {set(regimes)}"
        )
        assert set(lags) == {8, 2049, 4096}, f"[Fail-Closed Gate] Lags mismatch: {set(lags)}"
        expected_total = 20 * 4 * 3 * 22
        assert len(rows) == expected_total, (
            f"[Fail-Closed Gate] Expected {expected_total} rows, found {len(rows)}"
        )

    max_lag = max(lags) if lags else 4096
    w1_lag = 2049 if 2049 in lags else (lags[-2] if len(lags) > 1 else max_lag)
    short_lag = 8 if 8 in lags else (lags[0] if lags else 0)

    # Compute pair-level vectors
    pair_vectors = []

    for p in pairs:
        reg_metrics = []
        for reg in regimes:
            # Helper to get symmetric average of A->B and B->A (FAILS CLOSED ON MISSING CELL)
            def get_sym_metric(lag_val: int, cond_ab: str, cond_ba: str, key: str) -> float:
                key_ab = (p, reg, lag_val, cond_ab)
                key_ba = (p, reg, lag_val, cond_ba)
                if key_ab not in cell_map:
                    raise KeyError(f"[Fail-Closed Gate] Missing confirmatory cell: {key_ab}")
                if key_ba not in cell_map:
                    raise KeyError(f"[Fail-Closed Gate] Missing confirmatory cell: {key_ba}")
                row_ab = cell_map[key_ab]
                row_ba = cell_map[key_ba]
                val_ab = row_ab[key]
                val_ba = row_ba[key]
                return 0.5 * (val_ab + val_ba)

            # Directional Displacements P_C
            p_match_2w = get_sym_metric(max_lag, "rglru_only_a_into_b", "rglru_only_b_into_a", "directional_displacement")
            p_match_w1 = get_sym_metric(w1_lag, "rglru_only_a_into_b", "rglru_only_b_into_a", "directional_displacement")
            p_match_l8 = get_sym_metric(short_lag, "rglru_only_a_into_b", "rglru_only_b_into_a", "directional_displacement")

            p_unrel_2w = get_sym_metric(max_lag, "unrelated_rglru_a_into_b", "unrelated_rglru_b_into_a", "directional_displacement")
            p_perm_2w = get_sym_metric(max_lag, "permuted_rglru_a_into_b", "permuted_rglru_b_into_a", "directional_displacement")
            
            p_noise_s1_2w = get_sym_metric(max_lag, "noise_rglru_a_into_b_s1", "noise_rglru_b_into_a_s1", "directional_displacement")
            p_noise_s2_2w = get_sym_metric(max_lag, "noise_rglru_a_into_b_s2", "noise_rglru_b_into_a_s2", "directional_displacement")
            p_noise_2w = 0.5 * (p_noise_s1_2w + p_noise_s2_2w)

            p_kv_2w = get_sym_metric(max_lag, "kv_only_a_into_b", "kv_only_b_into_a", "directional_displacement")
            p_whole_2w = get_sym_metric(max_lag, "whole_swap_a_into_b", "whole_swap_b_into_a", "directional_displacement")

            # Logit Directional Projections alpha_C^logit
            alpha_match_2w = get_sym_metric(max_lag, "rglru_only_a_into_b", "rglru_only_b_into_a", "logit_directional_projection")
            alpha_kv_2w = get_sym_metric(max_lag, "kv_only_a_into_b", "kv_only_b_into_a", "logit_directional_projection")
            alpha_unrel_2w = get_sym_metric(max_lag, "unrelated_rglru_a_into_b", "unrelated_rglru_b_into_a", "logit_directional_projection")

            # Signed Graft Effects Delta_C
            delta_match_2w = get_sym_metric(max_lag, "rglru_only_a_into_b", "rglru_only_b_into_a", "signed_graft_effect")
            delta_kv_2w = get_sym_metric(max_lag, "kv_only_a_into_b", "kv_only_b_into_a", "signed_graft_effect")
            delta_unrel_2w = get_sym_metric(max_lag, "unrelated_rglru_a_into_b", "unrelated_rglru_b_into_a", "signed_graft_effect")

            reg_metrics.append({
                "p_match_2w": p_match_2w,
                "p_match_w1": p_match_w1,
                "p_match_l8": p_match_l8,
                "p_unrel_2w": p_unrel_2w,
                "p_perm_2w": p_perm_2w,
                "p_noise_2w": p_noise_2w,
                "p_kv_2w": p_kv_2w,
                "p_whole_2w": p_whole_2w,
                "delta_p_spec_unrel_2w": p_match_2w - p_unrel_2w,
                "delta_p_spec_perm_2w": p_match_2w - p_perm_2w,
                "delta_p_spec_noise_2w": p_match_2w - p_noise_2w,
                "delta_p_growth_2w_minus_w1": p_match_2w - p_match_w1,
                "delta_p_kv_minus_rglru_2w": p_kv_2w - p_match_2w,
                "alpha_match_2w": alpha_match_2w,
                "alpha_kv_2w": alpha_kv_2w,
                "alpha_unrel_2w": alpha_unrel_2w,
                "delta_match_2w": delta_match_2w,
                "delta_kv_2w": delta_kv_2w,
            })

        # Average over the 4 fixed regimes within pair cluster p
        pair_summary = {}
        keys = list(reg_metrics[0].keys())
        for k in keys:
            pair_summary[k] = sum(rm[k] for rm in reg_metrics) / len(reg_metrics)
        pair_vectors.append(pair_summary)

    # Compute observed point estimates directly from pair_vectors
    n_pairs = len(pair_vectors)
    observed_estimates = {}
    for k in pair_vectors[0].keys():
        observed_estimates[k] = sum(pv[k] for pv in pair_vectors) / n_pairs

    # 10,000-draw bootstrap over pair summaries
    rng = random.Random(seed)
    boot_samples = defaultdict(list)

    for _ in range(n_boot):
        sample = [rng.choice(pair_vectors) for _ in range(n_pairs)]
        for k in pair_vectors[0].keys():
            mean_stat = sum(s[k] for s in sample) / n_pairs
            boot_samples[k].append(mean_stat)

    ci_results: Dict[str, Dict[str, float]] = {}
    for stat_name, vals in boot_samples.items():
        sorted_v = sorted(vals)
        low_idx = int(0.025 * len(sorted_v))
        high_idx = int(0.975 * len(sorted_v))
        ci_results[stat_name] = {
            "estimate": round(observed_estimates[stat_name], 4),
            "bootstrap_mean": round(sum(sorted_v) / len(sorted_v), 4),
            "ci_low": round(sorted_v[low_idx], 4),
            "ci_high": round(sorted_v[high_idx], 4),
        }

    return ci_results


def analyze_run(run_dir: str, n_boot: int = 10000) -> Dict[str, Any]:
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

    is_confirmatory = (run_meta.get("phase") == "confirmatory")

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
            mean_dir_disp = sum(x.get("directional_displacement", x.get("absolute_displacement", 0.0)) for x in items) / len(items)
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
                "mean_directional_displacement": round(mean_dir_disp, 4),
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

    # Fail-closed mediational checks for confirmatory
    if is_confirmatory:
        assert len(med_results) == 160, f"[Fail-Closed Gate] Expected 160 mediation records, found {len(med_results)}"
        med_horizons = {m["future_tokens"] for m in med_results}
        assert med_horizons == {512, 2048}, f"[Fail-Closed Gate] Expected horizons {{512, 2048}}, found {med_horizons}"

    # Pair-Cluster Bootstrap CIs
    bootstrap_cis = compute_s12_pair_cluster_bootstrap(
        rows,
        n_boot=n_boot,
        is_confirmatory=is_confirmatory,
    )

    analysis_result = {
        "run_dir": str(run_path),
        "phase": run_meta.get("phase", "scout"),
        "model_provenance": run_meta.get("model_provenance", {}),
        "protocol_metadata": run_meta.get("protocol", {}),
        "git_provenance": run_meta.get("git_provenance", {}),
        "protocol_code_sha256": run_meta.get("protocol_code_sha256", "unknown"),
        "donor_mapping_sha256": run_meta.get("donor_mapping_sha256", "unknown"),
        "bootstrap_metadata": {
            "method": "Pair-Cluster Bootstrap",
            "B": n_boot,
            "cluster_unit": "pair_id",
            "conditioning": "frozen filler panel / deterministic seed assignment",
        },
        "lags": lags,
        "conditions": conditions,
        "causal_table": dict(causal_table),
        "bootstrap_cis": bootstrap_cis,
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
        f_rep.write(f"**Phase:** `{run_meta.get('phase', 'scout')}`\n")
        f_rep.write(f"**Run Path:** `{run_path}`\n\n")
        f_rep.write(f"**Bootstrap Inference:** Pair-Cluster Bootstrap ($B={n_boot:,}$) conditional on frozen filler panel / deterministic seed assignment.\n\n")

        f_rep.write("## 1. Primary S12 Estimands & 95% Pair-Cluster Bootstrap CIs\n\n")
        f_rep.write("| Estimand | Description | Observed Estimate | 95% Bootstrap CI |\n")
        f_rep.write("| :--- | :--- | :---: | :---: |\n")
        
        descriptions = {
            "p_match_2w": "Primary Physical Causal Endpoint: $P_{\\text{RGLRU}}(2W)$",
            "delta_p_spec_unrel_2w": "Primary Paired Specificity Contrast: $P_{\\text{match}}(2W) - P_{\\text{unrel}}(2W)$",
            "delta_p_spec_perm_2w": "Secondary Paired Specificity Contrast: $P_{\\text{match}}(2W) - P_{\\text{perm}}(2W)$",
            "delta_p_spec_noise_2w": "Matched Frobenius Noise Contrast: $P_{\\text{match}}(2W) - P_{\\text{noise}}(2W)$",
            "delta_p_growth_2w_minus_w1": "Temporal Causal Growth: $P_{\\text{match}}(2W) - P_{\\text{match}}(W+1)$",
            "delta_p_kv_minus_rglru_2w": "Store Causal Contrast: $P_{\\text{KV}}(2W) - P_{\\text{match}}(2W)$",
            "alpha_match_2w": "RG-LRU Relative Directional Share: $\\alpha_{\\text{RGLRU}}^{\\text{logit}}(2W)$",
            "alpha_kv_2w": "KV Relative Directional Share: $\\alpha_{\\text{KV}}^{\\text{logit}}(2W)$",
            "p_match_w1": "RG-LRU Displacement at $W+1$: $P_{\\text{RGLRU}}(W+1)$",
            "p_match_l8": "RG-LRU Displacement at $L=8$: $P_{\\text{RGLRU}}(L=8)$",
        }

        for k in sorted(bootstrap_cis.keys()):
            desc = descriptions.get(k, k)
            ci = bootstrap_cis[k]
            f_rep.write(f"| `{k}` | {desc} | {ci['estimate']:+.4f} | [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] |\n")

        f_rep.write("\n## 2. Causal Factorial Panel & Directional Logit Displacement Across Lags\n\n")
        f_rep.write("| Lag $L$ | Condition | Signed Graft $\\bar{\\Delta}_C$ | Directional Displacement $P_C$ | Logit Proj $\\alpha_C^{\\text{logit}}$ | Attrib Index $\\alpha_C^{\\text{cloze}}$ | Eligible N | Donor Concord |\n")
        f_rep.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for lag in lags:
            for cond in sorted(causal_table[lag].keys()):
                d = causal_table[lag][cond]
                alpha_str = f"{d['mean_attribution_index']:.3f}" if d['mean_attribution_index'] is not None else "N/A"
                elig_str = f"{d['n_eligible']}/{d['n_samples']}"
                f_rep.write(f"| {lag} | `{cond}` | {d['mean_signed_graft_effect']:+.2f} | {d['mean_directional_displacement']:+.2f} | {d['mean_logit_projection']:+.3f} | {alpha_str} | {elig_str} | {d['donor_concordance_rate']*100:.1f}% |\n")

        if med_results:
            f_rep.write("\n## 3. Mediational Forward Dynamic Propagation ($R^B \\to K_{\\text{future}}^B$)\n\n")
            f_rep.write("| Pair ID | Regime | Init Lag | Future Tokens | Turnover? | Raw Post Dist Rec A | Raw Post Dist Don B | Post Migr $\\mathcal{M}_{\\text{post}}$ | Full Migr $\\mathcal{M}_{\\text{full}}$ |\n")
            f_rep.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for m in med_results:
                turn_str = "Yes" if m.get("is_full_window_turnover") else "No"
                post_migr = m.get("post_migration_index", m.get("kv_migration_index", 0.0))
                full_migr = m.get("full_migration_index", 0.0)
                d_post_a = m.get("d_post_med_to_a", m.get("d_med_to_a", 0.0))
                d_post_b = m.get("d_post_med_to_b", m.get("d_med_to_b", 0.0))
                f_rep.write(f"| `{m['pair_id']}` | `{m['regime']}` | {m['initial_lag']} | {m['future_tokens']} | {turn_str} | {d_post_a:.4f} | {d_post_b:.4f} | {post_migr:+.4f} | {full_migr:+.4f} |\n")

        f_rep.write("\n## 4. Epistemic Assessment & Causal Framework\n\n")
        f_rep.write("1. **Directional Displacement ($P_C$) as Primary Causal Endpoint:** Directional displacement $P_C = (z_G - z_R) \\cdot \\frac{z_D - z_R}{\\|z_D - z_R\\|}$ distinguishes true causal steering magnitude from normalized share $\\alpha_C^{\\text{logit}}$ when the total donor-recipient contrast $\\|z_D - z_R\\|$ collapses at deep lags.\n")
        f_rep.write("2. **Historical Specificity Contrast:** Primary inference tests $P_{\\text{matching}} > P_{\\text{unrelated}}$ across balanced cyclic derangements conditional on frozen filler streams.\n")
        f_rep.write("3. **Dynamic Post-Graft KV Mediation:** Measures distances strictly over newly generated post-graft cache entries to determine whether continuous recurrent state propagates historical steering into downstream attention representations.\n")

    print(f"[E11] Analysis complete! Wrote calibrated report to {report_file}")
    return analysis_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze E11 Surgical Swaps Run")
    parser.add_argument("run_dir", type=str, help="Path to run output directory")
    parser.add_argument("--n_boot", type=int, default=10000, help="Number of bootstrap replicates")
    args = parser.parse_args()
    analyze_run(args.run_dir, n_boot=args.n_boot)
