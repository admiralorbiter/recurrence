"""Sprint S12c: Specificity Microscope Analysis & Pair-Cluster Bootstrap.

Computes 10,000-draw Pair-Cluster Bootstrap CIs for value-specific vs same-template vs cross-template
recurrent state steering at 2W = 4096 tokens.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np


def compute_s12c_pair_cluster_bootstrap(
    rows: List[Dict[str, Any]],
    n_boot: int = 10000,
    seed: int = 42,
    is_confirmatory: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Compute Pair-Cluster Bootstrap across the 24 value pairs."""
    # Build cell map: (pair_id, regime, condition)
    cell_map = {}
    for r in rows:
        cell_key = (r["pair_id"], r["regime"], r["condition"])
        if cell_key in cell_map:
            raise ValueError(f"[Fail-Closed Gate] Duplicate cell detected: {cell_key}")
        cell_map[cell_key] = r

    pairs = sorted(list({r["pair_id"] for r in rows}))
    regimes = sorted(list({r["regime"] for r in rows}))
    families = sorted(list({r["family_id"] for r in rows}))

    if is_confirmatory or len(pairs) == 24:
        assert len(pairs) == 24, f"[Fail-Closed Gate] Expected 24 pairs, found {len(pairs)}"
        assert set(regimes) == {"constant", "interfering", "natural", "random"}, f"Regimes mismatch: {set(regimes)}"
        expected_rows = 24 * 4 * 14
        assert len(rows) == expected_rows, f"[Fail-Closed Gate] Expected {expected_rows} rows, found {len(rows)}"

    def get_metric(pair_id: str, reg: str, cond: str) -> float:
        key = (pair_id, reg, cond)
        if key not in cell_map:
            raise KeyError(f"[Fail-Closed Gate] Missing required cell: {key}")
        return cell_map[key]["directional_displacement"]

    # Compute pair-level summaries across 4 regimes
    pair_summaries = []
    for p_id in pairs:
        # Average over 4 regimes
        p_match_vals = []
        p_wrong_vals = []
        p_cross_vals = []
        p_noise_vals = []
        p_whole_vals = []

        for reg in regimes:
            # Matching: average of A->B and B->A
            m_a2b = get_metric(p_id, reg, "matching_rglru_a_into_b")
            m_b2a = get_metric(p_id, reg, "matching_rglru_b_into_a")
            p_match_vals.append((m_a2b + m_b2a) / 2.0)

            # Same-template wrong values: average of C->B, D->B, C->A, D->A
            w_c2b = get_metric(p_id, reg, "same_template_wrong_c_into_b")
            w_d2b = get_metric(p_id, reg, "same_template_wrong_d_into_b")
            w_c2a = get_metric(p_id, reg, "same_template_wrong_c_into_a")
            w_d2a = get_metric(p_id, reg, "same_template_wrong_d_into_a")
            p_wrong_vals.append((w_c2b + w_d2b + w_c2a + w_d2a) / 4.0)

            # Cross-template: average of E->B and E->A
            cr_e2b = get_metric(p_id, reg, "cross_template_e_into_b")
            cr_e2a = get_metric(p_id, reg, "cross_template_e_into_a")
            p_cross_vals.append((cr_e2b + cr_e2a) / 2.0)

            # Noise: average of noise A->B and B->A
            n_a2b = get_metric(p_id, reg, "noise_rglru_a_into_b")
            n_b2a = get_metric(p_id, reg, "noise_rglru_b_into_a")
            p_noise_vals.append((n_a2b + n_b2a) / 2.0)

            # Whole swap
            wh_a2b = get_metric(p_id, reg, "whole_swap_a_into_b")
            wh_b2a = get_metric(p_id, reg, "whole_swap_b_into_a")
            p_whole_vals.append((wh_a2b + wh_b2a) / 2.0)

        mean_match = float(np.mean(p_match_vals))
        mean_wrong = float(np.mean(p_wrong_vals))
        mean_cross = float(np.mean(p_cross_vals))
        mean_noise = float(np.mean(p_noise_vals))
        mean_whole = float(np.mean(p_whole_vals))

        pair_summaries.append({
            "pair_id": p_id,
            "p_match": mean_match,
            "p_wrong_val": mean_wrong,
            "p_cross": mean_cross,
            "p_noise": mean_noise,
            "p_whole": mean_whole,
            "delta_p_value_spec": mean_match - mean_wrong,
            "delta_p_template_align": mean_wrong - mean_cross,
            "delta_p_template_vs_noise": mean_wrong - mean_noise,
            "delta_p_match_vs_cross": mean_match - mean_cross,
            "delta_p_match_vs_noise": mean_match - mean_noise,
        })

    estimand_keys = [
        "p_match",
        "p_wrong_val",
        "p_cross",
        "p_noise",
        "p_whole",
        "delta_p_value_spec",
        "delta_p_template_align",
        "delta_p_template_vs_noise",
        "delta_p_match_vs_cross",
        "delta_p_match_vs_noise",
    ]

    # Calculate observed point estimates directly from the pair summaries
    observed_estimates = {}
    for k in estimand_keys:
        observed_estimates[k] = float(np.mean([p[k] for p in pair_summaries]))

    # Bootstrap
    rng = np.random.default_rng(seed)
    n_pairs = len(pair_summaries)
    boot_dist: Dict[str, List[float]] = {k: [] for k in estimand_keys}

    for _ in range(n_boot):
        sample_indices = rng.choice(n_pairs, size=n_pairs, replace=True)
        resampled_pairs = [pair_summaries[idx] for idx in sample_indices]
        for k in estimand_keys:
            boot_dist[k].append(float(np.mean([p[k] for p in resampled_pairs])))

    results = {}
    for k in estimand_keys:
        obs_val = observed_estimates[k]
        dist = np.array(boot_dist[k])
        results[k] = {
            "estimate": obs_val,
            "bootstrap_mean": float(np.mean(dist)),
            "ci_low": float(np.percentile(dist, 2.5)),
            "ci_high": float(np.percentile(dist, 97.5)),
        }

    meta = {
        "n_pairs": n_pairs,
        "n_boot": n_boot,
        "seed": seed,
        "regimes": regimes,
        "families": families,
    }

    return results, meta


def analyze_run(run_dir: str, n_boot: int = 10000, seed: int = 42) -> None:
    run_path = Path(run_dir)
    trace_file = run_path / "microscope_trace.jsonl"
    summary_file = run_path / "summary.json"
    analysis_file = run_path / "analysis_summary.json"
    report_file = run_path / "report.md"

    if not trace_file.exists() or not summary_file.exists():
        raise FileNotFoundError(f"Missing required files in {run_dir}")

    with open(summary_file, "r") as f:
        run_manifest = json.load(f)

    rows = []
    with open(trace_file, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    is_confirmatory = (run_manifest.get("phase") == "confirmatory")
    results, meta = compute_s12c_pair_cluster_bootstrap(
        rows, n_boot=n_boot, seed=seed, is_confirmatory=is_confirmatory
    )

    analysis_summary = {
        "run_manifest": run_manifest,
        "meta": meta,
        "results": results,
    }

    with open(analysis_file, "w") as f:
        json.dump(analysis_summary, f, indent=2)

    # Render report.md
    md = []
    md.append("# Sprint S12c Specificity Microscope Causal Attribution Report\n")
    md.append(f"**Model Target:** `{run_manifest.get('model_provenance', {}).get('model_id', 'unknown')}`  ")
    md.append(f"**Phase:** `{run_manifest.get('phase', 'unknown')}`  ")
    md.append(f"**Run Path:** `{run_path}`  \n")
    md.append(f"**Inference:** Pair-Cluster Bootstrap ($B={n_boot:,}$) across 24 value pairs conditional on frozen filler panel.\n")

    md.append("## 1. Primary S12c Estimands & 95% Pair-Cluster Bootstrap CIs\n")
    md.append("| Estimand | Description | Observed Estimate | 95% Bootstrap CI | Confirmatory Inference |")
    md.append("| :--- | :--- | :---: | :---: | :--- |")

    descriptions = {
        "delta_p_value_spec": ("**Value-Specific Retention Contrast:** $P_{\\text{match}} - P_{\\text{same\\_template\\_wrong\\_val}}$", "Primary Test"),
        "delta_p_template_align": ("**Template Alignment Contrast:** $P_{\\text{same\\_template\\_wrong\\_val}} - P_{\\text{cross}}$", "Shared Syntax"),
        "delta_p_template_vs_noise": ("**Template vs. Noise Contrast:** $P_{\\text{same\\_template\\_wrong\\_val}} - P_{\\text{noise}}$", "Structure vs Noise"),
        "delta_p_match_vs_cross": ("Matching vs. Cross-Template Contrast: $P_{\\text{match}} - P_{\\text{cross}}$", "Total Increment"),
        "delta_p_match_vs_noise": ("Matching vs. Noise Contrast: $P_{\\text{match}} - P_{\\text{noise}}$", "Physical Lever"),
        "p_match": ("Matching Historical State: $P_{\\text{match}}(2W)$", "Target Value"),
        "p_wrong_val": ("Same-Template Wrong-Value State: $P_{\\text{wrong\\_val}}(2W)$", "Alternate Value"),
        "p_cross": ("Cross-Template Historical State: $P_{\\text{cross}}(2W)$", "Different Syntax"),
        "p_noise": ("Matched Frobenius Gaussian Noise: $P_{\\text{noise}}(2W)$", "Noise Baseline"),
        "p_whole": ("Whole State Positive Control: $P_{\\text{whole}}(2W)$", "Ceiling"),
    }

    for k, (desc, label) in descriptions.items():
        res = results[k]
        est = res["estimate"]
        ci_l = res["ci_low"]
        ci_h = res["ci_high"]
        status = "Positive; excludes zero" if ci_l > 0 else ("Negative; excludes zero" if ci_h < 0 else "Unresolved; spans zero")
        md.append(f"| `{k}` | {desc} | **{est:+.4f}** | **[{ci_l:+.4f}, {ci_h:+.4f}]** | {status} |")

    md.append("\n## 2. Scientific Interpretation\n")
    v_spec = results["delta_p_value_spec"]
    t_align = results["delta_p_template_align"]
    if v_spec["ci_low"] > 0:
        md.append(f"- **Value-Specific Retention Confirmed:** Matching historical state provides a resolved directional increment of **{v_spec['estimate']:+.2f}** (95% CI [{v_spec['ci_low']:+.2f}, {v_spec['ci_high']:+.2f}]) over same-template wrong-value states. Recurrent state carries token-level historical binding beyond syntactic template alignment.\n")
    else:
        md.append(f"- **Template-Level Steering Dominates:** The value-specific contrast ({v_spec['estimate']:+.2f}, 95% CI [{v_spec['ci_low']:+.2f}, {v_spec['ci_high']:+.2f}]) spans zero. Recurrent state steering is driven primarily by syntactic template / event-type representation.\n")

    if t_align["ci_low"] > 0:
        md.append(f"- **Syntactic Template Alignment Confirmed:** Same-template wrong-value states provide a resolved increment of **{t_align['estimate']:+.2f}** (95% CI [{t_align['ci_low']:+.2f}, {t_align['ci_high']:+.2f}]) over cross-template states, proving that shared sentence structure constitutes a major component of recurrent state steering.\n")

    with open(report_file, "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"[E12c] Analysis complete! Wrote report to {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint S12c Specificity Microscope Analyzer")
    parser.add_argument("run_dir", type=str, help="Directory of the E12c run")
    parser.add_argument("--n_boot", type=int, default=10000, help="Number of bootstrap replicates")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analyze_run(args.run_dir, n_boot=args.n_boot, seed=args.seed)
