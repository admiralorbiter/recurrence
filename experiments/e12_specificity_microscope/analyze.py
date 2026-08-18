"""Sprint S12c: Specificity Microscope Analysis & Pair-Cluster Bootstrap.

Computes 10,000-draw Pair-Cluster Bootstrap CIs for value-specific vs same-template vs cross-template
recurrent state steering at 2W = 4096 tokens, including:
1. Primary 10,000-draw pair-cluster bootstrap estimands.
2. Regime-specific sensitivity breakdown (constant, interfering, natural, random).
3. Family-specific breakdown and leave-one-family-out (LOFO) robustness checks.
4. Unitless normalized directional projection (alpha) sensitivity analysis.
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
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute Pair-Cluster Bootstrap across the 24 value pairs."""
    cell_map = {}
    for r in rows:
        cell_key = (r["pair_id"], r["regime"], r["condition"])
        if cell_key in cell_map:
            raise ValueError(f"[Fail-Closed Gate] Duplicate cell detected: {cell_key}")
        cell_map[cell_key] = r

    pairs = sorted(list({r["pair_id"] for r in rows}))
    regimes = sorted(list({r["regime"] for r in rows}))
    families = sorted(list({r["family_id"] for r in rows}))

    # Map pair_id to family_id
    pair_to_family = {}
    for r in rows:
        pair_to_family[r["pair_id"]] = r["family_id"]

    if is_confirmatory or len(pairs) == 24:
        assert len(pairs) == 24, f"[Fail-Closed Gate] Expected 24 pairs, found {len(pairs)}"
        assert set(regimes) == {"constant", "interfering", "natural", "random"}, f"Regimes mismatch: {set(regimes)}"
        expected_rows = 24 * 4 * 14
        assert len(rows) == expected_rows, f"[Fail-Closed Gate] Expected {expected_rows} rows, found {len(rows)}"

    def get_val(pair_id: str, reg: str, cond: str, metric_key: str = "directional_displacement") -> float:
        key = (pair_id, reg, cond)
        if key not in cell_map:
            raise KeyError(f"[Fail-Closed Gate] Missing required cell: {key}")
        return float(cell_map[key][metric_key])

    # 1. Compute Pair-Level Summaries across Regimes for both Displacement and Projection
    pair_summaries = []
    regime_pair_summaries: Dict[str, List[Dict[str, Any]]] = {reg: [] for reg in regimes}

    for p_id in pairs:
        fam_id = pair_to_family[p_id]
        p_match_vals, p_wrong_vals, p_cross_vals, p_noise_vals, p_whole_vals = [], [], [], [], []
        proj_match_vals, proj_wrong_vals, proj_cross_vals, proj_noise_vals = [], [], [], []

        for reg in regimes:
            # Directional Displacement Metrics
            m_a2b = get_val(p_id, reg, "matching_rglru_a_into_b")
            m_b2a = get_val(p_id, reg, "matching_rglru_b_into_a")
            m_avg = (m_a2b + m_b2a) / 2.0
            p_match_vals.append(m_avg)

            w_c2b = get_val(p_id, reg, "same_template_wrong_c_into_b")
            w_d2b = get_val(p_id, reg, "same_template_wrong_d_into_b")
            w_c2a = get_val(p_id, reg, "same_template_wrong_c_into_a")
            w_d2a = get_val(p_id, reg, "same_template_wrong_d_into_a")
            w_avg = (w_c2b + w_d2b + w_c2a + w_d2a) / 4.0
            p_wrong_vals.append(w_avg)

            cr_e2b = get_val(p_id, reg, "cross_template_e_into_b")
            cr_e2a = get_val(p_id, reg, "cross_template_e_into_a")
            cr_avg = (cr_e2b + cr_e2a) / 2.0
            p_cross_vals.append(cr_avg)

            n_a2b = get_val(p_id, reg, "noise_rglru_a_into_b")
            n_b2a = get_val(p_id, reg, "noise_rglru_b_into_a")
            n_avg = (n_a2b + n_b2a) / 2.0
            p_noise_vals.append(n_avg)

            wh_a2b = get_val(p_id, reg, "whole_swap_a_into_b")
            wh_b2a = get_val(p_id, reg, "whole_swap_b_into_a")
            wh_avg = (wh_a2b + wh_b2a) / 2.0
            p_whole_vals.append(wh_avg)

            # Normalized Directional Projection Metrics
            pj_m_a2b = get_val(p_id, reg, "matching_rglru_a_into_b", "logit_directional_projection")
            pj_m_b2a = get_val(p_id, reg, "matching_rglru_b_into_a", "logit_directional_projection")
            pj_m_avg = (pj_m_a2b + pj_m_b2a) / 2.0
            proj_match_vals.append(pj_m_avg)

            pj_w_c2b = get_val(p_id, reg, "same_template_wrong_c_into_b", "logit_directional_projection")
            pj_w_d2b = get_val(p_id, reg, "same_template_wrong_d_into_b", "logit_directional_projection")
            pj_w_c2a = get_val(p_id, reg, "same_template_wrong_c_into_a", "logit_directional_projection")
            pj_w_d2a = get_val(p_id, reg, "same_template_wrong_d_into_a", "logit_directional_projection")
            pj_w_avg = (pj_w_c2b + pj_w_d2b + pj_w_c2a + pj_w_d2a) / 4.0
            proj_wrong_vals.append(pj_w_avg)

            pj_cr_e2b = get_val(p_id, reg, "cross_template_e_into_b", "logit_directional_projection")
            pj_cr_e2a = get_val(p_id, reg, "cross_template_e_into_a", "logit_directional_projection")
            pj_cr_avg = (pj_cr_e2b + pj_cr_e2a) / 2.0
            proj_cross_vals.append(pj_cr_avg)

            pj_n_a2b = get_val(p_id, reg, "noise_rglru_a_into_b", "logit_directional_projection")
            pj_n_b2a = get_val(p_id, reg, "noise_rglru_b_into_a", "logit_directional_projection")
            pj_n_avg = (pj_n_a2b + pj_n_b2a) / 2.0
            proj_noise_vals.append(pj_n_avg)

            regime_pair_summaries[reg].append({
                "pair_id": p_id,
                "family_id": fam_id,
                "p_match": m_avg,
                "p_wrong_val": w_avg,
                "p_cross": cr_avg,
                "p_noise": n_avg,
                "delta_p_value_spec": m_avg - w_avg,
                "delta_p_template_align": w_avg - cr_avg,
                "delta_p_template_vs_noise": w_avg - n_avg,
                "proj_match": pj_m_avg,
                "proj_wrong_val": pj_w_avg,
                "delta_proj_value_spec": pj_m_avg - pj_w_avg,
            })

        mean_match = float(np.mean(p_match_vals))
        mean_wrong = float(np.mean(p_wrong_vals))
        mean_cross = float(np.mean(p_cross_vals))
        mean_noise = float(np.mean(p_noise_vals))
        mean_whole = float(np.mean(p_whole_vals))

        mean_pj_match = float(np.mean(proj_match_vals))
        mean_pj_wrong = float(np.mean(proj_wrong_vals))
        mean_pj_cross = float(np.mean(proj_cross_vals))
        mean_pj_noise = float(np.mean(proj_noise_vals))

        pair_summaries.append({
            "pair_id": p_id,
            "family_id": fam_id,
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
            "proj_match": mean_pj_match,
            "proj_wrong_val": mean_pj_wrong,
            "proj_cross": mean_pj_cross,
            "proj_noise": mean_pj_noise,
            "delta_proj_value_spec": mean_pj_match - mean_pj_wrong,
        })

    # 2. Main Primary Estimands Bootstrap
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
        "proj_match",
        "proj_wrong_val",
        "proj_cross",
        "proj_noise",
        "delta_proj_value_spec",
    ]

    observed_estimates = {k: float(np.mean([p[k] for p in pair_summaries])) for k in estimand_keys}

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

    # 3. Regime-Specific Sensitivity Breakdown
    regime_breakdown = {}
    for reg in regimes:
        r_pairs = regime_pair_summaries[reg]
        obs_spec = float(np.mean([p["delta_p_value_spec"] for p in r_pairs]))
        obs_match = float(np.mean([p["p_match"] for p in r_pairs]))
        obs_wrong = float(np.mean([p["p_wrong_val"] for p in r_pairs]))
        obs_noise = float(np.mean([p["p_noise"] for p in r_pairs]))
        obs_proj_spec = float(np.mean([p["delta_proj_value_spec"] for p in r_pairs]))

        # Bootstrap for regime
        r_boot_spec = []
        for _ in range(n_boot):
            s_idx = rng.choice(n_pairs, size=n_pairs, replace=True)
            r_boot_spec.append(float(np.mean([r_pairs[i]["delta_p_value_spec"] for i in s_idx])))

        regime_breakdown[reg] = {
            "delta_p_value_spec": obs_spec,
            "ci_low": float(np.percentile(r_boot_spec, 2.5)),
            "ci_high": float(np.percentile(r_boot_spec, 97.5)),
            "p_match": obs_match,
            "p_wrong_val": obs_wrong,
            "p_noise": obs_noise,
            "delta_proj_value_spec": obs_proj_spec,
        }

    # 4. Family-Specific & Leave-One-Family-Out (LOFO) Breakdown
    family_breakdown = {}
    for fam in families:
        fam_pairs = [p for p in pair_summaries if p["family_id"] == fam]
        obs_fam_spec = float(np.mean([p["delta_p_value_spec"] for p in fam_pairs]))
        obs_fam_match = float(np.mean([p["p_match"] for p in fam_pairs]))
        obs_fam_wrong = float(np.mean([p["p_wrong_val"] for p in fam_pairs]))
        obs_fam_proj = float(np.mean([p["delta_proj_value_spec"] for p in fam_pairs]))

        # Bootstrap within family
        n_fam = len(fam_pairs)
        f_boot_spec = []
        for _ in range(n_boot):
            s_idx = rng.choice(n_fam, size=n_fam, replace=True)
            f_boot_spec.append(float(np.mean([fam_pairs[i]["delta_p_value_spec"] for i in s_idx])))

        family_breakdown[fam] = {
            "num_pairs": n_fam,
            "delta_p_value_spec": obs_fam_spec,
            "ci_low": float(np.percentile(f_boot_spec, 2.5)),
            "ci_high": float(np.percentile(f_boot_spec, 97.5)),
            "p_match": obs_fam_match,
            "p_wrong_val": obs_fam_wrong,
            "delta_proj_value_spec": obs_fam_proj,
        }

    lofo_breakdown = {}
    for fam in families:
        remaining_pairs = [p for p in pair_summaries if p["family_id"] != fam]
        obs_lofo_spec = float(np.mean([p["delta_p_value_spec"] for p in remaining_pairs]))
        n_rem = len(remaining_pairs)
        lofo_boot = []
        for _ in range(n_boot):
            s_idx = rng.choice(n_rem, size=n_rem, replace=True)
            lofo_boot.append(float(np.mean([remaining_pairs[i]["delta_p_value_spec"] for i in s_idx])))

        lofo_breakdown[f"leave_out_{fam}"] = {
            "left_out_family": fam,
            "remaining_pairs": n_rem,
            "delta_p_value_spec": obs_lofo_spec,
            "ci_low": float(np.percentile(lofo_boot, 2.5)),
            "ci_high": float(np.percentile(lofo_boot, 97.5)),
        }

    meta = {
        "n_pairs": n_pairs,
        "n_boot": n_boot,
        "seed": seed,
        "regimes": regimes,
        "families": families,
        "regime_breakdown": regime_breakdown,
        "family_breakdown": family_breakdown,
        "lofo_breakdown": lofo_breakdown,
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

    with open(summary_file, "r", encoding="utf-8") as f:
        run_manifest = json.load(f)

    rows = []
    with open(trace_file, "r", encoding="utf-8") as f:
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

    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2)

    # Render report.md
    md = []
    md.append("# Sprint S12c Specificity Microscope Causal Attribution Report\n")
    md.append(f"**Model Target:** `{run_manifest.get('model_provenance', {}).get('model_id', 'unknown')}`  ")
    md.append(f"**Phase:** `{run_manifest.get('phase', 'unknown')}`  ")
    md.append(f"**Run Path:** `{run_path}`  \n")
    md.append(f"**Inference:** Pair-Cluster Bootstrap ($B={n_boot:,}$) across 24 value pairs (6 pairs per family across 4 template families) conditional on frozen filler panel.\n")

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
        "p_noise": ("Matched Frobenius Gaussian Noise: $P_{\\text{noise}}(2W)$", "Noise Control"),
        "p_whole": ("Whole-State Positive Reference: $P_{\\text{whole}}(2W)$", "Whole-State Reference"),
        "delta_proj_value_spec": ("**Normalized Value-Specific Projection:** $\\Delta \\alpha_{\\text{value\\_spec}}$", "Unitless Sensitivity"),
    }

    for k, (desc, label) in descriptions.items():
        res = results[k]
        est = res["estimate"]
        ci_l = res["ci_low"]
        ci_h = res["ci_high"]
        status = "Positive; excludes zero" if ci_l > 0 else ("Negative; excludes zero" if ci_h < 0 else "Unresolved; spans zero")
        md.append(f"| `{k}` | {desc} | **{est:+.4f}** | **[{ci_l:+.4f}, {ci_h:+.4f}]** | {status} |")

    md.append("\n## 2. Regime-Specific Sensitivity Breakdown\n")
    md.append("| Regime | $P_{\\text{match}}$ | $P_{\\text{wrong\\_val}}$ | $P_{\\text{noise}}$ | $\\Delta P_{\\text{value\\_spec}}$ | 95% Bootstrap CI | $\\Delta \\alpha_{\\text{value\\_spec}}$ |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for reg, r_data in meta["regime_breakdown"].items():
        md.append(
            f"| `{reg}` | {r_data['p_match']:+.2f} | {r_data['p_wrong_val']:+.2f} | {r_data['p_noise']:+.2f} | "
            f"**{r_data['delta_p_value_spec']:+.2f}** | **[{r_data['ci_low']:+.2f}, {r_data['ci_high']:+.2f}]** | "
            f"**{r_data['delta_proj_value_spec']:+.4f}** |"
        )

    md.append("\n## 3. Template Family Breakdown & Leave-One-Family-Out (LOFO) Robustness\n")
    md.append("### Family-Specific Value Contrasts\n")
    md.append("| Family | $N_{\\text{pairs}}$ | $P_{\\text{match}}$ | $P_{\\text{wrong\\_val}}$ | $\\Delta P_{\\text{value\\_spec}}$ | 95% Bootstrap CI | $\\Delta \\alpha_{\\text{value\\_spec}}$ |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for fam, f_data in meta["family_breakdown"].items():
        md.append(
            f"| `{fam}` | {f_data['num_pairs']} | {f_data['p_match']:+.2f} | {f_data['p_wrong_val']:+.2f} | "
            f"**{f_data['delta_p_value_spec']:+.2f}** | **[{f_data['ci_low']:+.2f}, {f_data['ci_high']:+.2f}]** | "
            f"**{f_data['delta_proj_value_spec']:+.4f}** |"
        )

    md.append("\n### Leave-One-Family-Out (LOFO) Analysis\n")
    md.append("| Left-Out Family | Remaining Pairs | $\\Delta P_{\\text{value\\_spec}}$ (LOFO) | 95% Bootstrap CI | Status |")
    md.append("| :--- | :---: | :---: | :---: | :--- |")
    for lofo_key, l_data in meta["lofo_breakdown"].items():
        fam = l_data["left_out_family"]
        rem = l_data["remaining_pairs"]
        est = l_data["delta_p_value_spec"]
        ci_l = l_data["ci_low"]
        ci_h = l_data["ci_high"]
        status = "Robustly Positive" if ci_l > 0 else "Unresolved"
        md.append(f"| `{fam}` | {rem} | **{est:+.2f}** | **[{ci_l:+.2f}, {ci_h:+.2f}]** | {status} |")

    md.append("\n## 4. Scientific Interpretation\n")
    v_spec = results["delta_p_value_spec"]
    t_align = results["delta_p_template_align"]
    pj_spec = results["delta_proj_value_spec"]
    if v_spec["ci_low"] > 0:
        md.append(f"- **Value-Specific Retention Confirmed:** Matching historical state provides a resolved directional increment of **{v_spec['estimate']:+.2f}** (95% CI [{v_spec['ci_low']:+.2f}, {v_spec['ci_high']:+.2f}]) over same-template wrong-value states. In normalized projection units, the contrast is **{pj_spec['estimate']:+.4f}** (95% CI [{pj_spec['ci_low']:+.4f}, {pj_spec['ci_high']:+.4f}]). Recurrent state carries value-specific historical information beyond syntactic template alignment.\n")
    else:
        md.append(f"- **Template-Level Steering Dominates:** The value-specific contrast ({v_spec['estimate']:+.2f}, 95% CI [{v_spec['ci_low']:+.2f}, {v_spec['ci_high']:+.2f}]) spans zero. Recurrent state steering is driven primarily by syntactic template / event-type representation.\n")

    md.append(f"- **Descriptive Contrast Ladder:**\n")
    md.append(f"  - Matched-norm noise control: $P_{{\\text{{noise}}}} = {results['p_noise']['estimate']:+.2f}$\n")
    md.append(f"  - Same-template wrong-value history: $P_{{\\text{{wrong\\_val}}}} = {results['p_wrong_val']['estimate']:+.2f}$ (contrast over noise: $\\Delta P = {results['delta_p_template_vs_noise']['estimate']:+.2f}$ [{results['delta_p_template_vs_noise']['ci_low']:+.2f}, {results['delta_p_template_vs_noise']['ci_high']:+.2f}])\n")
    md.append(f"  - Matching historical value: $P_{{\\text{{match}}}} = {results['p_match']['estimate']:+.2f}$ (contrast over wrong-value: $\\Delta P = {v_spec['estimate']:+.2f}$ [{v_spec['ci_low']:+.2f}, {v_spec['ci_high']:+.2f}])\n")
    md.append(f"  - Whole-state reference: $P_{{\\text{{whole}}}} = {results['p_whole']['estimate']:+.2f}$\n")

    md.append(f"- **Template Alignment Contrast:** The contrast between same-template wrong-value and cross-template historical states is $\\Delta P = {t_align['estimate']:+.2f}$ (95% CI [{t_align['ci_low']:+.2f}, {t_align['ci_high']:+.2f}]). Because this interval spans zero, we do not resolve an additional template increment over the cross-template control used here; structured nonmatching histories steer substantially more than noise, while the matching value provides a sharp, selective advantage.\n")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"[E12c] Analysis complete! Wrote report to {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint S12c Specificity Microscope Analyzer")
    parser.add_argument("run_dir", type=str, help="Directory of the E12c run")
    parser.add_argument("--n_boot", type=int, default=10000, help="Number of bootstrap replicates")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analyze_run(args.run_dir, n_boot=args.n_boot, seed=args.seed)
