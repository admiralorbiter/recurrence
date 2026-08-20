"""Permutation Null Test & Regime Breakdown for S14 Bidirectional Screen.

Zero-compute statistical analysis evaluating:
1. Regime-specific breakdown: Separating constant, random, and natural regimes
   to examine where antisymmetric effects are concentrated.
2. Permutation null distribution: Shuffling forward and reverse pairs within regime
   (and cluster-bootstrapping / permuting at the 24-pair level) to test whether
   the observed role/residual ratio (1.82) and paired correlation rho(Delta_F, Delta_R)
   reflect true forward/reverse pairing or merely marginal shift distributions.

Terminology alignment:
- R_role: Antisymmetric (role-associated) component
- R_residual: Symmetric component (capturing role-invariant influences)
- Donor-oriented antisymmetric component: R_role > +0.01
- Anti-donor-oriented antisymmetric component: R_role < -0.01
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from scipy import stats

MANIFEST = Path("results/e14_latent_metacognition/counterfactual_screen/bidirectional_provenance_manifest.json")


def analyze_cell_subset(cells: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    """Compute decomposition statistics for a subset of cells."""
    df_arr = np.array([c["forward"]["margin_shift"] for c in cells], dtype=np.float64)
    dr_arr = np.array([c["reverse"]["margin_shift"] for c in cells], dtype=np.float64)

    r_role = (df_arr - dr_arr) / 2.0
    r_resid = (df_arr + dr_arr) / 2.0

    mean_role = float(np.mean(r_role))
    mean_abs_role = float(np.mean(np.abs(r_role)))
    mean_resid = float(np.mean(r_resid))
    mean_abs_resid = float(np.mean(np.abs(r_resid)))
    ratio = mean_abs_role / mean_abs_resid if mean_abs_resid > 1e-6 else float("inf")

    n_donor_oriented = int(np.sum(r_role > 0.01))
    n_anti_donor_oriented = int(np.sum(r_role < -0.01))
    n_near_zero = int(np.sum(np.abs(r_role) <= 0.01))

    # Pearson & Spearman correlation between forward and reverse shifts
    if len(cells) > 2 and np.std(df_arr) > 1e-6 and np.std(dr_arr) > 1e-6:
        pearson_r, pearson_p = stats.pearsonr(df_arr, dr_arr)
        spearman_r, spearman_p = stats.spearmanr(df_arr, dr_arr)
    else:
        pearson_r, pearson_p = 0.0, 1.0
        spearman_r, spearman_p = 0.0, 1.0

    return {
        "name": name,
        "n_cells": len(cells),
        "mean_r_role": mean_role,
        "mean_abs_r_role": mean_abs_role,
        "mean_r_residual": mean_resid,
        "mean_abs_r_residual": mean_abs_resid,
        "role_residual_ratio": ratio,
        "n_donor_oriented": n_donor_oriented,
        "n_anti_donor_oriented": n_anti_donor_oriented,
        "n_near_zero": n_near_zero,
        "pearson_r_fwd_rev": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r_fwd_rev": float(spearman_r),
        "spearman_p": float(spearman_p),
        "df_arr": df_arr,
        "dr_arr": dr_arr,
    }


def run_permutation_null(
    cells: List[Dict[str, Any]],
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run within-regime permutation test of forward-reverse pairing."""
    rng = np.random.default_rng(seed)

    # Group by regime
    regimes = ["constant", "random", "natural"]
    regime_cells = {reg: [c for c in cells if c["regime"] == reg] for reg in regimes}

    # Observed metrics across all 72 cells
    obs_all = analyze_cell_subset(cells, "All 72 Cells")
    obs_ratio = obs_all["role_residual_ratio"]
    obs_pearson = obs_all["pearson_r_fwd_rev"]

    # Observed metrics per regime
    obs_by_regime = {reg: analyze_cell_subset(regime_cells[reg], reg) for reg in regimes}

    null_ratios = np.zeros(n_permutations)
    null_pearsons = np.zeros(n_permutations)
    null_regime_ratios = {reg: np.zeros(n_permutations) for reg in regimes}
    null_regime_pearsons = {reg: np.zeros(n_permutations) for reg in regimes}

    # Pair-level cluster structure: 24 pairs
    pairs_list = sorted(list(set(c["pair_id"] for c in cells)))
    n_pairs = len(pairs_list)

    null_cluster_ratios = np.zeros(n_permutations)
    null_cluster_pearsons = np.zeros(n_permutations)

    for p in range(n_permutations):
        # 1. Stratified within-regime shuffle of reverse shifts
        shuffled_df = []
        shuffled_dr = []
        for reg in regimes:
            df_r = np.array([c["forward"]["margin_shift"] for c in regime_cells[reg]])
            dr_r = np.array([c["reverse"]["margin_shift"] for c in regime_cells[reg]])
            dr_shuffled = rng.permutation(dr_r)

            shuffled_df.append(df_r)
            shuffled_dr.append(dr_shuffled)

            # Per-regime null stats
            r_role_k = (df_r - dr_shuffled) / 2.0
            r_res_k = (df_r + dr_shuffled) / 2.0
            m_role_k = np.mean(np.abs(r_role_k))
            m_res_k = np.mean(np.abs(r_res_k))
            null_regime_ratios[reg][p] = m_role_k / m_res_k if m_res_k > 1e-6 else 1.0
            if np.std(df_r) > 1e-6 and np.std(dr_shuffled) > 1e-6:
                null_regime_pearsons[reg][p] = stats.pearsonr(df_r, dr_shuffled)[0]

        all_df_shuff = np.concatenate(shuffled_df)
        all_dr_shuff = np.concatenate(shuffled_dr)

        r_role_all = (all_df_shuff - all_dr_shuff) / 2.0
        r_res_all = (all_df_shuff + all_dr_shuff) / 2.0
        m_role_all = np.mean(np.abs(r_role_all))
        m_res_all = np.mean(np.abs(r_res_all))
        null_ratios[p] = m_role_all / m_res_all if m_res_all > 1e-6 else 1.0
        null_pearsons[p] = stats.pearsonr(all_df_shuff, all_dr_shuff)[0]

        # 2. Cluster-level permutation (permuting reverse shifts across the 24 pairs as whole blocks)
        pair_perm = rng.permutation(n_pairs)
        pair_to_shuffled = {pairs_list[i]: pairs_list[pair_perm[i]] for i in range(n_pairs)}
        
        clust_df = []
        clust_dr = []
        for c in cells:
            clust_df.append(c["forward"]["margin_shift"])
            # lookup reverse shift of the mapped pair under same regime
            src_pair = pair_to_shuffled[c["pair_id"]]
            matched = next(x for x in cells if x["pair_id"] == src_pair and x["regime"] == c["regime"])
            clust_dr.append(matched["reverse"]["margin_shift"])

        c_df = np.array(clust_df)
        c_dr = np.array(clust_dr)
        r_role_c = (c_df - c_dr) / 2.0
        r_res_c = (c_df + c_dr) / 2.0
        m_role_c = np.mean(np.abs(r_role_c))
        m_res_c = np.mean(np.abs(r_res_c))
        null_cluster_ratios[p] = m_role_c / m_res_c if m_res_c > 1e-6 else 1.0
        null_cluster_pearsons[p] = stats.pearsonr(c_df, c_dr)[0]

    # Calculate p-values (one-tailed for expected directions: ratio > null, pearson < null)
    p_val_ratio = float(np.mean(null_ratios >= obs_ratio))
    p_val_pearson = float(np.mean(null_pearsons <= obs_pearson))
    p_val_clust_ratio = float(np.mean(null_cluster_ratios >= obs_ratio))
    p_val_clust_pearson = float(np.mean(null_cluster_pearsons <= obs_pearson))

    regime_pvals = {}
    for reg in regimes:
        regime_pvals[reg] = {
            "p_val_ratio_greater": float(np.mean(null_regime_ratios[reg] >= obs_by_regime[reg]["role_residual_ratio"])),
            "p_val_pearson_negative": float(np.mean(null_regime_pearsons[reg] <= obs_by_regime[reg]["pearson_r_fwd_rev"])),
            "null_ratio_mean": float(np.mean(null_regime_ratios[reg])),
            "null_ratio_95th": float(np.percentile(null_regime_ratios[reg], 95)),
            "null_pearson_mean": float(np.mean(null_regime_pearsons[reg])),
            "null_pearson_5th": float(np.percentile(null_regime_pearsons[reg], 5)),
        }

    return {
        "n_permutations": n_permutations,
        "observed_overall": {
            "role_residual_ratio": obs_ratio,
            "pearson_r_fwd_rev": obs_pearson,
            "spearman_r_fwd_rev": obs_all["spearman_r_fwd_rev"],
        },
        "within_regime_null": {
            "p_val_ratio_greater": p_val_ratio,
            "p_val_pearson_negative": p_val_pearson,
            "null_ratio_mean": float(np.mean(null_ratios)),
            "null_ratio_95th": float(np.percentile(null_ratios, 95)),
            "null_pearson_mean": float(np.mean(null_pearsons)),
            "null_pearson_5th": float(np.percentile(null_pearsons, 5)),
        },
        "cluster_pair_null": {
            "p_val_ratio_greater": p_val_clust_ratio,
            "p_val_pearson_negative": p_val_clust_pearson,
            "null_ratio_mean": float(np.mean(null_cluster_ratios)),
            "null_ratio_95th": float(np.percentile(null_cluster_ratios, 95)),
            "null_pearson_mean": float(np.mean(null_cluster_pearsons)),
            "null_pearson_5th": float(np.percentile(null_cluster_pearsons, 5)),
        },
        "regime_breakdowns": {reg: {k: v for k, v in obs_by_regime[reg].items() if not k.endswith("_arr")} for reg in regimes},
        "regime_permutation_tests": regime_pvals,
    }


def main():
    print("\n" + "=" * 115)
    print("S14 BIDIRECTIONAL SCREEN: PERMUTATION NULL TEST & REGIME DECOMPOSITION")
    print("=" * 115)

    with open(MANIFEST, "r", encoding="utf-8") as f:
        data = json.load(f)
    cells = data["all_cells"]

    # 1. Regime breakdowns
    regimes = ["constant", "random", "natural"]
    regime_subsets = {reg: [c for c in cells if c["regime"] == reg] for reg in regimes}

    overall_stats = analyze_cell_subset(cells, "All 72 Cells")
    print("\n--- 1. OVERALL & REGIME-BY-REGIME DECOMPOSITION ---")
    print(f"{'Regime / Subset':<18} {'N':>4} | {'Mean R_role':>12} {'Mean |R_role|':>14} {'Mean |R_res|':>13} {'Ratio':>7} | {'rho(F,R)':>9} {'p-val':>8} | {'Donor-dir':>10} {'Anti-dir':>9}")
    print("-" * 115)

    for subset in [overall_stats] + [analyze_cell_subset(regime_subsets[reg], f"Regime: {reg}") for reg in regimes]:
        print(
            f"{subset['name']:<18} {subset['n_cells']:>4} | "
            f"{subset['mean_r_role']:>+12.4f} {subset['mean_abs_r_role']:>14.4f} {subset['mean_abs_r_residual']:>13.4f} {subset['role_residual_ratio']:>7.2f} | "
            f"{subset['pearson_r_fwd_rev']:>+9.3f} {subset['pearson_p']:>8.4f} | "
            f"{subset['n_donor_oriented']:>10} {subset['n_anti_donor_oriented']:>9}"
        )
    print("-" * 115)

    # 2. Run Permutation Tests
    print("\n--- 2. PERMUTATION NULL HYPOTHESIS TESTS (10,000 iterations) ---")
    results = run_permutation_null(cells, n_permutations=10000, seed=42)

    ov = results["observed_overall"]
    wr = results["within_regime_null"]
    cp = results["cluster_pair_null"]

    print("\nA. Stratified Within-Regime Shuffling Null:")
    print(f"   Observed Role/Residual Ratio:     {ov['role_residual_ratio']:.3f} (Null mean: {wr['null_ratio_mean']:.3f}, 95th pct: {wr['null_ratio_95th']:.3f}) -> p = {wr['p_val_ratio_greater']:.4f}")
    print(f"   Observed Pearson rho(Delta_F, Delta_R): {ov['pearson_r_fwd_rev']:+.3f} (Null mean: {wr['null_pearson_mean']:+.3f}, 5th pct: {wr['null_pearson_5th']:+.3f}) -> p = {wr['p_val_pearson_negative']:.4f}")

    print("\nB. 24-Pair Cluster-Level Block Shuffling Null:")
    print(f"   Observed Role/Residual Ratio:     {ov['role_residual_ratio']:.3f} (Null mean: {cp['null_ratio_mean']:.3f}, 95th pct: {cp['null_ratio_95th']:.3f}) -> p = {cp['p_val_ratio_greater']:.4f}")
    print(f"   Observed Pearson rho(Delta_F, Delta_R): {ov['pearson_r_fwd_rev']:+.3f} (Null mean: {cp['null_pearson_mean']:+.3f}, 5th pct: {cp['null_pearson_5th']:+.3f}) -> p = {cp['p_val_pearson_negative']:.4f}")

    print("\nC. Per-Regime Permutation Breakdown:")
    for reg in regimes:
        r_obs = results["regime_breakdowns"][reg]
        r_p = results["regime_permutation_tests"][reg]
        print(f"   [{reg.upper():<8}] Ratio: {r_obs['role_residual_ratio']:.2f} (p={r_p['p_val_ratio_greater']:.4f}) | rho(F,R): {r_obs['pearson_r_fwd_rev']:+.3f} (p={r_p['p_val_pearson_negative']:.4f})")

    # Save artifact
    out_dir = Path("results/e14_latent_metacognition/counterfactual_screen")
    out_file = out_dir / "permutation_null_and_regime_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to {out_file}\n")


if __name__ == "__main__":
    main()
