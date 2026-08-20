"""Reanalyze the bidirectional provenance screen with R_role / R_residual decomposition.

No model run needed — reads from the saved manifest.

R_role     = (Delta_F - Delta_R) / 2  — antisymmetric (role-dependent) component
R_residual = (Delta_F + Delta_R) / 2  — symmetric (lexical/global bias) component
Balance    = min(|Delta_F|, |Delta_R|) / max(|Delta_F|, |Delta_R|)

Donor-congruent reversal: R_role > 0 (A<-B shifts toward B, B<-A shifts toward A)
Anti-donor reversal:      R_role < 0 (opposite direction)
"""

import json
from pathlib import Path

MANIFEST = Path("results/e14_latent_metacognition/counterfactual_screen/bidirectional_provenance_manifest.json")

def main():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        data = json.load(f)

    cells = data["all_cells"]

    reanalyzed = []
    for c in cells:
        df = c["forward"]["margin_shift"]
        dr = c["reverse"]["margin_shift"]

        r_role = (df - dr) / 2.0
        r_residual = (df + dr) / 2.0
        abs_f = abs(df)
        abs_r = abs(dr)
        max_abs = max(abs_f, abs_r)
        balance = (min(abs_f, abs_r) / max_abs) if max_abs > 1e-6 else 0.0

        # Donor-congruent: A<-B should shift D toward B (positive), B<-A toward A (negative)
        # So R_role > 0 means donor-congruent
        donor_congruent = r_role > 0.01
        anti_donor = r_role < -0.01

        reanalyzed.append({
            "pair_id": c["pair_id"],
            "regime": c["regime"],
            "delta_fwd": df,
            "delta_rev": dr,
            "r_role": r_role,
            "r_residual": r_residual,
            "balance": balance,
            "donor_congruent": donor_congruent,
            "anti_donor": anti_donor,
        })

    # Sort by |R_role| descending
    reanalyzed.sort(key=lambda x: abs(x["r_role"]), reverse=True)

    # Counts
    n_donor = sum(1 for r in reanalyzed if r["donor_congruent"])
    n_anti = sum(1 for r in reanalyzed if r["anti_donor"])
    n_neutral = len(reanalyzed) - n_donor - n_anti

    print("=" * 120)
    print("BIDIRECTIONAL PROVENANCE SCREEN: R_role / R_residual REANALYSIS")
    print("=" * 120)
    print(f"Total Cells: {len(reanalyzed)}")
    print(f"Donor-Congruent Reversals (R_role > +0.01): {n_donor} ({100*n_donor/len(reanalyzed):.1f}%)")
    print(f"Anti-Donor Reversals      (R_role < -0.01): {n_anti} ({100*n_anti/len(reanalyzed):.1f}%)")
    print(f"Neutral                   (|R_role| <= 0.01): {n_neutral} ({100*n_neutral/len(reanalyzed):.1f}%)")
    print()

    mean_r_role = sum(r["r_role"] for r in reanalyzed) / len(reanalyzed)
    mean_r_residual = sum(r["r_residual"] for r in reanalyzed) / len(reanalyzed)
    mean_abs_r_role = sum(abs(r["r_role"]) for r in reanalyzed) / len(reanalyzed)
    mean_abs_r_residual = sum(abs(r["r_residual"]) for r in reanalyzed) / len(reanalyzed)

    print(f"Mean R_role:              {mean_r_role:+.4f} logits")
    print(f"Mean |R_role|:            {mean_abs_r_role:.4f} logits")
    print(f"Mean R_residual:          {mean_r_residual:+.4f} logits")
    print(f"Mean |R_residual|:        {mean_abs_r_residual:.4f} logits")
    print(f"Role/Residual Ratio:      {mean_abs_r_role / mean_abs_r_residual:.2f}" if mean_abs_r_residual > 1e-6 else "Role/Residual Ratio: inf")
    print()

    print(f"{'Pair':<38} {'Regime':<10} {'Delta_F':>8} {'Delta_R':>8} | {'R_role':>8} {'R_resid':>8} {'Bal':>5} | {'Dir':<8}")
    print("-" * 120)
    for r in reanalyzed[:30]:
        direction = "DONOR" if r["donor_congruent"] else ("ANTI" if r["anti_donor"] else "~0")
        print(
            f"{r['pair_id']:<38} {r['regime']:<10} "
            f"{r['delta_fwd']:+8.3f} {r['delta_rev']:+8.3f} | "
            f"{r['r_role']:+8.3f} {r['r_residual']:+8.3f} {r['balance']:5.2f} | "
            f"{direction:<8}"
        )
    print("=" * 120)

    # Save reanalysis
    out_dir = Path("results/e14_latent_metacognition/counterfactual_screen")
    out_file = out_dir / "role_residual_reanalysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "n_donor_congruent": n_donor,
            "n_anti_donor": n_anti,
            "n_neutral": n_neutral,
            "mean_r_role": mean_r_role,
            "mean_abs_r_role": mean_abs_r_role,
            "mean_r_residual": mean_r_residual,
            "mean_abs_r_residual": mean_abs_r_residual,
            "cells_sorted_by_abs_r_role": reanalyzed,
        }, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
