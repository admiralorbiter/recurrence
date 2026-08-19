"""Sprint S13.3: Same-Four Paired Sensitivity Analysis (B=1 vs B=5).

Performs strict, unconfounded paired comparison:
E_B1(p, r, N, a) vs E_B5(p, r, N, a)

Extracts the exact same 4 pairs from the 11,520-row B=5 confirmatory dataset (run_e13_confirmatory_20260819_140139)
and computes paired differences Delta_batch E = E_B1 - E_B5 for:
- V^(0)(N): Raw directional displacement contrast along u_0
- Delta V_carry^(0)(N): Causal carry contrast
- V^(N)(N): Contemporaneous steerability
- C_R(N), Q_R(N), C_logit(N): Geometric invariants

Classifies each finding into:
1. Batch robust
2. Qualitatively robust but numerically sensitive
3. Execution dependent
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def load_b5_trace_records(b5_trace_path: Path, target_pair_ids: List[str]) -> Dict[Tuple[str, str, str, int], Dict]:
    """Load B=5 confirmatory records for the 4 target pairs and compute pair-level contrasts."""
    # (pair_id, regime, arm, horizon, condition, direction) -> record
    raw_cells = {}
    with open(b5_trace_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["pair_id"] in target_pair_ids:
                k = (r["pair_id"], r["regime"], r["arm"], r["horizon"], r["condition"], r["direction"])
                raw_cells[k] = r

    # Compute pair-level estimands for each (pair_id, regime, arm, horizon)
    results = {}
    horizons = [0, 16, 64, 256, 1024, 2048]
    regimes = ["constant", "random", "natural", "interfering"]
    arms = ["intact_recurrence", "rglru_carry_clamped"]

    for p_id in target_pair_ids:
        for reg in regimes:
            for arm in arms:
                for h in horizons:
                    # Direction a_into_b
                    m_a2b = raw_cells[(p_id, reg, arm, h, "matching_rglru_a_into_b", "a_into_b")]
                    wc_a2b = raw_cells[(p_id, reg, arm, h, "same_template_wrong_c_into_b", "a_into_b")]
                    wd_a2b = raw_cells[(p_id, reg, arm, h, "same_template_wrong_d_into_b", "a_into_b")]

                    disp_m_a2b = m_a2b["directional_displacement_u0"]
                    disp_c_a2b = wc_a2b["directional_displacement_u0"]
                    disp_d_a2b = wd_a2b["directional_displacement_u0"]
                    v0_a2b = disp_m_a2b - 0.5 * (disp_c_a2b + disp_d_a2b)

                    disp_m_a2b_N = m_a2b["directional_displacement_uN"]
                    disp_c_a2b_N = wc_a2b["directional_displacement_uN"]
                    disp_d_a2b_N = wd_a2b["directional_displacement_uN"]
                    vN_a2b = disp_m_a2b_N - 0.5 * (disp_c_a2b_N + disp_d_a2b_N)

                    # Direction b_into_a
                    m_b2a = raw_cells[(p_id, reg, arm, h, "matching_rglru_b_into_a", "b_into_a")]
                    wc_b2a = raw_cells[(p_id, reg, arm, h, "same_template_wrong_c_into_a", "b_into_a")]
                    wd_b2a = raw_cells[(p_id, reg, arm, h, "same_template_wrong_d_into_a", "b_into_a")]

                    disp_m_b2a = m_b2a["directional_displacement_u0"]
                    disp_c_b2a = wc_b2a["directional_displacement_u0"]
                    disp_d_b2a = wd_b2a["directional_displacement_u0"]
                    v0_b2a = disp_m_b2a - 0.5 * (disp_c_b2a + disp_d_b2a)

                    disp_m_b2a_N = m_b2a["directional_displacement_uN"]
                    disp_c_b2a_N = wc_b2a["directional_displacement_uN"]
                    disp_d_b2a_N = wd_b2a["directional_displacement_uN"]
                    vN_b2a = disp_m_b2a_N - 0.5 * (disp_c_b2a_N + disp_d_b2a_N)

                    v0_pooled = 0.5 * (v0_a2b + v0_b2a)
                    vN_pooled = 0.5 * (vN_a2b + vN_b2a)

                    results[(p_id, reg, arm, h)] = {
                        "pair_id": p_id,
                        "regime": reg,
                        "arm": arm,
                        "horizon": h,
                        "v0": v0_pooled,
                        "vN": vN_pooled,
                        "c_r": m_a2b["c_r"],
                        "q_r": m_a2b["q_r"],
                        "c_logit": m_a2b["c_logit"],
                    }

    return results


def main():
    b5_trace_path = Path("results/e13_controlled_recurrent_dynamics/run_e13_confirmatory_20260819_140139/dynamics_trace.jsonl")
    b1_results_path = Path("results/e13_controlled_recurrent_dynamics/b1_sensitivity_panel_4pairs/b1_panel_results.json")

    if not b1_results_path.exists():
        print(f"[Wait] B=1 panel results file not found yet at {b1_results_path}. Waiting for task-21997 to finish.")
        return

    with open(b1_results_path, "r", encoding="utf-8") as f:
        b1_raw = json.load(f)

    # Build B=1 dict
    b1_dict = {}
    for r in b1_raw:
        b1_dict[(r["pair_id"], r["regime"], r["arm"], r["horizon"])] = r

    target_pairs = sorted(list(set(r["pair_id"] for r in b1_raw)))
    b5_dict = load_b5_trace_records(b5_trace_path, target_pairs)

    print("=" * 115)
    print("STRICT SAME-FOUR PAIRED SENSITIVITY ANALYSIS: B=1 vs B=5")
    print(f"Target Pairs (4 Families): {target_pairs}")
    print("=" * 115)

    horizons = [0, 16, 64, 256, 1024, 2048]
    regimes = ["constant", "random", "natural", "interfering"]

    # 1. Primary Endpoint Comparison across Horizons (Pooled across 4 pairs and 4 regimes)
    print("\n" + "-" * 115)
    print(f"{'Horizon':<8} | {'V_intact^(0) B=1':<18} | {'V_intact^(0) B=5':<18} | {'Delta V_carry B=1':<18} | {'Delta V_carry B=5':<18} | {'C_R B=1':<10} | {'C_R B=5':<10}")
    print("-" * 115)

    for h in horizons:
        v0_intact_b1_vals = [b1_dict[(p, r, "intact_recurrence", h)]["v0"] for p in target_pairs for r in regimes]
        v0_intact_b5_vals = [b5_dict[(p, r, "intact_recurrence", h)]["v0"] for p in target_pairs for r in regimes]

        v0_clamped_b1_vals = [b1_dict[(p, r, "rglru_carry_clamped", h)]["v0"] for p in target_pairs for r in regimes]
        v0_clamped_b5_vals = [b5_dict[(p, r, "rglru_carry_clamped", h)]["v0"] for p in target_pairs for r in regimes]

        d_carry_b1 = np.mean(v0_intact_b1_vals) - np.mean(v0_clamped_b1_vals)
        d_carry_b5 = np.mean(v0_intact_b5_vals) - np.mean(v0_clamped_b5_vals)

        cr_b1 = np.mean([b1_dict[(p, r, "intact_recurrence", h)]["c_r"] for p in target_pairs for r in regimes])
        cr_b5 = np.mean([b5_dict[(p, r, "intact_recurrence", h)]["c_r"] for p in target_pairs for r in regimes])

        print(f"N={h:<6} | {np.mean(v0_intact_b1_vals):<+18.2f} | {np.mean(v0_intact_b5_vals):<+18.2f} | {d_carry_b1:<+18.2f} | {d_carry_b5:<+18.2f} | {cr_b1:<10.4f} | {cr_b5:<10.4f}")

    # 2. Per-Pair Primary Endpoints at N=2048
    print("\n" + "=" * 115)
    print("PER-PAIR PAIRED COMPARISON AT N=2048 (Averaged across 4 Regimes)")
    print("=" * 115)
    print(f"{'Pair ID':<38} | {'V^(0) B=1':<12} | {'V^(0) B=5':<12} | {'Delta V_carry B1':<16} | {'Delta V_carry B5':<16} | {'V^(N) B1':<10} | {'V^(N) B5':<10}")
    print("-" * 115)

    for p in target_pairs:
        v0_b1 = np.mean([b1_dict[(p, r, "intact_recurrence", 2048)]["v0"] for r in regimes])
        v0_b5 = np.mean([b5_dict[(p, r, "intact_recurrence", 2048)]["v0"] for r in regimes])

        dc_b1 = v0_b1 - np.mean([b1_dict[(p, r, "rglru_carry_clamped", 2048)]["v0"] for r in regimes])
        dc_b5 = v0_b5 - np.mean([b5_dict[(p, r, "rglru_carry_clamped", 2048)]["v0"] for r in regimes])

        vN_b1 = np.mean([b1_dict[(p, r, "intact_recurrence", 2048)]["vN"] for r in regimes])
        vN_b5 = np.mean([b5_dict[(p, r, "intact_recurrence", 2048)]["vN"] for r in regimes])

        print(f"{p:<38} | {v0_b1:<+12.2f} | {v0_b5:<+12.2f} | {dc_b1:<+16.2f} | {dc_b5:<+16.2f} | {vN_b1:<+10.2f} | {vN_b5:<+10.2f}")

    print("=" * 115)


if __name__ == "__main__":
    main()
