"""Sprint S13: Controlled Task-Irrelevant Recurrent Dynamics Analyzer.

Computes 10,000-draw longitudinal Pair-Cluster Bootstrap CIs for:
1. Primary longitudinal value specificity trajectory V^(0)(N) on frozen baseline axis u_0.
2. Causal arm comparison: Delta V_carry_effect^(0)(N) = V_intact^(0)(N) - V_carry_clamped^(0)(N).
3. Contemporaneous steerability trajectory V^(N)(N) on contemporaneous axis u_N.
4. Zero-cost geometric diagnostics: C_logit(N), C_R(N), and Q_R(N).
5. Regime-specific trajectory disaggregations across constant, random, natural, interfering.
6. Structured history vs noise retention contrast at N=0 and N=2048.
7. Leave-One-Family-Out (LOFO) robustness tables at primary endpoint N=2048.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np


def compute_s13_pair_cluster_bootstrap(
    rows: List[Dict[str, Any]],
    n_boot: int = 10000,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute longitudinal Pair-Cluster Bootstrap across pairs, preserving horizon covariance."""
    # Build cell map: (pair_id, regime, arm, horizon, condition)
    cell_map = {}
    for r in rows:
        cell_key = (r["pair_id"], r["regime"], r["arm"], r["horizon"], r["condition"])
        if cell_key in cell_map:
            raise ValueError(f"[E13 Fail-Closed Gate] Duplicate cell record detected in trace: {cell_key}")
        cell_map[cell_key] = r

    pairs = sorted(list({r["pair_id"] for r in rows}))
    regimes = sorted(list({r["regime"] for r in rows}))
    arms = sorted(list({r["arm"] for r in rows}))
    horizons = sorted(list({r["horizon"] for r in rows}))
    families = sorted(list({r["family_id"] for r in rows}))

    pair_to_family = {r["pair_id"]: r["family_id"] for r in rows}

    def get_val(p_id: str, reg: str, arm: str, h: int, cond: str, metric: str = "directional_displacement_u0", default: float = 0.0) -> float:
        key = (p_id, reg, arm, h, cond)
        if key not in cell_map:
            raise KeyError(f"[E13 Fail-Closed Gate] Missing required cell: {key}")
        val = cell_map[key].get(metric, default)
        return float(val) if val is not None else default

    # 1. Compute Pair-Level Trajectory Summaries
    pair_trajectories = []
    for p_id in pairs:
        fam_id = pair_to_family[p_id]
        p_data: Dict[str, Any] = {"pair_id": p_id, "family_id": fam_id}

        # Pooled computations across arms and horizons
        for arm in arms:
            for h in horizons:
                v0_match_list, v0_wrong_list, v0_noise_list, v0_cross_list = [], [], [], []
                vN_match_list, vN_wrong_list = [], []
                proj0_match_list, proj0_wrong_list = [], []
                cloze_list = []
                c_logit_list, c_r_list, q_r_list = [], [], []

                for reg in regimes:
                    m_a2b = get_val(p_id, reg, arm, h, "matching_rglru_a_into_b", "directional_displacement_u0")
                    m_b2a = get_val(p_id, reg, arm, h, "matching_rglru_b_into_a", "directional_displacement_u0")
                    v0_match_list.append((m_a2b + m_b2a) / 2.0)

                    w_c2b = get_val(p_id, reg, arm, h, "same_template_wrong_c_into_b", "directional_displacement_u0")
                    w_d2b = get_val(p_id, reg, arm, h, "same_template_wrong_d_into_b", "directional_displacement_u0")
                    w_c2a = get_val(p_id, reg, arm, h, "same_template_wrong_c_into_a", "directional_displacement_u0")
                    w_d2a = get_val(p_id, reg, arm, h, "same_template_wrong_d_into_a", "directional_displacement_u0")
                    v0_wrong_list.append((w_c2b + w_d2b + w_c2a + w_d2a) / 4.0)

                    # Contemporaneous uN
                    m_a2b_N = get_val(p_id, reg, arm, h, "matching_rglru_a_into_b", "directional_displacement_uN")
                    m_b2a_N = get_val(p_id, reg, arm, h, "matching_rglru_b_into_a", "directional_displacement_uN")
                    vN_match_list.append((m_a2b_N + m_b2a_N) / 2.0)

                    w_c2b_N = get_val(p_id, reg, arm, h, "same_template_wrong_c_into_b", "directional_displacement_uN")
                    w_d2b_N = get_val(p_id, reg, arm, h, "same_template_wrong_d_into_b", "directional_displacement_uN")
                    w_c2a_N = get_val(p_id, reg, arm, h, "same_template_wrong_c_into_a", "directional_displacement_uN")
                    w_d2a_N = get_val(p_id, reg, arm, h, "same_template_wrong_d_into_a", "directional_displacement_uN")
                    vN_wrong_list.append((w_c2b_N + w_d2b_N + w_c2a_N + w_d2a_N) / 4.0)

                    # Normalized projection u0
                    pj_m_a2b = get_val(p_id, reg, arm, h, "matching_rglru_a_into_b", "normalized_projection_u0")
                    pj_m_b2a = get_val(p_id, reg, arm, h, "matching_rglru_b_into_a", "normalized_projection_u0")
                    proj0_match_list.append((pj_m_a2b + pj_m_b2a) / 2.0)

                    pj_w_c2b = get_val(p_id, reg, arm, h, "same_template_wrong_c_into_b", "normalized_projection_u0")
                    pj_w_d2b = get_val(p_id, reg, arm, h, "same_template_wrong_d_into_b", "normalized_projection_u0")
                    pj_w_c2a = get_val(p_id, reg, arm, h, "same_template_wrong_c_into_a", "normalized_projection_u0")
                    pj_w_d2a = get_val(p_id, reg, arm, h, "same_template_wrong_d_into_a", "normalized_projection_u0")
                    proj0_wrong_list.append((pj_w_c2b + pj_w_d2b + pj_w_c2a + pj_w_d2a) / 4.0)

                    # Cloze margin
                    clz_a2b = get_val(p_id, reg, arm, h, "matching_rglru_a_into_b", "cloze_margin")
                    clz_b2a = get_val(p_id, reg, arm, h, "matching_rglru_b_into_a", "cloze_margin")
                    cloze_list.append((clz_a2b + clz_b2a) / 2.0)

                    # Geometric metrics
                    clog_a2b = get_val(p_id, reg, arm, h, "matching_rglru_a_into_b", "c_logit", default=np.nan)
                    clog_b2a = get_val(p_id, reg, arm, h, "matching_rglru_b_into_a", "c_logit", default=np.nan)
                    if not np.isnan(clog_a2b) and not np.isnan(clog_b2a):
                        c_logit_list.append((clog_a2b + clog_b2a) / 2.0)

                    cr_a2b = get_val(p_id, reg, arm, h, "matching_rglru_a_into_b", "c_r", default=np.nan)
                    cr_b2a = get_val(p_id, reg, arm, h, "matching_rglru_b_into_a", "c_r", default=np.nan)
                    if not np.isnan(cr_a2b) and not np.isnan(cr_b2a):
                        c_r_list.append((cr_a2b + cr_b2a) / 2.0)

                    qr_a2b = get_val(p_id, reg, arm, h, "matching_rglru_a_into_b", "q_r", default=np.nan)
                    qr_b2a = get_val(p_id, reg, arm, h, "matching_rglru_b_into_a", "q_r", default=np.nan)
                    if not np.isnan(qr_a2b) and not np.isnan(qr_b2a):
                        q_r_list.append((qr_a2b + qr_b2a) / 2.0)

                    # Secondary controls at N=0 and N=2048
                    if h in (0, 2048):
                        n_a2b = get_val(p_id, reg, arm, h, "noise_rglru_a_into_b", "directional_displacement_u0")
                        n_b2a = get_val(p_id, reg, arm, h, "noise_rglru_b_into_a", "directional_displacement_u0")
                        v0_noise_list.append((n_a2b + n_b2a) / 2.0)

                        cr_a2b = get_val(p_id, reg, arm, h, "cross_template_e_into_b", "directional_displacement_u0")
                        cr_b2a = get_val(p_id, reg, arm, h, "cross_template_e_into_a", "directional_displacement_u0")
                        v0_cross_list.append((cr_a2b + cr_b2a) / 2.0)

                # Store pair means across regimes
                mean_m0 = float(np.mean(v0_match_list))
                mean_w0 = float(np.mean(v0_wrong_list))
                mean_mN = float(np.mean(vN_match_list))
                mean_wN = float(np.mean(vN_wrong_list))
                mean_pjm = float(np.mean(proj0_match_list))
                mean_pjw = float(np.mean(proj0_wrong_list))
                mean_clz = float(np.mean(cloze_list))

                p_data[f"{arm}_N{h}_v0_match"] = mean_m0
                p_data[f"{arm}_N{h}_v0_wrong"] = mean_w0
                p_data[f"{arm}_N{h}_v0_spec"] = mean_m0 - mean_w0
                p_data[f"{arm}_N{h}_vN_spec"] = mean_mN - mean_wN
                p_data[f"{arm}_N{h}_proj0_spec"] = mean_pjm - mean_pjw
                p_data[f"{arm}_N{h}_cloze"] = mean_clz

                if c_logit_list:
                    p_data[f"{arm}_N{h}_c_logit"] = float(np.mean(c_logit_list))
                if c_r_list:
                    p_data[f"{arm}_N{h}_c_r"] = float(np.mean(c_r_list))
                if q_r_list:
                    p_data[f"{arm}_N{h}_q_r"] = float(np.mean(q_r_list))

                if h in (0, 2048):
                    mean_n0 = float(np.mean(v0_noise_list))
                    mean_cr0 = float(np.mean(v0_cross_list))
                    p_data[f"{arm}_N{h}_v0_noise"] = mean_n0
                    p_data[f"{arm}_N{h}_v0_cross"] = mean_cr0
                    p_data[f"{arm}_N{h}_v0_struct_vs_noise"] = mean_w0 - mean_n0

        # Pooled Carry Effect for each horizon
        for h in horizons:
            p_data[f"delta_v0_carry_effect_N{h}"] = (
                p_data[f"intact_recurrence_N{h}_v0_spec"] - p_data[f"rglru_carry_clamped_N{h}_v0_spec"]
            )

        # Regime-Specific Disaggregated Pair Metrics
        for reg in regimes:
            for h in horizons:
                # Intact in this regime
                m0_int = (get_val(p_id, reg, "intact_recurrence", h, "matching_rglru_a_into_b", "directional_displacement_u0") +
                          get_val(p_id, reg, "intact_recurrence", h, "matching_rglru_b_into_a", "directional_displacement_u0")) / 2.0
                w0_int = (get_val(p_id, reg, "intact_recurrence", h, "same_template_wrong_c_into_b", "directional_displacement_u0") +
                          get_val(p_id, reg, "intact_recurrence", h, "same_template_wrong_d_into_b", "directional_displacement_u0") +
                          get_val(p_id, reg, "intact_recurrence", h, "same_template_wrong_c_into_a", "directional_displacement_u0") +
                          get_val(p_id, reg, "intact_recurrence", h, "same_template_wrong_d_into_a", "directional_displacement_u0")) / 4.0
                v0_int = m0_int - w0_int

                # Clamped in this regime
                m0_clm = (get_val(p_id, reg, "rglru_carry_clamped", h, "matching_rglru_a_into_b", "directional_displacement_u0") +
                          get_val(p_id, reg, "rglru_carry_clamped", h, "matching_rglru_b_into_a", "directional_displacement_u0")) / 2.0
                w0_clm = (get_val(p_id, reg, "rglru_carry_clamped", h, "same_template_wrong_c_into_b", "directional_displacement_u0") +
                          get_val(p_id, reg, "rglru_carry_clamped", h, "same_template_wrong_d_into_b", "directional_displacement_u0") +
                          get_val(p_id, reg, "rglru_carry_clamped", h, "same_template_wrong_c_into_a", "directional_displacement_u0") +
                          get_val(p_id, reg, "rglru_carry_clamped", h, "same_template_wrong_d_into_a", "directional_displacement_u0")) / 4.0
                v0_clm = m0_clm - w0_clm

                # Contemporaneous uN intact in this regime
                mN_int = (get_val(p_id, reg, "intact_recurrence", h, "matching_rglru_a_into_b", "directional_displacement_uN") +
                          get_val(p_id, reg, "intact_recurrence", h, "matching_rglru_b_into_a", "directional_displacement_uN")) / 2.0
                wN_int = (get_val(p_id, reg, "intact_recurrence", h, "same_template_wrong_c_into_b", "directional_displacement_uN") +
                          get_val(p_id, reg, "intact_recurrence", h, "same_template_wrong_d_into_b", "directional_displacement_uN") +
                          get_val(p_id, reg, "intact_recurrence", h, "same_template_wrong_c_into_a", "directional_displacement_uN") +
                          get_val(p_id, reg, "intact_recurrence", h, "same_template_wrong_d_into_a", "directional_displacement_uN")) / 4.0
                vN_int = mN_int - wN_int

                p_data[f"reg_{reg}_intact_N{h}_v0_spec"] = v0_int
                p_data[f"reg_{reg}_clamped_N{h}_v0_spec"] = v0_clm
                p_data[f"reg_{reg}_delta_v0_carry_effect_N{h}"] = v0_int - v0_clm
                p_data[f"reg_{reg}_intact_N{h}_vN_spec"] = vN_int

                # Geometric diagnostics per regime
                clog_a2b = get_val(p_id, reg, "intact_recurrence", h, "matching_rglru_a_into_b", "c_logit", default=np.nan)
                clog_b2a = get_val(p_id, reg, "intact_recurrence", h, "matching_rglru_b_into_a", "c_logit", default=np.nan)
                if not np.isnan(clog_a2b) and not np.isnan(clog_b2a):
                    p_data[f"reg_{reg}_intact_N{h}_c_logit"] = (clog_a2b + clog_b2a) / 2.0

                cr_a2b = get_val(p_id, reg, "intact_recurrence", h, "matching_rglru_a_into_b", "c_r", default=np.nan)
                cr_b2a = get_val(p_id, reg, "intact_recurrence", h, "matching_rglru_b_into_a", "c_r", default=np.nan)
                if not np.isnan(cr_a2b) and not np.isnan(cr_b2a):
                    p_data[f"reg_{reg}_intact_N{h}_c_r"] = (cr_a2b + cr_b2a) / 2.0

                qr_a2b = get_val(p_id, reg, "intact_recurrence", h, "matching_rglru_a_into_b", "q_r", default=np.nan)
                qr_b2a = get_val(p_id, reg, "intact_recurrence", h, "matching_rglru_b_into_a", "q_r", default=np.nan)
                if not np.isnan(qr_a2b) and not np.isnan(qr_b2a):
                    p_data[f"reg_{reg}_intact_N{h}_q_r"] = (qr_a2b + qr_b2a) / 2.0

        pair_trajectories.append(p_data)

    # 2. Extract All Estimand Keys
    all_keys = [k for k in pair_trajectories[0].keys() if k not in ("pair_id", "family_id")]
    observed_estimates = {k: float(np.mean([p[k] for p in pair_trajectories])) for k in all_keys}

    # 3. Longitudinal Pair-Cluster Bootstrap
    rng = np.random.default_rng(seed)
    n_pairs = len(pair_trajectories)
    boot_dist: Dict[str, List[float]] = {k: [] for k in all_keys}

    for _ in range(n_boot):
        sample_idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        resampled_pairs = [pair_trajectories[i] for i in sample_idx]
        for k in all_keys:
            boot_dist[k].append(float(np.mean([p[k] for p in resampled_pairs])))

    results = {}
    for k in all_keys:
        dist = np.array(boot_dist[k])
        results[k] = {
            "estimate": observed_estimates[k],
            "bootstrap_mean": float(np.mean(dist)),
            "ci_low": float(np.percentile(dist, 2.5)),
            "ci_high": float(np.percentile(dist, 97.5)),
        }

    # 4. Leave-One-Family-Out (LOFO) at Primary Endpoint N=2048
    lofo_n2048 = {}
    primary_key = "intact_recurrence_N2048_v0_spec"
    for fam in families:
        rem_pairs = [p for p in pair_trajectories if p["family_id"] != fam]
        n_rem = len(rem_pairs)
        obs_lofo = float(np.mean([p[primary_key] for p in rem_pairs]))
        lofo_boot = []
        for _ in range(n_boot):
            s_idx = rng.choice(n_rem, size=n_rem, replace=True)
            lofo_boot.append(float(np.mean([rem_pairs[i][primary_key] for i in s_idx])))
        lofo_n2048[f"leave_out_{fam}"] = {
            "left_out_family": fam,
            "remaining_pairs": n_rem,
            "estimate": obs_lofo,
            "ci_low": float(np.percentile(lofo_boot, 2.5)),
            "ci_high": float(np.percentile(lofo_boot, 97.5)),
        }

    meta = {
        "n_pairs": n_pairs,
        "n_boot": n_boot,
        "seed": seed,
        "horizons": horizons,
        "regimes": regimes,
        "arms": arms,
        "families": families,
        "lofo_n2048": lofo_n2048,
    }

    return results, meta


def analyze_run(run_dir: str, n_boot: int = 10000, seed: int = 42) -> None:
    run_path = Path(run_dir)
    trace_file = run_path / "dynamics_trace.jsonl"
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

    results, meta = compute_s13_pair_cluster_bootstrap(rows, n_boot=n_boot, seed=seed)

    analysis_summary = {
        "run_manifest": run_manifest,
        "meta": meta,
        "results": results,
    }

    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2)

    # Render report.md
    md = []
    md.append("# Sprint S13 Controlled Recurrent Dynamics Report\n")
    md.append(f"**Model Target:** `{run_manifest.get('model_provenance', {}).get('model_id', 'unknown')}`  ")
    md.append(f"**Phase:** `{run_manifest.get('phase', 'unknown')}`  ")
    md.append(f"**Run Path:** `{run_path}`  \n")
    md.append(f"**Standardized Origin:** Random 2W Baseline ($L_0 = 4096$)  ")
    md.append(f"**Longitudinal Ruler:** Frozen Baseline Axis $u_0 = (z_D(0) - z_R(0)) / \\|z_D(0) - z_R(0)\\|$  ")
    md.append(f"**Inference:** Longitudinal Pair-Cluster Bootstrap ($B={n_boot:,}$) across {meta['n_pairs']} value pairs.\n")

    # Preregistered Endpoint Summary
    md.append("## Preregistered Confirmatory Endpoint Hierarchy\n")
    md.append("1. **Primary Endpoint A (Fate of Historical Representation):** $V_{\\text{intact}}^{(0)}(2048)$")
    res_a = results["intact_recurrence_N2048_v0_spec"]
    md.append(f"   - **Result:** $V_{{\\text{{intact}}}}^{{(0)}}(2048) = \\mathbf{{{res_a['estimate']:+.2f}}}$ [{res_a['ci_low']:+.2f}, {res_a['ci_high']:+.2f}]\n")

    md.append("2. **Primary Endpoint B (Causal Carry-Evolution Effect):** $\\Delta V_{\\text{carry\\_effect}}^{(0)}(2048) = V_{\\text{intact}}^{(0)}(2048) - V_{\\text{clamped}}^{(0)}(2048)$")
    res_b = results["delta_v0_carry_effect_N2048"]
    md.append(f"   - **Result:** $\\Delta V_{{\\text{{carry\\_effect}}}}^{{(0)}}(2048) = \\mathbf{{{res_b['estimate']:+.2f}}}$ [{res_b['ci_low']:+.2f}, {res_b['ci_high']:+.2f}]\n")

    if "intact_recurrence_N2048_c_r" in results:
        res_cr = results["intact_recurrence_N2048_c_r"]
        md.append(f"3. **Primary Geometric Diagnostic:** $C_R(2048) = \\mathbf{{{res_cr['estimate']:+.4f}}}$ [{res_cr['ci_low']:+.4f}, {res_cr['ci_high']:+.4f}]\n")

    md.append("4. **Trajectory Anatomy:** Horizons $N \\in \\{16, 64, 256, 1024\\}$ on frozen baseline axis $u_0$.\n")
    md.append("5. **Secondary Re-expression:** Contemporaneous steerability $V^{(N)}(N)$ and logit axis cosine $C_{\\text{logit}}(N)$.\n")

    # Section 1: Primary Value Specificity Trajectory
    md.append("\n## 1. Primary Value Specificity Trajectory $V^{(0)}(N)$ across Horizons\n")
    md.append("| Horizon $N$ | $P_{\\text{match}}^{(0)}(N)$ | $P_{\\text{wrong\\_val}}^{(0)}(N)$ | $V_{\\text{intact}}^{(0)}(N)$ | 95% Bootstrap CI | $\\Delta \\alpha_{\\text{value\\_spec}}^{(0)}(N)$ | Status |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for h in meta["horizons"]:
        m_val = results[f"intact_recurrence_N{h}_v0_match"]["estimate"]
        w_val = results[f"intact_recurrence_N{h}_v0_wrong"]["estimate"]
        res_v = results[f"intact_recurrence_N{h}_v0_spec"]
        res_pj = results[f"intact_recurrence_N{h}_proj0_spec"]
        v_est = res_v["estimate"]
        ci_l = res_v["ci_low"]
        ci_h = res_v["ci_high"]
        status = "Positive (Resolved)" if ci_l > 0 else ("Negative (Resolved)" if ci_h < 0 else "Unresolved")
        md.append(f"| $N={h}$ | {m_val:+.2f} | {w_val:+.2f} | **{v_est:+.2f}** | **[{ci_l:+.2f}, {ci_h:+.2f}]** | **{res_pj['estimate']:+.4f}** | {status} |")

    # Section 2: Causal Arm Comparison
    md.append("\n## 2. Causal Arm Comparison: Intact Recurrence vs. RG-LRU Carry Clamped\n")
    md.append("| Horizon $N$ | $V_{\\text{intact}}^{(0)}(N)$ | $V_{\\text{clamped}}^{(0)}(N)$ | $\\Delta V_{\\text{carry\\_effect}}^{(0)}(N)$ | 95% Bootstrap CI | Dynamical Interpretation |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :--- |")

    for h in meta["horizons"]:
        v_int = results[f"intact_recurrence_N{h}_v0_spec"]["estimate"]
        v_clm = results[f"rglru_carry_clamped_N{h}_v0_spec"]["estimate"]
        res_eff = results[f"delta_v0_carry_effect_N{h}"]
        eff_est = res_eff["estimate"]
        ci_l = res_eff["ci_low"]
        ci_h = res_eff["ci_high"]
        if ci_l > 0:
            interp = "Amplification / Positive Carry Effect"
        elif ci_h < 0:
            interp = "Old-axis attenuation relative to carry-clamped control"
        else:
            interp = "No Resolved Arm Difference"
        md.append(f"| $N={h}$ | {v_int:+.2f} | {v_clm:+.2f} | **{eff_est:+.2f}** | **[{ci_l:+.2f}, {ci_h:+.2f}]** | {interp} |")

    # Section 3: Regime-Specific Disaggregation
    md.append("\n## 3. Regime-Specific Longitudinal Trajectories & Causal Carry Effects\n")
    md.append("### 3.1 Intact Specificity $V_{\\text{intact}}^{(0)}(N)$ by Drive Regime\n")
    md.append("| Horizon $N$ | `constant` [95% CI] | `random` [95% CI] | `natural` [95% CI] | `interfering` [95% CI] |")
    md.append("| :---: | :---: | :---: | :---: | :---: |")
    for h in meta["horizons"]:
        cols = []
        for reg in meta["regimes"]:
            res = results[f"reg_{reg}_intact_N{h}_v0_spec"]
            cols.append(f"**{res['estimate']:+.2f}** [{res['ci_low']:+.2f}, {res['ci_high']:+.2f}]")
        md.append(f"| $N={h}$ | " + " | ".join(cols) + " |")

    md.append("\n### 3.2 Causal Carry Effect $\\Delta V_{\\text{carry\\_effect}}^{(0)}(N)$ by Drive Regime\n")
    md.append("| Horizon $N$ | `constant` [95% CI] | `random` [95% CI] | `natural` [95% CI] | `interfering` [95% CI] |")
    md.append("| :---: | :---: | :---: | :---: | :---: |")
    for h in meta["horizons"]:
        cols = []
        for reg in meta["regimes"]:
            res = results[f"reg_{reg}_delta_v0_carry_effect_N{h}"]
            cols.append(f"**{res['estimate']:+.2f}** [{res['ci_low']:+.2f}, {res['ci_high']:+.2f}]")
        md.append(f"| $N={h}$ | " + " | ".join(cols) + " |")

    md.append("\n### 3.3 Contemporaneous Steerability $V_{\\text{intact}}^{(N)}(N)$ (Projection on $u_N$)\n")
    md.append("| Horizon $N$ | Pooled [95% CI] | `constant` [95% CI] | `random` [95% CI] | `natural` [95% CI] | `interfering` [95% CI] |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for h in meta["horizons"]:
        res_pool = results[f"intact_recurrence_N{h}_vN_spec"]
        pool_str = f"**{res_pool['estimate']:+.2f}** [{res_pool['ci_low']:+.2f}, {res_pool['ci_high']:+.2f}]"
        cols = [pool_str]
        for reg in meta["regimes"]:
            res = results[f"reg_{reg}_intact_N{h}_vN_spec"]
            cols.append(f"**{res['estimate']:+.2f}** [{res['ci_low']:+.2f}, {res['ci_high']:+.2f}]")
        md.append(f"| $N={h}$ | " + " | ".join(cols) + " |")

    # Section 4: Geometric Diagnostics (if present)
    if "intact_recurrence_N0_c_r" in results:
        md.append("\n## 4. Geometric Diagnostics: State-Space Rotation ($C_R$) & Logit Alignment ($C_{\\text{logit}}$)\n")
        md.append("| Horizon $N$ | State Rotation $C_R(N)$ [95% CI] | State Quotient $Q_R(N)$ [95% CI] | Logit Alignment $C_{\\text{logit}}(N)$ [95% CI] |")
        md.append("| :---: | :---: | :---: | :---: |")
        for h in meta["horizons"]:
            cr_res = results[f"intact_recurrence_N{h}_c_r"]
            qr_res = results[f"intact_recurrence_N{h}_q_r"]
            clog_res = results[f"intact_recurrence_N{h}_c_logit"]
            md.append(
                f"| $N={h}$ | "
                f"**{cr_res['estimate']:+.4f}** [{cr_res['ci_low']:+.4f}, {cr_res['ci_high']:+.4f}] | "
                f"**{qr_res['estimate']:+.4f}** [{qr_res['ci_low']:+.4f}, {qr_res['ci_high']:+.4f}] | "
                f"**{clog_res['estimate']:+.4f}** [{clog_res['ci_low']:+.4f}, {clog_res['ci_high']:+.4f}] |"
            )

    # Section 5: Secondary Reference Controls
    md.append("\n## 5. Secondary Controls & Structural Contrast ($N=0$ and $N=2048$)\n")
    md.append("| Horizon $N$ | $P_{\\text{wrong\\_val}}^{(0)}$ | $P_{\\text{noise}}^{(0)}$ | $\\Delta P_{\\text{struct\\_vs\\_noise}}^{(0)}$ | 95% Bootstrap CI | Status |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :--- |")
    for h in (0, 2048):
        w_val = results[f"intact_recurrence_N{h}_v0_wrong"]["estimate"]
        n_val = results[f"intact_recurrence_N{h}_v0_noise"]["estimate"]
        res_sn = results[f"intact_recurrence_N{h}_v0_struct_vs_noise"]
        status = "Positive (Resolved)" if res_sn["ci_low"] > 0 else "Unresolved"
        md.append(f"| $N={h}$ | {w_val:+.2f} | {n_val:+.2f} | **{res_sn['estimate']:+.2f}** | **[{res_sn['ci_low']:+.2f}, {res_sn['ci_high']:+.2f}]** | {status} |")

    # Section 6: Leave-One-Family-Out (LOFO) Robustness
    md.append("\n## 6. Primary Endpoint ($N=2048$) Leave-One-Family-Out (LOFO) Robustness\n")
    md.append("| Left-Out Family | Remaining Pairs | $V_{\\text{intact}}^{(0)}(2048)$ (LOFO) | 95% Bootstrap CI | Robustness |")
    md.append("| :--- | :---: | :---: | :---: | :--- |")
    for lofo_k, l_data in meta["lofo_n2048"].items():
        fam = l_data["left_out_family"]
        rem = l_data["remaining_pairs"]
        est = l_data["estimate"]
        ci_l = l_data["ci_low"]
        ci_h = l_data["ci_high"]
        status = "Robustly Positive" if ci_l > 0 else ("Robustly Negative" if ci_h < 0 else "Unresolved")
        md.append(f"| `{fam}` | {rem} | **{est:+.2f}** | **[{ci_l:+.2f}, {ci_h:+.2f}]** | {status} |")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"[E13] Analysis complete! Wrote report to {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint S13 Controlled Recurrent Dynamics Analyzer")
    parser.add_argument("run_dir", type=str, help="Directory of the E13 run")
    parser.add_argument("--n_boot", type=int, default=10000, help="Number of bootstrap replicates")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analyze_run(args.run_dir, n_boot=args.n_boot, seed=args.seed)
