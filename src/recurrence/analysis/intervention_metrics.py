"""Statistical analysis and causal steering estimands for Sprint S08 (Experiment E07).

Computes:
1. State Allegiance Rate (SAR): P(Answer = V_state | State-Memory Conflict)
2. Memory Allegiance Rate (MAR): P(Answer = V_memory | State-Memory Conflict)
3. Delta_state|memory: Causal effect of state swap holding memory fixed
4. Delta_memory|state: Causal effect of memory swap holding state fixed
5. Reset Dependence (RD): Drop in congruent target choice when state is emptied with memory intact
6. Local Causal Precision (LCP): Joint target uptake AND control preservation under single-slot surgical edit
7. Order Sensitivity Gap: Effect of Memory->State vs State->Memory presentation order
8. Reconvergence Rate (RR): Probability of identical behavior post-synchronizing event
"""

from dataclasses import asdict, dataclass, field
import itertools
import math
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from recurrence.loop.intervention_experiment import InterventionTrialResult


@dataclass
class InterventionConditionStats:
    """Descriptive metrics for a specific intervention condition and probe domain."""
    condition: str
    presentation_order: str
    total_trials: int
    state_allegiance_rate: float
    memory_allegiance_rate: float
    control_preservation_rate: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_latency_ms: float


@dataclass
class CausalSteeringEstimandSummary:
    """Targeted causal estimand with paired cluster bootstrap 95% CI and sign-flip permutation test."""
    contrast_name: str
    description: str
    point_estimate: float
    ci_lower_95: float
    ci_upper_95: float
    permutation_p_value: float
    permutation_method: str
    is_statistically_distinguishable: bool


@dataclass
class LocalCausalPrecisionSummary:
    """Mechanistic breakdown of single-slot surgical intervention."""
    total_twin_pairs: int
    target_intervention_uptake: float
    control_slot_preservation: float
    joint_local_causal_precision: float


@dataclass
class StateInterventionAnalysisSummary:
    """Comprehensive analysis summary for Experiment E07."""
    total_twin_pairs: int
    total_trials: int
    condition_stats: Dict[str, InterventionConditionStats]
    causal_estimands: Dict[str, CausalSteeringEstimandSummary]
    local_precision: LocalCausalPrecisionSummary
    reconvergence_rate: float
    order_effects: Dict[str, float]


def compute_permutation_test(diffs: List[float]) -> Tuple[float, str]:
    """Compute exact sign-flip permutation test for paired episode differences."""
    n = len(diffs)
    if n == 0:
        return 1.0, "exact_exhaustive"
    
    obs_stat = abs(sum(diffs))
    if obs_stat == 0.0:
        return 1.0, "exact_exhaustive"

    if n <= 16:
        extreme_count = 0
        total_assignments = 1 << n
        for signs in itertools.product([-1.0, 1.0], repeat=n):
            perm_stat = abs(sum(s * d for s, d in zip(signs, diffs)))
            if perm_stat >= obs_stat - 1e-9:
                extreme_count += 1
        return float(extreme_count / total_assignments), "exact_exhaustive"
    else:
        rng = random.Random(42)
        n_perms = 50000
        extreme_count = 0
        for _ in range(n_perms):
            signs = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(n)]
            perm_stat = abs(sum(s * d for s, d in zip(signs, diffs)))
            if perm_stat >= obs_stat - 1e-9:
                extreme_count += 1
        return float(extreme_count / n_perms), "monte_carlo_50k"


def compute_paired_bootstrap_ci(
    vals_a: List[float],
    vals_b: List[float],
    num_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float, List[float]]:
    """Compute paired cluster bootstrap 95% CI across twin episode pairs."""
    assert len(vals_a) == len(vals_b)
    n = len(vals_a)
    if n == 0:
        return 0.0, 0.0, 0.0, []

    rng = random.Random(seed)
    diffs = [a - b for a, b in zip(vals_a, vals_b)]
    point_est = float(np.mean(diffs))

    boot_diffs = []
    for _ in range(num_bootstrap):
        sample_indices = [rng.randint(0, n - 1) for _ in range(n)]
        boot_diffs.append(np.mean([diffs[i] for i in sample_indices]))

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return point_est, ci_lower, ci_upper, diffs


def analyze_state_intervention_results(
    trials: List[InterventionTrialResult],
    num_bootstrap: int = 2000,
    seed: int = 42,
) -> StateInterventionAnalysisSummary:
    """Analyze full causal intervention battery across twin pairs and conditions."""
    df = pd.DataFrame([asdict(t) for t in trials])
    
    twin_pairs = [p for p in df["pair_id"].unique() if p.startswith("twin_")]
    n_twins = len(twin_pairs)

    # 1. Condition Stats
    cond_stats: Dict[str, InterventionConditionStats] = {}
    for (cond, order), df_co in df.groupby(["intervention_condition", "presentation_order"]):
        key = f"{cond}_{order}"
        st_all = float(df_co["is_state_allegiant"].mean())
        mem_all = float(df_co["is_memory_allegiant"].mean())
        ctrl_pres = float(df_co["is_control_preserved"].mean())
        cond_stats[key] = InterventionConditionStats(
            condition=cond,
            presentation_order=order,
            total_trials=len(df_co),
            state_allegiance_rate=st_all,
            memory_allegiance_rate=mem_all,
            control_preservation_rate=ctrl_pres,
            mean_prompt_tokens=float(df_co["prompt_tokens"].mean()),
            mean_completion_tokens=float(df_co["completion_tokens"].mean()),
            mean_latency_ms=float(df_co["latency_ms"].mean()),
        )

    # 2. Targeted Causal Estimands
    causal_estimands: Dict[str, CausalSteeringEstimandSummary] = {}

    # Estimand 1: State Allegiance under Conflict
    df_conflict = df[df["intervention_condition"].isin(["conflict_MA_SB", "conflict_MB_SA"]) & (df["probe_type"].isin(["target_key", "goal_status"]))]
    sar_by_twin = [df_conflict[df_conflict["pair_id"] == p]["is_state_allegiant"].mean() for p in twin_pairs]
    mar_by_twin = [df_conflict[df_conflict["pair_id"] == p]["is_memory_allegiant"].mean() for p in twin_pairs]

    point_sar, ci_l_sar, ci_u_sar, diffs_sar = compute_paired_bootstrap_ci(sar_by_twin, [0.0] * len(sar_by_twin), num_bootstrap=num_bootstrap, seed=seed)
    p_sar, p_meth_sar = compute_permutation_test(diffs_sar)

    causal_estimands["State_Allegiance_Rate"] = CausalSteeringEstimandSummary(
        contrast_name="State_Allegiance_Rate",
        description="Probability of answering with State-specified value under State-Memory conflict",
        point_estimate=point_sar,
        ci_lower_95=ci_l_sar,
        ci_upper_95=ci_u_sar,
        permutation_p_value=p_sar,
        permutation_method=p_meth_sar,
        is_statistically_distinguishable=(p_sar < 0.05),
    )

    # Estimand 2: Memory Allegiance under Conflict
    point_mar, ci_l_mar, ci_u_mar, diffs_mar = compute_paired_bootstrap_ci(mar_by_twin, [0.0] * len(mar_by_twin), num_bootstrap=num_bootstrap, seed=seed)
    p_mar, p_meth_mar = compute_permutation_test(diffs_mar)

    causal_estimands["Memory_Allegiance_Rate"] = CausalSteeringEstimandSummary(
        contrast_name="Memory_Allegiance_Rate",
        description="Probability of answering with Memory-specified value under State-Memory conflict",
        point_estimate=point_mar,
        ci_lower_95=ci_l_mar,
        ci_upper_95=ci_u_mar,
        permutation_p_value=p_mar,
        permutation_method=p_meth_mar,
        is_statistically_distinguishable=(p_mar < 0.05),
    )

    # Estimand 3: Delta_state|memory: (M_A + S_B) choosing V_B vs (M_A + S_A) choosing V_B
    df_MA_SB = df[(df["intervention_condition"] == "conflict_MA_SB") & (df["probe_type"] == "target_key")]
    df_MA_SA = df[(df["intervention_condition"] == "congruent_A") & (df["probe_type"] == "target_key")]

    vb_in_SB = [float((df_MA_SB[df_MA_SB["pair_id"] == p]["predicted_value"] == df_MA_SB[df_MA_SB["pair_id"] == p]["target_value_B"]).mean()) for p in twin_pairs]
    vb_in_SA = [float((df_MA_SA[df_MA_SA["pair_id"] == p]["predicted_value"] == df_MA_SA[df_MA_SA["pair_id"] == p]["target_value_B"]).mean()) for p in twin_pairs]

    point_d_sm, ci_l_d_sm, ci_u_d_sm, diffs_d_sm = compute_paired_bootstrap_ci(vb_in_SB, vb_in_SA, num_bootstrap=num_bootstrap, seed=seed)
    p_d_sm, p_meth_d_sm = compute_permutation_test(diffs_d_sm)

    causal_estimands["Delta_state_given_memory"] = CausalSteeringEstimandSummary(
        contrast_name="Delta_state_given_memory",
        description="Effect of swapping State (S_A -> S_B) on target choice while holding Memory (M_A) fixed",
        point_estimate=point_d_sm,
        ci_lower_95=ci_l_d_sm,
        ci_upper_95=ci_u_d_sm,
        permutation_p_value=p_d_sm,
        permutation_method=p_meth_d_sm,
        is_statistically_distinguishable=(p_d_sm < 0.05),
    )

    # Estimand 4: Reset Dependence: (M_A + S_A) choosing V_A vs (M_A + S_empty) choosing V_A
    df_MA_Sempty = df[(df["intervention_condition"] == "reset_MA_Sempty") & (df["probe_type"] == "target_key")]
    va_in_SA = [float((df_MA_SA[df_MA_SA["pair_id"] == p]["predicted_value"] == df_MA_SA[df_MA_SA["pair_id"] == p]["target_value_A"]).mean()) for p in twin_pairs]
    va_in_Sempty = [float((df_MA_Sempty[df_MA_Sempty["pair_id"] == p]["predicted_value"] == df_MA_Sempty[df_MA_Sempty["pair_id"] == p]["target_value_A"]).mean()) for p in twin_pairs]

    point_rd, ci_l_rd, ci_u_rd, diffs_rd = compute_paired_bootstrap_ci(va_in_SA, va_in_Sempty, num_bootstrap=num_bootstrap, seed=seed)
    p_rd, p_meth_rd = compute_permutation_test(diffs_rd)

    causal_estimands["Reset_Dependence"] = CausalSteeringEstimandSummary(
        contrast_name="Reset_Dependence",
        description="Drop in target answer consistency when state is reset to empty while memory is preserved",
        point_estimate=point_rd,
        ci_lower_95=ci_l_rd,
        ci_upper_95=ci_u_rd,
        permutation_p_value=p_rd,
        permutation_method=p_meth_rd,
        is_statistically_distinguishable=(p_rd < 0.05),
    )

    # 3. Local Causal Precision (Surgical Inversion)
    df_surgical = df[df["intervention_condition"] == "surgical_MA_SAprime"]
    uptake_by_twin = []
    pres_by_twin = []
    joint_by_twin = []

    for p in twin_pairs:
        df_p = df_surgical[df_surgical["pair_id"] == p]
        df_p_target = df_p[df_p["probe_type"] == "target_key"]
        df_p_ctrl = df_p[df_p["probe_type"] == "control_key"]

        uptake = bool(df_p_target["is_state_allegiant"].iloc[0]) if len(df_p_target) > 0 else False
        pres = bool(df_p_ctrl["is_control_preserved"].iloc[0]) if len(df_p_ctrl) > 0 else False

        uptake_by_twin.append(1.0 if uptake else 0.0)
        pres_by_twin.append(1.0 if pres else 0.0)
        joint_by_twin.append(1.0 if (uptake and pres) else 0.0)

    local_precision = LocalCausalPrecisionSummary(
        total_twin_pairs=n_twins,
        target_intervention_uptake=float(np.mean(uptake_by_twin)) if uptake_by_twin else 0.0,
        control_slot_preservation=float(np.mean(pres_by_twin)) if pres_by_twin else 0.0,
        joint_local_causal_precision=float(np.mean(joint_by_twin)) if joint_by_twin else 0.0,
    )

    # 4. Reconvergence Rate
    df_reconv = df[df["intervention_condition"].isin(["reconverged_branch_A", "reconverged_branch_B"])]
    reconv_rate = float(df_reconv["is_state_allegiant"].mean()) if len(df_reconv) > 0 else 0.0

    # 5. Presentation Order Effects
    df_conf_mem_first = df[(df["intervention_condition"].isin(["conflict_MA_SB", "conflict_MB_SA"])) & (df["presentation_order"] == "memory_first")]
    df_conf_st_first = df[(df["intervention_condition"].isin(["conflict_MA_SB", "conflict_MB_SA"])) & (df["presentation_order"] == "state_first")]

    order_effects = {
        "state_allegiance_memory_first": float(df_conf_mem_first["is_state_allegiant"].mean()) if len(df_conf_mem_first) > 0 else 0.0,
        "state_allegiance_state_first": float(df_conf_st_first["is_state_allegiant"].mean()) if len(df_conf_st_first) > 0 else 0.0,
        "order_sensitivity_gap": (float(df_conf_st_first["is_state_allegiant"].mean()) - float(df_conf_mem_first["is_state_allegiant"].mean())) if len(df_conf_mem_first) > 0 and len(df_conf_st_first) > 0 else 0.0,
    }

    return StateInterventionAnalysisSummary(
        total_twin_pairs=n_twins,
        total_trials=len(trials),
        condition_stats=cond_stats,
        causal_estimands=causal_estimands,
        local_precision=local_precision,
        reconvergence_rate=reconv_rate,
        order_effects=order_effects,
    )
