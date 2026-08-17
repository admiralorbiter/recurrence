"""Statistical analysis and causal steering estimands for Sprint S08 (Experiment E07).

Computes:
1. Delta_allegiance: Primary contrast (State Allegiance Rate - Memory Allegiance Rate)
2. Conflict 3-way partition: Follows State, Follows Memory, Neither / Other
3. Conditional State Preference: P(State | Answer is State-or-Memory candidate)
4. Directional conflict breakdown: (M_A + S_B) vs (M_B + S_A)
5. Full 2x2 Causal Contrasts:
   - Delta_state|memory_A: Effect of state swap S_A -> S_B holding M_A fixed
   - Delta_state|memory_B: Effect of state swap S_B -> S_A holding M_B fixed
   - Delta_memory|state_A: Effect of memory swap M_A -> M_B holding S_A fixed
   - Delta_memory|state_B: Effect of memory swap M_B -> M_A holding S_B fixed
   - Average Marginal State Effect & Average Marginal Memory Effect
6. Reset Dependence (RD): Drop in congruent target choice when state is emptied with memory intact
7. Local Causal Precision (LCP): Joint target uptake AND control preservation under single-slot surgical edit
8. Presentation Order Sensitivity Gap: Effect of Memory->State vs State->Memory
9. Reconvergence Concordance Rate: Paired behavioral concordance P(Answer_A == Answer_B) post-synchronization
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
    """Descriptive metrics for a specific intervention condition and presentation order."""
    condition: str
    presentation_order: str
    total_trials: int
    target_state_allegiance: float
    target_memory_allegiance: float
    goal_state_allegiance: float
    goal_memory_allegiance: float
    control_correctness: float
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
class ConflictPartitionSummary:
    """3-way response partition under State-Memory conflict."""
    total_conflict_trials: int
    follows_state_rate: float
    follows_memory_rate: float
    chooses_neither_rate: float
    conditional_state_preference: float  # P(State | State or Memory)
    delta_allegiance: float  # SAR - MAR
    directional_MA_SB_state_rate: float
    directional_MA_SB_memory_rate: float
    directional_MB_SA_state_rate: float
    directional_MB_SA_memory_rate: float


@dataclass
class StateInterventionAnalysisSummary:
    """Comprehensive analysis summary for Experiment E07."""
    total_twin_pairs: int
    total_trials: int
    condition_stats: Dict[str, InterventionConditionStats]
    causal_estimands: Dict[str, CausalSteeringEstimandSummary]
    conflict_partition: ConflictPartitionSummary
    local_precision: LocalCausalPrecisionSummary
    reconvergence_concordance_rate: float
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

    # 1. Disaggregated Condition Stats
    cond_stats: Dict[str, InterventionConditionStats] = {}
    for (cond, order), df_co in df.groupby(["intervention_condition", "presentation_order"]):
        key = f"{cond}_{order}"
        
        df_target = df_co[df_co["probe_type"] == "target_key"]
        df_ctrl = df_co[df_co["probe_type"] == "control_key"]
        df_goal = df_co[df_co["probe_type"] == "goal_status"]

        tgt_st = float(df_target["is_state_allegiant"].mean()) if len(df_target) > 0 else 0.0
        tgt_mem = float(df_target["is_memory_allegiant"].mean()) if len(df_target) > 0 else 0.0
        goal_st = float(df_goal["is_state_allegiant"].mean()) if len(df_goal) > 0 else 0.0
        goal_mem = float(df_goal["is_memory_allegiant"].mean()) if len(df_goal) > 0 else 0.0
        ctrl_corr = float(df_ctrl["is_control_preserved"].mean()) if len(df_ctrl) > 0 else 0.0

        cond_stats[key] = InterventionConditionStats(
            condition=cond,
            presentation_order=order,
            total_trials=len(df_co),
            target_state_allegiance=tgt_st,
            target_memory_allegiance=tgt_mem,
            goal_state_allegiance=goal_st,
            goal_memory_allegiance=goal_mem,
            control_correctness=ctrl_corr,
            mean_prompt_tokens=float(df_co["prompt_tokens"].mean()),
            mean_completion_tokens=float(df_co["completion_tokens"].mean()),
            mean_latency_ms=float(df_co["latency_ms"].mean()),
        )

    # 2. Conflict 3-Way Partition & Directional Breakdown
    df_conflict = df[df["intervention_condition"].isin(["conflict_MA_SB", "conflict_MB_SA"]) & (df["probe_type"].isin(["target_key", "goal_status"]))]
    tot_conf = len(df_conflict)
    
    st_count = int(df_conflict["is_state_allegiant"].sum())
    mem_count = int(df_conflict["is_memory_allegiant"].sum())
    neither_count = sum(1 for _, row in df_conflict.iterrows() if not row["is_state_allegiant"] and not row["is_memory_allegiant"])

    sar_all = st_count / max(1, tot_conf)
    mar_all = mem_count / max(1, tot_conf)
    neither_all = neither_count / max(1, tot_conf)
    cond_st_pref = st_count / max(1, st_count + mem_count)

    df_MA_SB = df_conflict[df_conflict["intervention_condition"] == "conflict_MA_SB"]
    df_MB_SA = df_conflict[df_conflict["intervention_condition"] == "conflict_MB_SA"]

    conflict_partition = ConflictPartitionSummary(
        total_conflict_trials=tot_conf,
        follows_state_rate=sar_all,
        follows_memory_rate=mar_all,
        chooses_neither_rate=neither_all,
        conditional_state_preference=cond_st_pref,
        delta_allegiance=sar_all - mar_all,
        directional_MA_SB_state_rate=float(df_MA_SB["is_state_allegiant"].mean()) if len(df_MA_SB) > 0 else 0.0,
        directional_MA_SB_memory_rate=float(df_MA_SB["is_memory_allegiant"].mean()) if len(df_MA_SB) > 0 else 0.0,
        directional_MB_SA_state_rate=float(df_MB_SA["is_state_allegiant"].mean()) if len(df_MB_SA) > 0 else 0.0,
        directional_MB_SA_memory_rate=float(df_MB_SA["is_memory_allegiant"].mean()) if len(df_MB_SA) > 0 else 0.0,
    )

    # 3. Targeted Causal Estimands (Full 2x2 Matrix)
    causal_estimands: Dict[str, CausalSteeringEstimandSummary] = {}

    # Estimand 1: Primary Conflict Contrast: Delta_allegiance = SAR - MAR
    sar_by_twin = [df_conflict[df_conflict["pair_id"] == p]["is_state_allegiant"].mean() for p in twin_pairs]
    mar_by_twin = [df_conflict[df_conflict["pair_id"] == p]["is_memory_allegiant"].mean() for p in twin_pairs]

    point_d_all, ci_l_all, ci_u_all, diffs_all = compute_paired_bootstrap_ci(sar_by_twin, mar_by_twin, num_bootstrap=num_bootstrap, seed=seed)
    p_d_all, p_meth_all = compute_permutation_test(diffs_all)

    causal_estimands["Delta_allegiance"] = CausalSteeringEstimandSummary(
        contrast_name="Delta_allegiance",
        description="Primary Conflict Contrast (State Allegiance Rate - Memory Allegiance Rate)",
        point_estimate=point_d_all,
        ci_lower_95=ci_l_all,
        ci_upper_95=ci_u_all,
        permutation_p_value=p_d_all,
        permutation_method=p_meth_all,
        is_statistically_distinguishable=(p_d_all < 0.05),
    )

    # Estimand 2: Delta_state|memory_A: (M_A + S_B) choosing V_B vs (M_A + S_A) choosing V_B holding M_A fixed
    df_MA_SB_tgt = df[(df["intervention_condition"] == "conflict_MA_SB") & (df["probe_type"] == "target_key")]
    df_MA_SA_tgt = df[(df["intervention_condition"] == "congruent_A") & (df["probe_type"] == "target_key")]

    vb_in_SB = [float((df_MA_SB_tgt[df_MA_SB_tgt["pair_id"] == p]["predicted_value"] == df_MA_SB_tgt[df_MA_SB_tgt["pair_id"] == p]["target_value_B"]).mean()) for p in twin_pairs]
    vb_in_SA = [float((df_MA_SA_tgt[df_MA_SA_tgt["pair_id"] == p]["predicted_value"] == df_MA_SA_tgt[df_MA_SA_tgt["pair_id"] == p]["target_value_B"]).mean()) for p in twin_pairs]

    point_d_sm_A, ci_l_d_sm_A, ci_u_d_sm_A, diffs_d_sm_A = compute_paired_bootstrap_ci(vb_in_SB, vb_in_SA, num_bootstrap=num_bootstrap, seed=seed)
    p_d_sm_A, p_meth_d_sm_A = compute_permutation_test(diffs_d_sm_A)

    causal_estimands["Delta_state_given_memory_A"] = CausalSteeringEstimandSummary(
        contrast_name="Delta_state_given_memory_A",
        description="Effect of swapping State (S_A -> S_B) on target choice holding Memory (M_A) fixed",
        point_estimate=point_d_sm_A,
        ci_lower_95=ci_l_d_sm_A,
        ci_upper_95=ci_u_d_sm_A,
        permutation_p_value=p_d_sm_A,
        permutation_method=p_meth_d_sm_A,
        is_statistically_distinguishable=(p_d_sm_A < 0.05),
    )

    # Estimand 3: Delta_state|memory_B: (M_B + S_A) choosing V_A vs (M_B + S_B) choosing V_A holding M_B fixed
    df_MB_SA_tgt = df[(df["intervention_condition"] == "conflict_MB_SA") & (df["probe_type"] == "target_key")]
    df_MB_SB_tgt = df[(df["intervention_condition"] == "congruent_B") & (df["probe_type"] == "target_key")]

    va_in_MB_SA = [float((df_MB_SA_tgt[df_MB_SA_tgt["pair_id"] == p]["predicted_value"] == df_MB_SA_tgt[df_MB_SA_tgt["pair_id"] == p]["target_value_A"]).mean()) for p in twin_pairs]
    va_in_MB_SB = [float((df_MB_SB_tgt[df_MB_SB_tgt["pair_id"] == p]["predicted_value"] == df_MB_SB_tgt[df_MB_SB_tgt["pair_id"] == p]["target_value_A"]).mean()) for p in twin_pairs]

    point_d_sm_B, ci_l_d_sm_B, ci_u_d_sm_B, diffs_d_sm_B = compute_paired_bootstrap_ci(va_in_MB_SA, va_in_MB_SB, num_bootstrap=num_bootstrap, seed=seed)
    p_d_sm_B, p_meth_d_sm_B = compute_permutation_test(diffs_d_sm_B)

    causal_estimands["Delta_state_given_memory_B"] = CausalSteeringEstimandSummary(
        contrast_name="Delta_state_given_memory_B",
        description="Effect of swapping State (S_B -> S_A) on target choice holding Memory (M_B) fixed",
        point_estimate=point_d_sm_B,
        ci_lower_95=ci_l_d_sm_B,
        ci_upper_95=ci_u_d_sm_B,
        permutation_p_value=p_d_sm_B,
        permutation_method=p_meth_d_sm_B,
        is_statistically_distinguishable=(p_d_sm_B < 0.05),
    )

    # Estimand 4: Delta_memory|state_A: (M_B + S_A) choosing V_B vs (M_A + S_A) choosing V_B holding S_A fixed
    point_d_ms_A, ci_l_d_ms_A, ci_u_d_ms_A, diffs_d_ms_A = compute_paired_bootstrap_ci(
        [float((df_MB_SA_tgt[df_MB_SA_tgt["pair_id"] == p]["predicted_value"] == df_MB_SA_tgt[df_MB_SA_tgt["pair_id"] == p]["target_value_B"]).mean()) for p in twin_pairs],
        vb_in_SA,
        num_bootstrap=num_bootstrap,
        seed=seed,
    )
    p_d_ms_A, p_meth_d_ms_A = compute_permutation_test(diffs_d_ms_A)

    causal_estimands["Delta_memory_given_state_A"] = CausalSteeringEstimandSummary(
        contrast_name="Delta_memory_given_state_A",
        description="Effect of swapping Memory (M_A -> M_B) on target choice holding State (S_A) fixed",
        point_estimate=point_d_ms_A,
        ci_lower_95=ci_l_d_ms_A,
        ci_upper_95=ci_u_d_ms_A,
        permutation_p_value=p_d_ms_A,
        permutation_method=p_meth_d_ms_A,
        is_statistically_distinguishable=(p_d_ms_A < 0.05),
    )

    # Estimand 5: Delta_memory|state_B: (M_A + S_B) choosing V_A vs (M_B + S_B) choosing V_A holding S_B fixed
    va_in_MA_SB = [float((df_MA_SB_tgt[df_MA_SB_tgt["pair_id"] == p]["predicted_value"] == df_MA_SB_tgt[df_MA_SB_tgt["pair_id"] == p]["target_value_A"]).mean()) for p in twin_pairs]
    point_d_ms_B, ci_l_d_ms_B, ci_u_d_ms_B, diffs_d_ms_B = compute_paired_bootstrap_ci(
        va_in_MA_SB,
        va_in_MB_SB,
        num_bootstrap=num_bootstrap,
        seed=seed,
    )
    p_d_ms_B, p_meth_d_ms_B = compute_permutation_test(diffs_d_ms_B)

    causal_estimands["Delta_memory_given_state_B"] = CausalSteeringEstimandSummary(
        contrast_name="Delta_memory_given_state_B",
        description="Effect of swapping Memory (M_B -> M_A) on target choice holding State (S_B) fixed",
        point_estimate=point_d_ms_B,
        ci_lower_95=ci_l_d_ms_B,
        ci_upper_95=ci_u_d_ms_B,
        permutation_p_value=p_d_ms_B,
        permutation_method=p_meth_d_ms_B,
        is_statistically_distinguishable=(p_d_ms_B < 0.05),
    )

    # Estimand 6: Average Marginal Effects for State and Memory
    avg_d_state = [(a + b) / 2.0 for a, b in zip(diffs_d_sm_A, diffs_d_sm_B)]
    point_avg_st, ci_l_avg_st, ci_u_avg_st, _ = compute_paired_bootstrap_ci(avg_d_state, [0.0] * len(avg_d_state), num_bootstrap=num_bootstrap, seed=seed)
    p_avg_st, p_meth_avg_st = compute_permutation_test(avg_d_state)

    causal_estimands["Average_Marginal_State_Effect"] = CausalSteeringEstimandSummary(
        contrast_name="Average_Marginal_State_Effect",
        description="Pooled Average Marginal Effect of State Swaps across both memory contexts",
        point_estimate=point_avg_st,
        ci_lower_95=ci_l_avg_st,
        ci_upper_95=ci_u_avg_st,
        permutation_p_value=p_avg_st,
        permutation_method=p_meth_avg_st,
        is_statistically_distinguishable=(p_avg_st < 0.05),
    )

    avg_d_mem = [(a + b) / 2.0 for a, b in zip(diffs_d_ms_A, diffs_d_ms_B)]
    point_avg_mem, ci_l_avg_mem, ci_u_avg_mem, _ = compute_paired_bootstrap_ci(avg_d_mem, [0.0] * len(avg_d_mem), num_bootstrap=num_bootstrap, seed=seed)
    p_avg_mem, p_meth_avg_mem = compute_permutation_test(avg_d_mem)

    causal_estimands["Average_Marginal_Memory_Effect"] = CausalSteeringEstimandSummary(
        contrast_name="Average_Marginal_Memory_Effect",
        description="Pooled Average Marginal Effect of Memory Swaps across both state contexts",
        point_estimate=point_avg_mem,
        ci_lower_95=ci_l_avg_mem,
        ci_upper_95=ci_u_avg_mem,
        permutation_p_value=p_avg_mem,
        permutation_method=p_meth_avg_mem,
        is_statistically_distinguishable=(p_avg_mem < 0.05),
    )

    # Estimand 7: Reset Dependence: (M_A + S_A) choosing V_A vs (M_A + S_empty) choosing V_A
    df_MA_Sempty = df[(df["intervention_condition"] == "reset_MA_Sempty") & (df["probe_type"] == "target_key")]
    va_in_SA = [float((df_MA_SA_tgt[df_MA_SA_tgt["pair_id"] == p]["predicted_value"] == df_MA_SA_tgt[df_MA_SA_tgt["pair_id"] == p]["target_value_A"]).mean()) for p in twin_pairs]
    va_in_Sempty = [float((df_MA_Sempty[df_MA_Sempty["pair_id"] == p]["predicted_value"] == df_MA_Sempty[df_MA_Sempty["pair_id"] == p]["target_value_A"]).mean()) for p in twin_pairs]

    point_rd, ci_l_rd, ci_u_rd, diffs_rd = compute_paired_bootstrap_ci(va_in_SA, va_in_Sempty, num_bootstrap=num_bootstrap, seed=seed)
    p_rd, p_meth_rd = compute_permutation_test(diffs_rd)

    causal_estimands["Reset_Dependence"] = CausalSteeringEstimandSummary(
        contrast_name="Reset_Dependence",
        description="Drop in target answer consistency when state is reset to empty with memory preserved",
        point_estimate=point_rd,
        ci_lower_95=ci_l_rd,
        ci_upper_95=ci_u_rd,
        permutation_p_value=p_rd,
        permutation_method=p_meth_rd,
        is_statistically_distinguishable=(p_rd < 0.05),
    )

    # 4. Local Causal Precision (Surgical Inversion)
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

    # 5. Genuine Behavioral Reconvergence Concordance Rate
    df_reconv_A = df[df["intervention_condition"] == "reconverged_branch_A"]
    df_reconv_B = df[df["intervention_condition"] == "reconverged_branch_B"]

    concordance_matches = 0
    total_reconv_pairs = 0
    reconv_specs = df_reconv_A["pair_id"].unique()

    for spec_id in reconv_specs:
        ans_A = df_reconv_A[df_reconv_A["pair_id"] == spec_id]["predicted_value"].tolist()
        ans_B = df_reconv_B[df_reconv_B["pair_id"] == spec_id]["predicted_value"].tolist()
        for a, b in zip(ans_A, ans_B):
            total_reconv_pairs += 1
            if a == b:
                concordance_matches += 1

    reconv_concordance_rate = (concordance_matches / max(1, total_reconv_pairs)) if total_reconv_pairs > 0 else 1.0

    # 6. Presentation Order Effects
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
        conflict_partition=conflict_partition,
        local_precision=local_precision,
        reconvergence_concordance_rate=reconv_concordance_rate,
        order_effects=order_effects,
    )
