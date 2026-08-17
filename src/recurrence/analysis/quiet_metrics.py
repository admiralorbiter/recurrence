"""Statistical analysis, regime-specific estimands, and derived inference metrics for Sprint S07.1 (Experiment E06b).

Evaluates:
1. Delta_derivation-available: Multi-hop difference under selective reflection when complete evidence was available pre-null
2. Delta_derivation-missing: Multi-hop difference when evidence was incomplete pre-null (resistance to premature inference)
3. Delta_derivation-nowrite-avail: Persistent writing vs discarded semantic invocation control
4. Delta_goal-consolidation: Goal verification difference across quiet intervals
5. Delta_clock-cue: Sensitivity to elapsed timestamp cues
6. Delta_evidence-integrity: Retention of unaffected working memory facts
7. Persistent Derivation Write Failure / Derived-State Consolidation Failure metrics (precision, recall, hallucination rate)
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import itertools
import math
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from recurrence.loop.quiet_experiment import QuietTrialResult, ReflectionTickTrace


@dataclass
class QuietConditionStats:
    """Performance and computational cost metrics for a specific experimental condition (pooled strictly over K > 0)."""
    condition: str
    total_trials: int
    correct_trials: int
    accuracy_micro: float
    probe_accuracies: Dict[str, float]
    mean_query_prompt_tokens: float
    mean_amortized_prompt_tokens: float
    mean_query_latency_ms: float
    mean_amortized_latency_ms: float


@dataclass
class QuietCausalEstimandSummary:
    """Summary of a targeted causal contrast with cluster bootstrap CI and exact permutation tests."""
    contrast_name: str
    target_probe_domain: str
    regime: str  # 'available_inference', 'missing_premise_control', 'all'
    condition_a: str
    condition_b: str
    delta_accuracy: float
    ci_lower_95: float
    ci_upper_95: float
    discordance_b: int
    discordance_c: int
    exact_mcnemar_p_value: float
    permutation_p_value: float
    permutation_method: str
    is_statistically_distinguishable: bool


@dataclass
class DerivedInferenceMetrics:
    """Mechanistic quality metrics for model-generated derived inferences."""
    total_reflection_ticks: int
    valid_schema_rate: float
    selective_schema_valid_rate: float
    unconstrained_schema_valid_rate: float
    semantic_nowrite_schema_valid_rate: float
    overall_schema_valid_rate: float
    inferences_written_available_regime: int
    correct_inferences_available_regime: int
    derived_precision_available: float
    derived_recall_available: float
    inferences_written_missing_regime: int
    premature_hallucination_rate_missing: float


@dataclass
class IntervalContrastSummary:
    """Contrast summary evaluated specifically within a single quiet interval duration K."""
    interval_k: int
    contrast_name: str
    target_probe_domain: str
    regime: str
    condition_a: str
    condition_b: str
    delta_accuracy: float
    ci_lower_95: float
    ci_upper_95: float
    exact_mcnemar_p_value: float
    permutation_p_value: float
    permutation_method: str


@dataclass
class QuietIntervalAnalysisSummary:
    """Comprehensive analysis summary for Experiment E06b."""
    total_episodes: int
    total_trials: int
    intervals_evaluated: List[int]
    baseline_k0_accuracies: Dict[str, float]
    condition_stats: Dict[str, QuietConditionStats]
    causal_estimands: Dict[str, QuietCausalEstimandSummary]
    derived_metrics: DerivedInferenceMetrics
    interval_breakdown: Dict[int, Dict[str, float]]
    interval_probe_breakdown: Dict[int, Dict[str, Dict[str, float]]]
    interval_contrasts: List[IntervalContrastSummary]
    unconstrained_evidence_drift_rate: float
    selective_evidence_drift_rate: float


def compute_exact_mcnemar_test(
    paired_outcomes_a: List[bool],
    paired_outcomes_b: List[bool],
) -> Tuple[int, int, float]:
    """Compute exact two-sided binomial McNemar test for paired binary outcomes."""
    assert len(paired_outcomes_a) == len(paired_outcomes_b)
    b = 0
    c = 0
    for ya, yb in zip(paired_outcomes_a, paired_outcomes_b):
        if ya and not yb:
            b += 1
        elif not ya and yb:
            c += 1

    n_disc = b + c
    if n_disc == 0:
        return 0, 0, 1.0

    k = min(b, c)
    prob_tail = sum(math.comb(n_disc, i) for i in range(k + 1)) * (0.5 ** n_disc)
    p_val = min(1.0, 2.0 * prob_tail)
    return b, c, float(p_val)


def compute_permutation_test(
    diffs: List[float],
) -> Tuple[float, str]:
    """Compute exact sign-flip permutation p-value (exact for N <= 16, Monte Carlo for N > 16)."""
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


def compute_episode_clustered_bootstrap(
    df: pd.DataFrame,
    cond_a: str,
    cond_b: str,
    probe_filter: Optional[str] = None,
    regime_filter: Optional[str] = None,
    num_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float, List[float]]:
    """Compute episode-clustered paired bootstrap 95% CI for Delta = Acc_A - Acc_B."""
    rng = random.Random(seed)
    
    df_sub = df
    if probe_filter is not None:
        df_sub = df_sub[df_sub["probe_type"] == probe_filter]
    if regime_filter is not None:
        df_sub = df_sub[df_sub["regime"] == regime_filter]

    episodes = df_sub["episode_id"].unique()
    n_episodes = len(episodes)
    if n_episodes == 0:
        return 0.0, 0.0, 0.0, []

    ep_acc_a: Dict[str, float] = {}
    ep_acc_b: Dict[str, float] = {}
    ep_diffs: List[float] = []

    for ep in episodes:
        df_ep = df_sub[df_sub["episode_id"] == ep]
        acc_a = df_ep[df_ep["condition"] == cond_a]["is_correct"].mean()
        acc_b = df_ep[df_ep["condition"] == cond_b]["is_correct"].mean()
        val_a = float(acc_a) if not pd.isna(acc_a) else 0.0
        val_b = float(acc_b) if not pd.isna(acc_b) else 0.0
        ep_acc_a[ep] = val_a
        ep_acc_b[ep] = val_b
        ep_diffs.append(val_a - val_b)

    point_delta = np.mean([ep_acc_a[ep] for ep in episodes]) - np.mean([ep_acc_b[ep] for ep in episodes])

    boot_deltas: List[float] = []
    for _ in range(num_bootstrap):
        sampled_eps = [episodes[rng.randint(0, n_episodes - 1)] for _ in range(n_episodes)]
        mean_a = np.mean([ep_acc_a[ep] for ep in sampled_eps])
        mean_b = np.mean([ep_acc_b[ep] for ep in sampled_eps])
        boot_deltas.append(mean_a - mean_b)

    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))

    return float(point_delta), ci_lower, ci_upper, ep_diffs


def compute_derived_inference_metrics(
    reflection_traces: List[ReflectionTickTrace],
    trials: List[QuietTrialResult],
) -> DerivedInferenceMetrics:
    """Analyze reflection traces to evaluate derivation precision, recall, and premature hallucination rate."""
    if not reflection_traces:
        return DerivedInferenceMetrics(
            total_reflection_ticks=0,
            valid_schema_rate=1.0,
            selective_schema_valid_rate=1.0,
            unconstrained_schema_valid_rate=1.0,
            semantic_nowrite_schema_valid_rate=1.0,
            overall_schema_valid_rate=1.0,
            inferences_written_available_regime=0,
            correct_inferences_available_regime=0,
            derived_precision_available=1.0,
            derived_recall_available=1.0,
            inferences_written_missing_regime=0,
            premature_hallucination_rate_missing=0.0,
        )

    # Build ground-truth mapping for root_key -> terminal_val from trials
    ep_truth: Dict[str, Tuple[str, str]] = {}
    for t in trials:
        if t.probe_type == "derivation_multihop":
            ep_truth[t.episode_id] = (t.metadata.get("root_key", ""), t.metadata.get("terminal_val", ""))

    sel_traces = [tr for tr in reflection_traces if tr.condition == "selective_reflection"]
    uncon_traces = [tr for tr in reflection_traces if tr.condition == "unconstrained_reflection"]
    nowrite_traces = [tr for tr in reflection_traces if tr.condition == "semantic_no_write"]

    sel_valid_rate = sum(1 for tr in sel_traces if tr.schema_valid) / max(1, len(sel_traces))
    uncon_valid_rate = sum(1 for tr in uncon_traces if tr.schema_valid) / max(1, len(uncon_traces))
    nowrite_valid_rate = sum(1 for tr in nowrite_traces if tr.schema_valid) / max(1, len(nowrite_traces))
    overall_valid_rate = sum(1 for tr in reflection_traces if tr.schema_valid) / max(1, len(reflection_traces))

    avail_traces = [tr for tr in sel_traces if tr.regime == "available_inference"]
    missing_traces = [tr for tr in sel_traces if tr.regime == "missing_premise_control"]

    avail_inferences_total = 0
    avail_inferences_correct = 0
    episodes_with_correct_avail = set()
    avail_episodes = set(tr.episode_id for tr in avail_traces)

    for tr in avail_traces:
        inferences = tr.parsed_write.get("derived_inferences", {})
        root_k, term_v = ep_truth.get(tr.episode_id, ("", ""))
        for k, v in inferences.items():
            avail_inferences_total += 1
            if k == root_k and v == term_v:
                avail_inferences_correct += 1
                episodes_with_correct_avail.add(tr.episode_id)

    prec_avail = (avail_inferences_correct / max(1, avail_inferences_total)) if avail_inferences_total > 0 else 0.0
    rec_avail = len(episodes_with_correct_avail) / max(1, len(avail_episodes))

    missing_inferences_total = 0
    for tr in missing_traces:
        inferences = tr.parsed_write.get("derived_inferences", {})
        root_k, term_v = ep_truth.get(tr.episode_id, ("", ""))
        for k, v in inferences.items():
            if k == root_k or v == term_v:
                missing_inferences_total += 1

    missing_episodes = set(tr.episode_id for tr in missing_traces)
    halluc_rate = (missing_inferences_total / max(1, len(missing_episodes)))

    return DerivedInferenceMetrics(
        total_reflection_ticks=len(reflection_traces),
        valid_schema_rate=sel_valid_rate,
        selective_schema_valid_rate=sel_valid_rate,
        unconstrained_schema_valid_rate=uncon_valid_rate,
        semantic_nowrite_schema_valid_rate=nowrite_valid_rate,
        overall_schema_valid_rate=overall_valid_rate,
        inferences_written_available_regime=avail_inferences_total,
        correct_inferences_available_regime=avail_inferences_correct,
        derived_precision_available=prec_avail,
        derived_recall_available=rec_avail,
        inferences_written_missing_regime=missing_inferences_total,
        premature_hallucination_rate_missing=halluc_rate,
    )


def analyze_quiet_interval_results(
    trials: List[QuietTrialResult],
    reflection_traces: Optional[List[ReflectionTickTrace]] = None,
    num_bootstrap: int = 2000,
    seed: int = 42,
) -> QuietIntervalAnalysisSummary:
    """Perform comprehensive statistical analysis across conditions, regimes, and quiet intervals."""
    df = pd.DataFrame([asdict(t) for t in trials])
    
    episodes = df["episode_id"].unique().tolist()
    intervals = sorted(df["interval_k"].unique().tolist())
    conditions = df["condition"].unique().tolist()

    # K=0 baseline accuracies
    df_k0 = df[df["interval_k"] == 0]
    k0_accs = {cond: float(df_k0[df_k0["condition"] == cond]["is_correct"].mean()) for cond in df_k0["condition"].unique()}

    # 1. Condition Stats Pooled Strictly Over K > 0
    df_k_pos = df[df["interval_k"] > 0]
    condition_stats: Dict[str, QuietConditionStats] = {}
    for cond in conditions:
        df_c = df_k_pos[df_k_pos["condition"] == cond]
        if len(df_c) == 0:
            continue
        total_n = len(df_c)
        corr_n = int(df_c["is_correct"].sum())
        micro_acc = corr_n / max(1, total_n)

        probe_accs = {}
        for ptype in df_c["probe_type"].unique():
            df_cp = df_c[df_c["probe_type"] == ptype]
            probe_accs[ptype] = float(df_cp["is_correct"].mean())

        condition_stats[cond] = QuietConditionStats(
            condition=cond,
            total_trials=total_n,
            correct_trials=corr_n,
            accuracy_micro=micro_acc,
            probe_accuracies=probe_accs,
            mean_query_prompt_tokens=float(df_c["query_prompt_tokens"].mean()),
            mean_amortized_prompt_tokens=float(df_c["amortized_prompt_tokens"].mean()),
            mean_query_latency_ms=float(df_c["query_latency_ms"].mean()),
            mean_amortized_latency_ms=float(df_c["amortized_latency_ms"].mean()),
        )

    # 2. Targeted Causal Estimands
    contrasts = [
        ("Delta_derivation-available", "derivation_multihop", "available_inference", "selective_reflection", "strict_identity"),
        ("Delta_derivation-missing", "derivation_multihop", "missing_premise_control", "selective_reflection", "strict_identity"),
        ("Delta_derivation-nowrite-avail", "derivation_multihop", "available_inference", "selective_reflection", "semantic_no_write"),
        ("Delta_goal-consolidation", "goal_activation", "all", "selective_reflection", "strict_identity"),
        ("Delta_clock-cue", "all", "all", "clock_only", "strict_identity"),
        ("Delta_evidence-integrity", "stable_kv", "all", "selective_reflection", "strict_identity"),
        ("Delta_unconstrained-drift", "all", "all", "unconstrained_reflection", "strict_identity"),
    ]

    causal_estimands: Dict[str, QuietCausalEstimandSummary] = {}
    for name, pfilter, rfilter, c_a, c_b in contrasts:
        probe_f = None if pfilter == "all" else pfilter
        regime_f = None if rfilter == "all" else rfilter

        if c_a in conditions and c_b in conditions:
            point_d, ci_l, ci_u, ep_diffs = compute_episode_clustered_bootstrap(
                df=df_k_pos, cond_a=c_a, cond_b=c_b, probe_filter=probe_f, regime_filter=regime_f, num_bootstrap=num_bootstrap, seed=seed
            )

            df_sub = df_k_pos
            if probe_f is not None:
                df_sub = df_sub[df_sub["probe_type"] == probe_f]
            if regime_f is not None:
                df_sub = df_sub[df_sub["regime"] == regime_f]

            df_a = df_sub[df_sub["condition"] == c_a].sort_values(["episode_id", "interval_k", "probe_id"])
            df_b = df_sub[df_sub["condition"] == c_b].sort_values(["episode_id", "interval_k", "probe_id"])

            b, c, mcnemar_p = compute_exact_mcnemar_test(
                df_a["is_correct"].tolist(),
                df_b["is_correct"].tolist(),
            )
            perm_p, perm_method = compute_permutation_test(ep_diffs)
            is_dist = (perm_p < 0.05)

            causal_estimands[name] = QuietCausalEstimandSummary(
                contrast_name=name,
                target_probe_domain=pfilter,
                regime=rfilter,
                condition_a=c_a,
                condition_b=c_b,
                delta_accuracy=point_d,
                ci_lower_95=ci_l,
                ci_upper_95=ci_u,
                discordance_b=b,
                discordance_c=c,
                exact_mcnemar_p_value=mcnemar_p,
                permutation_p_value=perm_p,
                permutation_method=perm_method,
                is_statistically_distinguishable=is_dist,
            )

    # 3. Mechanistic Derived Inference Metrics
    derived_metrics = compute_derived_inference_metrics(reflection_traces or [], trials)

    # 4. Interval Breakdown
    interval_breakdown: Dict[int, Dict[str, float]] = {}
    interval_probe_breakdown: Dict[int, Dict[str, Dict[str, float]]] = {}
    interval_contrasts: List[IntervalContrastSummary] = []

    for k in intervals:
        df_k = df[df["interval_k"] == k]
        k_dict: Dict[str, float] = {}
        k_probe_dict: Dict[str, Dict[str, float]] = {}

        for cond in conditions:
            df_kc = df_k[df_k["condition"] == cond]
            if len(df_kc) > 0:
                k_dict[cond] = float(df_kc["is_correct"].mean())
                k_probe_dict[cond] = {
                    pt: float(df_kc[df_kc["probe_type"] == pt]["is_correct"].mean())
                    for pt in df_kc["probe_type"].unique()
                }

        interval_breakdown[k] = k_dict
        interval_probe_breakdown[k] = k_probe_dict

        if k > 0 and "selective_reflection" in conditions and "strict_identity" in conditions:
            pt_d, ci_l, ci_u, ep_d_k = compute_episode_clustered_bootstrap(
                df=df_k, cond_a="selective_reflection", cond_b="strict_identity", probe_filter="derivation_multihop", regime_filter="available_inference", num_bootstrap=num_bootstrap, seed=seed
            )
            df_ka = df_k[(df_k["condition"] == "selective_reflection") & (df_k["probe_type"] == "derivation_multihop") & (df_k["regime"] == "available_inference")].sort_values(["episode_id", "probe_id"])
            df_kb = df_k[(df_k["condition"] == "strict_identity") & (df_k["probe_type"] == "derivation_multihop") & (df_k["regime"] == "available_inference")].sort_values(["episode_id", "probe_id"])
            _, _, mcn_p = compute_exact_mcnemar_test(df_ka["is_correct"].tolist(), df_kb["is_correct"].tolist())
            perm_p_k, perm_meth_k = compute_permutation_test(ep_d_k)

            interval_contrasts.append(IntervalContrastSummary(
                interval_k=k,
                contrast_name="Delta_derivation-available",
                target_probe_domain="derivation_multihop",
                regime="available_inference",
                condition_a="selective_reflection",
                condition_b="strict_identity",
                delta_accuracy=pt_d,
                ci_lower_95=ci_l,
                ci_upper_95=ci_u,
                exact_mcnemar_p_value=mcn_p,
                permutation_p_value=perm_p_k,
                permutation_method=perm_meth_k,
            ))

    # 5. Evidence Drift Rates
    df_uncon = df[df["condition"] == "unconstrained_reflection"]
    uncon_drift_rate = float(df_uncon["evidence_drift_detected"].mean()) if len(df_uncon) > 0 else 0.0

    df_sel = df[df["condition"] == "selective_reflection"]
    sel_drift_rate = float(df_sel["evidence_drift_detected"].mean()) if len(df_sel) > 0 else 0.0

    return QuietIntervalAnalysisSummary(
        total_episodes=len(episodes),
        total_trials=len(trials),
        intervals_evaluated=intervals,
        baseline_k0_accuracies=k0_accs,
        condition_stats=condition_stats,
        causal_estimands=causal_estimands,
        derived_metrics=derived_metrics,
        interval_breakdown=interval_breakdown,
        interval_probe_breakdown=interval_probe_breakdown,
        interval_contrasts=interval_contrasts,
        unconstrained_evidence_drift_rate=uncon_drift_rate,
        selective_evidence_drift_rate=sel_drift_rate,
    )
