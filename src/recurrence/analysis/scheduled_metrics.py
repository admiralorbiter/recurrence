"""Statistical analysis, causal estimands, and exact statistical estimators for Experiment E05c.

Computes:
1. Delta_online-direct (incremental_state vs replay_transcript)
2. Delta_reconstruction (incremental_state vs replay_state_model)
3. Delta_schedule (incremental_state vs replay_state_deterministic)
4. Delta_representation (replay_state_deterministic vs replay_transcript)
5. Episode-clustered paired bootstrap (95% CI)
6. True exact two-sided McNemar binomial tests
7. Exact sign-flip permutation tests (N <= 16) / Monte Carlo permutation tests (N > 16)
8. Horizon-specific paired contrasts and scaling statistics
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import itertools
import math
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from recurrence.loop.scheduled_experiment import ScheduledTrialResult


@dataclass
class ConditionAccuracyStats:
    """Accuracy and token metrics for a single experimental condition."""
    condition: str
    total_trials: int
    correct_trials: int
    accuracy_micro: float
    accuracy_macro_by_probe: float
    probe_accuracies: Dict[str, float]
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_latency_ms: float
    mean_amortized_prompt_tokens: float
    mean_amortized_latency_ms: float


@dataclass
class CausalEstimandSummary:
    """Summary of a causal contrast (Delta = Acc_A - Acc_B) with paired bootstrap CI and exact tests."""
    contrast_name: str
    condition_a: str
    condition_b: str
    delta_accuracy: float
    ci_lower_95: float
    ci_upper_95: float
    discordance_b: int  # A correct, B incorrect
    discordance_c: int  # A incorrect, B correct
    exact_mcnemar_p_value: float
    permutation_p_value: float
    permutation_method: str  # 'exact_exhaustive' or 'monte_carlo_50k'
    is_statistically_distinguishable: bool


@dataclass
class HorizonContrastSummary:
    """Paired contrast summary evaluated specifically within a single horizon."""
    horizon_ticks: int
    contrast_name: str
    condition_a: str
    condition_b: str
    delta_accuracy: float
    ci_lower_95: float
    ci_upper_95: float
    exact_mcnemar_p_value: float


@dataclass
class ScheduledReplayAnalysisSummary:
    """Comprehensive S06.2 benchmark analysis summary across horizons and conditions."""
    total_episodes: int
    total_trials: int
    horizons_evaluated: List[int]
    condition_stats: Dict[str, ConditionAccuracyStats]
    causal_estimands: Dict[str, CausalEstimandSummary]
    horizon_breakdown: Dict[int, Dict[str, float]]
    horizon_contrasts: List[HorizonContrastSummary]
    token_cost_crossover_ticks: Optional[float]
    descriptive_accuracy_crossover_ticks: Optional[int]


def compute_exact_mcnemar_test(
    paired_outcomes_a: List[bool],
    paired_outcomes_b: List[bool],
) -> Tuple[int, int, float]:
    """Compute exact two-sided binomial McNemar test for paired binary outcomes."""
    assert len(paired_outcomes_a) == len(paired_outcomes_b)
    
    b = 0  # A correct, B incorrect
    c = 0  # A incorrect, B correct

    for ya, yb in zip(paired_outcomes_a, paired_outcomes_b):
        if ya and not yb:
            b += 1
        elif not ya and yb:
            c += 1

    n_disc = b + c
    if n_disc == 0:
        return 0, 0, 1.0

    k = min(b, c)
    # Exact two-tailed binomial CDF: 2 * sum(comb(n, i) * 0.5^n for i in 0..k)
    prob_tail = sum(math.comb(n_disc, i) for i in range(k + 1)) * (0.5 ** n_disc)
    p_val = min(1.0, 2.0 * prob_tail)
    
    return b, c, float(p_val)


def compute_permutation_test(
    diffs: List[float],
) -> Tuple[float, str]:
    """Compute sign-flip permutation p-value (exact for N <= 16, Monte Carlo for N > 16)."""
    n = len(diffs)
    if n == 0:
        return 1.0, "exact_exhaustive"
    
    obs_stat = abs(sum(diffs))
    if obs_stat == 0.0:
        return 1.0, "exact_exhaustive"

    if n <= 16:
        # Full exhaustive exact permutation
        extreme_count = 0
        total_assignments = 1 << n  # 2^n
        for signs in itertools.product([-1.0, 1.0], repeat=n):
            perm_stat = abs(sum(s * d for s, d in zip(signs, diffs)))
            if perm_stat >= obs_stat - 1e-9:
                extreme_count += 1
        return float(extreme_count / total_assignments), "exact_exhaustive"
    else:
        # Monte Carlo permutation test for n > 16 (B=50,000)
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
    num_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float, List[float]]:
    """Compute episode-clustered paired bootstrap 95% confidence interval for Delta = Acc_A - Acc_B."""
    rng = random.Random(seed)
    
    episodes = df["episode_id"].unique()
    n_episodes = len(episodes)
    if n_episodes == 0:
        return 0.0, 0.0, 0.0, []

    ep_acc_a: Dict[str, float] = {}
    ep_acc_b: Dict[str, float] = {}
    ep_diffs: List[float] = []

    for ep in episodes:
        df_ep = df[df["episode_id"] == ep]
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


def analyze_scheduled_replay_results(
    trials: List[ScheduledTrialResult],
    num_bootstrap: int = 2000,
    seed: int = 42,
) -> ScheduledReplayAnalysisSummary:
    """Perform comprehensive statistical and causal analysis of Experiment E05c trials."""
    df = pd.DataFrame([asdict(t) for t in trials])
    
    episodes = df["episode_id"].unique().tolist()
    horizons = sorted(df["horizon_ticks"].unique().tolist())
    conditions = df["condition"].unique().tolist()

    # 1. Condition Accuracy & Cost Profiles
    condition_stats: Dict[str, ConditionAccuracyStats] = {}
    for cond in conditions:
        df_c = df[df["condition"] == cond]
        total_n = len(df_c)
        corr_n = int(df_c["is_correct"].sum())
        micro_acc = corr_n / max(1, total_n)

        probe_accs = {}
        for ptype in df_c["probe_type"].unique():
            df_cp = df_c[df_c["probe_type"] == ptype]
            probe_accs[ptype] = float(df_cp["is_correct"].mean())

        macro_acc = float(np.mean(list(probe_accs.values()))) if probe_accs else micro_acc
        p_tok = float(df_c["prompt_tokens"].mean())
        c_tok = float(df_c["completion_tokens"].mean())
        lat = float(df_c["latency_ms"].mean())
        amort_tok = float(df_c["amortized_prompt_tokens"].mean())
        amort_lat = float(df_c["amortized_latency_ms"].mean())

        condition_stats[cond] = ConditionAccuracyStats(
            condition=cond,
            total_trials=total_n,
            correct_trials=corr_n,
            accuracy_micro=micro_acc,
            accuracy_macro_by_probe=macro_acc,
            probe_accuracies=probe_accs,
            mean_prompt_tokens=p_tok,
            mean_completion_tokens=c_tok,
            mean_latency_ms=lat,
            mean_amortized_prompt_tokens=amort_tok,
            mean_amortized_latency_ms=amort_lat,
        )

    # 2. Causal Estimands & Exact Inferences
    contrasts = [
        ("Delta_online-direct", "incremental_state", "replay_transcript"),
        ("Delta_reconstruction", "incremental_state", "replay_state_model"),
        ("Delta_schedule", "incremental_state", "replay_state_deterministic"),
        ("Delta_representation", "replay_state_deterministic", "replay_transcript"),
    ]

    causal_estimands: Dict[str, CausalEstimandSummary] = {}
    for name, c_a, c_b in contrasts:
        if c_a in conditions and c_b in conditions:
            point_d, ci_l, ci_u, ep_diffs = compute_episode_clustered_bootstrap(
                df=df, cond_a=c_a, cond_b=c_b, num_bootstrap=num_bootstrap, seed=seed
            )
            # Exact paired tests
            df_a = df[df["condition"] == c_a].sort_values(["episode_id", "probe_id"])
            df_b = df[df["condition"] == c_b].sort_values(["episode_id", "probe_id"])
            b, c, mcnemar_p = compute_exact_mcnemar_test(
                df_a["is_correct"].tolist(),
                df_b["is_correct"].tolist(),
            )
            perm_p, perm_method = compute_permutation_test(ep_diffs)
            is_dist = (ci_l > 0.0 or ci_u < 0.0) or (mcnemar_p < 0.05) or (perm_p < 0.05)

            causal_estimands[name] = CausalEstimandSummary(
                contrast_name=name,
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

    # 3. Horizon Breakdown & Horizon-Specific Contrasts
    horizon_breakdown: Dict[int, Dict[str, float]] = {}
    horizon_contrasts: List[HorizonContrastSummary] = []

    for h in horizons:
        df_h = df[df["horizon_ticks"] == h]
        h_dict: Dict[str, float] = {}
        for cond in conditions:
            h_dict[cond] = float(df_h[df_h["condition"] == cond]["is_correct"].mean())
        horizon_breakdown[h] = h_dict

        # Compute horizon-specific Delta_online-direct
        if "incremental_state" in conditions and "replay_transcript" in conditions:
            pt_d, ci_l, ci_u, _ = compute_episode_clustered_bootstrap(
                df=df_h, cond_a="incremental_state", cond_b="replay_transcript", num_bootstrap=num_bootstrap, seed=seed
            )
            df_h_a = df_h[df_h["condition"] == "incremental_state"].sort_values(["episode_id", "probe_id"])
            df_h_b = df_h[df_h["condition"] == "replay_transcript"].sort_values(["episode_id", "probe_id"])
            _, _, mcn_p = compute_exact_mcnemar_test(df_h_a["is_correct"].tolist(), df_h_b["is_correct"].tolist())
            
            horizon_contrasts.append(HorizonContrastSummary(
                horizon_ticks=h,
                contrast_name="Delta_online-direct",
                condition_a="incremental_state",
                condition_b="replay_transcript",
                delta_accuracy=pt_d,
                ci_lower_95=ci_l,
                ci_upper_95=ci_u,
                exact_mcnemar_p_value=mcn_p,
            ))

    # 4. Token Cost Crossover Point (Interpolated)
    t_star_cost: Optional[float] = None
    if "incremental_state" in conditions and "replay_transcript" in conditions:
        h_tokens = []
        for h in horizons:
            df_h = df[df["horizon_ticks"] == h]
            p_inc = df_h[df_h["condition"] == "incremental_state"]["prompt_tokens"].mean()
            p_rep = df_h[df_h["condition"] == "replay_transcript"]["prompt_tokens"].mean()
            h_tokens.append((h, p_inc, p_rep))
        
        for i in range(len(h_tokens) - 1):
            h1, inc1, rep1 = h_tokens[i]
            h2, inc2, rep2 = h_tokens[i+1]
            diff1 = rep1 - inc1
            diff2 = rep2 - inc2
            if diff1 * diff2 <= 0 and (diff2 - diff1) != 0:
                t_star_cost = h1 + (0.0 - diff1) * (h2 - h1) / (diff2 - diff1)
                break

    # 5. Descriptive Accuracy Crossover
    t_cross_acc: Optional[int] = None
    for h in horizons:
        acc_inc = horizon_breakdown[h].get("incremental_state", 0.0)
        acc_rep = horizon_breakdown[h].get("replay_transcript", 0.0)
        if acc_inc > acc_rep:
            t_cross_acc = h
            break

    return ScheduledReplayAnalysisSummary(
        total_episodes=len(episodes),
        total_trials=len(trials),
        horizons_evaluated=horizons,
        condition_stats=condition_stats,
        causal_estimands=causal_estimands,
        horizon_breakdown=horizon_breakdown,
        horizon_contrasts=horizon_contrasts,
        token_cost_crossover_ticks=t_star_cost,
        descriptive_accuracy_crossover_ticks=t_cross_acc,
    )
