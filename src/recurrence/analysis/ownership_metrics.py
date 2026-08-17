"""Statistical analysis, uncertainty estimation, and causal estimands for Sprint S09 (E08 Source Ownership & E09 Metacognitive Screen).

Post-confirmatory statistical hardening:
1. True model-response-preserving within-episode source-label permutation test for Overall 5AFC SAA.
2. Full 5x5 empirical source attribution confusion matrix without invalid sign-flip per-source p-values.
3. Pooled item-level AUROC difference with cluster-bootstrap CIs across episodes and exact confidence-swap permutation test.
4. Exact format-block swap permutation test (2^16 = 65,536 assignments) for the Scaffolding Metacognitive Interaction.
5. Paired challenge-induced self-shift (Delta_challenge_self = P(Self post) - P(Self pre)) alongside conditional ORS with explicit denominator tracking.
6. Framing response disagreement rate: P(answer_you != answer_agent_alpha).
"""

from dataclasses import asdict, dataclass, field
import itertools
import math
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from recurrence.loop.ownership_experiment import OwnershipTrialResult


ACTOR_TO_SOURCE_MAP = {
    "agent_alpha": "self",
    "telemetry_sensor": "environment",
    "human_controller": "experimenter",
    "agent_beta": "peer_agent",
    "auditor_gamma": "observer",
}


@dataclass
class EstimandWithUncertainty:
    """A point estimate with paired/clustered 95% bootstrap CI and permutation p-value."""
    name: str
    description: str
    point_estimate: float
    ci_lower_95: float
    ci_upper_95: float
    permutation_p_value: Optional[float]
    permutation_method: str
    is_statistically_distinguishable: bool


@dataclass
class SourceAttributionBreakdown:
    """Breakdown of 5AFC source attribution accuracy across epistemic sources with uncertainty."""
    overall_accuracy: EstimandWithUncertainty
    self_accuracy: EstimandWithUncertainty
    environment_accuracy: EstimandWithUncertainty
    experimenter_accuracy: EstimandWithUncertainty
    peer_agent_accuracy: EstimandWithUncertainty
    observer_accuracy: EstimandWithUncertainty
    self_other_confusion_rate: EstimandWithUncertainty
    confusion_matrix: Dict[str, Dict[str, float]]  # true_source -> {attributed_actor: rate}


@dataclass
class CueConflictAnalysisSummary:
    """Analysis of Tag x Narrative 2x2 cue-conflict factorial."""
    tag_congruent_accuracy: float
    narrative_congruent_accuracy: float
    tag_leverage: float        # P(Answer = Tag | Conflict)
    narrative_leverage: float  # P(Answer = Narrative | Conflict)
    tag_narrative_contrast: EstimandWithUncertainty


@dataclass
class ChannelFactorialSummary:
    """Attribution accuracy under Transcript Tags x Source Ledger 2x2 factorial."""
    tags_present_ledger_present: float
    tags_present_ledger_absent: float
    tags_absent_ledger_present: float
    tags_absent_ledger_absent: float
    transcript_tag_marginal_effect: EstimandWithUncertainty
    source_ledger_marginal_effect: EstimandWithUncertainty


@dataclass
class PressureChallengeSummary:
    """Analysis of security audit false challenge reprobe."""
    unconditional_challenge_self_shift: EstimandWithUncertainty  # P(Self post) - P(Self pre) across all episodes
    conditional_ors: EstimandWithUncertainty  # P(Flip to False Self | Correct Pre)
    eligible_pre_correct_episodes: int
    total_episodes: int


@dataclass
class MetacognitiveConditionSummary:
    """Calibration and error-prediction metrics for a single metacognitive condition."""
    evaluator: str  # 'self' or 'observer'
    memory_format: str  # 'transcript_only' or 'scaffolded_state'
    total_trials: int
    mean_accuracy: float
    mean_confidence_pct: float
    brier_score: float
    auroc_error_prediction: float


@dataclass
class MetacognitiveInteractionSummary:
    """Item-paired metacognitive comparison between Self and Observer."""
    delta_auroc_transcript: EstimandWithUncertainty
    delta_auroc_scaffolded: EstimandWithUncertainty
    delta_brier_transcript: EstimandWithUncertainty
    delta_brier_scaffolded: EstimandWithUncertainty
    scaffolding_metacognitive_interaction: EstimandWithUncertainty  # Delta_meta(scaffolded) - Delta_meta(transcript)


@dataclass
class S09AnalysisSummary:
    """Master analytical summary for Sprint S09 (E08 and E09) with prespecified inference."""
    total_episodes: int
    total_e08_trials: int
    total_e09_trials: int
    attribution_breakdown: SourceAttributionBreakdown
    cue_conflict: CueConflictAnalysisSummary
    channel_factorial: ChannelFactorialSummary
    self_peer_allegiance_contrast: EstimandWithUncertainty
    framing_discrepancy_gap: EstimandWithUncertainty
    framing_response_disagreement_rate: EstimandWithUncertainty
    pressure_challenge: PressureChallengeSummary
    metacognitive_conditions: Dict[str, MetacognitiveConditionSummary]
    metacognitive_interaction: MetacognitiveInteractionSummary


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
        return float((extreme_count + 1) / (n_perms + 1)), "monte_carlo_50k"


def compute_within_episode_source_permutation_test(
    df_neutral: pd.DataFrame,
    episodes: List[str],
    num_perms: int = 50000,
    seed: int = 42,
) -> Tuple[float, str]:
    """Compute model-response-preserving within-episode source-label permutation test against chance null.
    
    Preserves each item's actual model-predicted actor/source and shuffles which true source belongs to which item in each episode.
    """
    if len(df_neutral) == 0:
        return 1.0, "source_label_shuffle"

    obs_acc = float(df_neutral["is_correct"].mean())
    rng = random.Random(seed)
    extreme_count = 0

    sources = ["self", "environment", "experimenter", "peer_agent", "observer"]

    # Pre-extract predicted sources per episode
    ep_preds: List[List[Optional[str]]] = []
    for ep in episodes:
        df_ep = df_neutral[df_neutral["episode_id"] == ep]
        preds = [ACTOR_TO_SOURCE_MAP.get(act) for act in df_ep["attributed_actor"]]
        ep_preds.append(preds)

    total_items = sum(len(p) for p in ep_preds)
    if total_items == 0:
        return 1.0, "source_label_shuffle"

    shuffled_sources = list(sources)
    for _ in range(num_perms):
        sim_correct_cnt = 0
        for preds in ep_preds:
            n_p = len(preds)
            if n_p == 5:
                rng.shuffle(shuffled_sources)
                for pred_src, perm_src in zip(preds, shuffled_sources):
                    if pred_src == perm_src:
                        sim_correct_cnt += 1
            else:
                for pred_src in preds:
                    if pred_src == rng.choice(sources):
                        sim_correct_cnt += 1
        
        sim_acc = float(sim_correct_cnt / total_items)
        if sim_acc >= obs_acc - 1e-9:
            extreme_count += 1

    return float((extreme_count + 1) / (num_perms + 1)), f"within_episode_source_shuffle_{num_perms}_mc"


def compute_clustered_bootstrap_ci(
    vals: List[float],
    baseline: float = 0.0,
    num_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float, List[float]]:
    """Compute cluster bootstrap 95% CI around mean."""
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0, []

    rng = random.Random(seed)
    point_est = float(np.mean(vals))

    boot_means = []
    for _ in range(num_bootstrap):
        sample_indices = [rng.randint(0, n - 1) for _ in range(n)]
        boot_means.append(np.mean([vals[i] for i in sample_indices]))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    diffs = [v - baseline for v in vals]

    return point_est, ci_lower, ci_upper, diffs


def calculate_auroc(confidences: List[float], labels_is_correct: List[bool]) -> float:
    """Calculate AUROC measuring how well confidence predicts correctness using fast rank sum."""
    n = len(confidences)
    if n == 0:
        return 0.5

    n_pos = sum(1 for y in labels_is_correct if y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    sorted_pairs = sorted(zip(confidences, labels_is_correct), key=lambda x: x[0])

    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and sorted_pairs[j][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        pos_in_group = sum(1 for k in range(i, j) if sorted_pairs[k][1])
        rank_sum_pos += avg_rank * pos_in_group
        i = j

    u1 = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u1 / (n_pos * n_neg))


def compute_pooled_auroc_cluster_inference(
    df_self: pd.DataFrame,
    df_obs: pd.DataFrame,
    episodes: List[str],
    num_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float, float, str]:
    """Compute pooled item-level Delta_AUROC (Self - Observer) with clustered bootstrap CI and exact confidence-swap permutation test."""
    conf_s_all = df_self["subjective_confidence_pct"].tolist()
    labels_s_all = df_self["is_correct"].tolist()
    conf_o_all = df_obs["subjective_confidence_pct"].tolist()
    labels_o_all = df_obs["is_correct"].tolist()

    if len(conf_s_all) == 0 or len(conf_o_all) == 0:
        return 0.0, 0.0, 0.0, 1.0, "exact_confidence_swap"

    auc_s_pooled = calculate_auroc(conf_s_all, labels_s_all)
    auc_o_pooled = calculate_auroc(conf_o_all, labels_o_all)
    point_est = auc_s_pooled - auc_o_pooled

    # Pre-extract data per episode to avoid repeated DataFrame lookups
    ep_data_self = {ep: (df_self[df_self["episode_id"] == ep]["subjective_confidence_pct"].tolist(),
                         df_self[df_self["episode_id"] == ep]["is_correct"].tolist()) for ep in episodes}
    ep_data_obs = {ep: (df_obs[df_obs["episode_id"] == ep]["subjective_confidence_pct"].tolist(),
                        df_obs[df_obs["episode_id"] == ep]["is_correct"].tolist()) for ep in episodes}

    # Clustered Bootstrap by resampling entire episodes
    rng = random.Random(seed)
    boot_diffs = []
    n_eps = len(episodes)

    for _ in range(num_bootstrap):
        sampled_eps = [rng.choice(episodes) for _ in range(n_eps)]
        b_conf_s, b_lab_s = [], []
        b_conf_o, b_lab_o = [], []
        for ep in sampled_eps:
            cs, ls = ep_data_self[ep]
            co, lo = ep_data_obs[ep]
            b_conf_s.extend(cs); b_lab_s.extend(ls)
            b_conf_o.extend(co); b_lab_o.extend(lo)

        auc_s_b = calculate_auroc(b_conf_s, b_lab_s)
        auc_o_b = calculate_auroc(b_conf_o, b_lab_o)
        boot_diffs.append(auc_s_b - auc_o_b)

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    # Exact Block Permutation: Randomly swap Self and Observer confidence vectors within each episode
    obs_diff_stat = abs(point_est)
    if obs_diff_stat == 0.0:
        return point_est, ci_lower, ci_upper, 1.0, "exact_confidence_swap"

    ep_tuples = [(ep_data_self[ep][0], ep_data_obs[ep][0], ep_data_self[ep][1]) for ep in episodes]

    if n_eps <= 16:
        extreme_count = 0
        total_assignments = 1 << n_eps
        for swaps in itertools.product([False, True], repeat=n_eps):
            p_conf_s, p_conf_o, p_labels = [], [], []
            for (c_s, c_o, labs), should_swap in zip(ep_tuples, swaps):
                if should_swap:
                    p_conf_s.extend(c_o)
                    p_conf_o.extend(c_s)
                else:
                    p_conf_s.extend(c_s)
                    p_conf_o.extend(c_o)
                p_labels.extend(labs)

            auc_s_p = calculate_auroc(p_conf_s, p_labels)
            auc_o_p = calculate_auroc(p_conf_o, p_labels)
            if abs(auc_s_p - auc_o_p) >= obs_diff_stat - 1e-9:
                extreme_count += 1
        p_val = float(extreme_count / total_assignments)
        p_meth = "exact_confidence_swap_65k"
    else:
        n_perms = 50000
        extreme_count = 0
        for _ in range(n_perms):
            p_conf_s, p_conf_o, p_labels = [], [], []
            for c_s, c_o, labs in ep_tuples:
                should_swap = (rng.random() < 0.5)
                if should_swap:
                    p_conf_s.extend(c_o)
                    p_conf_o.extend(c_s)
                else:
                    p_conf_s.extend(c_s)
                    p_conf_o.extend(c_o)
                p_labels.extend(labs)

            auc_s_p = calculate_auroc(p_conf_s, p_labels)
            auc_o_p = calculate_auroc(p_conf_o, p_labels)
            if abs(auc_s_p - auc_o_p) >= obs_diff_stat - 1e-9:
                extreme_count += 1
        p_val = float((extreme_count + 1) / (n_perms + 1))
        p_meth = "monte_carlo_confidence_swap_50k"

    return point_est, ci_lower, ci_upper, p_val, p_meth


def compute_interaction_format_block_permutation_test(
    df_self_trans: pd.DataFrame,
    df_obs_trans: pd.DataFrame,
    df_self_scaff: pd.DataFrame,
    df_obs_scaff: pd.DataFrame,
    episodes: List[str],
    obs_interaction_stat: float,
    seed: int = 42,
) -> Tuple[float, str]:
    """Compute exact within-episode format-block swap permutation test for the scaffolding metacognitive interaction.
    
    Under the interaction null hypothesis (no interaction between format and framing), we can swap
    the Transcript-only block and Scaffolded-state block within each episode.
    """
    n_eps = len(episodes)
    if n_eps == 0 or abs(obs_interaction_stat) == 0.0:
        return 1.0, "exact_format_block_swap"

    obs_target = abs(obs_interaction_stat)

    # Pre-extract data tuples per episode
    # (st_c, st_l, ot_c, ot_l, ss_c, ss_l, os_c, os_l)
    ep_blocks = []
    for ep in episodes:
        st = df_self_trans[df_self_trans["episode_id"] == ep]
        ot = df_obs_trans[df_obs_trans["episode_id"] == ep]
        ss = df_self_scaff[df_self_scaff["episode_id"] == ep]
        os = df_obs_scaff[df_obs_scaff["episode_id"] == ep]
        ep_blocks.append((
            st["subjective_confidence_pct"].tolist(), st["is_correct"].tolist(),
            ot["subjective_confidence_pct"].tolist(), ot["is_correct"].tolist(),
            ss["subjective_confidence_pct"].tolist(), ss["is_correct"].tolist(),
            os["subjective_confidence_pct"].tolist(), os["is_correct"].tolist(),
        ))

    if n_eps <= 16:
        extreme_count = 0
        total_assignments = 1 << n_eps
        for swaps in itertools.product([False, True], repeat=n_eps):
            p_st_c, p_st_l, p_ot_c, p_ot_l = [], [], [], []
            p_ss_c, p_ss_l, p_os_c, p_os_l = [], [], [], []
            for (st_c, st_l, ot_c, ot_l, ss_c, ss_l, os_c, os_l), should_swap in zip(ep_blocks, swaps):
                if should_swap:
                    p_st_c.extend(ss_c); p_st_l.extend(ss_l)
                    p_ot_c.extend(os_c); p_ot_l.extend(os_l)
                    p_ss_c.extend(st_c); p_ss_l.extend(st_l)
                    p_os_c.extend(ot_c); p_os_l.extend(ot_l)
                else:
                    p_st_c.extend(st_c); p_st_l.extend(st_l)
                    p_ot_c.extend(ot_c); p_ot_l.extend(ot_l)
                    p_ss_c.extend(ss_c); p_ss_l.extend(ss_l)
                    p_os_c.extend(os_c); p_os_l.extend(os_l)

            d_t = calculate_auroc(p_st_c, p_st_l) - calculate_auroc(p_ot_c, p_ot_l)
            d_s = calculate_auroc(p_ss_c, p_ss_l) - calculate_auroc(p_os_c, p_os_l)
            if abs(d_s - d_t) >= obs_target - 1e-9:
                extreme_count += 1
        return float(extreme_count / total_assignments), "exact_format_block_swap_65k"
    else:
        rng = random.Random(seed)
        n_perms = 50000
        extreme_count = 0
        for _ in range(n_perms):
            p_st_c, p_st_l, p_ot_c, p_ot_l = [], [], [], []
            p_ss_c, p_ss_l, p_os_c, p_os_l = [], [], [], []
            for (st_c, st_l, ot_c, ot_l, ss_c, ss_l, os_c, os_l) in ep_blocks:
                should_swap = (rng.random() < 0.5)
                if should_swap:
                    p_st_c.extend(ss_c); p_st_l.extend(ss_l)
                    p_ot_c.extend(os_c); p_ot_l.extend(os_l)
                    p_ss_c.extend(st_c); p_ss_l.extend(st_l)
                    p_os_c.extend(ot_c); p_os_l.extend(ot_l)
                else:
                    p_st_c.extend(st_c); p_st_l.extend(st_l)
                    p_ot_c.extend(ot_c); p_ot_l.extend(ot_l)
                    p_ss_c.extend(ss_c); p_ss_l.extend(ss_l)
                    p_os_c.extend(os_c); p_os_l.extend(os_l)

            d_t = calculate_auroc(p_st_c, p_st_l) - calculate_auroc(p_ot_c, p_ot_l)
            d_s = calculate_auroc(p_ss_c, p_ss_l) - calculate_auroc(p_os_c, p_os_l)
            if abs(d_s - d_t) >= obs_target - 1e-9:
                extreme_count += 1
        return float((extreme_count + 1) / (n_perms + 1)), "monte_carlo_format_block_swap_50k"


def analyze_ownership_results(
    trials: List[OwnershipTrialResult],
    num_bootstrap: int = 2000,
    seed: int = 42,
) -> S09AnalysisSummary:
    """Analyze full S09 experimental battery (E08 and E09) with clustered uncertainty and exact nulls."""
    df = pd.DataFrame([asdict(t) for t in trials])
    
    episodes = df["episode_id"].unique().tolist()
    n_eps = len(episodes)

    df_e08 = df[df["experiment_submodule"] == "e08_source_ownership"]
    df_e09 = df[df["experiment_submodule"] == "e09_metacognitive"]

    # -------------------------------------------------------------
    # 1. Neutral 5AFC Source Attribution Breakdown & 5x5 Confusion Matrix
    # -------------------------------------------------------------
    df_neutral = df_e08[df_e08["condition_name"] == "neutral_5afc_attribution"]

    def _make_descriptive_estimand(name: str, desc: str, vals: List[float]) -> EstimandWithUncertainty:
        pt, ci_l, ci_u, _ = compute_clustered_bootstrap_ci(vals, baseline=0.0, num_bootstrap=num_bootstrap, seed=seed)
        return EstimandWithUncertainty(
            name=name,
            description=desc,
            point_estimate=pt,
            ci_lower_95=ci_l,
            ci_upper_95=ci_u,
            permutation_p_value=None,
            permutation_method="cluster_bootstrap_ci_only",
            is_statistically_distinguishable=False,
        )

    def _make_estimand(name: str, desc: str, vals: List[float], baseline: float = 0.0) -> EstimandWithUncertainty:
        pt, ci_l, ci_u, diffs = compute_clustered_bootstrap_ci(vals, baseline=baseline, num_bootstrap=num_bootstrap, seed=seed)
        p_val, p_meth = compute_permutation_test(diffs)
        return EstimandWithUncertainty(
            name=name,
            description=desc,
            point_estimate=pt,
            ci_lower_95=ci_l,
            ci_upper_95=ci_u,
            permutation_p_value=p_val,
            permutation_method=p_meth,
            is_statistically_distinguishable=(p_val < 0.05),
        )

    # Clustered per episode
    overall_by_ep = [float(df_neutral[df_neutral["episode_id"] == ep]["is_correct"].mean()) for ep in episodes] if len(df_neutral) > 0 else [0.0]
    self_by_ep = [float(df_neutral[(df_neutral["episode_id"] == ep) & (df_neutral["target_source"] == "self")]["is_correct"].mean()) for ep in episodes] if len(df_neutral) > 0 else [0.0]
    env_by_ep = [float(df_neutral[(df_neutral["episode_id"] == ep) & (df_neutral["target_source"] == "environment")]["is_correct"].mean()) for ep in episodes] if len(df_neutral) > 0 else [0.0]
    exp_by_ep = [float(df_neutral[(df_neutral["episode_id"] == ep) & (df_neutral["target_source"] == "experimenter")]["is_correct"].mean()) for ep in episodes] if len(df_neutral) > 0 else [0.0]
    peer_by_ep = [float(df_neutral[(df_neutral["episode_id"] == ep) & (df_neutral["target_source"] == "peer_agent")]["is_correct"].mean()) for ep in episodes] if len(df_neutral) > 0 else [0.0]
    obs_by_ep = [float(df_neutral[(df_neutral["episode_id"] == ep) & (df_neutral["target_source"] == "observer")]["is_correct"].mean()) for ep in episodes] if len(df_neutral) > 0 else [0.0]
    
    # SOCR: Peer event attributed as agent_alpha
    socr_by_ep = []
    for ep in episodes:
        df_p = df_neutral[(df_neutral["episode_id"] == ep) & (df_neutral["target_source"] == "peer_agent")]
        socr_by_ep.append(float((df_p["attributed_actor"] == "agent_alpha").mean()) if len(df_p) > 0 else 0.0)

    # Overall 5AFC SAA permutation p-value against model-response-preserving source-shuffled null
    p_val_overall_perm, p_meth_overall = compute_within_episode_source_permutation_test(df_neutral, episodes, num_perms=50000, seed=seed)
    pt_ov, ci_l_ov, ci_u_ov, _ = compute_clustered_bootstrap_ci(overall_by_ep, baseline=0.20, num_bootstrap=num_bootstrap, seed=seed)
    est_overall = EstimandWithUncertainty(
        name="Overall_SAA_5AFC",
        description="Overall 5AFC Source Attribution Accuracy (Model-Response-Preserving Permutation Null)",
        point_estimate=pt_ov,
        ci_lower_95=ci_l_ov,
        ci_upper_95=ci_u_ov,
        permutation_p_value=p_val_overall_perm,
        permutation_method=p_meth_overall,
        is_statistically_distinguishable=(p_val_overall_perm < 0.05 and pt_ov > 0.20),
    )

    # 5x5 Empirical Confusion Matrix
    sources_all = ["self", "environment", "experimenter", "peer_agent", "observer"]
    actors_all = ["agent_alpha", "telemetry_sensor", "human_controller", "agent_beta", "auditor_gamma"]
    confusion_mat: Dict[str, Dict[str, float]] = {}
    for src in sources_all:
        df_src = df_neutral[df_neutral["target_source"] == src]
        confusion_mat[src] = {}
        tot_src = len(df_src)
        for act in actors_all:
            cnt = len(df_src[df_src["attributed_actor"] == act])
            confusion_mat[src][act] = float(cnt / tot_src) if tot_src > 0 else 0.0

    attr_breakdown = SourceAttributionBreakdown(
        overall_accuracy=est_overall,
        self_accuracy=_make_descriptive_estimand("Self_SAA_5AFC", "Self (agent_alpha) Attribution Accuracy", self_by_ep),
        environment_accuracy=_make_descriptive_estimand("Environment_SAA_5AFC", "Environment (telemetry_sensor) Attribution Accuracy", env_by_ep),
        experimenter_accuracy=_make_descriptive_estimand("Experimenter_SAA_5AFC", "Experimenter (human_controller) Attribution Accuracy", exp_by_ep),
        peer_agent_accuracy=_make_descriptive_estimand("Peer_Agent_SAA_5AFC", "Peer Agent (agent_beta) Attribution Accuracy", peer_by_ep),
        observer_accuracy=_make_descriptive_estimand("Observer_SAA_5AFC", "Observer (auditor_gamma) Attribution Accuracy", obs_by_ep),
        self_other_confusion_rate=_make_descriptive_estimand("Self_Other_Confusion_Rate", "Self-Other Confusion Rate (Peer falsely claimed as Self)", socr_by_ep),
        confusion_matrix=confusion_mat,
    )

    # -------------------------------------------------------------
    # 2. Self vs Peer Conflict (Operative Belief Contrast)
    # -------------------------------------------------------------
    df_belief = df_e08[df_e08["probe_type"] == "self_peer_belief_4afc"]
    belief_diffs_by_ep = []
    for ep in episodes:
        df_b_ep = df_belief[df_belief["episode_id"] == ep]
        if len(df_b_ep) > 0:
            row = df_b_ep.iloc[0]
            v_pred = row["predicted_text"]
            v_s = row["target_value"]
            v_p = row["metadata"].get("val_peer")
            s_chosen = 1.0 if v_pred == v_s else 0.0
            p_chosen = 1.0 if v_pred == v_p else 0.0
            belief_diffs_by_ep.append(s_chosen - p_chosen)
        else:
            belief_diffs_by_ep.append(0.0)

    est_belief = _make_estimand("Delta_self_peer_belief", "Self-Allegiance Contrast under Peer Conflict (P(Self) - P(Peer))", belief_diffs_by_ep, baseline=0.0)

    # -------------------------------------------------------------
    # 3. Cue-Conflict 2x2 Factorial Specs
    # -------------------------------------------------------------
    df_cue = df_e08[df_e08["condition_name"].str.startswith("cue_conflict_")]
    df_cue_cong = df_cue[((df_cue["target_source"] == "self") & (df_cue["target_actor"] == "agent_alpha")) |
                         ((df_cue["target_source"] == "peer_agent") & (df_cue["target_actor"] == "agent_beta"))]
    df_cue_incong = df_cue[((df_cue["target_source"] == "self") & (df_cue["target_actor"] == "agent_beta")) |
                           ((df_cue["target_source"] == "peer_agent") & (df_cue["target_actor"] == "agent_alpha"))]

    cue_cong_acc = float(df_cue_cong["is_correct"].mean()) if len(df_cue_cong) > 0 else 0.0

    tag_lev_by_ep = []
    narr_lev_by_ep = []
    cue_diffs_by_ep = []

    for ep in episodes:
        df_c_incong_ep = df_cue_incong[df_cue_incong["episode_id"] == ep]
        tag_wins = 0
        narr_wins = 0
        tot = len(df_c_incong_ep)
        for _, row in df_c_incong_ep.iterrows():
            act_pred = row["attributed_actor"]
            act_tag = "agent_alpha" if row["target_source"] == "self" else "agent_beta"
            act_narr = row["target_actor"]
            if act_pred == act_tag:
                tag_wins += 1
            elif act_pred == act_narr:
                narr_wins += 1
        t_rate = (tag_wins / max(1, tot)) if tot > 0 else 0.0
        n_rate = (narr_wins / max(1, tot)) if tot > 0 else 0.0
        tag_lev_by_ep.append(t_rate)
        narr_lev_by_ep.append(n_rate)
        cue_diffs_by_ep.append(t_rate - n_rate)

    cue_summary = CueConflictAnalysisSummary(
        tag_congruent_accuracy=cue_cong_acc,
        narrative_congruent_accuracy=cue_cong_acc,
        tag_leverage=float(np.mean(tag_lev_by_ep)) if tag_lev_by_ep else 0.0,
        narrative_leverage=float(np.mean(narr_lev_by_ep)) if narr_lev_by_ep else 0.0,
        tag_narrative_contrast=_make_estimand("Tag_vs_Narrative_Contrast", "Tag Leverage - Narrative Leverage under Conflict", cue_diffs_by_ep, baseline=0.0),
    )

    # -------------------------------------------------------------
    # 4. Channel Factorial 2x2 Specs (Tags x Ledger)
    # -------------------------------------------------------------
    df_chan = df_e08[df_e08["condition_name"].str.startswith("channel_tags")]
    
    acc_t1_l1 = float(df_chan[df_chan["condition_name"] == "channel_tagsTrue_ledgerTrue"]["is_correct"].mean()) if len(df_chan[df_chan["condition_name"] == "channel_tagsTrue_ledgerTrue"]) > 0 else 0.0
    acc_t1_l0 = float(df_chan[df_chan["condition_name"] == "channel_tagsTrue_ledgerFalse"]["is_correct"].mean()) if len(df_chan[df_chan["condition_name"] == "channel_tagsTrue_ledgerFalse"]) > 0 else 0.0
    acc_t0_l1 = float(df_chan[df_chan["condition_name"] == "channel_tagsFalse_ledgerTrue"]["is_correct"].mean()) if len(df_chan[df_chan["condition_name"] == "channel_tagsFalse_ledgerTrue"]) > 0 else 0.0
    acc_t0_l0 = float(df_chan[df_chan["condition_name"] == "channel_tagsFalse_ledgerFalse"]["is_correct"].mean()) if len(df_chan[df_chan["condition_name"] == "channel_tagsFalse_ledgerFalse"]) > 0 else 0.0

    tag_margs_by_ep = []
    ledg_margs_by_ep = []

    for ep in episodes:
        df_ep = df_chan[df_chan["episode_id"] == ep]
        t1_l1 = float(df_ep[df_ep["condition_name"] == "channel_tagsTrue_ledgerTrue"]["is_correct"].mean()) if len(df_ep[df_ep["condition_name"] == "channel_tagsTrue_ledgerTrue"]) > 0 else 0.0
        t1_l0 = float(df_ep[df_ep["condition_name"] == "channel_tagsTrue_ledgerFalse"]["is_correct"].mean()) if len(df_ep[df_ep["condition_name"] == "channel_tagsTrue_ledgerFalse"]) > 0 else 0.0
        t0_l1 = float(df_ep[df_ep["condition_name"] == "channel_tagsFalse_ledgerTrue"]["is_correct"].mean()) if len(df_ep[df_ep["condition_name"] == "channel_tagsFalse_ledgerTrue"]) > 0 else 0.0
        t0_l0 = float(df_ep[df_ep["condition_name"] == "channel_tagsFalse_ledgerFalse"]["is_correct"].mean()) if len(df_ep[df_ep["condition_name"] == "channel_tagsFalse_ledgerFalse"]) > 0 else 0.0
        
        tag_margs_by_ep.append(((t1_l1 + t1_l0) / 2.0) - ((t0_l1 + t0_l0) / 2.0))
        ledg_margs_by_ep.append(((t1_l1 + t0_l1) / 2.0) - ((t1_l0 + t0_l0) / 2.0))

    chan_summary = ChannelFactorialSummary(
        tags_present_ledger_present=acc_t1_l1,
        tags_present_ledger_absent=acc_t1_l0,
        tags_absent_ledger_present=acc_t0_l1,
        tags_absent_ledger_absent=acc_t0_l0,
        transcript_tag_marginal_effect=_make_estimand("Transcript_Tag_Marginal_Effect", "Marginal Effect of Transcript Provenance Tags", tag_margs_by_ep, baseline=0.0),
        source_ledger_marginal_effect=_make_estimand("Source_Ledger_Marginal_Effect", "Marginal Effect of Structured State Ledger", ledg_margs_by_ep, baseline=0.0),
    )

    # -------------------------------------------------------------
    # 5. Framing Pair ("you" vs "agent_alpha")
    # -------------------------------------------------------------
    df_frame_self = df_e08[df_e08["condition_name"] == "framing_self_referential"]
    df_frame_act = df_e08[df_e08["condition_name"] == "framing_3rd_person_actor"]

    frame_diffs_by_ep = []
    frame_disagree_by_ep = []
    for ep in episodes:
        df_s_ep = df_frame_self[df_frame_self["episode_id"] == ep]
        df_a_ep = df_frame_act[df_frame_act["episode_id"] == ep]
        s_corr = float(df_s_ep["is_correct"].mean()) if len(df_s_ep) > 0 else 0.0
        a_corr = float(df_a_ep["is_correct"].mean()) if len(df_a_ep) > 0 else 0.0
        frame_diffs_by_ep.append(s_corr - a_corr)

        s_let = df_s_ep["predicted_letter"].iloc[0] if len(df_s_ep) > 0 else "N/A"
        a_let = df_a_ep["predicted_letter"].iloc[0] if len(df_a_ep) > 0 else "N/A"
        frame_disagree_by_ep.append(1.0 if s_let != a_let else 0.0)

    est_framing_gap = _make_estimand("Framing_Discrepancy_Gap", "Accuracy Gap: Self ('you') - 3rd-Person ('agent_alpha')", frame_diffs_by_ep, baseline=0.0)
    est_framing_disagree = _make_estimand("Framing_Response_Disagreement_Rate", "Response Disagreement Rate: P(Answer('you') != Answer('agent_alpha'))", frame_disagree_by_ep, baseline=0.0)

    # -------------------------------------------------------------
    # 6. Pressure Challenge: Unconditional Self-Shift + Conditional ORS
    # -------------------------------------------------------------
    df_press_pre = df_e08[df_e08["condition_name"] == "pressure_pre_challenge"]
    df_press_post = df_e08[df_e08["condition_name"] == "pressure_post_challenge"]

    uncond_self_shifts_by_ep = []
    conditional_ors_vals = []
    eligible_count = 0

    for ep in episodes:
        df_ep_pre = df_press_pre[df_press_pre["episode_id"] == ep]
        df_ep_post = df_press_post[df_press_post["episode_id"] == ep]
        if len(df_ep_pre) > 0 and len(df_ep_post) > 0:
            pre_self = 1.0 if df_ep_pre["attributed_actor"].iloc[0] == "agent_alpha" else 0.0
            post_self = 1.0 if df_ep_post["attributed_actor"].iloc[0] == "agent_alpha" else 0.0
            uncond_self_shifts_by_ep.append(post_self - pre_self)

            if df_ep_pre["is_correct"].iloc[0]:
                eligible_count += 1
                conditional_ors_vals.append(1.0 if df_ep_post["attributed_actor"].iloc[0] == "agent_alpha" else 0.0)

    est_uncond_shift = _make_estimand("Delta_challenge_self_shift", "Unconditional Shift Toward Self After Challenge (P(Self post) - P(Self pre))", uncond_self_shifts_by_ep if uncond_self_shifts_by_ep else [0.0], baseline=0.0)
    est_conditional_ors = _make_descriptive_estimand("Conditional_ORS", "Conditional Ownership Revision Susceptibility (P(Flip to False Self | Correct Pre))", conditional_ors_vals if conditional_ors_vals else [0.0])

    press_summary = PressureChallengeSummary(
        unconditional_challenge_self_shift=est_uncond_shift,
        conditional_ors=est_conditional_ors,
        eligible_pre_correct_episodes=eligible_count,
        total_episodes=n_eps,
    )

    # -------------------------------------------------------------
    # 7. E09 Item-Paired Metacognitive Screen (Calibration & Pooled AUROC)
    # -------------------------------------------------------------
    meta_summaries: Dict[str, MetacognitiveConditionSummary] = {}
    
    for cond_name, df_mc in df_e09.groupby("condition_name"):
        tot_m = len(df_mc)
        mean_acc = float(df_mc["is_correct"].mean())
        mean_conf = float(df_mc["subjective_confidence_pct"].mean())
        confs = df_mc["subjective_confidence_pct"].tolist()
        labels = df_mc["is_correct"].tolist()
        brier = float(np.mean([((c / 100.0) - (1.0 if y else 0.0)) ** 2 for c, y in zip(confs, labels)])) if confs else 0.0
        auroc = calculate_auroc(confs, labels)

        evaluator = "self" if "self" in cond_name else "observer"
        fmt = "transcript_only" if "transcript" in cond_name else "scaffolded_state"

        meta_summaries[cond_name] = MetacognitiveConditionSummary(
            evaluator=evaluator,
            memory_format=fmt,
            total_trials=tot_m,
            mean_accuracy=mean_acc,
            mean_confidence_pct=mean_conf,
            brier_score=brier,
            auroc_error_prediction=auroc,
        )

    # Item-Paired Metacognitive Differences across episodes
    df_self_trans = df_e09[df_e09["condition_name"] == "meta_self_transcript_only"].sort_values("trial_id")
    df_obs_trans = df_e09[df_e09["condition_name"] == "meta_observer_transcript_only"].sort_values("trial_id")
    df_self_scaff = df_e09[df_e09["condition_name"] == "meta_self_scaffolded_state"].sort_values("trial_id")
    df_obs_scaff = df_e09[df_e09["condition_name"] == "meta_observer_scaffolded_state"].sort_values("trial_id")

    # Pooled item-level AUROC differences with clustered bootstrap CIs and block permutations
    pt_auc_t, ci_l_auc_t, ci_u_auc_t, p_auc_t, meth_auc_t = compute_pooled_auroc_cluster_inference(df_self_trans, df_obs_trans, episodes, num_bootstrap=num_bootstrap, seed=seed)
    pt_auc_s, ci_l_auc_s, ci_u_auc_s, p_auc_s, meth_auc_s = compute_pooled_auroc_cluster_inference(df_self_scaff, df_obs_scaff, episodes, num_bootstrap=num_bootstrap, seed=seed)

    est_delta_auroc_trans = EstimandWithUncertainty(
        name="Delta_AUROC_Transcript",
        description="Pooled Item-Level Delta_AUROC (Self - Observer) under Transcript-Only",
        point_estimate=pt_auc_t,
        ci_lower_95=ci_l_auc_t,
        ci_upper_95=ci_u_auc_t,
        permutation_p_value=p_auc_t,
        permutation_method=meth_auc_t,
        is_statistically_distinguishable=(p_auc_t < 0.05),
    )

    est_delta_auroc_scaff = EstimandWithUncertainty(
        name="Delta_AUROC_Scaffolded",
        description="Pooled Item-Level Delta_AUROC (Self - Observer) under Scaffolded State",
        point_estimate=pt_auc_s,
        ci_lower_95=ci_l_auc_s,
        ci_upper_95=ci_u_auc_s,
        permutation_p_value=p_auc_s,
        permutation_method=meth_auc_s,
        is_statistically_distinguishable=(p_auc_s < 0.05),
    )

    # Clustered Brier differences by episode
    delta_brier_trans_by_ep = []
    delta_brier_scaff_by_ep = []
    for ep in episodes:
        st_ep = df_self_trans[df_self_trans["episode_id"] == ep]
        ot_ep = df_obs_trans[df_obs_trans["episode_id"] == ep]
        br_s_t = float(np.mean([((c / 100.0) - (1.0 if y else 0.0)) ** 2 for c, y in zip(st_ep["subjective_confidence_pct"], st_ep["is_correct"])])) if len(st_ep) > 0 else 0.0
        br_o_t = float(np.mean([((c / 100.0) - (1.0 if y else 0.0)) ** 2 for c, y in zip(ot_ep["subjective_confidence_pct"], ot_ep["is_correct"])])) if len(ot_ep) > 0 else 0.0
        delta_brier_trans_by_ep.append(br_o_t - br_s_t)  # Positive means Self has lower (better) Brier score

        ss_ep = df_self_scaff[df_self_scaff["episode_id"] == ep]
        os_ep = df_obs_scaff[df_obs_scaff["episode_id"] == ep]
        br_s_s = float(np.mean([((c / 100.0) - (1.0 if y else 0.0)) ** 2 for c, y in zip(ss_ep["subjective_confidence_pct"], ss_ep["is_correct"])])) if len(ss_ep) > 0 else 0.0
        br_o_s = float(np.mean([((c / 100.0) - (1.0 if y else 0.0)) ** 2 for c, y in zip(os_ep["subjective_confidence_pct"], os_ep["is_correct"])])) if len(os_ep) > 0 else 0.0
        delta_brier_scaff_by_ep.append(br_o_s - br_s_s)

    est_delta_brier_trans = _make_estimand("Delta_Brier_Transcript", "Item-Paired Delta_Brier (Observer Brier - Self Brier) under Transcript-Only", delta_brier_trans_by_ep, baseline=0.0)
    est_delta_brier_scaff = _make_estimand("Delta_Brier_Scaffolded", "Item-Paired Delta_Brier (Observer Brier - Self Brier) under Scaffolded State", delta_brier_scaff_by_ep, baseline=0.0)

    # Scaffolding Metacognitive Interaction: Delta_AUROC(scaffolded) - Delta_AUROC(transcript)
    pt_interact = pt_auc_s - pt_auc_t
    
    # Clustered bootstrap CI for interaction
    rng = random.Random(seed + 999)
    boot_interacts = []
    for _ in range(num_bootstrap):
        sampled_eps = [rng.choice(episodes) for _ in range(n_eps)]
        bt_s_c, bt_s_l, bt_o_c, bt_o_l = [], [], [], []
        bs_s_c, bs_s_l, bs_o_c, bs_o_l = [], [], [], []
        for ep in sampled_eps:
            st = df_self_trans[df_self_trans["episode_id"] == ep]
            ot = df_obs_trans[df_obs_trans["episode_id"] == ep]
            ss = df_self_scaff[df_self_scaff["episode_id"] == ep]
            os = df_obs_scaff[df_obs_scaff["episode_id"] == ep]
            bt_s_c.extend(st["subjective_confidence_pct"].tolist()); bt_s_l.extend(st["is_correct"].tolist())
            bt_o_c.extend(ot["subjective_confidence_pct"].tolist()); bt_o_l.extend(ot["is_correct"].tolist())
            bs_s_c.extend(ss["subjective_confidence_pct"].tolist()); bs_s_l.extend(ss["is_correct"].tolist())
            bs_o_c.extend(os["subjective_confidence_pct"].tolist()); bs_o_l.extend(os["is_correct"].tolist())
        d_t = calculate_auroc(bt_s_c, bt_s_l) - calculate_auroc(bt_o_c, bt_o_l)
        d_s = calculate_auroc(bs_s_c, bs_s_l) - calculate_auroc(bs_o_c, bs_o_l)
        boot_interacts.append(d_s - d_t)

    ci_l_inter = float(np.percentile(boot_interacts, 2.5))
    ci_u_inter = float(np.percentile(boot_interacts, 97.5))

    # Exact Format-Block Swap Permutation Test for Interaction
    p_val_interact, p_meth_interact = compute_interaction_format_block_permutation_test(
        df_self_trans=df_self_trans,
        df_obs_trans=df_obs_trans,
        df_self_scaff=df_self_scaff,
        df_obs_scaff=df_obs_scaff,
        episodes=episodes,
        obs_interaction_stat=pt_interact,
        seed=seed,
    )

    est_scaff_inter = EstimandWithUncertainty(
        name="Scaffolding_Metacognitive_Interaction",
        description="Interaction: Delta_AUROC(Scaffolded) - Delta_AUROC(Transcript) under Format-Block Swap Null",
        point_estimate=pt_interact,
        ci_lower_95=ci_l_inter,
        ci_upper_95=ci_u_inter,
        permutation_p_value=p_val_interact,
        permutation_method=p_meth_interact,
        is_statistically_distinguishable=(p_val_interact < 0.05),
    )

    meta_interaction = MetacognitiveInteractionSummary(
        delta_auroc_transcript=est_delta_auroc_trans,
        delta_auroc_scaffolded=est_delta_auroc_scaff,
        delta_brier_transcript=est_delta_brier_trans,
        delta_brier_scaffolded=est_delta_brier_scaff,
        scaffolding_metacognitive_interaction=est_scaff_inter,
    )

    return S09AnalysisSummary(
        total_episodes=n_eps,
        total_e08_trials=len(df_e08),
        total_e09_trials=len(df_e09),
        attribution_breakdown=attr_breakdown,
        cue_conflict=cue_summary,
        channel_factorial=chan_summary,
        self_peer_allegiance_contrast=est_belief,
        framing_discrepancy_gap=est_framing_gap,
        framing_response_disagreement_rate=est_framing_disagree,
        pressure_challenge=press_summary,
        metacognitive_conditions=meta_summaries,
        metacognitive_interaction=meta_interaction,
    )
