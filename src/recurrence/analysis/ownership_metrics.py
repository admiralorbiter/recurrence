"""Statistical analysis, uncertainty estimation, and causal estimands for Sprint S09 (E08 Source Ownership & E09 Metacognitive Screen).

Computes:
1. Source Attribution Accuracy (SAA): Overall and per-source (Self, Environment, Experimenter, Peer Agent, Observer) with 95% bootstrap CIs.
2. Self-Other Confusion Rate (SOCR): P(Attributed as Self | True Source = Peer Agent) with 95% CI.
3. Self-Allegiance Contrast under Peer Conflict: P(Belief = V_self) - P(Belief = V_peer) with sign-flip permutation test.
4. Cue-Conflict Marginal Effects: Tag Leverage vs Narrative Identity Leverage with exact permutation test.
5. Channel Factorial Effects: Transcript Tags Marginal Effect vs Source Ledger Marginal Effect with bootstrap CIs.
6. Framing Discrepancy Gap: |Acc("you") - Acc("agent_alpha")| with paired permutation test.
7. Ownership Revision Susceptibility (ORS): P(Flip to False Self | Pressure Challenge).
8. Metacognitive Calibration & Post-Choice Error Prediction:
   - Paired Delta_AUROC (Self - Observer) predicting identical target correctness.
   - Paired Delta_Brier (Self - Observer).
   - Interaction Contrast: Delta_meta(Scaffolded) - Delta_meta(Transcript).
"""

from dataclasses import asdict, dataclass, field
import itertools
import math
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from recurrence.loop.ownership_experiment import OwnershipTrialResult


@dataclass
class EstimandWithUncertainty:
    """A point estimate with paired/clustered 95% bootstrap CI and permutation p-value."""
    name: str
    description: str
    point_estimate: float
    ci_lower_95: float
    ci_upper_95: float
    permutation_p_value: float
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
    ownership_revision_susceptibility: EstimandWithUncertainty
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
        return float(extreme_count / n_perms), "monte_carlo_50k"


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
    """Calculate AUROC measuring how well confidence predicts correctness (Mann-Whitney U)."""
    assert len(confidences) == len(labels_is_correct)
    n = len(confidences)
    if n == 0:
        return 0.5

    pos = [c for c, y in zip(confidences, labels_is_correct) if y]
    neg = [c for c, y in zip(confidences, labels_is_correct) if not y]

    n_pos = len(pos)
    n_neg = len(neg)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    wins = 0.0
    for p in pos:
        for m in neg:
            if p > m:
                wins += 1.0
            elif p == m:
                wins += 0.5

    return float(wins / (n_pos * n_neg))


def analyze_ownership_results(
    trials: List[OwnershipTrialResult],
    num_bootstrap: int = 2000,
    seed: int = 42,
) -> S09AnalysisSummary:
    """Analyze full S09 experimental battery (E08 and E09) with clustered uncertainty."""
    df = pd.DataFrame([asdict(t) for t in trials])
    
    episodes = df["episode_id"].unique().tolist()
    n_eps = len(episodes)

    df_e08 = df[df["experiment_submodule"] == "e08_source_ownership"]
    df_e09 = df[df["experiment_submodule"] == "e09_metacognitive"]

    # -------------------------------------------------------------
    # 1. Neutral 5AFC Source Attribution Breakdown
    # -------------------------------------------------------------
    df_neutral = df_e08[df_e08["condition_name"] == "neutral_5afc_attribution"]

    def _make_estimand(name: str, desc: str, vals: List[float], baseline: float = 0.20) -> EstimandWithUncertainty:
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

    attr_breakdown = SourceAttributionBreakdown(
        overall_accuracy=_make_estimand("Overall_SAA_5AFC", "Overall 5AFC Source Attribution Accuracy", overall_by_ep, baseline=0.20),
        self_accuracy=_make_estimand("Self_SAA_5AFC", "Self (agent_alpha) Attribution Accuracy", self_by_ep, baseline=0.20),
        environment_accuracy=_make_estimand("Environment_SAA_5AFC", "Environment (telemetry_sensor) Attribution Accuracy", env_by_ep, baseline=0.20),
        experimenter_accuracy=_make_estimand("Experimenter_SAA_5AFC", "Experimenter (human_controller) Attribution Accuracy", exp_by_ep, baseline=0.20),
        peer_agent_accuracy=_make_estimand("Peer_Agent_SAA_5AFC", "Peer Agent (agent_beta) Attribution Accuracy", peer_by_ep, baseline=0.20),
        observer_accuracy=_make_estimand("Observer_SAA_5AFC", "Observer (auditor_gamma) Attribution Accuracy", obs_by_ep, baseline=0.20),
        self_other_confusion_rate=_make_estimand("Self_Other_Confusion_Rate", "Self-Other Confusion Rate (Peer falsely claimed as Self)", socr_by_ep, baseline=0.0),
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
    for ep in episodes:
        df_s_ep = df_frame_self[df_frame_self["episode_id"] == ep]
        df_a_ep = df_frame_act[df_frame_act["episode_id"] == ep]
        s_corr = float(df_s_ep["is_correct"].mean()) if len(df_s_ep) > 0 else 0.0
        a_corr = float(df_a_ep["is_correct"].mean()) if len(df_a_ep) > 0 else 0.0
        frame_diffs_by_ep.append(s_corr - a_corr)

    est_framing = _make_estimand("Framing_Discrepancy_Gap", "Accuracy Gap: Self ('you') - 3rd-Person ('agent_alpha')", frame_diffs_by_ep, baseline=0.0)

    # -------------------------------------------------------------
    # 6. Pressure-Induced Revision Susceptibility
    # -------------------------------------------------------------
    df_press_pre = df_e08[df_e08["condition_name"] == "pressure_pre_challenge"]
    df_press_post = df_e08[df_e08["condition_name"] == "pressure_post_challenge"]

    ors_by_ep = []
    for ep in episodes:
        df_ep_pre = df_press_pre[df_press_pre["episode_id"] == ep]
        df_ep_post = df_press_post[df_press_post["episode_id"] == ep]
        if len(df_ep_pre) > 0 and len(df_ep_post) > 0:
            if df_ep_pre["is_correct"].iloc[0]:
                ors_by_ep.append(1.0 if df_ep_post["attributed_actor"].iloc[0] == "agent_alpha" else 0.0)

    est_ors = _make_estimand("Ownership_Revision_Susceptibility", "P(Flip to False Self | Pressure Challenge)", ors_by_ep if ors_by_ep else [0.0], baseline=0.0)

    # -------------------------------------------------------------
    # 7. E09 Item-Paired Metacognitive Screen (Calibration & AUROC)
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

    # Clustered AUROC & Brier differences by episode
    delta_auroc_trans_by_ep = []
    delta_auroc_scaff_by_ep = []
    delta_brier_trans_by_ep = []
    delta_brier_scaff_by_ep = []
    interact_by_ep = []

    for ep in episodes:
        # Transcript format
        st_ep = df_self_trans[df_self_trans["episode_id"] == ep]
        ot_ep = df_obs_trans[df_obs_trans["episode_id"] == ep]
        
        auc_s_t = calculate_auroc(st_ep["subjective_confidence_pct"].tolist(), st_ep["is_correct"].tolist())
        auc_o_t = calculate_auroc(ot_ep["subjective_confidence_pct"].tolist(), ot_ep["is_correct"].tolist())
        d_auc_t = auc_s_t - auc_o_t
        delta_auroc_trans_by_ep.append(d_auc_t)

        br_s_t = float(np.mean([((c / 100.0) - (1.0 if y else 0.0)) ** 2 for c, y in zip(st_ep["subjective_confidence_pct"], st_ep["is_correct"])])) if len(st_ep) > 0 else 0.0
        br_o_t = float(np.mean([((c / 100.0) - (1.0 if y else 0.0)) ** 2 for c, y in zip(ot_ep["subjective_confidence_pct"], ot_ep["is_correct"])])) if len(ot_ep) > 0 else 0.0
        delta_brier_trans_by_ep.append(br_o_t - br_s_t)  # Positive means Self has lower (better) Brier score

        # Scaffolded format
        ss_ep = df_self_scaff[df_self_scaff["episode_id"] == ep]
        os_ep = df_obs_scaff[df_obs_scaff["episode_id"] == ep]
        
        auc_s_s = calculate_auroc(ss_ep["subjective_confidence_pct"].tolist(), ss_ep["is_correct"].tolist())
        auc_o_s = calculate_auroc(os_ep["subjective_confidence_pct"].tolist(), os_ep["is_correct"].tolist())
        d_auc_s = auc_s_s - auc_o_s
        delta_auroc_scaff_by_ep.append(d_auc_s)

        br_s_s = float(np.mean([((c / 100.0) - (1.0 if y else 0.0)) ** 2 for c, y in zip(ss_ep["subjective_confidence_pct"], ss_ep["is_correct"])])) if len(ss_ep) > 0 else 0.0
        br_o_s = float(np.mean([((c / 100.0) - (1.0 if y else 0.0)) ** 2 for c, y in zip(os_ep["subjective_confidence_pct"], os_ep["is_correct"])])) if len(os_ep) > 0 else 0.0
        delta_brier_scaff_by_ep.append(br_o_s - br_s_s)

        interact_by_ep.append(d_auc_s - d_auc_t)

    meta_interaction = MetacognitiveInteractionSummary(
        delta_auroc_transcript=_make_estimand("Delta_AUROC_Transcript", "Item-Paired Delta_AUROC (Self - Observer) under Transcript-Only", delta_auroc_trans_by_ep, baseline=0.0),
        delta_auroc_scaffolded=_make_estimand("Delta_AUROC_Scaffolded", "Item-Paired Delta_AUROC (Self - Observer) under Scaffolded State", delta_auroc_scaff_by_ep, baseline=0.0),
        delta_brier_transcript=_make_estimand("Delta_Brier_Transcript", "Item-Paired Delta_Brier (Observer Brier - Self Brier) under Transcript-Only", delta_brier_trans_by_ep, baseline=0.0),
        delta_brier_scaffolded=_make_estimand("Delta_Brier_Scaffolded", "Item-Paired Delta_Brier (Observer Brier - Self Brier) under Scaffolded State", delta_brier_scaff_by_ep, baseline=0.0),
        scaffolding_metacognitive_interaction=_make_estimand("Scaffolding_Metacognitive_Interaction", "Interaction: Delta_AUROC(Scaffolded) - Delta_AUROC(Transcript)", interact_by_ep, baseline=0.0),
    )

    return S09AnalysisSummary(
        total_episodes=n_eps,
        total_e08_trials=len(df_e08),
        total_e09_trials=len(df_e09),
        attribution_breakdown=attr_breakdown,
        cue_conflict=cue_summary,
        channel_factorial=chan_summary,
        self_peer_allegiance_contrast=est_belief,
        framing_discrepancy_gap=est_framing,
        ownership_revision_susceptibility=est_ors,
        metacognitive_conditions=meta_summaries,
        metacognitive_interaction=meta_interaction,
    )
