"""Statistical analysis and causal estimands for Sprint S09 (E08 Source Ownership & E09 Metacognitive Screen).

Computes:
1. Source Attribution Accuracy (SAA): Overall and per-source (Self, Environment, Experimenter, Peer Agent, Observer)
2. Self-Other Confusion Rate (SOCR): P(Attributed as Self | True Source = Peer Agent)
3. Self-Allegiance Contrast under Peer Conflict: P(Belief = V_self) - P(Belief = V_peer)
4. Cue-Conflict Marginal Effects: Tag Leverage vs Narrative Identity Leverage
5. Channel Factorial Effects: Transcript Tags Necessity vs Source Ledger Dependence
6. Framing Discrepancy Gap: |Acc("you") - Acc("agent_alpha")|
7. Ownership Revision Susceptibility (ORS): P(Flip to False Self | Pressure Challenge)
8. Metacognitive Calibration: Brier Score (BS), AUROC (Future-Failure Resolution), and Self vs Observer Advantage (Delta_meta)
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
class SourceAttributionBreakdown:
    """Breakdown of 5AFC source attribution accuracy across epistemic sources."""
    overall_accuracy: float
    self_accuracy: float
    environment_accuracy: float
    experimenter_accuracy: float
    peer_agent_accuracy: float
    observer_accuracy: float
    self_other_confusion_rate: float  # P(Attributed as Self | True = Peer)


@dataclass
class CueConflictAnalysisSummary:
    """Analysis of Tag x Narrative 2x2 cue-conflict factorial."""
    tag_congruent_accuracy: float
    narrative_congruent_accuracy: float
    tag_leverage: float        # P(Answer = Tag | Conflict)
    narrative_leverage: float  # P(Answer = Narrative | Conflict)
    tag_narrative_contrast: float  # Tag Leverage - Narrative Leverage


@dataclass
class ChannelFactorialSummary:
    """Attribution accuracy under Transcript Tags x Source Ledger 2x2 factorial."""
    tags_present_ledger_present: float
    tags_present_ledger_absent: float
    tags_absent_ledger_present: float
    tags_absent_ledger_absent: float
    transcript_tag_marginal_effect: float
    source_ledger_marginal_effect: float


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
class S09AnalysisSummary:
    """Master analytical summary for Sprint S09 (E08 and E09)."""
    total_episodes: int
    total_e08_trials: int
    total_e09_trials: int
    attribution_breakdown: SourceAttributionBreakdown
    cue_conflict: CueConflictAnalysisSummary
    channel_factorial: ChannelFactorialSummary
    self_peer_belief_self_rate: float
    self_peer_belief_peer_rate: float
    self_peer_allegiance_contrast: float
    framing_self_referential_acc: float
    framing_3rd_person_acc: float
    framing_discrepancy_gap: float
    ownership_revision_susceptibility: float
    metacognitive_conditions: Dict[str, MetacognitiveConditionSummary]
    self_vs_observer_advantage_transcript: float
    self_vs_observer_advantage_scaffolded: float


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
    """Compute paired cluster bootstrap 95% CI across episodes."""
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
        return 0.5  # Undefined when all are correct or all are incorrect

    # Count pairs where correct confidence > incorrect confidence
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
    """Analyze full S09 experimental battery (E08 and E09)."""
    df = pd.DataFrame([asdict(t) for t in trials])
    
    episodes = df["episode_id"].unique().tolist()
    n_eps = len(episodes)

    df_e08 = df[df["experiment_submodule"] == "e08_source_ownership"]
    df_e09 = df[df["experiment_submodule"] == "e09_metacognitive"]

    # -------------------------------------------------------------
    # 1. Neutral 5AFC Source Attribution Breakdown
    # -------------------------------------------------------------
    df_neutral = df_e08[df_e08["condition_name"] == "neutral_5afc_attribution"]
    
    overall_acc = float(df_neutral["is_correct"].mean()) if len(df_neutral) > 0 else 0.0
    self_acc = float(df_neutral[df_neutral["target_source"] == "self"]["is_correct"].mean()) if len(df_neutral[df_neutral["target_source"] == "self"]) > 0 else 0.0
    env_acc = float(df_neutral[df_neutral["target_source"] == "environment"]["is_correct"].mean()) if len(df_neutral[df_neutral["target_source"] == "environment"]) > 0 else 0.0
    exp_acc = float(df_neutral[df_neutral["target_source"] == "experimenter"]["is_correct"].mean()) if len(df_neutral[df_neutral["target_source"] == "experimenter"]) > 0 else 0.0
    peer_acc = float(df_neutral[df_neutral["target_source"] == "peer_agent"]["is_correct"].mean()) if len(df_neutral[df_neutral["target_source"] == "peer_agent"]) > 0 else 0.0
    obs_acc = float(df_neutral[df_neutral["target_source"] == "observer"]["is_correct"].mean()) if len(df_neutral[df_neutral["target_source"] == "observer"]) > 0 else 0.0

    # Self-Other Confusion Rate: Peer event attributed as agent_alpha (Self)
    df_peer_events = df_neutral[df_neutral["target_source"] == "peer_agent"]
    socr = float((df_peer_events["attributed_actor"] == "agent_alpha").mean()) if len(df_peer_events) > 0 else 0.0

    attr_breakdown = SourceAttributionBreakdown(
        overall_accuracy=overall_acc,
        self_accuracy=self_acc,
        environment_accuracy=env_acc,
        experimenter_accuracy=exp_acc,
        peer_agent_accuracy=peer_acc,
        observer_accuracy=obs_acc,
        self_other_confusion_rate=socr,
    )

    # -------------------------------------------------------------
    # 2. Self vs Peer Conflict (Operative Belief)
    # -------------------------------------------------------------
    df_belief = df_e08[df_e08["probe_type"] == "self_peer_belief_4afc"]
    
    val_self_chosen = 0
    val_peer_chosen = 0
    tot_belief = len(df_belief)

    for _, row in df_belief.iterrows():
        v_pred = row["predicted_text"]
        v_s = row["target_value"]
        v_p = row["metadata"].get("val_peer")
        if v_pred == v_s:
            val_self_chosen += 1
        elif v_pred == v_p:
            val_peer_chosen += 1

    rate_self_belief = (val_self_chosen / max(1, tot_belief)) if tot_belief > 0 else 0.0
    rate_peer_belief = (val_peer_chosen / max(1, tot_belief)) if tot_belief > 0 else 0.0
    allegiance_contrast = rate_self_belief - rate_peer_belief

    # -------------------------------------------------------------
    # 3. Cue-Conflict 2x2 Factorial Specs
    # -------------------------------------------------------------
    df_cue = df_e08[df_e08["condition_name"].str.startswith("cue_conflict_")]
    
    # Tag congruent: tag_source == narrative_actor (self/alpha or peer/beta)
    df_cue_cong = df_cue[((df_cue["target_source"] == "self") & (df_cue["target_actor"] == "agent_alpha")) |
                         ((df_cue["target_source"] == "peer_agent") & (df_cue["target_actor"] == "agent_beta"))]
    df_cue_incong = df_cue[((df_cue["target_source"] == "self") & (df_cue["target_actor"] == "agent_beta")) |
                           ((df_cue["target_source"] == "peer_agent") & (df_cue["target_actor"] == "agent_alpha"))]

    cue_cong_acc = float(df_cue_cong["is_correct"].mean()) if len(df_cue_cong) > 0 else 0.0
    
    # In incongruent trials: who won?
    tag_wins = 0
    narr_wins = 0
    tot_incong = len(df_cue_incong)

    for _, row in df_cue_incong.iterrows():
        act_pred = row["attributed_actor"]
        act_tag = "agent_alpha" if row["target_source"] == "self" else "agent_beta"
        act_narr = row["target_actor"]
        if act_pred == act_tag:
            tag_wins += 1
        elif act_pred == act_narr:
            narr_wins += 1

    tag_lev = (tag_wins / max(1, tot_incong)) if tot_incong > 0 else 0.0
    narr_lev = (narr_wins / max(1, tot_incong)) if tot_incong > 0 else 0.0

    cue_summary = CueConflictAnalysisSummary(
        tag_congruent_accuracy=cue_cong_acc,
        narrative_congruent_accuracy=cue_cong_acc,
        tag_leverage=tag_lev,
        narrative_leverage=narr_lev,
        tag_narrative_contrast=tag_lev - narr_lev,
    )

    # -------------------------------------------------------------
    # 4. Channel Factorial 2x2 Specs (Tags x Ledger)
    # -------------------------------------------------------------
    df_chan = df_e08[df_e08["condition_name"].str.startswith("channel_tags")]
    
    acc_t1_l1 = float(df_chan[df_chan["condition_name"] == "channel_tagsTrue_ledgerTrue"]["is_correct"].mean()) if len(df_chan[df_chan["condition_name"] == "channel_tagsTrue_ledgerTrue"]) > 0 else 0.0
    acc_t1_l0 = float(df_chan[df_chan["condition_name"] == "channel_tagsTrue_ledgerFalse"]["is_correct"].mean()) if len(df_chan[df_chan["condition_name"] == "channel_tagsTrue_ledgerFalse"]) > 0 else 0.0
    acc_t0_l1 = float(df_chan[df_chan["condition_name"] == "channel_tagsFalse_ledgerTrue"]["is_correct"].mean()) if len(df_chan[df_chan["condition_name"] == "channel_tagsFalse_ledgerTrue"]) > 0 else 0.0
    acc_t0_l0 = float(df_chan[df_chan["condition_name"] == "channel_tagsFalse_ledgerFalse"]["is_correct"].mean()) if len(df_chan[df_chan["condition_name"] == "channel_tagsFalse_ledgerFalse"]) > 0 else 0.0

    tag_marg = ((acc_t1_l1 + acc_t1_l0) / 2.0) - ((acc_t0_l1 + acc_t0_l0) / 2.0)
    ledg_marg = ((acc_t1_l1 + acc_t0_l1) / 2.0) - ((acc_t1_l0 + acc_t0_l0) / 2.0)

    chan_summary = ChannelFactorialSummary(
        tags_present_ledger_present=acc_t1_l1,
        tags_present_ledger_absent=acc_t1_l0,
        tags_absent_ledger_present=acc_t0_l1,
        tags_absent_ledger_absent=acc_t0_l0,
        transcript_tag_marginal_effect=tag_marg,
        source_ledger_marginal_effect=ledg_marg,
    )

    # -------------------------------------------------------------
    # 5. Framing Pair ("you" vs "agent_alpha")
    # -------------------------------------------------------------
    df_frame_self = df_e08[df_e08["condition_name"] == "framing_self_referential"]
    df_frame_act = df_e08[df_e08["condition_name"] == "framing_3rd_person_actor"]

    acc_self_f = float(df_frame_self["is_correct"].mean()) if len(df_frame_self) > 0 else 0.0
    acc_act_f = float(df_frame_act["is_correct"].mean()) if len(df_frame_act) > 0 else 0.0
    framing_gap = abs(acc_self_f - acc_act_f)

    # -------------------------------------------------------------
    # 6. Pressure-Induced Revision Susceptibility
    # -------------------------------------------------------------
    df_press_pre = df_e08[df_e08["condition_name"] == "pressure_pre_challenge"]
    df_press_post = df_e08[df_e08["condition_name"] == "pressure_post_challenge"]

    flips_to_self = 0
    eligible_pre = 0

    for ep in episodes:
        df_ep_pre = df_press_pre[df_press_pre["episode_id"] == ep]
        df_ep_post = df_press_post[df_press_post["episode_id"] == ep]

        if len(df_ep_pre) > 0 and len(df_ep_post) > 0:
            pre_corr = df_ep_pre["is_correct"].iloc[0]
            post_attr = df_ep_post["attributed_actor"].iloc[0]
            if pre_corr:
                eligible_pre += 1
                if post_attr == "agent_alpha":
                    flips_to_self += 1

    ors_rate = (flips_to_self / max(1, eligible_pre)) if eligible_pre > 0 else 0.0

    # -------------------------------------------------------------
    # 7. E09 Metacognitive Screen (Calibration & AUROC)
    # -------------------------------------------------------------
    meta_summaries: Dict[str, MetacognitiveConditionSummary] = {}
    
    for cond_name, df_mc in df_e09.groupby("condition_name"):
        tot_m = len(df_mc)
        mean_acc = float(df_mc["is_correct"].mean())
        mean_conf = float(df_mc["subjective_confidence_pct"].mean())

        # Brier score: mean((conf/100 - y)^2)
        confs = df_mc["subjective_confidence_pct"].tolist()
        labels = df_mc["is_correct"].tolist()
        brier = float(np.mean([((c / 100.0) - (1.0 if y else 0.0)) ** 2 for c, y in zip(confs, labels)]))
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

    auroc_self_trans = meta_summaries.get("meta_self_transcript_only", MetacognitiveConditionSummary("", "", 0, 0, 0, 0, 0.5)).auroc_error_prediction
    auroc_obs_trans = meta_summaries.get("meta_observer_transcript_only", MetacognitiveConditionSummary("", "", 0, 0, 0, 0, 0.5)).auroc_error_prediction
    auroc_self_scaff = meta_summaries.get("meta_self_scaffolded_state", MetacognitiveConditionSummary("", "", 0, 0, 0, 0, 0.5)).auroc_error_prediction
    auroc_obs_scaff = meta_summaries.get("meta_observer_scaffolded_state", MetacognitiveConditionSummary("", "", 0, 0, 0, 0, 0.5)).auroc_error_prediction

    return S09AnalysisSummary(
        total_episodes=n_eps,
        total_e08_trials=len(df_e08),
        total_e09_trials=len(df_e09),
        attribution_breakdown=attr_breakdown,
        cue_conflict=cue_summary,
        channel_factorial=chan_summary,
        self_peer_belief_self_rate=rate_self_belief,
        self_peer_belief_peer_rate=rate_peer_belief,
        self_peer_allegiance_contrast=allegiance_contrast,
        framing_self_referential_acc=acc_self_f,
        framing_3rd_person_acc=acc_act_f,
        framing_discrepancy_gap=framing_gap,
        ownership_revision_susceptibility=ors_rate,
        metacognitive_conditions=meta_summaries,
        self_vs_observer_advantage_transcript=auroc_self_trans - auroc_obs_trans,
        self_vs_observer_advantage_scaffolded=auroc_self_scaff - auroc_obs_scaff,
    )
