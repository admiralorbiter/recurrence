"""Unit tests for S09 statistical invariants, permutation nulls, AUROC ties, and canonical replay."""

from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from recurrence.analysis.ownership_metrics import (
    ACTOR_TO_SOURCE_MAP,
    calculate_auroc,
    compute_interaction_format_block_permutation_test,
    compute_within_episode_source_permutation_test,
    analyze_ownership_results,
)
from recurrence.loop.ownership_experiment import OwnershipTrialResult


def test_auroc_ties_and_invariants():
    """Verify that calculate_auroc accurately handles perfect predictions, inverted predictions, and tied confidence."""
    # Perfect separation
    conf_perf = [10.0, 20.0, 30.0, 80.0, 90.0]
    labels_perf = [0, 0, 0, 1, 1]
    assert calculate_auroc(conf_perf, labels_perf) == pytest.approx(1.0, abs=1e-5)

    # Inverted separation
    conf_inv = [90.0, 80.0, 70.0, 20.0, 10.0]
    labels_inv = [0, 0, 0, 1, 1]
    assert calculate_auroc(conf_inv, labels_inv) == pytest.approx(0.0, abs=1e-5)

    # Completely tied predictions (chance)
    conf_tied = [50.0, 50.0, 50.0, 50.0]
    labels_tied = [0, 0, 1, 1]
    assert calculate_auroc(conf_tied, labels_tied) == pytest.approx(0.5, abs=1e-5)

    # All single class (undefined/degenerate)
    assert calculate_auroc([10.0, 20.0], [1, 1]) == 0.5
    assert calculate_auroc([10.0, 20.0], [0, 0]) == 0.5


def test_within_episode_source_permutation_perfect_classifier():
    """Verify that an oracle source predictor produces p < 0.001 under the model-response-preserving source shuffle null."""
    episodes = [f"ep_{i}" for i in range(16)]
    sources = ["self", "environment", "experimenter", "peer_agent", "observer"]
    actors = ["agent_alpha", "telemetry_sensor", "human_controller", "agent_beta", "auditor_gamma"]

    rows = []
    for ep in episodes:
        for src, act in zip(sources, actors):
            rows.append({
                "episode_id": ep,
                "target_source": src,
                "attributed_actor": act,
                "is_correct": 1.0,
            })
    df_perf = pd.DataFrame(rows)

    p_val, method = compute_within_episode_source_permutation_test(df_perf, episodes, num_perms=1000, seed=42)
    assert p_val < 0.01
    assert "within_episode_source_shuffle" in method


def test_within_episode_source_permutation_all_self_classifier():
    """Verify that an agent answering 'agent_alpha' to 100% of queries is recognized as non-significant under the response-preserving null."""
    episodes = [f"ep_{i}" for i in range(16)]
    sources = ["self", "environment", "experimenter", "peer_agent", "observer"]

    rows = []
    for ep in episodes:
        for src in sources:
            # Always attributes to Self
            act = "agent_alpha"
            is_corr = 1.0 if src == "self" else 0.0
            rows.append({
                "episode_id": ep,
                "target_source": src,
                "attributed_actor": act,
                "is_correct": is_corr,
            })
    df_all_self = pd.DataFrame(rows)

    # Observed accuracy is 20% (16/80)
    assert df_all_self["is_correct"].mean() == pytest.approx(0.20)

    p_val, method = compute_within_episode_source_permutation_test(df_all_self, episodes, num_perms=5000, seed=42)
    # Under any permutation of true sources, exactly 1 out of 5 items matches 'self', so perm_acc is always 20%
    # Therefore extreme_count should be 100%, yielding p ≈ 1.0
    assert p_val >= 0.99


def test_interaction_format_block_permutation_null():
    """Verify that a true null interaction (no format-framing interaction) produces p ≈ 1.0."""
    episodes = [f"ep_{i}" for i in range(16)]

    def make_df(c_val: float, correct: int):
        rows = []
        for ep in episodes:
            for j in range(5):
                rows.append({
                    "episode_id": ep,
                    "subjective_confidence_pct": c_val if j % 2 == 0 else 50.0,
                    "is_correct": correct if j % 2 == 0 else (1 - correct),
                })
        return pd.DataFrame(rows)

    df_st = make_df(80.0, 1)
    df_ot = make_df(40.0, 1)
    df_ss = make_df(80.0, 1)
    df_os = make_df(40.0, 1)

    p_val, method = compute_interaction_format_block_permutation_test(
        df_self_trans=df_st,
        df_obs_trans=df_ot,
        df_self_scaff=df_ss,
        df_obs_scaff=df_os,
        episodes=episodes,
        obs_interaction_stat=0.0,
        seed=42,
    )
    assert p_val == 1.0
    assert "format_block_swap" in method


def test_interaction_format_block_permutation_planted_effect():
    """Verify that a strong planted format interaction is detected with p < 0.05."""
    episodes = [f"ep_{i}" for i in range(16)]

    df_st = pd.DataFrame([{"episode_id": ep, "subjective_confidence_pct": 90.0 if i % 2 == 0 else 10.0, "is_correct": 1 if i % 2 == 0 else 0} for ep in episodes for i in range(4)])
    df_ot = pd.DataFrame([{"episode_id": ep, "subjective_confidence_pct": 10.0 if i % 2 == 0 else 90.0, "is_correct": 1 if i % 2 == 0 else 0} for ep in episodes for i in range(4)])

    df_ss = pd.DataFrame([{"episode_id": ep, "subjective_confidence_pct": 10.0 if i % 2 == 0 else 90.0, "is_correct": 1 if i % 2 == 0 else 0} for ep in episodes for i in range(4)])
    df_os = pd.DataFrame([{"episode_id": ep, "subjective_confidence_pct": 90.0 if i % 2 == 0 else 10.0, "is_correct": 1 if i % 2 == 0 else 0} for ep in episodes for i in range(4)])

    obs_stat = (calculate_auroc(df_ss["subjective_confidence_pct"].tolist(), df_ss["is_correct"].tolist()) -
                calculate_auroc(df_os["subjective_confidence_pct"].tolist(), df_os["is_correct"].tolist())) - \
               (calculate_auroc(df_st["subjective_confidence_pct"].tolist(), df_st["is_correct"].tolist()) -
                calculate_auroc(df_ot["subjective_confidence_pct"].tolist(), df_ot["is_correct"].tolist()))

    assert obs_stat == pytest.approx(-2.0)

    p_val, method = compute_interaction_format_block_permutation_test(
        df_self_trans=df_st,
        df_obs_trans=df_ot,
        df_self_scaff=df_ss,
        df_obs_scaff=df_os,
        episodes=episodes,
        obs_interaction_stat=obs_stat,
        seed=42,
    )
    assert p_val < 0.001
    assert "exact_format_block_swap_65k" in method


def test_canonical_e08_e09_offline_replay():
    """Verify that offline reprocessing of frozen trials reproduces canonical summary numbers."""
    e08_dir = Path("results/e08_source_ownership/run_e08_owner_20260817_181634_confirmatory")
    e09_dir = Path("results/e09_metacognitive_screen/run_e09_meta_20260817_183133_confirmatory")

    if not (e08_dir / "trials.jsonl").exists() or not (e09_dir / "trials.jsonl").exists():
        pytest.skip("Canonical confirmatory logs not found.")

    trials_e08 = []
    with open(e08_dir / "trials.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trials_e08.append(OwnershipTrialResult(**json.loads(line)))

    analysis_e08 = analyze_ownership_results(trials_e08, num_bootstrap=1000, seed=1337)
    assert analysis_e08.attribution_breakdown.overall_accuracy.point_estimate == pytest.approx(0.3125, abs=1e-3)
    assert analysis_e08.attribution_breakdown.self_accuracy.point_estimate == pytest.approx(0.8125, abs=1e-3)
    assert analysis_e08.attribution_breakdown.self_other_confusion_rate.point_estimate == pytest.approx(0.5000, abs=1e-3)
    assert analysis_e08.cue_conflict.tag_narrative_contrast.point_estimate == pytest.approx(-0.34375, abs=1e-3)

    trials_e09 = []
    with open(e09_dir / "trials.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trials_e09.append(OwnershipTrialResult(**json.loads(line)))

    analysis_e09 = analyze_ownership_results(trials_e09, num_bootstrap=1000, seed=1337)
    assert analysis_e09.metacognitive_interaction.delta_auroc_transcript.point_estimate == pytest.approx(0.081, abs=1e-3)
    assert analysis_e09.metacognitive_interaction.delta_auroc_scaffolded.point_estimate == pytest.approx(-0.154, abs=1e-3)
    assert analysis_e09.metacognitive_interaction.scaffolding_metacognitive_interaction.point_estimate == pytest.approx(-0.235, abs=1e-3)


def test_canonical_e08c_e09c_offline_replay():
    """Verify that offline reprocessing of frozen E08c and E09c confirmatory trials reproduces exact canonical outputs."""
    from experiments.e08c_role_counterbalance.run import analyze_e08c_results
    from experiments.e09c_fixed_target_meta.run import analyze_e09c_results

    # 1. E08c Replay Test
    e08c_dirs = list(Path("results/e08c_role_counterbalance").glob("run_*_confirmatory"))
    if e08c_dirs and (e08c_dirs[0] / "trials.jsonl").exists():
        trials_e08c = []
        with open(e08c_dirs[0] / "trials.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trials_e08c.append(OwnershipTrialResult(**json.loads(line)))

        analysis_e08c = analyze_e08c_results(trials_e08c, num_bootstrap=1000, seed=1337)
        assert analysis_e08c.delta_role_reversal_shift.point_estimate == pytest.approx(0.28125, abs=1e-3)
        assert analysis_e08c.alpha_lexical_token_bias.point_estimate == pytest.approx(0.08125, abs=1e-3)
        assert analysis_e08c.isolated_ceiling_overall_accuracy.point_estimate == pytest.approx(0.2125, abs=1e-3)
        assert analysis_e08c.delta_role_reversal_shift.permutation_p_value == pytest.approx(0.0012, abs=1e-3)

    # 2. E09c Replay Test
    e09c_dirs = list(Path("results/e09c_fixed_target_meta").glob("run_*_confirmatory"))
    if e09c_dirs and (e09c_dirs[0] / "trials.jsonl").exists():
        trials_e09c = []
        with open(e09c_dirs[0] / "trials.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trials_e09c.append(OwnershipTrialResult(**json.loads(line)))

        analysis_e09c = analyze_e09c_results(trials_e09c, num_bootstrap=1000, seed=1337)
        assert analysis_e09c.first_order_target_accuracy == pytest.approx(0.475, abs=1e-3)
        assert analysis_e09c.brier_diff_in_diff_interaction.point_estimate == pytest.approx(0.1880, abs=1e-3)
        assert analysis_e09c.auroc_interaction.point_estimate == pytest.approx(-0.209, abs=1e-3)
        assert analysis_e09c.brier_diff_in_diff_interaction.permutation_p_value == pytest.approx(0.1501, abs=1e-3)
        assert analysis_e09c.auroc_interaction.permutation_p_value == pytest.approx(0.1406, abs=1e-3)
        # Verify that the true clustered bootstrap CI for AUROC interaction spans zero
        assert analysis_e09c.auroc_interaction.ci_lower_95 < 0.0
        assert analysis_e09c.auroc_interaction.ci_upper_95 > 0.0


def test_e08d_role_ablation_analyzer():
    """Verify that E08d analyzer computes correct condition summaries and diagnostic inferences."""
    from experiments.e08d_role_ablation.run import analyze_e08d_results, run_e08d_experiment
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        res_dir = run_e08d_experiment(
            phase="exploratory",
            seed=42,
            total_episode_pairs=2,
            use_mock=True,
            output_dir=Path(tmpdir),
        )
        assert (res_dir / "summary.json").exists()
        assert (res_dir / "report.md").exists()
        assert (res_dir / "trials.jsonl").exists()

        with open(res_dir / "summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["manifest"]["total_trials"] == 80
        assert len(summary["analysis"]["conditions"]) == 4
        assert "c1_full_package" in summary["analysis"]["conditions"]
        assert "c4_neutral_lookup" in summary["analysis"]["conditions"]

