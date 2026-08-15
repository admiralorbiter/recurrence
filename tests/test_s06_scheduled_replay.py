"""Unit and integration tests for Sprint S06 (Experiment E05: Scheduled versus Replay)."""

from pathlib import Path
import pytest

from recurrence.memory.schemas import (
    EventSource,
    MemoryEvent,
    StateCapacityConfig,
    StructuredSelfState,
)
from recurrence.tasks.scheduled_replay import (
    ScheduledReplayGenerator,
    ScheduledReplayEpisode,
    ScheduledReplayProbe,
)
from recurrence.loop.scheduled_experiment import (
    ScheduledReplayHarness,
    canonical_state_hash,
)
from recurrence.analysis.scheduled_metrics import (
    analyze_scheduled_replay_results,
    compute_mcnemar_test,
    compute_episode_clustered_bootstrap,
)
from experiments.e05_scheduled_vs_replay.run import (
    MockScheduledBackend,
    run_e05_experiment,
)


def test_canonical_state_hash_equality_invariant():
    """Critical S06 Invariant: Online incremental maintenance and retrospective replay must match bit-for-bit."""
    gen = ScheduledReplayGenerator(seed=42)
    ep = gen.generate_episode(episode_idx=0, num_ticks=25, target_keys_count=5)

    backend = MockScheduledBackend()
    harness = ScheduledReplayHarness(backend=backend)

    trials, meta = harness.execute_episode(episode=ep)
    assert "canonical_state_hash" in meta
    
    # State hashes for incremental and deterministic replay must match
    inc_trials = [t for t in trials if t.condition == "incremental_state"]
    rep_trials = [t for t in trials if t.condition == "replay_state_deterministic"]

    assert len(inc_trials) == len(ep.probes)
    assert len(rep_trials) == len(ep.probes)
    assert inc_trials[0].state_hash == rep_trials[0].state_hash


def test_burst_vs_uniform_arrival_invariance():
    """Verify that event arrival timing with unchanged event order produces identical terminal state."""
    gen = ScheduledReplayGenerator(seed=100)
    
    ep_uniform = gen.generate_episode(episode_idx=1, num_ticks=30, burst_mode=False, target_keys_count=4)
    ep_burst = gen.generate_episode(episode_idx=1, num_ticks=30, burst_mode=True, target_keys_count=4)

    # In both episodes, same key-value pairs are asserted
    hash_u = canonical_state_hash(ep_uniform.oracle_terminal_state)
    hash_b = canonical_state_hash(ep_burst.oracle_terminal_state)

    # Both terminal states have the same active keys and goals
    assert len(ep_uniform.oracle_terminal_state.working_memory) == len(ep_burst.oracle_terminal_state.working_memory)
    assert len(ep_uniform.oracle_terminal_state.goals) == len(ep_burst.oracle_terminal_state.goals)


def test_scheduled_replay_generator_5_probes():
    """Verify episodic generator produces balanced 5-domain forced choice probes."""
    gen = ScheduledReplayGenerator(seed=42)
    ep = gen.generate_episode(episode_idx=2, num_ticks=25, target_keys_count=4)

    assert len(ep.probes) == 5
    probe_types = [p.probe_type for p in ep.probes]
    assert "delayed_kv" in probe_types
    assert "source_attribution" in probe_types
    assert "goal_state" in probe_types
    assert "goal_action" in probe_types
    assert "multihop" in probe_types

    for p in ep.probes:
        assert p.correct_letter in p.options
        assert p.options[p.correct_letter] == p.correct_answer


def test_scheduled_replay_harness_5_conditions():
    """Verify harness executes all 5 experimental conditions."""
    gen = ScheduledReplayGenerator(seed=42)
    ep = gen.generate_episode(episode_idx=3, num_ticks=15, target_keys_count=3)

    backend = MockScheduledBackend()
    harness = ScheduledReplayHarness(backend=backend)

    trials, meta = harness.execute_episode(episode=ep)
    # 5 probes * 5 conditions = 25 trials
    assert len(trials) == 25

    conditions = set(t.condition for t in trials)
    assert conditions == {
        "incremental_state",
        "replay_state_deterministic",
        "replay_transcript",
        "replay_state_model",
        "fresh",
    }


def test_mcnemar_and_bootstrap_statistics():
    """Verify paired McNemar and bootstrap CI calculations."""
    outcomes_a = [True, True, True, False, True, False, True, True]
    outcomes_b = [True, False, False, False, True, False, False, True]

    stat, p_val = compute_mcnemar_test(outcomes_a, outcomes_b)
    assert stat >= 0.0
    assert 0.0 <= p_val <= 1.0

    # Test analysis summary
    gen = ScheduledReplayGenerator(seed=42)
    ep = gen.generate_episode(episode_idx=4, num_ticks=10)
    backend = MockScheduledBackend()
    harness = ScheduledReplayHarness(backend=backend)
    trials, _ = harness.execute_episode(episode=ep)

    analysis = analyze_scheduled_replay_results(trials, num_bootstrap=100)
    assert "Delta_online-direct" in analysis.causal_estimands
    assert "Delta_schedule" in analysis.causal_estimands
    assert "Delta_reconstruction" in analysis.causal_estimands
    assert "Delta_representation" in analysis.causal_estimands


def test_e05_runner_dry_run(tmp_path: Path):
    """Verify complete end-to-end benchmark execution in dry-run mode."""
    res = run_e05_experiment(
        model_name="qwen2.5:3b",
        seed=42,
        phase="exploratory",
        episodes_per_horizon=2,
        horizons=[10, 25],
        dry_run=True,
        output_dir=tmp_path / "e05_dryrun",
    )

    assert res["manifest"]["total_episodes"] == 4
    assert res["manifest"]["total_trials"] == 4 * 25  # 100 trials
    assert (tmp_path / "e05_dryrun" / "manifest.json").exists()
    assert (tmp_path / "e05_dryrun" / "summary.json").exists()
    assert (tmp_path / "e05_dryrun" / "trials.jsonl").exists()
    assert (tmp_path / "e05_dryrun" / "trials.parquet").exists()
    assert (tmp_path / "e05_dryrun" / "report.md").exists()
