"""Hardened unit and integration tests for Sprint S06.2 (Experiment E05c: Scheduled versus Replay)."""

from pathlib import Path
import re
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
    compute_exact_mcnemar_test,
    compute_permutation_test,
    compute_episode_clustered_bootstrap,
)
from experiments.e05_scheduled_vs_replay.run import (
    MockScheduledBackend,
    run_e05_experiment,
)


def test_canonical_state_hash_and_prompt_equality_invariant():
    """Critical S06 Invariant: Online incremental and retrospective replay must match bit-for-bit in state and prompt."""
    gen = ScheduledReplayGenerator(seed=42)
    ep = gen.generate_episode(episode_idx=0, num_ticks=25, target_keys_count=5)

    backend = MockScheduledBackend()
    harness = ScheduledReplayHarness(backend=backend)

    trials, meta = harness.execute_episode(episode=ep)
    assert "canonical_state_hash" in meta
    
    inc_trials = [t for t in trials if t.condition == "incremental_state"]
    rep_trials = [t for t in trials if t.condition == "replay_state_deterministic"]

    assert len(inc_trials) == len(ep.probes)
    assert len(rep_trials) == len(ep.probes)

    for t_inc, t_rep in zip(inc_trials, rep_trials):
        assert t_inc.state_hash == t_rep.state_hash
        assert t_inc.prompt_hash == t_rep.prompt_hash


def test_burst_vs_uniform_arrival_invariance():
    """Verify that event arrival timing with unchanged event order produces identical terminal state."""
    gen = ScheduledReplayGenerator(seed=100)
    
    ep_uniform = gen.generate_episode(episode_idx=1, num_ticks=30, burst_mode=False, target_keys_count=4)
    ep_burst = gen.generate_episode(episode_idx=1, num_ticks=30, burst_mode=True, target_keys_count=4)

    hash_u = canonical_state_hash(ep_uniform.oracle_terminal_state)
    hash_b = canonical_state_hash(ep_burst.oracle_terminal_state)

    # State hashes must be strictly identical
    assert hash_u == hash_b
    assert ep_uniform.oracle_terminal_state.working_memory == ep_burst.oracle_terminal_state.working_memory


def test_in_context_foils_for_kv_and_multihop():
    """Verify that 100% of candidate options in Delayed KV and Multi-Hop appear in the episode's context."""
    gen = ScheduledReplayGenerator(seed=42)

    for ep_idx in range(12):
        ep = gen.generate_episode(episode_idx=ep_idx, num_ticks=25)
        
        # Collect all values present in this episode's scheduled event stream
        episode_values = set()
        for ev in ep.scheduled_events:
            for v in ev.key_bindings.values():
                episode_values.add(v)
            if "final_val" in ev.metadata:
                episode_values.add(ev.metadata["final_val"])

        for p in ep.probes:
            if p.probe_type in ("delayed_kv", "multihop"):
                # Every single candidate option must be in episode_values
                for opt_letter, opt_val in p.options.items():
                    assert opt_val in episode_values, (
                        f"Foil value '{opt_val}' in probe {p.probe_id} was NOT present in episode context!"
                    )


def test_explicit_pending_goal_in_oracle_state():
    """Verify that when chosen_sec_status is 'pending', goal_secondary exists with status pending in state."""
    gen = ScheduledReplayGenerator(seed=42)

    for ep_idx in range(8):
        ep = gen.generate_episode(episode_idx=ep_idx, num_ticks=25)
        if ep.metadata.get("counterbalanced_goal_status") == "pending":
            sec_goals = [g for g in ep.oracle_terminal_state.goals if g.goal_id == "goal_secondary"]
            assert len(sec_goals) == 1, f"Episode {ep.episode_id} missing goal_secondary in oracle state!"
            sec_st = sec_goals[0].status
            st_val = sec_st.value if hasattr(sec_st, "value") else str(sec_st)
            assert st_val == "pending", f"Expected pending goal status, got {st_val}"


def test_no_numerical_or_role_suffix_shortcuts():
    """Verify that all keys, values, and foils have zero numerical suffixes or role markers."""
    gen = ScheduledReplayGenerator(seed=42)
    digit_pattern = re.compile(r"\d")

    for ep_idx in range(10):
        ep = gen.generate_episode(episode_idx=ep_idx, num_ticks=25)
        
        for ev in ep.scheduled_events:
            for k, v in ev.key_bindings.items():
                assert not digit_pattern.search(k), f"Leaked digit in key: {k}"
                assert not digit_pattern.search(v), f"Leaked digit in val: {v}"

        for p in ep.probes:
            if p.probe_type in ("delayed_kv", "multihop"):
                assert not digit_pattern.search(p.correct_answer), f"Leaked digit in answer: {p.correct_answer}"
                for foil in p.options.values():
                    assert not digit_pattern.search(foil), f"Leaked digit in foil: {foil}"


def test_balanced_goal_and_source_counterbalancing():
    """Verify goal statuses and source targets are evenly counterbalanced across episodes."""
    gen = ScheduledReplayGenerator(seed=42)
    
    goal_statuses = []
    source_targets = []

    for ep_idx in range(12):
        ep = gen.generate_episode(episode_idx=ep_idx, num_ticks=25)
        goal_probe = [p for p in ep.probes if p.probe_type == "goal_state"][0]
        src_probe = [p for p in ep.probes if p.probe_type == "source_attribution"][0]
        
        goal_statuses.append(goal_probe.correct_answer)
        source_targets.append(src_probe.correct_answer)

    for st in ["active", "suspended", "completed", "pending"]:
        assert goal_statuses.count(st) == 3, f"Goal status {st} count: {goal_statuses.count(st)}"

    for src in ["environment", "self", "experimenter"]:
        assert source_targets.count(src) == 4, f"Source target {src} count: {source_targets.count(src)}"


def test_exact_mcnemar_and_permutation_statistics():
    """Verify exact two-sided binomial McNemar and permutation test calculation."""
    outcomes_a = [True, True, True, False, True, False, True, True]
    outcomes_b = [True, False, False, False, True, False, False, True]

    b, c, p_val = compute_exact_mcnemar_test(outcomes_a, outcomes_b)
    assert b == 3
    assert c == 0
    assert p_val == pytest.approx(0.25)

    diffs = [0.25, -0.25, 0.5, 0.0]
    p_perm, method = compute_permutation_test(diffs)
    assert 0.0 <= p_perm <= 1.0
    assert method == "exact_exhaustive"


def test_e05_runner_dry_run(tmp_path: Path):
    """Verify complete end-to-end benchmark execution in dry-run mode."""
    res = run_e05_experiment(
        model_name="qwen2.5:3b",
        seed=42,
        phase="exploratory",
        episodes_per_horizon=2,
        horizons=[10, 25],
        dry_run=True,
        output_dir=tmp_path / "e05c_dryrun",
    )

    assert res["manifest"]["total_episodes"] == 4
    assert res["manifest"]["total_trials"] == 4 * 20  # 80 trials
    assert (tmp_path / "e05c_dryrun" / "manifest.json").exists()
    assert (tmp_path / "e05c_dryrun" / "summary.json").exists()
    assert (tmp_path / "e05c_dryrun" / "trials.jsonl").exists()
    assert (tmp_path / "e05c_dryrun" / "trials.parquet").exists()
    assert (tmp_path / "e05c_dryrun" / "report.md").exists()
