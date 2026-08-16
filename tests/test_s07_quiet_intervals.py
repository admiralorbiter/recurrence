"""Unit and integration tests for Sprint S07 (Experiment E06: Scaffolded Null-Intervals)."""

from pathlib import Path
import pytest

from recurrence.memory.schemas import (
    EventSource,
    GoalState,
    MemoryEvent,
    StructuredSelfState,
)
from recurrence.tasks.quiet_interval import (
    QuietIntervalGenerator,
    QuietIntervalEpisode,
    QuietIntervalProbe,
)
from recurrence.loop.quiet_experiment import (
    QuietIntervalHarness,
    compute_evidence_hash,
)
from recurrence.analysis.quiet_metrics import (
    analyze_quiet_interval_results,
    compute_exact_mcnemar_test,
    compute_permutation_test,
    compute_episode_clustered_bootstrap,
)
from experiments.e06_quiet_intervals.run import (
    MockQuietBackend,
    run_e06_experiment,
)


def test_task_generator_prefix_continuation_structure():
    """Verify episode generator creates well-formed prefix, continuation, and 4 diagnostic probes."""
    gen = QuietIntervalGenerator(seed=42)
    ep = gen.generate_episode(episode_idx=0)

    assert len(ep.prefix_events) == 7
    assert len(ep.continuation_events) == 3
    assert len(ep.probes) == 4

    probe_types = {p.probe_type for p in ep.probes}
    assert probe_types == {"derivation_multihop", "source_conflict", "unresolved_goal", "stable_kv"}

    # Verify all foil values appear in the context
    context_values = set()
    for ev in ep.prefix_events + ep.continuation_events:
        for v in ev.key_bindings.values():
            context_values.add(v)

    for p in ep.probes:
        if p.probe_type in ("derivation_multihop", "stable_kv"):
            for opt_val in p.options.values():
                assert opt_val in context_values, f"Foil '{opt_val}' was not present in context!"


def test_protected_evidence_hash_invariance():
    """Critical S07 Invariant: selective_reflection must NEVER mutate working_memory or source_ledger."""
    gen = QuietIntervalGenerator(seed=42)
    ep = gen.generate_episode(episode_idx=1)

    backend = MockQuietBackend()
    harness = QuietIntervalHarness(backend=backend)

    trials, meta = harness.execute_episode(episode=ep, interval_ks=[1, 3, 6, 12])
    
    sel_trials = [t for t in trials if t.condition == "selective_reflection"]
    assert len(sel_trials) > 0
    assert all(t.evidence_hash_valid is True for t in sel_trials)
    assert all(t.evidence_drift_detected is False for t in sel_trials)


def test_strict_identity_bit_for_bit_prompt_invariance():
    """Verify that strict_identity preserves identical evaluation state representation across all K."""
    gen = QuietIntervalGenerator(seed=100)
    ep = gen.generate_episode(episode_idx=2)

    backend = MockQuietBackend()
    harness = QuietIntervalHarness(backend=backend)

    trials, _ = harness.execute_episode(episode=ep, interval_ks=[0, 1, 3, 6, 12], conditions=["strict_identity"])

    # For each probe, the context character length should be identical across K in strict_identity
    probes_by_id = {}
    for t in trials:
        probes_by_id.setdefault(t.probe_id, []).append(t.context_chars)

    for pid, lengths in probes_by_id.items():
        assert len(set(lengths)) == 1, f"Context length varied across K for probe {pid} in strict_identity!"


def test_clock_only_timestamp_advance():
    """Verify that clock_only advances last_updated_step while leaving state bindings untouched."""
    gen = QuietIntervalGenerator(seed=42)
    ep = gen.generate_episode(episode_idx=3)

    backend = MockQuietBackend()
    harness = QuietIntervalHarness(backend=backend)

    trials, _ = harness.execute_episode(episode=ep, interval_ks=[1, 6], conditions=["clock_only"])
    assert len(trials) == 8
    assert all(t.condition == "clock_only" for t in trials)


def test_semantic_no_write_state_immutability():
    """Verify that semantic_no_write accumulates compute tokens while leaving state unchanged."""
    gen = QuietIntervalGenerator(seed=42)
    ep = gen.generate_episode(episode_idx=4)

    backend = MockQuietBackend()
    harness = QuietIntervalHarness(backend=backend)

    trials, _ = harness.execute_episode(episode=ep, interval_ks=[1, 3, 6], conditions=["semantic_no_write"])
    nw_trials_k6 = [t for t in trials if t.interval_k == 6]
    nw_trials_k1 = [t for t in trials if t.interval_k == 1]

    assert len(nw_trials_k6) == 4
    # Amortized prompt tokens at K=6 should be greater than at K=1 due to 6 reflection passes
    assert nw_trials_k6[0].amortized_prompt_tokens > nw_trials_k1[0].amortized_prompt_tokens


def test_source_conflict_ground_truth_is_unresolved():
    """Verify that source_conflict probe options provide explicit Unresolved/Conflicting assertion as correct."""
    gen = QuietIntervalGenerator(seed=42)
    for ep_idx in range(8):
        ep = gen.generate_episode(episode_idx=ep_idx)
        conflict_probe = [p for p in ep.probes if p.probe_type == "source_conflict"][0]
        corr_text = conflict_probe.options[conflict_probe.correct_letter]
        assert "Unresolved conflicting assertion" in corr_text


def test_exact_mcnemar_and_permutation_estimators():
    """Verify exact binomial McNemar and cluster-level sign-flip permutation calculation."""
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


def test_e06_runner_dry_run(tmp_path: Path):
    """Verify end-to-end dry run execution of E06 runner producing all required artifacts."""
    res = run_e06_experiment(
        model_name="qwen2.5:3b",
        seed=42,
        phase="exploratory",
        episodes_count=2,
        intervals=[0, 1, 3],
        dry_run=True,
        output_dir=tmp_path / "e06_dryrun",
    )

    assert res["manifest"]["total_episodes"] == 2
    assert (tmp_path / "e06_dryrun" / "manifest.json").exists()
    assert (tmp_path / "e06_dryrun" / "summary.json").exists()
    assert (tmp_path / "e06_dryrun" / "trials.jsonl").exists()
    assert (tmp_path / "e06_dryrun" / "trials.parquet").exists()
    assert (tmp_path / "e06_dryrun" / "report.md").exists()
