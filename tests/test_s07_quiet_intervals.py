"""Unit and integration tests for Sprint S07.1 (Experiment E06b: Available-Inference Null Consolidation)."""

import hashlib
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
    compute_state_hash,
)
from recurrence.analysis.quiet_metrics import (
    analyze_quiet_interval_results,
    compute_exact_mcnemar_test,
    compute_permutation_test,
    compute_episode_clustered_bootstrap,
    compute_derived_inference_metrics,
)
from experiments.e06_quiet_intervals.run import (
    MockQuietBackend,
    run_e06b_experiment,
)


def test_task_generator_regimes_and_foil_integrity():
    """Verify episode generator creates well-formed available and missing-premise episodes."""
    gen = QuietIntervalGenerator(seed=42)
    
    # Available regime
    ep_avail = gen.generate_episode(episode_idx=0, regime="available_inference")
    assert len(ep_avail.prefix_events) == 7
    assert len(ep_avail.continuation_events) == 2
    assert len(ep_avail.probes) == 3

    # Missing premise regime
    ep_miss = gen.generate_episode(episode_idx=0, regime="missing_premise_control")
    assert len(ep_miss.prefix_events) == 7
    assert len(ep_miss.continuation_events) == 2
    assert len(ep_miss.probes) == 3

    # Verify all foil values appear in the context
    context_values = set()
    for ev in ep_avail.prefix_events + ep_avail.continuation_events:
        for v in ev.key_bindings.values():
            context_values.add(v)

    for p in ep_avail.probes:
        if p.probe_type in ("derivation_multihop", "stable_kv"):
            for opt_val in p.options.values():
                assert opt_val in context_values, f"Foil '{opt_val}' was not present in context!"


def test_reference_state_and_goal_consistency():
    """Critical S07.1 Fix: verify that goal authorization updates goal_beta to active in both state and probe."""
    gen = QuietIntervalGenerator(seed=42)
    ep = gen.generate_episode(episode_idx=1, regime="available_inference")

    backend = MockQuietBackend()
    harness = QuietIntervalHarness(backend=backend)

    trials, _, _ = harness.execute_episode(episode=ep, interval_ks=[0, 1])

    # Goal probe should expect goal_beta to be active
    goal_probe = [p for p in ep.probes if p.probe_type == "goal_activation"][0]
    corr_ans = goal_probe.options[goal_probe.correct_letter]
    assert "status: active, authorization satisfied" in corr_ans

    # Prefix state must have goal_beta as active following tick 3 authorization
    goal_beta = [g for g in ep.oracle_prefix_state.goals if g.goal_id == "goal_beta"][0]
    assert goal_beta.status == "active"


def test_strict_identity_bit_for_bit_sha256_prompt_invariance():
    """Critical S07.1 Invariant: strict_identity must produce identical SHA-256 evaluation prompts across all K."""
    gen = QuietIntervalGenerator(seed=100)
    ep = gen.generate_episode(episode_idx=2, regime="available_inference")

    backend = MockQuietBackend()
    harness = QuietIntervalHarness(backend=backend)

    trials, _, _ = harness.execute_episode(episode=ep, interval_ks=[0, 1, 3, 6, 12], conditions=["strict_identity"])

    # For each probe, the evaluation prompt SHA-256 hash must be bit-for-bit identical across K
    probes_by_id = {}
    for t in trials:
        probes_by_id.setdefault(t.probe_id, []).append(t.prompt_hash)

    for pid, hashes in probes_by_id.items():
        assert len(set(hashes)) == 1, f"Prompt hash varied across K for probe {pid} in strict_identity!"


def test_clock_only_vs_semantic_nowrite_context_hash_equality():
    """Critical S07.1 Invariant: clock_only and semantic_no_write must produce identical context strings across all K."""
    gen = QuietIntervalGenerator(seed=42)
    ep = gen.generate_episode(episode_idx=5, regime="available_inference")

    backend = MockQuietBackend()
    harness = QuietIntervalHarness(backend=backend)

    trials, _, _ = harness.execute_episode(
        episode=ep,
        interval_ks=[1, 3, 6, 12],
        conditions=["clock_only", "semantic_no_write"],
    )

    by_k_probe = {}
    for t in trials:
        by_k_probe.setdefault((t.interval_k, t.probe_id), {})[t.condition] = t.context_hash

    for (k, pid), cond_hashes in by_k_probe.items():
        assert "clock_only" in cond_hashes and "semantic_no_write" in cond_hashes
        assert cond_hashes["clock_only"] == cond_hashes["semantic_no_write"], (
            f"Context hash mismatch between clock_only and semantic_no_write at K={k} for probe {pid}!"
        )


def test_protected_evidence_hash_invariance():
    """Critical S07.1 Invariant: selective_reflection must NEVER mutate working_memory or source_ledger."""
    gen = QuietIntervalGenerator(seed=42)
    ep = gen.generate_episode(episode_idx=3, regime="available_inference")

    backend = MockQuietBackend()
    harness = QuietIntervalHarness(backend=backend)

    trials, traces, _ = harness.execute_episode(episode=ep, interval_ks=[1, 3, 6, 12])
    
    sel_trials = [t for t in trials if t.condition == "selective_reflection"]
    assert len(sel_trials) > 0
    assert all(t.evidence_hash_valid is True for t in sel_trials)
    assert all(t.evidence_drift_detected is False for t in sel_trials)
    assert len(traces) == 36  # 12 ticks x 3 reflection conditions


def test_reflection_trace_audit_logging():
    """Verify that all quiet reflection ticks record schema validity, prompt hash, and state transitions."""
    gen = QuietIntervalGenerator(seed=42)
    ep = gen.generate_episode(episode_idx=4, regime="available_inference")

    backend = MockQuietBackend()
    harness = QuietIntervalHarness(backend=backend)

    _, traces, _ = harness.execute_episode(episode=ep, interval_ks=[1, 3, 6, 12])
    assert len(traces) == 36

    for tr in traces:
        assert tr.schema_valid is True
        assert len(tr.prompt_hash) == 64
        assert len(tr.pre_state_hash) == 64
        assert len(tr.post_state_hash) == 64


def test_e06b_runner_dry_run(tmp_path: Path):
    """Verify end-to-end dry run execution of E06b runner producing all required artifacts."""
    res = run_e06b_experiment(
        model_name="qwen2.5:3b",
        seed=42,
        phase="exploratory",
        episodes_count=2,
        intervals=[0, 1, 3],
        dry_run=True,
        output_dir=tmp_path / "e06b_dryrun",
    )

    assert res["manifest"]["total_episodes"] == 4  # 2 episodes x 2 regimes
    assert (tmp_path / "e06b_dryrun" / "manifest.json").exists()
    assert (tmp_path / "e06b_dryrun" / "summary.json").exists()
    assert (tmp_path / "e06b_dryrun" / "trials.jsonl").exists()
    assert (tmp_path / "e06b_dryrun" / "reflection_traces.jsonl").exists()
    assert (tmp_path / "e06b_dryrun" / "trials.parquet").exists()
    assert (tmp_path / "e06b_dryrun" / "report.md").exists()
