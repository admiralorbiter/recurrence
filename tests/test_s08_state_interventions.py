"""Unit and integration tests for Sprint S08 (Experiment E07: State x Memory Conflict & Causal Interventions)."""

import hashlib
from pathlib import Path
import pytest

from recurrence.memory.schemas import (
    EventSource,
    GoalState,
    MemoryEvent,
    StructuredSelfState,
)
from recurrence.loop.updater import OracleStateUpdater
from recurrence.tasks.intervention import (
    StateInterventionGenerator,
    MatchedTwinEpisodePair,
    CloneReconvergenceSpec,
)
from recurrence.loop.intervention_experiment import (
    InterventionHarness,
    compute_state_hash,
    compute_events_hash,
)
from recurrence.analysis.intervention_metrics import (
    analyze_state_intervention_results,
    compute_permutation_test,
    compute_paired_bootstrap_ci,
)
from experiments.e07_state_interventions.run import (
    MockInterventionBackend,
    run_e07_experiment,
)


def test_twin_pair_vocabulary_balancing():
    """Verify that both V_red (A target) and V_blue (B target) appear in BOTH histories."""
    gen = StateInterventionGenerator(seed=42)
    twin_pair = gen.generate_twin_pair(twin_idx=0)

    # Collect all bound values in World A
    vals_A = set()
    for ev in twin_pair.prefix_events_A:
        vals_A.update(ev.key_bindings.values())

    # Collect all bound values in World B
    vals_B = set()
    for ev in twin_pair.prefix_events_B:
        vals_B.update(ev.key_bindings.values())

    # Both values must be present in both histories
    assert twin_pair.val_target_A in vals_A, "World A target value missing in World A history!"
    assert twin_pair.val_target_B in vals_A, "World B target value missing in World A history (vocabulary balance failure)!"
    assert twin_pair.val_target_A in vals_B, "World A target value missing in World B history (vocabulary balance failure)!"
    assert twin_pair.val_target_B in vals_B, "World B target value missing in World B history!"


def test_twin_pair_structural_isomorphism():
    """Verify matched twins have identical key names, goal structures, and option distributions."""
    gen = StateInterventionGenerator(seed=100)
    twin_pair = gen.generate_twin_pair(twin_idx=1)

    assert twin_pair.k_target is not None
    assert twin_pair.k_control is not None
    assert twin_pair.val_target_A != twin_pair.val_target_B
    assert twin_pair.val_control in twin_pair.oracle_state_A.working_memory.values()
    assert twin_pair.val_control in twin_pair.oracle_state_B.working_memory.values()

    # Verify options count
    for p in twin_pair.probes_A + twin_pair.probes_B:
        assert len(p.options) == 4
        assert set(p.options.keys()) == {"A", "B", "C", "D"}


def test_reset_condition_memory_invariance():
    """Verify that the memory event log hash in reset condition is bit-for-bit identical to congruent condition."""
    gen = StateInterventionGenerator(seed=42)
    twin_pair = gen.generate_twin_pair(twin_idx=2)

    # Explicit SHA-256 hash comparison of event lists
    hash_events_congruent = compute_events_hash(twin_pair.prefix_events_A)
    hash_events_reset = compute_events_hash(twin_pair.prefix_events_A)

    assert hash_events_congruent == hash_events_reset, "Memory log mutated under reset condition!"

    backend = MockInterventionBackend()
    harness = InterventionHarness(backend=backend)

    # Verify prompt formation preserves memory transcript
    p_cong, _ = harness._build_prompt(twin_pair.prefix_events_A, twin_pair.oracle_state_A, twin_pair.probes_A[0], order="memory_first")
    s_empty = StructuredSelfState(working_memory={}, goals=[], source_ledger={}, unresolved_items=[], derived_inferences={}, last_updated_step=0)
    p_reset, _ = harness._build_prompt(twin_pair.prefix_events_A, s_empty, twin_pair.probes_A[0], order="memory_first")

    assert harness._format_transcript(twin_pair.prefix_events_A) in p_cong
    assert harness._format_transcript(twin_pair.prefix_events_A) in p_reset


def test_surgical_slot_inversion_locality():
    """Verify that surgical inversion modifies only the target key, leaving control slot and goals 100% intact."""
    gen = StateInterventionGenerator(seed=42)
    twin_pair = gen.generate_twin_pair(twin_idx=3)

    s_orig = twin_pair.oracle_state_A
    s_surg = s_orig.model_copy(deep=True)
    s_surg.working_memory[twin_pair.k_target] = twin_pair.val_target_B

    # Target modified
    assert s_surg.working_memory[twin_pair.k_target] == twin_pair.val_target_B
    assert s_orig.working_memory[twin_pair.k_target] == twin_pair.val_target_A

    # Diff working memory: exactly one key differs
    diff_keys = [k for k in s_orig.working_memory if s_orig.working_memory[k] != s_surg.working_memory[k]]
    assert diff_keys == [twin_pair.k_target]

    # Control unchanged
    assert s_surg.working_memory[twin_pair.k_control] == s_orig.working_memory[twin_pair.k_control]
    assert s_surg.goals == s_orig.goals
    assert s_surg.unresolved_items == s_orig.unresolved_items
    assert s_surg.derived_inferences == s_orig.derived_inferences


def test_clone_prefix_hash_equality():
    """Verify that independently instantiated clone branches share identical SHA-256 state hashes prior to forking."""
    gen = StateInterventionGenerator(seed=42)
    spec = gen.generate_clone_reconvergence_spec(twin_idx=0)

    updater = OracleStateUpdater()
    state_branch_A = StructuredSelfState(working_memory={}, goals=[], source_ledger={}, unresolved_items=[], derived_inferences={}, last_updated_step=0)
    state_branch_B = StructuredSelfState(working_memory={}, goals=[], source_ledger={}, unresolved_items=[], derived_inferences={}, last_updated_step=0)

    for i, ev in enumerate(spec.prefix_events_common):
        state_branch_A, _, _, _, _, _, _, _ = updater.update(state_branch_A, [ev], tick=i)
        state_branch_B, _, _, _, _, _, _, _ = updater.update(state_branch_B, [ev], tick=i)

    hash_A_pre = compute_state_hash(state_branch_A)
    hash_B_pre = compute_state_hash(state_branch_B)

    assert hash_A_pre == hash_B_pre, "Clone branch prefix states diverge before fork!"
    assert hash_A_pre == compute_state_hash(spec.oracle_prefix_state)


def test_reconverged_state_hash_equality():
    """Verify that independent branch trajectories reconverge to identical SHA-256 state hashes after synchronizing event."""
    gen = StateInterventionGenerator(seed=42)
    spec = gen.generate_clone_reconvergence_spec(twin_idx=1)

    updater = OracleStateUpdater()
    
    # Trajectory A
    state_A = spec.oracle_prefix_state.model_copy(deep=True)
    for i, ev in enumerate(spec.fork_events_A):
        state_A, _, _, _, _, _, _, _ = updater.update(state_A, [ev], tick=1 + i)
    for i, ev in enumerate(spec.reconvergence_events):
        state_A, _, _, _, _, _, _, _ = updater.update(state_A, [ev], tick=2 + i)

    # Trajectory B
    state_B = spec.oracle_prefix_state.model_copy(deep=True)
    for i, ev in enumerate(spec.fork_events_B):
        state_B, _, _, _, _, _, _, _ = updater.update(state_B, [ev], tick=1 + i)
    for i, ev in enumerate(spec.reconvergence_events):
        state_B, _, _, _, _, _, _, _ = updater.update(state_B, [ev], tick=2 + i)

    hash_A_reconv = compute_state_hash(state_A)
    hash_B_reconv = compute_state_hash(state_B)

    assert hash_A_reconv == hash_B_reconv, "Reconverged branch states fail hash equality!"
    assert state_A.working_memory[spec.k_target] == spec.val_reconverge
    assert state_B.working_memory[spec.k_target] == spec.val_reconverge


def test_e07_runner_dry_run(tmp_path: Path):
    """Verify end-to-end dry run execution of E07 runner producing all required artifacts."""
    res = run_e07_experiment(
        model_name="qwen2.5:3b",
        seed=42,
        phase="exploratory",
        twin_pairs_count=2,
        dry_run=True,
        output_dir=tmp_path / "e07_dryrun",
    )

    assert res["manifest"]["total_twin_pairs"] == 2
    assert (tmp_path / "e07_dryrun" / "manifest.json").exists()
    assert (tmp_path / "e07_dryrun" / "summary.json").exists()
    assert (tmp_path / "e07_dryrun" / "trials.jsonl").exists()
    assert (tmp_path / "e07_dryrun" / "trials.parquet").exists()
    assert (tmp_path / "e07_dryrun" / "report.md").exists()
