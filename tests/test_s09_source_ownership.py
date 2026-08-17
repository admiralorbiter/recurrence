"""Unit and integration tests for Sprint S09 (E08 Source Ownership & E09 Metacognitive Screen)."""

from pathlib import Path
import pytest

from recurrence.memory.schemas import (
    EventSource,
    GoalState,
    MemoryEvent,
    StructuredSelfState,
)
from recurrence.tasks.ownership import (
    ACTOR_MAP,
    ACTOR_DISPLAY_NAMES,
    OwnershipTaskGenerator,
    OwnershipEpisode,
)
from recurrence.loop.ownership_experiment import (
    OwnershipHarness,
)
from recurrence.analysis.ownership_metrics import (
    analyze_ownership_results,
    calculate_auroc,
)
from experiments.e08_source_ownership.run import (
    MockOwnershipBackend,
    run_e08_experiment,
)
from experiments.e09_metacognitive_screen.run import (
    run_e09_experiment,
)


def test_eventsource_backwards_compatibility():
    """Verify EventSource enum maintains full backwards compatibility with S01-S08 and adds S09 sources."""
    assert EventSource.SELF.value == "self"
    assert EventSource.ENVIRONMENT.value == "environment"
    assert EventSource.EXPERIMENTER.value == "experimenter"
    assert EventSource.PEER_AGENT.value == "peer_agent"
    assert EventSource.OBSERVER.value == "observer"


def test_source_neutral_template_isomorphism():
    """Verify all 5 sources use isomorphic syntactic templates without semantic classification leaks."""
    gen = OwnershipTaskGenerator(seed=42)
    ep = gen.generate_episode(twin_idx=0)

    assert len(ep.events_neutral) == 5
    for ev in ep.events_neutral:
        assert ev.event_type == "state_assertion"
        assert "State binding registered:" in ev.content
        assert ev.actor_id in ACTOR_MAP.values()


def test_no_source_leakage_in_identifiers():
    """HARD CHECK: Ensure no source name, actor ID, or role substring appears in any generated key or value."""
    gen = OwnershipTaskGenerator(seed=42)
    forbidden_tokens = ["self", "peer", "environment", "experimenter", "observer", "alpha", "beta", "gamma", "sensor", "controller", "telemetry"]

    for ep_idx in range(5):
        ep = gen.generate_episode(twin_idx=ep_idx)
        all_keys = list(ep.oracle_state.working_memory.keys()) + [ep.k_target_self, ep.k_target_peer]
        all_vals = list(ep.oracle_state.working_memory.values()) + [ep.val_self, ep.val_peer]

        for k in all_keys:
            for tok in forbidden_tokens:
                assert tok not in k.lower(), f"Provenance leak in key '{k}': contains forbidden token '{tok}'"
        
        for v in all_vals:
            for tok in forbidden_tokens:
                assert tok not in v.lower(), f"Provenance leak in value '{v}': contains forbidden token '{tok}'"


def test_channel_factorial_clean_transcript_stripping():
    """Verify that when transcript tags are stripped, the transcript text contains NO actor identity or source token."""
    gen = OwnershipTaskGenerator(seed=42)
    ep = gen.generate_episode(twin_idx=0)
    harness = OwnershipHarness(backend=MockOwnershipBackend())

    stripped_trans = harness._format_transcript(ep.events_neutral, include_tags=False)
    for act in ACTOR_MAP.values():
        assert act not in stripped_trans, f"Stripped transcript leaked actor identity '{act}'"
    for src in [s.value for s in EventSource]:
        assert src not in stripped_trans, f"Stripped transcript leaked source name '{src}'"

    tagged_trans = harness._format_transcript(ep.events_neutral, include_tags=True)
    for act in ACTOR_MAP.values():
        assert act in tagged_trans, f"Tagged transcript missing actor '{act}'"


def test_cue_conflict_factorial_structure():
    """Verify the 2x2 Tag x Narrative cue conflict factorial generates all 4 conditions."""
    gen = OwnershipTaskGenerator(seed=42)
    ep = gen.generate_episode(twin_idx=1)

    assert len(ep.cue_conflict_specs) == 4
    tags = [s.tag_source for s in ep.cue_conflict_specs]
    narrs = [s.narrative_actor for s in ep.cue_conflict_specs]

    assert set(tags) == {EventSource.SELF, EventSource.PEER_AGENT}
    assert set(narrs) == {"agent_alpha", "agent_beta"}


def test_channel_factorial_structure():
    """Verify the 2x2 Transcript Tags x State Ledger channel factorial generates all 4 conditions."""
    gen = OwnershipTaskGenerator(seed=42)
    ep = gen.generate_episode(twin_idx=2)

    assert len(ep.channel_factorial_specs) == 4
    tag_flags = [s.has_transcript_tags for s in ep.channel_factorial_specs]
    ledg_flags = [s.has_state_ledger for s in ep.channel_factorial_specs]

    assert set(tag_flags) == {True, False}
    assert set(ledg_flags) == {True, False}


def test_self_vs_actor_framing_pair_isomorphism():
    """Verify paired framing probes share identical answer keys, values, and option distributions."""
    gen = OwnershipTaskGenerator(seed=42)
    ep = gen.generate_episode(twin_idx=3)

    probe_self, probe_actor = ep.probes_framing_pair
    assert probe_self.correct_option == probe_actor.correct_option
    assert probe_self.target_value == probe_actor.target_value
    assert probe_self.options == probe_actor.options
    assert "YOU" in probe_self.question
    assert "agent_alpha" in probe_actor.question


def test_calculate_auroc_metric():
    """Verify AUROC rank calculation with perfect, chance, and inverse rankings."""
    assert calculate_auroc([90, 80, 20, 10], [True, True, False, False]) == 1.0
    assert calculate_auroc([10, 20, 80, 90], [True, True, False, False]) == 0.0
    assert calculate_auroc([50, 50, 50, 50], [True, True, False, False]) == 0.5


def test_e08_runner_dry_run(tmp_path: Path):
    """Verify end-to-end dry run execution of E08 runner producing all required artifacts."""
    res = run_e08_experiment(
        model_name="qwen2.5:3b",
        seed=42,
        phase="exploratory",
        episodes_count=2,
        dry_run=True,
        output_dir=tmp_path / "e08_dryrun",
    )

    assert res["manifest"]["total_episodes"] == 2
    assert (tmp_path / "e08_dryrun" / "manifest.json").exists()
    assert (tmp_path / "e08_dryrun" / "summary.json").exists()
    assert (tmp_path / "e08_dryrun" / "trials.jsonl").exists()
    assert (tmp_path / "e08_dryrun" / "trials.parquet").exists()
    assert (tmp_path / "e08_dryrun" / "report.md").exists()


def test_e09_runner_dry_run(tmp_path: Path):
    """Verify end-to-end dry run execution of E09 runner producing all required artifacts."""
    res = run_e09_experiment(
        model_name="qwen2.5:3b",
        seed=42,
        phase="exploratory",
        episodes_count=2,
        dry_run=True,
        output_dir=tmp_path / "e09_dryrun",
    )

    assert res["manifest"]["total_episodes"] == 2
    assert (tmp_path / "e09_dryrun" / "manifest.json").exists()
    assert (tmp_path / "e09_dryrun" / "summary.json").exists()
    assert (tmp_path / "e09_dryrun" / "trials.jsonl").exists()
    assert (tmp_path / "e09_dryrun" / "trials.parquet").exists()
    assert (tmp_path / "e09_dryrun" / "report.md").exists()
