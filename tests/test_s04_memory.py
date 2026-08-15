"""Unit tests for Level 1 explicit memory representations, adapters, task battery, and metrics (Sprint S04)."""

import pytest
from recurrence.memory.schemas import (
    EventSource,
    GoalState,
    MemoryEvent,
    MemoryFormat,
    StructuredSelfState,
)
from recurrence.memory.adapters import (
    CombinedStateAdapter,
    DeterministicSummaryAdapter,
    FreshAdapter,
    ModelSummaryAdapter,
    StructuredStateAdapter,
    TranscriptAdapter,
    get_memory_adapter,
)
from recurrence.tasks.memory_battery import MemoryBatteryTask, MemoryProbeItem
from recurrence.analysis.memory_metrics import (
    compute_summary_distortion,
    compute_memory_format_benchmarks,
)


@pytest.fixture
def sample_events():
    return [
        MemoryEvent(
            event_id="ev_01",
            step_index=1,
            source=EventSource.EXPERIMENTER,
            event_type="goal_assertion",
            content="Primary objective assigned: Calibrate sensor array.",
            metadata={"goal_id": "goal_01"},
        ),
        MemoryEvent(
            event_id="ev_02",
            step_index=2,
            source=EventSource.ENVIRONMENT,
            event_type="binding_assertion",
            content="Sensor telemetry observed key_emerald_falcon = val_obsidian_river.",
            key_bindings={"key_emerald_falcon": "val_obsidian_river"},
        ),
        MemoryEvent(
            event_id="ev_03",
            step_index=3,
            source=EventSource.SELF,
            event_type="binding_assertion",
            content="I computed key_golden_tempest = val_crimson_glacier.",
            key_bindings={"key_golden_tempest": "val_crimson_glacier"},
        ),
    ]


@pytest.fixture
def sample_state():
    return StructuredSelfState(
        working_memory={
            "key_emerald_falcon": "val_obsidian_river",
            "key_golden_tempest": "val_crimson_glacier",
        },
        goals=[
            GoalState(
                goal_id="goal_01",
                description="Calibrate sensor array",
                status="completed",
                created_at_step=1,
                updated_at_step=3,
            ),
            GoalState(
                goal_id="goal_02",
                description="Process background telemetry archive",
                status="suspended",
                created_at_step=2,
                updated_at_step=2,
            ),
        ],
        source_ledger={
            "key_emerald_falcon": "environment",
            "key_golden_tempest": "self",
        },
        unresolved_items=["goal_02"],
        last_updated_step=3,
    )


def test_fresh_adapter(sample_events, sample_state):
    adapter = FreshAdapter()
    assert adapter.format_name == MemoryFormat.FRESH
    ctx = adapter.build_context_prompt(sample_events, sample_state)
    assert ctx == ""
    stats = adapter.compute_context_stats(ctx)
    assert stats["estimated_tokens"] == 0
    assert stats["byte_count"] == 0


def test_transcript_adapter(sample_events, sample_state):
    adapter = TranscriptAdapter()
    assert adapter.format_name == MemoryFormat.TRANSCRIPT
    ctx = adapter.build_context_prompt(sample_events, sample_state)
    assert "=== FULL EVENT TRANSCRIPT ===" in ctx
    assert "[Step 01 | Source: EXPERIMENTER" in ctx
    assert "[Step 02 | Source: ENVIRONMENT" in ctx
    assert "val_obsidian_river" in ctx
    assert "=== END TRANSCRIPT ===" in ctx

    # Empty event handling
    assert adapter.build_context_prompt([]) == ""


def test_deterministic_summary_adapter(sample_events, sample_state):
    adapter = DeterministicSummaryAdapter()
    assert adapter.format_name == MemoryFormat.DETERMINISTIC_SUMMARY
    ctx = adapter.build_context_prompt(sample_events, sample_state)
    assert "=== DETERMINISTIC FACTUAL SUMMARY ===" in ctx
    assert "key_emerald_falcon: val_obsidian_river (Source: environment)" in ctx
    assert "key_golden_tempest: val_crimson_glacier (Source: self)" in ctx
    assert "=== END FACTUAL SUMMARY ===" in ctx


def test_model_summary_adapter(sample_events):
    adapter = ModelSummaryAdapter()
    assert adapter.format_name == MemoryFormat.MODEL_SUMMARY
    summary_text = "I recorded emerald falcon with obsidian river and computed golden tempest."
    ctx = adapter.build_context_prompt(sample_events, cached_summary=summary_text)
    assert "=== AUTOBIOGRAPHICAL MODEL MEMORY SUMMARY ===" in ctx
    assert summary_text in ctx
    assert "=== END MEMORY SUMMARY ===" in ctx

    # Empty summary handling
    assert adapter.build_context_prompt(sample_events, cached_summary=None) == ""


def test_structured_state_adapter(sample_events, sample_state):
    adapter = StructuredStateAdapter()
    assert adapter.format_name == MemoryFormat.STRUCTURED_STATE
    ctx = adapter.build_context_prompt(sample_events, sample_state)
    assert "=== STRUCTURED SELF-STATE ===" in ctx
    assert '"working_memory": {' in ctx
    assert '"key_emerald_falcon": "val_obsidian_river"' in ctx
    assert '"unresolved_items": [' in ctx
    assert '"goal_02"' in ctx
    assert "=== END STRUCTURED STATE ===" in ctx


def test_combined_state_adapter(sample_events, sample_state):
    adapter = CombinedStateAdapter()
    assert adapter.format_name == MemoryFormat.COMBINED
    summary_text = "Autobiographical memory narrative."
    ctx = adapter.build_context_prompt(sample_events, sample_state, cached_summary=summary_text)
    assert "=== STRUCTURED SELF-STATE ===" in ctx
    assert "=== AUTOBIOGRAPHICAL MODEL MEMORY SUMMARY ===" in ctx
    assert summary_text in ctx


def test_memory_adapter_factory():
    for fmt in MemoryFormat:
        adapter = get_memory_adapter(fmt)
        assert adapter.format_name == fmt


def test_memory_battery_task_generation():
    task = MemoryBatteryTask(identifier_type="semantic", mode="forced_choice")
    ep = task.generate_episode(episode_idx=0, target_kv_count=3, distractor_count=6, seed=42)

    assert ep.episode_id == "ep_000"
    assert len(ep.kv_targets) == 3
    assert len(ep.events) > 8
    assert len(ep.goals) == 2

    # Verify positional strata are assigned
    strata = [t["stratum"] for t in ep.kv_targets.values()]
    assert "early" in strata
    assert "middle" in strata
    assert "late" in strata

    # Generate probes across 2 memory formats
    items_fresh = task.generate_probe_items([ep], memory_format=MemoryFormat.FRESH)
    items_transcript = task.generate_probe_items([ep], memory_format=MemoryFormat.TRANSCRIPT)

    # 3 delayed_kv + 3 source_attribution + 1 goal_resumption = 7 probes per episode
    assert len(items_fresh) == 7
    assert len(items_transcript) == 7

    # Fresh prompt should have no transcript header; Transcript prompt must have it
    assert "=== FULL EVENT TRANSCRIPT ===" not in items_fresh[0].prompt
    assert "=== FULL EVENT TRANSCRIPT ===" in items_transcript[0].prompt


def test_memory_battery_scoring():
    task = MemoryBatteryTask()
    ep = task.generate_episode(episode_idx=0, seed=42)
    items = task.generate_probe_items([ep], memory_format=MemoryFormat.TRANSCRIPT)
    probe = items[0]

    # 1. Exact valid answer
    resp_correct = f'{{"answer": "{probe.ground_truth}"}}'
    score = task.score_response(probe, resp_correct)
    assert score["correct"] is True
    assert score["schema_valid"] is True
    assert score["parsed_answer"] == probe.ground_truth

    # 2. Wrong answer letter
    wrong_letter = "B" if probe.ground_truth != "B" else "C"
    resp_wrong = f'{{"answer": "{wrong_letter}"}}'
    score_w = task.score_response(probe, resp_wrong)
    assert score_w["correct"] is False
    assert score_w["schema_valid"] is True
    assert score_w["parsed_answer"] == wrong_letter

    # 3. Extra keys violate strict schema
    resp_extra = f'{{"answer": "{probe.ground_truth}", "confidence": 90}}'
    score_extra = task.score_response(probe, resp_extra)
    assert score_extra["correct"] is True
    assert score_extra["schema_valid"] is False


def test_compute_summary_distortion():
    targets = {
        "key_emerald_falcon": "val_obsidian_river",
        "key_golden_tempest": "val_crimson_glacier",
        "key_amber_monolith": "val_sapphire_spire",
    }
    # Summary retains falcon/river, mutates tempest to wrong value, and completely omits monolith
    summary = "We saw key_emerald_falcon with val_obsidian_river. Also key_golden_tempest had val_wrong_value."

    metrics = compute_summary_distortion([], targets, summary)
    assert metrics.total_target_facts == 3
    assert metrics.retained_target_facts == 1
    assert metrics.mutated_facts == 1
    assert metrics.omitted_target_facts == 1
    assert metrics.omission_rate == pytest.approx(1 / 3)
    assert metrics.mutation_rate == pytest.approx(1 / 3)


def test_compute_memory_format_benchmarks():
    records = [
        {"memory_format": "fresh", "correct": False, "probe_type": "delayed_kv", "prompt_chars": 200, "estimated_tokens": 50, "byte_count": 200},
        {"memory_format": "fresh", "correct": True, "probe_type": "delayed_kv", "prompt_chars": 200, "estimated_tokens": 50, "byte_count": 200},
        {"memory_format": "transcript", "correct": True, "probe_type": "delayed_kv", "prompt_chars": 1000, "estimated_tokens": 250, "byte_count": 1000},
        {"memory_format": "transcript", "correct": True, "probe_type": "source_attribution", "prompt_chars": 1000, "estimated_tokens": 250, "byte_count": 1000},
    ]

    benchmarks = compute_memory_format_benchmarks(records)
    assert "fresh" in benchmarks
    assert "transcript" in benchmarks

    assert benchmarks["fresh"].overall_accuracy == pytest.approx(0.5)
    assert benchmarks["transcript"].overall_accuracy == pytest.approx(1.0)
    assert benchmarks["transcript"].accuracy_by_probe_type["source_attribution"] == pytest.approx(1.0)
