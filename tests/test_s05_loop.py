import json
from typing import Any, Dict, Optional, Tuple
import pytest
from recurrence.loop.clock import SimulatedClock
from recurrence.loop.queue import EventQueue
from recurrence.loop.state_manager import ImmutableEventLog, StateManager
from recurrence.loop.updater import (
    OracleStateUpdater,
    ModelStateUpdater,
    AutonomousUpdateLoop,
)
from recurrence.memory.schemas import (
    EventSource,
    MemoryEvent,
    GoalState,
    StructuredSelfState,
    StateCapacityConfig,
)
from recurrence.tasks.stream_scenarios import StreamScenarioGenerator
from recurrence.analysis.drift_metrics import (
    evaluate_tick_state,
    compute_scenario_stability,
)


class MockModelBackend:
    """Mock backend returning deterministic structured state updates."""

    def __init__(self, response_json: dict) -> None:
        self.response_json = response_json

    def step(self, prompt: str, format: Optional[Dict[str, Any]] = None) -> Tuple[str, str, Dict[str, Any]]:
        text = json.dumps(self.response_json)
        metadata = {
            "prompt_eval_count": 50,
            "eval_count": 30,
            "total_duration_ms": 10.0,
        }
        return text, "mock_hash_123", metadata



def test_simulated_clock():
    clock = SimulatedClock(start_tick=0, tick_duration_ms=1000)
    assert clock.current_tick == 0
    assert clock.advance(1) == 1
    assert clock.advance(5) == 6
    assert clock.current_tick == 6
    clock.reset(0)
    assert clock.current_tick == 0
    assert "2026-01-01" in clock.get_timestamp_str()


def test_event_queue_priority_and_dispatch():
    queue = EventQueue()
    ev1 = MemoryEvent(
        event_id="ev_01", step_index=1, source=EventSource.ENVIRONMENT,
        event_type="observation", content="Observation 1"
    )
    ev2 = MemoryEvent(
        event_id="ev_02", step_index=1, source=EventSource.SELF,
        event_type="action", content="Action 1"
    )
    ev3 = MemoryEvent(
        event_id="ev_03", step_index=5, source=EventSource.EXPERIMENTER,
        event_type="statement", content="Statement 1"
    )

    queue.schedule(ev1, tick=1, priority=1)
    queue.schedule(ev2, tick=1, priority=0)  # Higher priority
    queue.schedule(ev3, tick=5, priority=0)

    assert queue.pending_count == 3
    assert queue.peek_next_tick() == 1

    # Pop tick 0 -> None
    dispatched_0 = queue.pop_events_for_tick(0)
    assert len(dispatched_0) == 0

    # Pop tick 1 -> ev2 then ev1
    dispatched_1 = queue.pop_events_for_tick(1)
    assert len(dispatched_1) == 2
    assert dispatched_1[0].event_id == "ev_02"
    assert dispatched_1[1].event_id == "ev_01"

    # Pop tick 5 -> ev3
    dispatched_5 = queue.pop_events_for_tick(5)
    assert len(dispatched_5) == 1
    assert dispatched_5[0].event_id == "ev_03"
    assert not queue.has_pending_events()


def test_immutable_event_log_integrity():
    log = ImmutableEventLog()
    ev1 = MemoryEvent(event_id="ev_01", step_index=0, source=EventSource.SELF, event_type="action", content="Start")
    ev2 = MemoryEvent(event_id="ev_02", step_index=1, source=EventSource.ENVIRONMENT, event_type="obs", content="Obs")

    h1 = log.append(ev1, tick=0)
    h2 = log.append(ev2, tick=1)

    assert log.entry_count == 2
    valid, err = log.verify_integrity()
    assert valid is True
    assert err is None

    # Tamper test
    log._entries[0]["payload"]["event"]["content"] = "Tampered Content"
    valid_tampered, err_tampered = log.verify_integrity()
    assert valid_tampered is False
    assert "Tampered entry" in str(err_tampered)


def test_state_manager_capacity_bounds():
    config = StateCapacityConfig(max_working_memory_items=3, max_goals=2, max_unresolved_items=2)
    manager = StateManager(capacity_config=config)

    # State with 5 working memory items
    large_state = StructuredSelfState(
        working_memory={f"key_{i}": f"val_{i}" for i in range(5)},
        goals=[
            GoalState(goal_id=f"g_{i}", description=f"Goal {i}", status="active", created_at_step=0, updated_at_step=0)
            for i in range(4)
        ],
        source_ledger={f"key_{i}": "self" for i in range(5)},
        unresolved_items=[f"g_{i}" for i in range(5)],
        last_updated_step=0,
    )

    manager.update_state(large_state, tick=0)
    curr = manager.current_state
    
    assert len(curr.working_memory) == 3
    assert len(curr.source_ledger) == 3
    assert len(curr.goals) == 2
    assert len(curr.unresolved_items) == 2
    assert len(manager.snapshots) == 1


def test_oracle_state_updater():
    updater = OracleStateUpdater()
    initial_state = StructuredSelfState()
    
    ev_kv = MemoryEvent(
        event_id="ev_01",
        step_index=1,
        source=EventSource.ENVIRONMENT,
        event_type="observation",
        content="Observed sensor data",
        key_bindings={"key_sensor_alpha": "val_nominal_status"},
    )
    ev_goal = MemoryEvent(
        event_id="ev_02",
        step_index=1,
        source=EventSource.EXPERIMENTER,
        event_type="goal_update",
        content="Assigned task",
        key_bindings={},
        metadata={"goal_id": "goal_scan", "goal_description": "Scan telemetry", "goal_status": "active"},
    )

    new_st, valid, p_tok, c_tok, lat, err = updater.update(initial_state, [ev_kv, ev_goal], tick=1)
    assert valid is True
    assert new_st.working_memory["key_sensor_alpha"] == "val_nominal_status"
    assert new_st.source_ledger["key_sensor_alpha"] == "environment"
    assert len(new_st.goals) == 1
    assert new_st.goals[0].goal_id == "goal_scan"
    assert new_st.goals[0].status == "active"


def test_autonomous_update_loop_execution():
    clock = SimulatedClock()
    queue = EventQueue()
    manager = StateManager()
    updater = OracleStateUpdater()

    ev = MemoryEvent(
        event_id="ev_01", step_index=0, source=EventSource.SELF,
        event_type="action", content="Action", key_bindings={"key_a": "val_1"}
    )
    queue.schedule(ev, tick=0)

    loop = AutonomousUpdateLoop(clock, queue, manager, updater, mode_name="oracle")
    snapshots = loop.run_until_complete(max_ticks=5)

    assert len(snapshots) == 1
    assert clock.current_tick >= 1
    assert manager.current_state.working_memory["key_a"] == "val_1"
    assert manager.event_log.entry_count == 1


def test_model_state_updater_with_mock():
    mock_payload = {
        "working_memory": {"key_obsidian_falcon": "val_azure_glacier"},
        "goals": [{"goal_id": "g_01", "description": "Run diagnostic", "status": "active"}],
        "source_ledger": {"key_obsidian_falcon": "environment"},
        "unresolved_items": [],
    }
    backend = MockModelBackend(mock_payload)
    updater = ModelStateUpdater(backend)
    
    prev_st = StructuredSelfState()
    ev = MemoryEvent(
        event_id="ev_01", step_index=1, source=EventSource.ENVIRONMENT,
        event_type="obs", content="Obs", key_bindings={"key_obsidian_falcon": "val_azure_glacier"}
    )

    new_st, valid, p_tok, c_tok, lat, err = updater.update(prev_st, [ev], tick=1)
    assert valid is True
    assert new_st.working_memory["key_obsidian_falcon"] == "val_azure_glacier"
    assert new_st.source_ledger["key_obsidian_falcon"] == "environment"
    assert len(new_st.goals) == 1
    assert new_st.goals[0].goal_id == "g_01"


def test_stream_scenario_generator_and_drift_metrics():
    gen = StreamScenarioGenerator(seed=42)
    scen = gen.generate_scenario(scenario_idx=0, num_ticks=10, target_keys_count=4)
    
    assert scen.total_ticks == 10
    assert len(scen.scheduled_events) > 0
    assert 0 in scen.oracle_states
    assert 10 in scen.oracle_states

    # Run Oracle loop on this scenario
    clock = SimulatedClock()
    queue = EventQueue()
    queue.schedule_batch(scen.scheduled_events)
    manager = StateManager()
    updater = OracleStateUpdater()

    loop = AutonomousUpdateLoop(clock, queue, manager, updater, mode_name="oracle")
    snapshots = loop.run_until_complete(max_ticks=20)

    summary = compute_scenario_stability(scen, snapshots, updater_mode="oracle")
    assert summary.schema_compliance_rate == 1.0
    assert summary.mean_retention_fidelity == pytest.approx(1.0)
    assert summary.terminal_retention_fidelity == pytest.approx(1.0)
    assert summary.mean_omission_rate == pytest.approx(0.0)
    assert summary.mean_mutation_rate == pytest.approx(0.0)
    assert summary.total_phantom_intrusions == 0
    assert summary.is_ossified is False
