"""Unit tests for Sprint S05 & S05.1 autonomous update loop, state manager, delta updater, and drift metrics."""

import json
from typing import Any, Dict, Optional, Tuple
import pytest
from recurrence.loop.clock import SimulatedClock
from recurrence.loop.queue import EventQueue
from recurrence.loop.state_manager import ImmutableEventLog, StateManager
from recurrence.loop.updater import (
    OracleStateUpdater,
    FullModelStateUpdater,
    DeltaModelStateUpdater,
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
    """Mock backend returning deterministic JSON updates."""

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
    queue.schedule(ev2, tick=1, priority=0)
    queue.schedule(ev3, tick=5, priority=0)

    assert queue.pending_count == 3
    assert queue.peek_next_tick() == 1

    dispatched_0 = queue.pop_events_for_tick(0)
    assert len(dispatched_0) == 0

    dispatched_1 = queue.pop_events_for_tick(1)
    assert len(dispatched_1) == 2
    assert dispatched_1[0].event_id == "ev_02"
    assert dispatched_1[1].event_id == "ev_01"


def test_immutable_event_log_integrity():
    log = ImmutableEventLog()
    ev1 = MemoryEvent(event_id="ev_01", step_index=0, source=EventSource.SELF, event_type="action", content="Start")
    ev2 = MemoryEvent(event_id="ev_02", step_index=1, source=EventSource.ENVIRONMENT, event_type="obs", content="Obs")

    log.append(ev1, tick=0)
    log.append(ev2, tick=1)

    assert log.entry_count == 2
    valid, err = log.verify_integrity()
    assert valid is True
    assert err is None

    # Tamper test
    log._entries[0]["payload"]["event"]["content"] = "Tampered Content"
    valid_tampered, err_tampered = log.verify_integrity()
    assert valid_tampered is False
    assert "Tampered entry" in str(err_tampered)


def test_state_manager_delta_merging_and_goal_validation():
    manager = StateManager()
    initial_state = StructuredSelfState()

    delta_1 = {
        "working_memory_upserts": {"key_a": "val_1", "key_b": "val_2"},
        "working_memory_deletions": [],
        "source_upserts": {"key_a": "environment", "key_b": "self"},
        "goal_updates": [{"goal_id": "g_01", "description": "Primary goal", "status": "active"}],
        "unresolved_items_add": ["task_audit"],
        "unresolved_items_remove": [],
    }

    state_1, warnings_1 = manager.apply_delta(initial_state, delta_1, tick=1)
    assert state_1.working_memory["key_a"] == "val_1"
    assert state_1.working_memory["key_b"] == "val_2"
    assert state_1.source_ledger["key_a"] == "environment"
    assert len(state_1.goals) == 1
    assert state_1.goals[0].status == "active"
    assert "task_audit" in state_1.unresolved_items

    # Test goal completion and illegal reactivation
    delta_complete = {
        "working_memory_upserts": {},
        "working_memory_deletions": [],
        "source_upserts": {},
        "goal_updates": [{"goal_id": "g_01", "description": "Primary goal", "status": "completed"}],
        "unresolved_items_add": [],
        "unresolved_items_remove": ["task_audit"],
    }
    state_2, warnings_2 = manager.apply_delta(state_1, delta_complete, tick=2)
    assert state_2.goals[0].status == "completed"

    # Illegal transition: completed -> active
    delta_illegal = {
        "working_memory_upserts": {},
        "working_memory_deletions": [],
        "source_upserts": {},
        "goal_updates": [{"goal_id": "g_01", "description": "Primary goal", "status": "active"}],
        "unresolved_items_add": [],
        "unresolved_items_remove": [],
    }
    state_3, warnings_3 = manager.apply_delta(state_2, delta_illegal, tick=3)
    assert state_3.goals[0].status == "completed"  # Stays completed
    assert len(warnings_3) == 1
    assert "Illegal goal transition rejected" in warnings_3[0]


def test_state_manager_capacity_bounds_and_lru_eviction():
    config = StateCapacityConfig(max_working_memory_items=3, max_goals=2, max_unresolved_items=2)
    manager = StateManager(capacity_config=config)

    # Insert 3 keys sequentially
    s = StructuredSelfState()
    for i in range(3):
        d = {"working_memory_upserts": {f"key_{i}": f"val_{i}"}, "working_memory_deletions": [], "source_upserts": {f"key_{i}": "self"}, "goal_updates": [], "unresolved_items_add": [], "unresolved_items_remove": []}
        s, _ = manager.apply_delta(s, d, tick=i)
        manager.update_state(s, tick=i, explicit_written_keys=[f"key_{i}"])

    assert len(manager.current_state.working_memory) == 3
    assert set(manager.current_state.working_memory.keys()) == {"key_0", "key_1", "key_2"}

    # Update key_0 so it becomes more recently updated than key_1
    d_touch = {"working_memory_upserts": {"key_0": "val_0_new"}, "working_memory_deletions": [], "source_upserts": {"key_0": "self"}, "goal_updates": [], "unresolved_items_add": [], "unresolved_items_remove": []}
    s, _ = manager.apply_delta(manager.current_state, d_touch, tick=3)
    manager.update_state(s, tick=3, explicit_written_keys=["key_0"])

    # Now add key_3 (exceeding capacity 3 -> should evict key_1, which is now least recently updated!)
    d_add = {"working_memory_upserts": {"key_3": "val_3"}, "working_memory_deletions": [], "source_upserts": {"key_3": "self"}, "goal_updates": [], "unresolved_items_add": [], "unresolved_items_remove": []}
    s, _ = manager.apply_delta(manager.current_state, d_add, tick=4)
    manager.update_state(s, tick=4, explicit_written_keys=["key_3"])

    assert len(manager.current_state.working_memory) == 3
    assert "key_1" not in manager.current_state.working_memory
    assert "key_0" in manager.current_state.working_memory
    assert "key_2" in manager.current_state.working_memory
    assert "key_3" in manager.current_state.working_memory


def test_delta_model_state_updater_with_mock():
    mock_delta = {
        "working_memory_upserts": {"key_obsidian_falcon": "val_azure_glacier"},
        "working_memory_deletions": [],
        "source_upserts": {"key_obsidian_falcon": "environment"},
        "goal_updates": [{"goal_id": "g_01", "description": "Run diagnostic", "status": "active"}],
        "unresolved_items_add": [],
        "unresolved_items_remove": [],
    }
    backend = MockModelBackend(mock_delta)
    updater = DeltaModelStateUpdater(backend)
    
    prev_st = StructuredSelfState()
    ev = MemoryEvent(
        event_id="ev_01", step_index=1, source=EventSource.ENVIRONMENT,
        event_type="obs", content="Obs", key_bindings={"key_obsidian_falcon": "val_azure_glacier"}
    )

    new_st, valid, p_tok, c_tok, lat, err, raw, parsed = updater.update(prev_st, [ev], tick=1)
    assert valid is True
    assert new_st.working_memory["key_obsidian_falcon"] == "val_azure_glacier"
    assert new_st.source_ledger["key_obsidian_falcon"] == "environment"
    assert len(new_st.goals) == 1


def test_autonomous_update_loop_run_for_ticks_and_quiet_ticks():
    clock = SimulatedClock()
    queue = EventQueue()
    manager = StateManager()
    updater = OracleStateUpdater(state_manager=manager)

    # Schedule event ONLY at tick 1
    ev = MemoryEvent(
        event_id="ev_01", step_index=1, source=EventSource.SELF,
        event_type="action", content="Action", key_bindings={"key_a": "val_1"}
    )
    queue.schedule(ev, tick=1)

    loop = AutonomousUpdateLoop(clock, queue, manager, updater, mode_name="oracle")
    snapshots = loop.run_for_ticks(total_ticks=5)

    # Exactly 5 snapshots for 5 logical ticks
    assert len(snapshots) == 5
    assert len(loop.state_traces) == 5

    # Tick 0 (quiet): 0 events, 0 prompt tokens, empty state
    assert snapshots[0].incoming_event_count == 0
    assert snapshots[0].prompt_tokens == 0
    assert len(snapshots[0].state.working_memory) == 0

    # Tick 1 (event arrived): key_a present
    assert snapshots[1].incoming_event_count == 1
    assert snapshots[1].state.working_memory["key_a"] == "val_1"

    # Ticks 2, 3, 4 (quiet): key_a preserved with 0 tokens consumed
    for t in [2, 3, 4]:
        assert snapshots[t].incoming_event_count == 0
        assert snapshots[t].prompt_tokens == 0
        assert snapshots[t].state.working_memory["key_a"] == "val_1"


def test_stream_scenario_generator_stress_scenarios():
    gen = StreamScenarioGenerator(seed=42)
    
    # 1. Full lifecycle scenario
    scen_gl = gen.generate_full_lifecycle_goal_scenario(scenario_idx=101, num_ticks=16)
    assert scen_gl.total_ticks == 16
    assert 0 in scen_gl.oracle_states
    assert 15 in scen_gl.oracle_states

    # 2. Capacity overflow scenario (>16 keys)
    scen_cap = gen.generate_capacity_overflow_scenario(scenario_idx=201, total_keys=24, max_capacity=16)
    assert scen_cap.total_ticks == 28
    # Terminal oracle state working memory must be bounded to 16
    assert len(scen_cap.oracle_states[27].working_memory) == 16

    # 3. 100-tick scenario
    scen_100 = gen.generate_long_horizon_scenario(scenario_idx=301, num_ticks=100)
    assert scen_100.total_ticks == 100
    assert len(scen_100.oracle_states) == 100
