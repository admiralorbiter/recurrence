"""Synthetic multi-tick stream scenario generator for autonomous update loop evaluation."""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from recurrence.memory.schemas import EventSource, MemoryEvent, GoalState, StructuredSelfState, StateCapacityConfig


@dataclass
class StreamScenario:
    """A synthetic multi-tick streaming scenario with ground-truth oracle transitions."""
    scenario_id: str
    total_ticks: int
    scheduled_events: List[Tuple[MemoryEvent, int, int]]  # (event, tick, priority)
    target_bindings_by_tick: Dict[int, Dict[str, str]]
    source_bindings_by_tick: Dict[int, Dict[str, str]]
    goals_by_tick: Dict[int, List[GoalState]]
    oracle_states: Dict[int, StructuredSelfState] = field(default_factory=dict)


class StreamScenarioGenerator:
    """Generates controlled multi-tick event streams with known ground-truth state trajectories."""

    ADJECTIVES = [
        "crimson", "obsidian", "azure", "emerald", "golden", "amber",
        "sapphire", "iron", "lunar", "solar", "velvet", "frost",
        "cobalt", "topaz", "garnet", "silver", "onyx", "bronze",
    ]
    NOUNS = [
        "glacier", "falcon", "tempest", "monolith", "river", "beacon",
        "citadel", "canyon", "spire", "shadow", "harbor", "meadow",
        "bastion", "summit", "forge", "haven", "vortex", "crag",
    ]
    GOAL_DESCRIPTIONS = [
        ("goal_primary", "Execute deep diagnostic scan on system telemetry bus"),
        ("goal_secondary", "Deploy emergency patch for subsystem thermal regulator"),
        ("goal_maintenance", "Consolidate transient state cache into archive partition"),
    ]

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _compute_oracle_states(
        self,
        scheduled_events: List[Tuple[MemoryEvent, int, int]],
        num_ticks: int,
        capacity_config: Optional[StateCapacityConfig] = None,
    ) -> Tuple[Dict[int, Dict[str, str]], Dict[int, Dict[str, str]], Dict[int, List[GoalState]], Dict[int, StructuredSelfState]]:
        """Compute exact ground truth oracle state trajectories across all ticks."""
        from recurrence.loop.state_manager import StateManager
        
        manager = StateManager(capacity_config=capacity_config or StateCapacityConfig(max_working_memory_items=64))
        target_bindings_by_tick: Dict[int, Dict[str, str]] = {}
        source_bindings_by_tick: Dict[int, Dict[str, str]] = {}
        goals_by_tick: Dict[int, List[GoalState]] = {}
        oracle_states: Dict[int, StructuredSelfState] = {}

        for t in range(num_ticks):
            tick_events = [ev for ev, ev_t, _ in scheduled_events if ev_t == t]
            
            # Build programmatic delta for oracle
            wm_upserts: Dict[str, str] = {}
            src_upserts: Dict[str, str] = {}
            goal_updates: List[Dict[str, Any]] = []

            for ev in tick_events:
                if ev.key_bindings:
                    for k, v in ev.key_bindings.items():
                        wm_upserts[k] = v
                        src_upserts[k] = ev.source.value

                if ev.event_type == "goal_update" or "goal_id" in ev.metadata:
                    gid = ev.metadata.get("goal_id", f"goal_{ev.event_id}")
                    desc = ev.metadata.get("goal_description", ev.content)
                    status = ev.metadata.get("goal_status", "active")
                    goal_updates.append({
                        "goal_id": gid,
                        "description": desc,
                        "status": status,
                    })

            if tick_events:
                delta_payload = {
                    "working_memory_upserts": wm_upserts,
                    "working_memory_deletions": [],
                    "source_upserts": src_upserts,
                    "goal_updates": goal_updates,
                    "unresolved_items_add": [],
                    "unresolved_items_remove": [],
                }
                new_state, _ = manager.apply_delta(manager.current_state, delta_payload, t)
                manager.update_state(new_state, tick=t, incoming_event_count=len(tick_events))
            else:
                quiet_state = manager.current_state.model_copy(deep=True)
                quiet_state.last_updated_step = t
                manager.update_state(quiet_state, tick=t, incoming_event_count=0)

            curr = manager.current_state
            target_bindings_by_tick[t] = dict(curr.working_memory)
            source_bindings_by_tick[t] = dict(curr.source_ledger)
            goals_by_tick[t] = [g.model_copy(deep=True) for g in curr.goals]
            oracle_states[t] = curr.model_copy(deep=True)

        return target_bindings_by_tick, source_bindings_by_tick, goals_by_tick, oracle_states

    def generate_scenario(
        self,
        scenario_idx: int = 0,
        num_ticks: int = 15,
        target_keys_count: int = 6,
        include_overwrites: bool = True,
        distractor_density: float = 0.5,
        capacity_config: Optional[StateCapacityConfig] = None,
    ) -> StreamScenario:
        """Generate a complete multi-tick stream scenario with ground-truth states."""
        rng = random.Random(self.seed + scenario_idx * 100)
        scenario_id = f"stream_scen_{scenario_idx:03d}"

        # 1. Create target key-value pairs
        keys = [
            f"key_{rng.choice(self.ADJECTIVES)}_{rng.choice(self.NOUNS)}_{i}"
            for i in range(target_keys_count)
        ]
        values = [
            f"val_{rng.choice(self.ADJECTIVES)}_{rng.choice(self.NOUNS)}_{i}"
            for i in range(target_keys_count)
        ]
        sources = [rng.choice(list(EventSource)) for _ in range(target_keys_count)]

        scheduled_events: List[Tuple[MemoryEvent, int, int]] = []
        event_counter = 0

        # Schedule initial primary goal at tick 0
        primary_gid, primary_desc = self.GOAL_DESCRIPTIONS[0]
        ev_goal_0 = MemoryEvent(
            event_id=f"ev_{event_counter:04d}",
            step_index=0,
            source=EventSource.EXPERIMENTER,
            event_type="goal_update",
            content=f"Initial goal assigned: {primary_desc}",
            key_bindings={},
            metadata={
                "goal_id": primary_gid,
                "goal_description": primary_desc,
                "goal_status": "active",
            },
        )
        scheduled_events.append((ev_goal_0, 0, 0))
        event_counter += 1

        # Schedule early, middle, late key-value assertions
        target_ticks = [
            1, 2,  # Early
            num_ticks // 2 - 1, num_ticks // 2,  # Middle
            num_ticks - 3, num_ticks - 2,  # Late
        ][:target_keys_count]

        for i, (k, v, src, t) in enumerate(zip(keys, values, sources, target_ticks)):
            ev_kv = MemoryEvent(
                event_id=f"ev_{event_counter:04d}",
                step_index=t,
                source=src,
                event_type="observation" if src == EventSource.ENVIRONMENT else "statement",
                content=f"Entity established: {k} is set to {v} by {src.value}.",
                key_bindings={k: v},
                metadata={"target_key": k, "target_value": v, "true_source": src.value},
            )
            scheduled_events.append((ev_kv, t, 0))
            event_counter += 1

        # Schedule key overwrite at late middle tick if enabled
        if include_overwrites and len(keys) >= 2:
            overwrite_key = keys[0]
            new_val = f"val_updated_{rng.choice(self.ADJECTIVES)}_{rng.choice(self.NOUNS)}"
            overwrite_tick = num_ticks // 2 + 1
            ev_overwrite = MemoryEvent(
                event_id=f"ev_{event_counter:04d}",
                step_index=overwrite_tick,
                source=EventSource.SELF,
                event_type="action",
                content=f"State updated: {overwrite_key} changed value to {new_val}.",
                key_bindings={overwrite_key: new_val},
                metadata={"target_key": overwrite_key, "target_value": new_val, "is_overwrite": True},
            )
            scheduled_events.append((ev_overwrite, overwrite_tick, 0))
            event_counter += 1

        # Schedule goal interruption at tick (num_ticks // 2)
        interrupt_tick = num_ticks // 2
        sec_gid, sec_desc = self.GOAL_DESCRIPTIONS[1]
        
        # Suspend primary
        ev_suspend = MemoryEvent(
            event_id=f"ev_{event_counter:04d}",
            step_index=interrupt_tick,
            source=EventSource.EXPERIMENTER,
            event_type="goal_update",
            content=f"High-priority interrupt received! Primary goal {primary_gid} is SUSPENDED.",
            key_bindings={},
            metadata={"goal_id": primary_gid, "goal_status": "suspended"},
        )
        scheduled_events.append((ev_suspend, interrupt_tick, 1))
        event_counter += 1

        # Activate secondary
        ev_sec = MemoryEvent(
            event_id=f"ev_{event_counter:04d}",
            step_index=interrupt_tick,
            source=EventSource.EXPERIMENTER,
            event_type="goal_update",
            content=f"Emergency task activated: {sec_desc}",
            key_bindings={},
            metadata={"goal_id": sec_gid, "goal_description": sec_desc, "goal_status": "active"},
        )
        scheduled_events.append((ev_sec, interrupt_tick, 2))
        event_counter += 1

        # Schedule distractors across ticks
        for t in range(num_ticks):
            if rng.random() < distractor_density:
                dist_content = f"Background monitor check at tick {t}: nominal subsystem throughput."
                ev_dist = MemoryEvent(
                    event_id=f"ev_{event_counter:04d}",
                    step_index=t,
                    source=EventSource.ENVIRONMENT,
                    event_type="distractor",
                    content=dist_content,
                    key_bindings={},
                    metadata={"is_distractor": True},
                )
                scheduled_events.append((ev_dist, t, 10))
                event_counter += 1

        # Sort scheduled events by tick then priority
        scheduled_events.sort(key=lambda x: (x[1], x[2]))

        # Compute ground truth states by tick
        tb, sb, gb, ost = self._compute_oracle_states(scheduled_events, num_ticks, capacity_config)

        return StreamScenario(
            scenario_id=scenario_id,
            total_ticks=num_ticks,
            scheduled_events=scheduled_events,
            target_bindings_by_tick=tb,
            source_bindings_by_tick=sb,
            goals_by_tick=gb,
            oracle_states=ost,
        )

    def generate_full_lifecycle_goal_scenario(
        self,
        scenario_idx: int = 101,
        num_ticks: int = 16,
    ) -> StreamScenario:
        """Generate a scenario testing complete goal lifecycle: pending -> active -> suspended -> secondary complete -> primary resume -> complete."""
        rng = random.Random(self.seed + scenario_idx * 100)
        scenario_id = f"scen_goal_lifecycle_{scenario_idx}"
        scheduled_events: List[Tuple[MemoryEvent, int, int]] = []
        
        p_gid, p_desc = "goal_primary", "Execute deep diagnostic scan"
        s_gid, s_desc = "goal_emergency", "Deploy thermal regulator patch"

        # Tick 0: Primary assigned
        scheduled_events.append((
            MemoryEvent(
                event_id="ev_gl_00", step_index=0, source=EventSource.EXPERIMENTER,
                event_type="goal_update", content=f"Assigned primary: {p_desc}",
                metadata={"goal_id": p_gid, "goal_description": p_desc, "goal_status": "active"},
            ), 0, 0
        ))

        # Tick 2: Early KV
        scheduled_events.append((
            MemoryEvent(
                event_id="ev_gl_01", step_index=2, source=EventSource.ENVIRONMENT,
                event_type="observation", content="Telemetry bus nominal",
                key_bindings={"key_telemetry_bus": "val_nominal_400"},
            ), 2, 0
        ))

        # Tick 4: Suspend primary & Activate emergency
        scheduled_events.append((
            MemoryEvent(
                event_id="ev_gl_02", step_index=4, source=EventSource.EXPERIMENTER,
                event_type="goal_update", content=f"EMERGENCY INTERRUPT: suspend {p_gid}",
                metadata={"goal_id": p_gid, "goal_status": "suspended"},
            ), 4, 0
        ))
        scheduled_events.append((
            MemoryEvent(
                event_id="ev_gl_03", step_index=4, source=EventSource.EXPERIMENTER,
                event_type="goal_update", content=f"Activate emergency: {s_desc}",
                metadata={"goal_id": s_gid, "goal_description": s_desc, "goal_status": "active"},
            ), 4, 1
        ))

        # Tick 8: Complete emergency & Resume primary
        scheduled_events.append((
            MemoryEvent(
                event_id="ev_gl_04", step_index=8, source=EventSource.SELF,
                event_type="goal_update", content=f"Emergency patch applied: {s_gid} completed",
                metadata={"goal_id": s_gid, "goal_status": "completed"},
            ), 8, 0
        ))
        scheduled_events.append((
            MemoryEvent(
                event_id="ev_gl_05", step_index=8, source=EventSource.EXPERIMENTER,
                event_type="goal_update", content=f"Resuming primary task: {p_gid} active",
                metadata={"goal_id": p_gid, "goal_status": "active"},
            ), 8, 1
        ))

        # Tick 12: Complete primary
        scheduled_events.append((
            MemoryEvent(
                event_id="ev_gl_06", step_index=12, source=EventSource.SELF,
                event_type="goal_update", content=f"Diagnostic complete: {p_gid} completed",
                metadata={"goal_id": p_gid, "goal_status": "completed"},
            ), 12, 0
        ))

        scheduled_events.sort(key=lambda x: (x[1], x[2]))
        tb, sb, gb, ost = self._compute_oracle_states(scheduled_events, num_ticks)

        return StreamScenario(
            scenario_id=scenario_id,
            total_ticks=num_ticks,
            scheduled_events=scheduled_events,
            target_bindings_by_tick=tb,
            source_bindings_by_tick=sb,
            goals_by_tick=gb,
            oracle_states=ost,
        )

    def generate_capacity_overflow_scenario(
        self,
        scenario_idx: int = 201,
        total_keys: int = 24,
        max_capacity: int = 16,
    ) -> StreamScenario:
        """Generate a scenario with >16 keys to verify LRU eviction under capacity bounds."""
        rng = random.Random(self.seed + scenario_idx * 100)
        scenario_id = f"scen_capacity_overflow_{scenario_idx}"
        num_ticks = total_keys + 4
        scheduled_events: List[Tuple[MemoryEvent, int, int]] = []

        for i in range(total_keys):
            k = f"key_entity_{i:02d}_{rng.choice(self.NOUNS)}"
            v = f"val_entity_{i:02d}_{rng.choice(self.ADJECTIVES)}"
            scheduled_events.append((
                MemoryEvent(
                    event_id=f"ev_cap_{i:03d}",
                    step_index=i,
                    source=EventSource.ENVIRONMENT,
                    event_type="observation",
                    content=f"Observed entity: {k} = {v}",
                    key_bindings={k: v},
                ), i, 0
            ))

        scheduled_events.sort(key=lambda x: (x[1], x[2]))
        cap_cfg = StateCapacityConfig(max_working_memory_items=max_capacity)
        tb, sb, gb, ost = self._compute_oracle_states(scheduled_events, num_ticks, capacity_config=cap_cfg)

        return StreamScenario(
            scenario_id=scenario_id,
            total_ticks=num_ticks,
            scheduled_events=scheduled_events,
            target_bindings_by_tick=tb,
            source_bindings_by_tick=sb,
            goals_by_tick=gb,
            oracle_states=ost,
        )

    def generate_long_horizon_scenario(
        self,
        scenario_idx: int = 301,
        num_ticks: int = 100,
    ) -> StreamScenario:
        """Generate a 100-tick long-horizon scenario with sparse events and many quiet ticks."""
        rng = random.Random(self.seed + scenario_idx * 100)
        scenario_id = f"scen_long_horizon_100_{scenario_idx}"
        scheduled_events: List[Tuple[MemoryEvent, int, int]] = []

        # Sparse assertion ticks: 5, 20, 45, 70, 90
        sparse_ticks = [5, 20, 45, 70, 90]
        keys = [f"key_sparse_{i}_{rng.choice(self.NOUNS)}" for i in range(len(sparse_ticks))]
        values = [f"val_sparse_{i}_{rng.choice(self.ADJECTIVES)}" for i in range(len(sparse_ticks))]

        for i, (t, k, v) in enumerate(zip(sparse_ticks, keys, values)):
            scheduled_events.append((
                MemoryEvent(
                    event_id=f"ev_sparse_{i}",
                    step_index=t,
                    source=EventSource.EXPERIMENTER,
                    event_type="statement",
                    content=f"Sparse assertion at tick {t}: {k} is {v}",
                    key_bindings={k: v},
                ), t, 0
            ))

        # Goal at tick 0
        scheduled_events.append((
            MemoryEvent(
                event_id="ev_sparse_goal",
                step_index=0,
                source=EventSource.EXPERIMENTER,
                event_type="goal_update",
                content="Long horizon persistent monitor",
                metadata={"goal_id": "goal_long_horizon", "goal_status": "active"},
            ), 0, 0
        ))

        scheduled_events.sort(key=lambda x: (x[1], x[2]))
        tb, sb, gb, ost = self._compute_oracle_states(scheduled_events, num_ticks)

        return StreamScenario(
            scenario_id=scenario_id,
            total_ticks=num_ticks,
            scheduled_events=scheduled_events,
            target_bindings_by_tick=tb,
            source_bindings_by_tick=sb,
            goals_by_tick=gb,
            oracle_states=ost,
        )
