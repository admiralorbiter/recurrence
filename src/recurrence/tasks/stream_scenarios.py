"""Synthetic multi-tick stream scenario generator for autonomous update loop evaluation."""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from recurrence.memory.schemas import EventSource, MemoryEvent, GoalState, StructuredSelfState


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
    ]
    NOUNS = [
        "glacier", "falcon", "tempest", "monolith", "river", "beacon",
        "citadel", "canyon", "spire", "shadow", "harbor", "meadow",
    ]
    GOAL_DESCRIPTIONS = [
        ("goal_primary", "Execute deep diagnostic scan on system telemetry bus"),
        ("goal_secondary", "Deploy emergency patch for subsystem thermal regulator"),
        ("goal_maintenance", "Consolidate transient state cache into archive partition"),
    ]

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def generate_scenario(
        self,
        scenario_idx: int = 0,
        num_ticks: int = 15,
        target_keys_count: int = 6,
        include_overwrites: bool = True,
        distractor_density: float = 0.5,
    ) -> StreamScenario:
        """Generate a complete multi-tick stream scenario with ground-truth states."""
        rng = random.Random(self.seed + scenario_idx * 100)
        scenario_id = f"stream_scen_{scenario_idx:03d}"

        # 1. Create target key-value pairs
        keys = [
            f"key_{rng.choice(self.ADJECTIVES)}_{rng.choice(self.NOUNS)}"
            for _ in range(target_keys_count)
        ]
        values = [
            f"val_{rng.choice(self.ADJECTIVES)}_{rng.choice(self.NOUNS)}"
            for _ in range(target_keys_count)
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
        oracle_wm: Dict[str, str] = {}
        oracle_src: Dict[str, str] = {}
        oracle_goals: Dict[str, GoalState] = {}
        
        target_bindings_by_tick: Dict[int, Dict[str, str]] = {}
        source_bindings_by_tick: Dict[int, Dict[str, str]] = {}
        goals_by_tick: Dict[int, List[GoalState]] = {}
        oracle_states: Dict[int, StructuredSelfState] = {}

        for t in range(num_ticks + 1):
            tick_events = [ev for ev, ev_t, _ in scheduled_events if ev_t == t]
            for ev in tick_events:
                if ev.key_bindings:
                    for k, v in ev.key_bindings.items():
                        oracle_wm[k] = v
                        oracle_src[k] = ev.source.value
                if ev.event_type == "goal_update" or "goal_id" in ev.metadata:
                    gid = ev.metadata.get("goal_id", f"goal_{ev.event_id}")
                    desc = ev.metadata.get("goal_description", ev.content)
                    status = ev.metadata.get("goal_status", "active")
                    if gid in oracle_goals:
                        oracle_goals[gid].status = status
                        oracle_goals[gid].updated_at_step = t
                    else:
                        oracle_goals[gid] = GoalState(
                            goal_id=gid,
                            description=desc,
                            status=status,
                            created_at_step=t,
                            updated_at_step=t,
                        )

            target_bindings_by_tick[t] = dict(oracle_wm)
            source_bindings_by_tick[t] = dict(oracle_src)
            goals_by_tick[t] = [g.model_copy(deep=True) for g in oracle_goals.values()]
            
            unresolved = [g.goal_id for g in oracle_goals.values() if g.status in ("pending", "suspended")]
            oracle_states[t] = StructuredSelfState(
                working_memory=dict(oracle_wm),
                goals=[g.model_copy(deep=True) for g in oracle_goals.values()],
                source_ledger=dict(oracle_src),
                unresolved_items=list(unresolved),
                last_updated_step=t,
            )

        return StreamScenario(
            scenario_id=scenario_id,
            total_ticks=num_ticks,
            scheduled_events=scheduled_events,
            target_bindings_by_tick=target_bindings_by_tick,
            source_bindings_by_tick=source_bindings_by_tick,
            goals_by_tick=goals_by_tick,
            oracle_states=oracle_states,
        )
