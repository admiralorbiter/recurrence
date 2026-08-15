"""Episodic stream and probe battery generator for Sprint S06 (Experiment E05).

Generates strictly controlled multi-stage streams across parameterized horizons,
arrival dynamics, distraction densities, and 5-domain forced-choice evaluation probes:
1. Delayed Key-Value Retrieval (4AFC)
2. Source Attribution (3AFC)
3. Goal State Identification (4AFC)
4. Goal Action Selection (4AFC)
5. Multi-Hop Associative Retrieval (4AFC)
"""

from dataclasses import dataclass, field
import hashlib
import json
import random
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel

from recurrence.memory.schemas import (
    EventSource,
    GoalState,
    MemoryEvent,
    StructuredSelfState,
    StateCapacityConfig,
)
from recurrence.loop.state_manager import StateManager
from recurrence.loop.updater import OracleStateUpdater


@dataclass
class ScheduledReplayProbe:
    """A standardized forced-choice evaluation probe evaluated at terminal tick T."""
    probe_id: str
    probe_type: Literal["delayed_kv", "source_attribution", "goal_state", "goal_action", "multihop"]
    question: str
    options: Dict[str, str]  # e.g., {"A": "val_1", "B": "val_2", "C": "val_3", "D": "val_4"}
    correct_letter: str
    correct_answer: str
    target_key: Optional[str] = None
    target_goal_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledReplayEpisode:
    """A complete episodic stream with scheduled events, oracle terminal state, and probes."""
    episode_id: str
    total_ticks: int
    scheduled_events: List[MemoryEvent]
    oracle_terminal_state: StructuredSelfState
    probes: List[ScheduledReplayProbe]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScheduledReplayGenerator:
    """Generator for controlled scheduled-vs-replay benchmark episodes and balanced probe suites."""

    NOUNS = [
        "falcon", "canyon", "river", "glacier", "prism", "tempest", "harbor", "citadel",
        "volcano", "compass", "meadow", "cascade", "spire", "cavern", "lagoon", "monolith",
        "beacon", "aurora", "stratus", "zenith", "vortex", "bastion", "sanctuary", "pinnacle"
    ]
    ADJECTIVES = [
        "obsidian", "velvet", "golden", "emerald", "crimson", "sapphire", "amber", "silver",
        "celestial", "shadow", "radiant", "frozen", "solar", "lunar", "mystic", "ancient",
        "stellar", "cobalt", "iron", "topaz", "crystal", "quartz", "twilight", "phantom"
    ]
    ACTIONS = [
        "Transmit status telemetry to ground station",
        "Recalibrate optical sensor array",
        "Purge thermal coolant buffer",
        "Synchronize navigational gyro beacon",
        "Compile subsystem integrity manifest",
        "Execute auxiliary thruster diagnostic",
        "Verify cryptographic hash chain",
        "Engage defensive shield matrix"
    ]

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _make_key_val(self, idx: int, rng: random.Random) -> Tuple[str, str]:
        adj1 = self.ADJECTIVES[rng.randint(0, len(self.ADJECTIVES) - 1)]
        noun1 = self.NOUNS[rng.randint(0, len(self.NOUNS) - 1)]
        adj2 = self.ADJECTIVES[rng.randint(0, len(self.ADJECTIVES) - 1)]
        noun2 = self.NOUNS[rng.randint(0, len(self.NOUNS) - 1)]
        return f"key_{adj1}_{noun1}_{idx}", f"val_{adj2}_{noun2}_{idx}"

    def generate_episode(
        self,
        episode_idx: int,
        num_ticks: int = 25,
        target_keys_count: int = 4,
        distractor_density: float = 0.5,
        burst_mode: bool = False,
        capacity_overflow: bool = False,
        seed: Optional[int] = None,
    ) -> ScheduledReplayEpisode:
        """Generate a single controlled episodic stream with balanced probes."""
        ep_seed = (seed or self.seed) + episode_idx * 10007
        rng = random.Random(ep_seed)
        episode_id = f"ep_{num_ticks}t_idx{episode_idx:03d}"

        # 1. Determine key count
        total_targets = 24 if capacity_overflow else target_keys_count
        target_pairs: List[Tuple[str, str, EventSource, int]] = []
        
        # Sources pool: environment (60%), self (20%), experimenter (20%)
        sources_pool = [
            EventSource.ENVIRONMENT,
            EventSource.ENVIRONMENT,
            EventSource.ENVIRONMENT,
            EventSource.SELF,
            EventSource.EXPERIMENTER,
        ]

        # 2. Select event ticks
        if burst_mode:
            # Clustered burst arrival at the start of the horizon
            target_ticks = list(range(1, min(total_targets + 1, num_ticks - 2)))
            while len(target_ticks) < total_targets:
                target_ticks.append(target_ticks[-1])
        else:
            # Spread evenly across the horizon
            step_stride = max(1, (num_ticks - 4) // max(1, total_targets))
            target_ticks = [1 + i * step_stride for i in range(total_targets)]
            target_ticks = [min(t, num_ticks - 3) for t in target_ticks]

        used_keys = set()
        for idx in range(total_targets):
            while True:
                k, v = self._make_key_val(idx, rng)
                if k not in used_keys:
                    used_keys.add(k)
                    break
            src = rng.choice(sources_pool)
            t_tick = target_ticks[idx] if idx < len(target_ticks) else (num_ticks - 3)
            target_pairs.append((k, v, src, t_tick))

        scheduled_events: List[MemoryEvent] = []
        ev_counter = 0

        # 3. Schedule Initial Primary Goal at Tick 0
        primary_goal_desc = "Execute comprehensive environmental diagnostic"
        primary_goal_action = self.ACTIONS[0]
        scheduled_events.append(MemoryEvent(
            event_id=f"ev_{ev_counter:04d}",
            step_index=0,
            source=EventSource.EXPERIMENTER,
            event_type="goal_update",
            content=f"Primary objective assigned: {primary_goal_desc}",
            key_bindings={},
            metadata={
                "goal_id": "goal_primary",
                "goal_description": primary_goal_desc,
                "goal_status": "active",
                "target_action": primary_goal_action,
            }
        ))
        ev_counter += 1

        # 4. Schedule Secondary Goal Lifecycle Events
        # Secondary goal assigned at tick ~ T/4, completed at ~ T/2
        sec_goal_desc = "Calibrate auxiliary power matrix"
        sec_goal_action = self.ACTIONS[1]
        t_sec_start = max(2, num_ticks // 4)
        t_sec_end = max(t_sec_start + 2, (num_ticks * 2) // 4)

        scheduled_events.append(MemoryEvent(
            event_id=f"ev_{ev_counter:04d}",
            step_index=t_sec_start,
            source=EventSource.SELF,
            event_type="goal_update",
            content=f"Secondary objective initiated: {sec_goal_desc}",
            key_bindings={},
            metadata={
                "goal_id": "goal_secondary",
                "goal_description": sec_goal_desc,
                "goal_status": "active",
                "target_action": sec_goal_action,
            }
        ))
        ev_counter += 1

        scheduled_events.append(MemoryEvent(
            event_id=f"ev_{ev_counter:04d}",
            step_index=t_sec_end,
            source=EventSource.SELF,
            event_type="goal_update",
            content=f"Secondary objective finalized: {sec_goal_desc}",
            key_bindings={},
            metadata={
                "goal_id": "goal_secondary",
                "goal_description": sec_goal_desc,
                "goal_status": "completed",
                "target_action": sec_goal_action,
            }
        ))
        ev_counter += 1

        # 5. Schedule Target Key-Value Observations
        for idx, (k, v, src, t_tick) in enumerate(target_pairs):
            scheduled_events.append(MemoryEvent(
                event_id=f"ev_{ev_counter:04d}",
                step_index=t_tick,
                source=src,
                event_type="observation" if src == EventSource.ENVIRONMENT else ("action" if src == EventSource.SELF else "experimenter"),
                content=f"Entity established: {k} is set to {v} by {src.value}.",
                key_bindings={k: v},
                metadata={
                    "target_key": k,
                    "target_value": v,
                    "true_source": src.value,
                    "target_index": idx,
                }
            ))
            ev_counter += 1

        # 6. Schedule Multi-Hop Link (Pair key_hop_A -> key_hop_B -> val_hop_target)
        t_hop_1 = max(1, num_ticks // 5)
        t_hop_2 = max(t_hop_1 + 3, (num_ticks * 3) // 5)
        hop_k1, hop_pointer = self._make_key_val(901, rng)
        hop_k2 = hop_pointer  # hop_k2 key is the value of hop_k1
        _, hop_final_val = self._make_key_val(902, rng)

        scheduled_events.append(MemoryEvent(
            event_id=f"ev_{ev_counter:04d}",
            step_index=t_hop_1,
            source=EventSource.ENVIRONMENT,
            event_type="observation",
            content=f"Associative pointer: {hop_k1} references entity {hop_k2}.",
            key_bindings={hop_k1: hop_k2},
            metadata={"is_multihop_step1": True, "source_key": hop_k1, "pointer_val": hop_k2}
        ))
        ev_counter += 1

        scheduled_events.append(MemoryEvent(
            event_id=f"ev_{ev_counter:04d}",
            step_index=t_hop_2,
            source=EventSource.ENVIRONMENT,
            event_type="observation",
            content=f"Associative target: {hop_k2} resolves to {hop_final_val}.",
            key_bindings={hop_k2: hop_final_val},
            metadata={"is_multihop_step2": True, "pointer_key": hop_k2, "final_val": hop_final_val}
        ))
        ev_counter += 1

        # 7. Schedule Background Distractors
        distractor_verbs = ["monitored", "logged", "polled", "verified", "inspected"]
        for t in range(num_ticks):
            if rng.random() < distractor_density:
                verb = rng.choice(distractor_verbs)
                scheduled_events.append(MemoryEvent(
                    event_id=f"ev_{ev_counter:04d}",
                    step_index=t,
                    source=EventSource.ENVIRONMENT,
                    event_type="distractor",
                    content=f"Telemetry bus at tick {t}: nominal subsystem throughput {verb}.",
                    key_bindings={},
                    metadata={"is_distractor": True}
                ))
                ev_counter += 1

        # Sort all events chronologically by step_index
        scheduled_events.sort(key=lambda e: (e.step_index, e.event_id))

        # 8. Compute Ground-Truth Terminal State via StateManager & OracleUpdater
        cap_cfg = StateCapacityConfig(max_working_memory_items=16 if capacity_overflow else 64)
        manager = StateManager(capacity_config=cap_cfg)
        updater = OracleStateUpdater(state_manager=manager)

        events_by_tick: Dict[int, List[MemoryEvent]] = {}
        for ev in scheduled_events:
            events_by_tick.setdefault(ev.step_index, []).append(ev)

        for t in range(num_ticks):
            evs = events_by_tick.get(t, [])
            if evs:
                new_st, _, _, _, _, _, _, _ = updater.update(manager.current_state, evs, t)
                manager.update_state(new_st, t, len(evs), schema_valid=True)
            else:
                quiet_st = manager.current_state.model_copy(deep=True)
                quiet_st.last_updated_step = t
                manager.update_state(quiet_st, t, 0, schema_valid=True)

        terminal_state = manager.current_state.model_copy(deep=True)

        # 9. Generate Standardized 5-Domain Forced-Choice Probes
        probes: List[ScheduledReplayProbe] = []

        # Probe 1: Delayed KV Retrieval (4AFC)
        # Select target that exists in terminal working memory
        valid_targets = [p for p in target_pairs if p[0] in terminal_state.working_memory]
        if valid_targets:
            t_key, t_val, _, _ = valid_targets[0]
            # Generate 3 distinct plausible foils
            foils = []
            while len(foils) < 3:
                _, foil_v = self._make_key_val(rng.randint(500, 900), rng)
                if foil_v != t_val and foil_v not in foils:
                    foils.append(foil_v)

            options_list = [t_val] + foils
            rng.shuffle(options_list)
            letters = ["A", "B", "C", "D"]
            opt_dict = {letters[i]: options_list[i] for i in range(4)}
            corr_letter = [l for l, v in opt_dict.items() if v == t_val][0]

            probes.append(ScheduledReplayProbe(
                probe_id=f"{episode_id}_p1_delayed_kv",
                probe_type="delayed_kv",
                question=f"What is the exact value bound to '{t_key}'?",
                options=opt_dict,
                correct_letter=corr_letter,
                correct_answer=t_val,
                target_key=t_key,
                metadata={"ground_truth_key": t_key, "ground_truth_val": t_val}
            ))

        # Probe 2: Source Attribution (3AFC)
        if valid_targets:
            t_key, _, t_src, _ = valid_targets[0]
            src_opts = {"A": "environment", "B": "self", "C": "experimenter"}
            corr_l = [l for l, s in src_opts.items() if s == t_src.value][0]

            probes.append(ScheduledReplayProbe(
                probe_id=f"{episode_id}_p2_source_attr",
                probe_type="source_attribution",
                question=f"Which entity or subsystem established the value for '{t_key}'?",
                options=src_opts,
                correct_letter=corr_l,
                correct_answer=t_src.value,
                target_key=t_key,
                metadata={"ground_truth_key": t_key, "ground_truth_source": t_src.value}
            ))

        # Probe 3: Goal State Identification (4AFC)
        goal_status_opts = {"A": "active", "B": "suspended", "C": "completed", "D": "pending"}
        corr_sec_status = "completed"
        corr_l = [l for l, st in goal_status_opts.items() if st == corr_sec_status][0]

        probes.append(ScheduledReplayProbe(
            probe_id=f"{episode_id}_p3_goal_state",
            probe_type="goal_state",
            question="What is the current operational status of objective 'goal_secondary' (Calibrate auxiliary power matrix)?",
            options=goal_status_opts,
            correct_letter=corr_l,
            correct_answer=corr_sec_status,
            target_goal_id="goal_secondary",
            metadata={"goal_id": "goal_secondary", "true_status": corr_sec_status}
        ))

        # Probe 4: Goal Action Selection (4AFC)
        # Given primary goal is active, what action executes next?
        action_foils = [a for a in self.ACTIONS if a != primary_goal_action][:3]
        action_opts_list = [primary_goal_action] + action_foils
        rng.shuffle(action_opts_list)
        action_opts = {letters[i]: action_opts_list[i] for i in range(4)}
        corr_action_l = [l for l, act in action_opts.items() if act == primary_goal_action][0]

        probes.append(ScheduledReplayProbe(
            probe_id=f"{episode_id}_p4_goal_action",
            probe_type="goal_action",
            question="Based on the currently active primary goal ('goal_primary'), which operational action should execute next?",
            options=action_opts,
            correct_letter=corr_action_l,
            correct_answer=primary_goal_action,
            target_goal_id="goal_primary",
            metadata={"goal_id": "goal_primary", "target_action": primary_goal_action}
        ))

        # Probe 5: Multi-Hop Associative Probe (4AFC)
        hop_foils = []
        while len(hop_foils) < 3:
            _, f_val = self._make_key_val(rng.randint(600, 950), rng)
            if f_val != hop_final_val and f_val not in hop_foils:
                hop_foils.append(f_val)

        hop_opts_list = [hop_final_val] + hop_foils
        rng.shuffle(hop_opts_list)
        hop_opts = {letters[i]: hop_opts_list[i] for i in range(4)}
        corr_hop_l = [l for l, v in hop_opts.items() if v == hop_final_val][0]

        probes.append(ScheduledReplayProbe(
            probe_id=f"{episode_id}_p5_multihop",
            probe_type="multihop",
            question=f"Following the associative chain starting at '{hop_k1}', what is the final resolved value?",
            options=hop_opts,
            correct_letter=corr_hop_l,
            correct_answer=hop_final_val,
            target_key=hop_k1,
            metadata={"chain_start": hop_k1, "chain_pointer": hop_k2, "final_val": hop_final_val}
        ))

        return ScheduledReplayEpisode(
            episode_id=episode_id,
            total_ticks=num_ticks,
            scheduled_events=scheduled_events,
            oracle_terminal_state=terminal_state,
            probes=probes,
            metadata={
                "seed": ep_seed,
                "target_keys_count": total_targets,
                "distractor_density": distractor_density,
                "burst_mode": burst_mode,
                "capacity_overflow": capacity_overflow,
            }
        )
