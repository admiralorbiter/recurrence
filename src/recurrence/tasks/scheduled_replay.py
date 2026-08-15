"""Hardened episodic stream and probe battery generator for Sprint S06.1 (Experiment E05b).

Generates strictly controlled multi-stage streams across parameterized horizons,
counterbalanced arrival dynamics, distraction densities, and 4 clean forced-choice evaluation probes:
1. Delayed Key-Value Retrieval (4AFC) - Zero suffix shortcuts
2. Source Attribution (3AFC) - Strictly counterbalanced across environment/self/experimenter
3. Goal State Identification (4AFC) - Strictly counterbalanced across active/suspended/completed/pending
4. Multi-Hop Associative Retrieval (4AFC) - Zero role-coded index tags
"""

from dataclasses import dataclass, field
import hashlib
import json
import random
from typing import Any, Dict, List, Literal, Optional, Tuple

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
    probe_type: Literal["delayed_kv", "source_attribution", "goal_state", "multihop"]
    question: str
    options: Dict[str, str]  # e.g., {"A": "val_crystal_monolith", "B": "val_amber_spire", ...}
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
    """Generator for hardened scheduled-vs-replay benchmark episodes with eradicated shortcuts."""

    NOUNS = [
        "falcon", "canyon", "river", "glacier", "prism", "tempest", "harbor", "citadel",
        "volcano", "compass", "meadow", "cascade", "spire", "cavern", "lagoon", "monolith",
        "beacon", "aurora", "stratus", "zenith", "vortex", "bastion", "sanctuary", "pinnacle",
        "summit", "horizon", "eclipse", "haven", "oasis", "solstice", "pulsar", "nebula"
    ]
    ADJECTIVES = [
        "obsidian", "velvet", "golden", "emerald", "crimson", "sapphire", "amber", "silver",
        "celestial", "shadow", "radiant", "frozen", "solar", "lunar", "mystic", "ancient",
        "stellar", "cobalt", "iron", "topaz", "crystal", "quartz", "twilight", "phantom",
        "spectral", "prismatic", "kinetic", "arcane", "radiant", "magnetic", "glacial"
    ]

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _make_key_val(self, rng: random.Random, used_keys: set, used_vals: set) -> Tuple[str, str]:
        """Generate key and value with NO numerical suffixes or role markers."""
        while True:
            adj1 = rng.choice(self.ADJECTIVES)
            noun1 = rng.choice(self.NOUNS)
            k = f"key_{adj1}_{noun1}"
            if k not in used_keys:
                used_keys.add(k)
                break

        while True:
            adj2 = rng.choice(self.ADJECTIVES)
            noun2 = rng.choice(self.NOUNS)
            v = f"val_{adj2}_{noun2}"
            if v not in used_vals and v != k:
                used_vals.add(v)
                break

        return k, v

    def _make_foil_val(self, rng: random.Random, used_vals: set) -> str:
        """Generate a distractor value with identical vocabulary structure and NO numerical suffix."""
        while True:
            adj = rng.choice(self.ADJECTIVES)
            noun = rng.choice(self.NOUNS)
            v = f"val_{adj}_{noun}"
            if v not in used_vals:
                used_vals.add(v)
                return v

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
        """Generate a single controlled episodic stream with 4 clean counterbalanced probes."""
        ep_seed = (seed or self.seed) + episode_idx * 10007
        rng = random.Random(ep_seed)
        episode_id = f"ep_{num_ticks}t_idx{episode_idx:03d}"

        used_keys: set = set()
        used_vals: set = set()

        total_targets = 24 if capacity_overflow else target_keys_count
        
        # 1. Counterbalance Source Allocation across targets
        # Ensure at least 1 target from environment, self, and experimenter
        sources_cycle = [EventSource.ENVIRONMENT, EventSource.SELF, EventSource.EXPERIMENTER]
        sources_list = [sources_cycle[i % 3] for i in range(total_targets)]
        rng.shuffle(sources_list)

        # 2. Select event arrival ticks
        if burst_mode:
            target_ticks = list(range(1, min(total_targets + 1, num_ticks - 2)))
            while len(target_ticks) < total_targets:
                target_ticks.append(target_ticks[-1])
        else:
            step_stride = max(1, (num_ticks - 4) // max(1, total_targets))
            target_ticks = [1 + i * step_stride for i in range(total_targets)]
            target_ticks = [min(t, num_ticks - 3) for t in target_ticks]

        target_pairs: List[Tuple[str, str, EventSource, int]] = []
        for idx in range(total_targets):
            k, v = self._make_key_val(rng, used_keys, used_vals)
            src = sources_list[idx]
            t_tick = target_ticks[idx] if idx < len(target_ticks) else (num_ticks - 3)
            target_pairs.append((k, v, src, t_tick))

        scheduled_events: List[MemoryEvent] = []
        ev_counter = 0

        # 3. Schedule Primary Goal at Tick 0
        primary_goal_desc = "Execute comprehensive environmental diagnostic"
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
            }
        ))
        ev_counter += 1

        # 4. Schedule Secondary Goal with Counterbalanced Terminal Status
        # Counterbalance across active, suspended, completed, pending
        target_goal_statuses = ["active", "suspended", "completed", "pending"]
        chosen_sec_status = target_goal_statuses[episode_idx % 4]
        sec_goal_desc = "Calibrate auxiliary power matrix"
        t_sec_start = max(2, num_ticks // 4)
        t_sec_mod = max(t_sec_start + 2, (num_ticks * 2) // 4)

        if chosen_sec_status != "pending":
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
                }
            ))
            ev_counter += 1

            if chosen_sec_status in ("suspended", "completed"):
                scheduled_events.append(MemoryEvent(
                    event_id=f"ev_{ev_counter:04d}",
                    step_index=t_sec_mod,
                    source=EventSource.SELF,
                    event_type="goal_update",
                    content=f"Secondary objective {chosen_sec_status}: {sec_goal_desc}",
                    key_bindings={},
                    metadata={
                        "goal_id": "goal_secondary",
                        "goal_description": sec_goal_desc,
                        "goal_status": chosen_sec_status,
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

        # 6. Schedule Multi-Hop Associative Link (K_hop_A -> K_hop_B, K_hop_B -> V_hop_final)
        # Eradicate role-coded IDs (901/902); use clean dictionary strings
        t_hop_1 = max(1, num_ticks // 5)
        t_hop_2 = max(t_hop_1 + 3, (num_ticks * 3) // 5)
        hop_k1, hop_pointer = self._make_key_val(rng, used_keys, used_vals)
        hop_k2 = hop_pointer  # Intermediate pointer key
        _, hop_final_val = self._make_key_val(rng, used_keys, used_vals)

        scheduled_events.append(MemoryEvent(
            event_id=f"ev_{ev_counter:04d}",
            step_index=t_hop_1,
            source=EventSource.ENVIRONMENT,
            event_type="observation",
            content=f"Associative pointer established: {hop_k1} references {hop_k2}.",
            key_bindings={hop_k1: hop_k2},
            metadata={"is_multihop_step1": True, "source_key": hop_k1, "pointer_val": hop_k2}
        ))
        ev_counter += 1

        scheduled_events.append(MemoryEvent(
            event_id=f"ev_{ev_counter:04d}",
            step_index=t_hop_2,
            source=EventSource.ENVIRONMENT,
            event_type="observation",
            content=f"Associative target established: {hop_k2} resolves to {hop_final_val}.",
            key_bindings={hop_k2: hop_final_val},
            metadata={"is_multihop_step2": True, "pointer_key": hop_k2, "final_val": hop_final_val}
        ))
        ev_counter += 1

        # 7. Schedule Background Distractors
        distractor_verbs = ["monitored", "logged", "polled", "verified", "inspected", "synchronized"]
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

        scheduled_events.sort(key=lambda e: (e.step_index, e.event_id))

        # 8. Compute Ground-Truth Terminal State
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

        # 9. Generate 4 Clean Forced-Choice Probes
        probes: List[ScheduledReplayProbe] = []
        letters_4 = ["A", "B", "C", "D"]
        letters_3 = ["A", "B", "C"]

        # Probe 1: Delayed KV Retrieval (4AFC) - Zero Suffix Shortcut
        valid_targets = [p for p in target_pairs if p[0] in terminal_state.working_memory]
        if valid_targets:
            # Rotate target selection across episodes
            t_idx = episode_idx % len(valid_targets)
            t_key, t_val, _, _ = valid_targets[t_idx]
            
            foils_kv = [self._make_foil_val(rng, used_vals) for _ in range(3)]
            opts_kv_list = [t_val] + foils_kv
            rng.shuffle(opts_kv_list)
            opts_kv = {letters_4[i]: opts_kv_list[i] for i in range(4)}
            corr_kv_l = [l for l, v in opts_kv.items() if v == t_val][0]

            probes.append(ScheduledReplayProbe(
                probe_id=f"{episode_id}_p1_delayed_kv",
                probe_type="delayed_kv",
                question=f"What is the exact value bound to '{t_key}'?",
                options=opts_kv,
                correct_letter=corr_kv_l,
                correct_answer=t_val,
                target_key=t_key,
                metadata={"ground_truth_key": t_key, "ground_truth_val": t_val}
            ))

        # Probe 2: Source Attribution (3AFC) - Counterbalanced Source Target
        if valid_targets:
            # Pick target whose source matches episode_idx % 3
            target_source = sources_cycle[episode_idx % 3]
            matching_targets = [p for p in valid_targets if p[2] == target_source]
            if not matching_targets:
                matching_targets = valid_targets
            t_key_src, _, true_src, _ = matching_targets[0]

            src_opts_list = ["environment", "self", "experimenter"]
            rng.shuffle(src_opts_list)
            opts_src = {letters_3[i]: src_opts_list[i] for i in range(3)}
            corr_src_l = [l for l, s in opts_src.items() if s == true_src.value][0]

            probes.append(ScheduledReplayProbe(
                probe_id=f"{episode_id}_p2_source_attr",
                probe_type="source_attribution",
                question=f"Which entity or subsystem established the value for '{t_key_src}'?",
                options=opts_src,
                correct_letter=corr_src_l,
                correct_answer=true_src.value,
                target_key=t_key_src,
                metadata={"ground_truth_key": t_key_src, "ground_truth_source": true_src.value}
            ))

        # Probe 3: Goal State Identification (4AFC) - Counterbalanced Status & Randomized Options
        status_opts_list = ["active", "suspended", "completed", "pending"]
        rng.shuffle(status_opts_list)
        opts_goal = {letters_4[i]: status_opts_list[i] for i in range(4)}
        corr_goal_l = [l for l, st in opts_goal.items() if st == chosen_sec_status][0]

        probes.append(ScheduledReplayProbe(
            probe_id=f"{episode_id}_p3_goal_state",
            probe_type="goal_state",
            question=f"What is the current operational status of objective 'goal_secondary' ({sec_goal_desc})?",
            options=opts_goal,
            correct_letter=corr_goal_l,
            correct_answer=chosen_sec_status,
            target_goal_id="goal_secondary",
            metadata={"goal_id": "goal_secondary", "true_status": chosen_sec_status}
        ))

        # Probe 4: Multi-Hop Associative Probe (4AFC) - Zero Role Suffix
        foils_hop = [self._make_foil_val(rng, used_vals) for _ in range(3)]
        opts_hop_list = [hop_final_val] + foils_hop
        rng.shuffle(opts_hop_list)
        opts_hop = {letters_4[i]: opts_hop_list[i] for i in range(4)}
        corr_hop_l = [l for l, v in opts_hop.items() if v == hop_final_val][0]

        probes.append(ScheduledReplayProbe(
            probe_id=f"{episode_id}_p4_multihop",
            probe_type="multihop",
            question=f"Following the associative chain starting at '{hop_k1}', what is the final resolved value?",
            options=opts_hop,
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
                "counterbalanced_goal_status": chosen_sec_status,
                "counterbalanced_source": sources_cycle[episode_idx % 3].value,
            }
        )
