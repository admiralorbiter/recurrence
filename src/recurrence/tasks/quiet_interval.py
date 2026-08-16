"""Task generator and episode specifications for Sprint S07 (Experiment E06: Quiet Intervals).

Constructs prefix -> null interval -> continuation episodes with 4 targeted diagnostic constructs:
1. derivation_multihop: Multi-hop transitive deduction across an intervening temporal gap
2. source_conflict: Unresolved conflicting assertions requiring consolidation rather than hallucination
3. unresolved_goal: Goal prioritization with prerequisites arriving across the boundary
4. stable_kv: Working memory retention / invariant negative control
"""

from dataclasses import dataclass, field
import random
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from recurrence.memory.schemas import (
    EventSource,
    GoalState,
    MemoryEvent,
    StructuredSelfState,
)


@dataclass
class QuietIntervalProbe:
    """Diagnostic probe evaluating representational status at terminal step."""
    probe_id: str
    probe_type: Literal["derivation_multihop", "source_conflict", "unresolved_goal", "stable_kv"]
    question: str
    options: Dict[str, str]
    correct_letter: str
    correct_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuietIntervalEpisode:
    """Episode specification with split prefix, continuation, and paired probes."""
    episode_id: str
    prefix_events: List[MemoryEvent]
    continuation_events: List[MemoryEvent]
    probes: List[QuietIntervalProbe]
    oracle_prefix_state: StructuredSelfState
    oracle_terminal_state: StructuredSelfState
    metadata: Dict[str, Any] = field(default_factory=dict)


# Vocabulary banks for structured thematic entity generation without shortcuts
NOUN_POOL = [
    "prism", "matrix", "beacon", "summit", "canyon", "spire", "nexus", "ridge",
    "orbit", "harbor", "portal", "sensor", "relay", "vortex", "cipher", "stratum",
    "pulsar", "zenith", "vertex", "glacier", "chronos", "vector", "radiance", "solstice"
]

COLOR_POOL = [
    "amber", "cobalt", "crimson", "emerald", "topaz", "amethyst", "indigo", "scarlet",
    "silver", "obsidian", "azure", "garnet", "jade", "onyx", "cerulean", "sapphire",
    "copper", "platinum", "coral", "violet", "quartz", "vermilion", "malachite", "bronze"
]


class QuietIntervalGenerator:
    """Generator for prefix -> null interval -> continuation benchmark episodes."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def _make_key(self, prefix: str) -> str:
        c = self.rng.choice(COLOR_POOL)
        n = self.rng.choice(NOUN_POOL)
        return f"key_{prefix}_{c}_{n}"

    def _make_val(self, prefix: str) -> str:
        c = self.rng.choice(COLOR_POOL)
        n = self.rng.choice(NOUN_POOL)
        return f"val_{prefix}_{c}_{n}"

    def generate_episode(
        self,
        episode_idx: int,
        prefix_ticks: int = 4,
        continuation_ticks: int = 3,
        seed: Optional[int] = None,
    ) -> QuietIntervalEpisode:
        """Generate a single controlled episode with common prefix, continuation, and 4 probes."""
        if seed is not None:
            self.rng = random.Random(seed + episode_idx * 997)

        ep_id = f"ep_s07_{episode_idx:03d}"

        # -------------------------------------------------------------
        # 1. Generate Entity Bindings for the 4 Diagnostic Domains
        # -------------------------------------------------------------
        # Domain 1: Multi-Hop (Hop 1: K_hop1 -> V_hop1; Hop 2: K_hop2 -> V_hop2 where K_hop2 == V_hop1)
        k_hop1 = self._make_key("hop")
        v_hop1 = self._make_val("mid")
        k_hop2 = f"key_{v_hop1.replace('val_', '')}"
        v_hop2 = self._make_val("term")

        # Domain 2: Source Conflict (Environment says V_env, Self says V_self)
        k_conflict = self._make_key("conflict")
        v_conflict_env = self._make_val("env")
        v_conflict_self = self._make_val("self")

        # Domain 3: Unresolved Goal (Goal Alpha active, Goal Beta pending prerequisite auth_token)
        gid_alpha = "goal_alpha"
        desc_alpha = "Calibrate primary diagnostic matrix"
        gid_beta = "goal_beta"
        desc_beta = "Engage secondary telemetry subsystem"
        auth_token_val = self._make_val("auth")

        # Domain 4: Stable Working Memory (Direct KV)
        k_stable = self._make_key("stable")
        v_stable = self._make_val("stable")

        # In-Context Distractor Pool
        k_dist1 = self._make_key("aux")
        v_dist1 = self._make_val("aux")
        k_dist2 = self._make_key("bg")
        v_dist2 = self._make_val("bg")

        all_context_values = {
            v_hop1, v_hop2, v_conflict_env, v_conflict_self, auth_token_val,
            v_stable, v_dist1, v_dist2,
        }

        # -------------------------------------------------------------
        # 2. Build Prefix Events (Discrete ticks 0 ... prefix_ticks - 1)
        # -------------------------------------------------------------
        prefix_events: List[MemoryEvent] = []

        # Tick 0: Stable KV & Initial Goal Alpha
        prefix_events.append(MemoryEvent(
            event_id=f"{ep_id}_ev00",
            step_index=0,
            source=EventSource.ENVIRONMENT,
            event_type="observation",
            content=f"Initial telemetry establishes {k_stable} = {v_stable}.",
            key_bindings={k_stable: v_stable},
            metadata={"domain": "stable_kv"},
        ))
        prefix_events.append(MemoryEvent(
            event_id=f"{ep_id}_ev01",
            step_index=0,
            source=EventSource.SELF,
            event_type="goal_update",
            content=f"Subsystem initialized. Primary objective: {desc_alpha}.",
            key_bindings={},
            metadata={"goal_id": gid_alpha, "goal_description": desc_alpha, "goal_status": "active"},
        ))

        # Tick 1: Multi-Hop Step 1 (A -> B)
        prefix_events.append(MemoryEvent(
            event_id=f"{ep_id}_ev02",
            step_index=1,
            source=EventSource.ENVIRONMENT,
            event_type="observation",
            content=f"Relay channel connects {k_hop1} to intermediate target {v_hop1}.",
            key_bindings={k_hop1: v_hop1},
            metadata={"domain": "multihop_step1"},
        ))

        # Tick 2: Goal Beta declared with pending prerequisite & background observation
        prefix_events.append(MemoryEvent(
            event_id=f"{ep_id}_ev03",
            step_index=2,
            source=EventSource.SELF,
            event_type="goal_update",
            content=f"Secondary objective registered: {desc_beta}. Status pending authorization token.",
            key_bindings={},
            metadata={"goal_id": gid_beta, "goal_description": desc_beta, "goal_status": "pending"},
        ))
        prefix_events.append(MemoryEvent(
            event_id=f"{ep_id}_ev03b",
            step_index=2,
            source=EventSource.ENVIRONMENT,
            event_type="distractor",
            content=f"Background monitoring registers {k_dist2} = {v_dist2}.",
            key_bindings={k_dist2: v_dist2},
            metadata={"domain": "distractor"},
        ))

        # Tick 3: Conflicting Assertions on k_conflict
        prefix_events.append(MemoryEvent(
            event_id=f"{ep_id}_ev04",
            step_index=3,
            source=EventSource.ENVIRONMENT,
            event_type="observation",
            content=f"External sensor scan asserts {k_conflict} = {v_conflict_env}.",
            key_bindings={k_conflict: v_conflict_env},
            metadata={"domain": "source_conflict_env"},
        ))
        prefix_events.append(MemoryEvent(
            event_id=f"{ep_id}_ev05",
            step_index=3,
            source=EventSource.SELF,
            event_type="statement",
            content=f"Internal calibration model asserts {k_conflict} = {v_conflict_self}.",
            key_bindings={k_conflict: v_conflict_self},
            metadata={"domain": "source_conflict_self"},
        ))

        # -------------------------------------------------------------
        # 3. Compute Oracle State at Prefix Boundary (S*)
        # -------------------------------------------------------------
        oracle_prefix_state = StructuredSelfState(
            working_memory={
                k_stable: v_stable,
                k_hop1: v_hop1,
                k_conflict: v_conflict_self,  # latest write
            },
            goals=[
                GoalState(goal_id=gid_alpha, description=desc_alpha, status="active", created_at_step=0, updated_at_step=0),
                GoalState(goal_id=gid_beta, description=desc_beta, status="pending", created_at_step=2, updated_at_step=2),
            ],
            source_ledger={
                k_stable: "environment",
                k_hop1: "environment",
                k_conflict: "self",
            },
            unresolved_items=[
                f"conflict:{k_conflict}",
                f"pending_prerequisite:{gid_beta}",
            ],
            derived_inferences={},
            last_updated_step=prefix_ticks - 1,
        )

        # -------------------------------------------------------------
        # 4. Build Common Continuation Events (Ticks T_cont ... T_cont + continuation_ticks - 1)
        # Note: step_index will be offset dynamically based on K
        # -------------------------------------------------------------
        continuation_events: List[MemoryEvent] = []

        # Cont 1: Multi-Hop Step 2 (B -> C)
        continuation_events.append(MemoryEvent(
            event_id=f"{ep_id}_ev06",
            step_index=0,  # relative offset
            source=EventSource.ENVIRONMENT,
            event_type="observation",
            content=f"Routing matrix updates: {k_hop2} links directly to terminal target {v_hop2}.",
            key_bindings={k_hop2: v_hop2},
            metadata={"domain": "multihop_step2"},
        ))

        # Cont 2: Prerequisite authorization arrives for Goal Beta
        k_auth = "key_telemetry_authorization"
        continuation_events.append(MemoryEvent(
            event_id=f"{ep_id}_ev07",
            step_index=1,  # relative offset
            source=EventSource.EXPERIMENTER,
            event_type="action",
            content=f"Experimenter provides verification token: {k_auth} = {auth_token_val}. Authorization satisfied for {gid_beta}.",
            key_bindings={k_auth: auth_token_val},
            metadata={"domain": "goal_auth", "satisfied_goal": gid_beta},
        ))

        # Cont 3: Distractor observation
        continuation_events.append(MemoryEvent(
            event_id=f"{ep_id}_ev08",
            step_index=2,  # relative offset
            source=EventSource.ENVIRONMENT,
            event_type="distractor",
            content=f"Auxiliary background scan notes {k_dist1} = {v_dist1}.",
            key_bindings={k_dist1: v_dist1},
            metadata={"domain": "distractor"},
        ))

        # -------------------------------------------------------------
        # 5. Compute Oracle Terminal State
        # -------------------------------------------------------------
        oracle_terminal_state = StructuredSelfState(
            working_memory={
                k_stable: v_stable,
                k_hop1: v_hop1,
                k_conflict: v_conflict_self,
                k_hop2: v_hop2,
                k_auth: auth_token_val,
                k_dist1: v_dist1,
            },
            goals=[
                GoalState(goal_id=gid_alpha, description=desc_alpha, status="active", created_at_step=0, updated_at_step=0),
                GoalState(goal_id=gid_beta, description=desc_beta, status="active", created_at_step=2, updated_at_step=prefix_ticks + 1),
            ],
            source_ledger={
                k_stable: "environment",
                k_hop1: "environment",
                k_conflict: "self",
                k_hop2: "environment",
                k_auth: "experimenter",
                k_dist1: "environment",
            },
            unresolved_items=[
                f"conflict:{k_conflict}",
            ],
            derived_inferences={
                k_hop1: v_hop2,  # derived bridge deduction: k_hop1 -> v_hop2
            },
            last_updated_step=prefix_ticks + continuation_ticks - 1,
        )

        # -------------------------------------------------------------
        # 6. Generate 4 Targeted Probes (with In-Context Foils & Counterbalanced Letters)
        # -------------------------------------------------------------
        probes: List[QuietIntervalProbe] = []

        # --- Probe 1: Intervening Multi-Hop Transitive Binding (4AFC) ---
        target_multihop_ans = v_hop2
        multihop_foils = list(all_context_values - {target_multihop_ans})
        self.rng.shuffle(multihop_foils)
        chosen_multihop_foils = multihop_foils[:3]
        all_multihop_opts = [target_multihop_ans] + chosen_multihop_foils
        self.rng.shuffle(all_multihop_opts)
        letters = ["A", "B", "C", "D"]
        multihop_opt_dict = {l: val for l, val in zip(letters, all_multihop_opts)}
        correct_multihop_letter = [l for l, val in multihop_opt_dict.items() if val == target_multihop_ans][0]

        probes.append(QuietIntervalProbe(
            probe_id=f"{ep_id}_p1_multihop",
            probe_type="derivation_multihop",
            question=f"Tracing all relational links starting from '{k_hop1}', what is the final terminal value reached?",
            options=multihop_opt_dict,
            correct_letter=correct_multihop_letter,
            correct_answer=target_multihop_ans,
            metadata={"domain": "derivation_multihop", "root_key": k_hop1, "terminal_val": target_multihop_ans},
        ))

        # --- Probe 2: Unresolved Source Conflict & Consolidation (3AFC) ---
        conflict_options = [
            f"Value '{v_conflict_env}' (asserted by environment)",
            f"Value '{v_conflict_self}' (asserted by self)",
            "Unresolved conflicting assertion (both hypotheses remain in active conflict)",
        ]
        self.rng.shuffle(conflict_options)
        conflict_letters = ["A", "B", "C"]
        conflict_opt_dict = {l: opt for l, opt in zip(conflict_letters, conflict_options)}
        correct_conflict_letter = [
            l for l, opt in conflict_opt_dict.items() if "Unresolved conflicting assertion" in opt
        ][0]

        probes.append(QuietIntervalProbe(
            probe_id=f"{ep_id}_p2_conflict",
            probe_type="source_conflict",
            question=f"Regarding entity '{k_conflict}', what is its verified authoritative status and binding?",
            options=conflict_opt_dict,
            correct_letter=correct_conflict_letter,
            correct_answer=conflict_opt_dict[correct_conflict_letter],
            metadata={"domain": "source_conflict", "target_key": k_conflict},
        ))

        # --- Probe 3: Unresolved Goal Prioritization across Interruption (4AFC) ---
        target_goal_ans = f"Goal '{gid_beta}': {desc_beta} (status: active, authorization satisfied)"
        goal_foils = [
            f"Goal '{gid_alpha}': {desc_alpha} (status: completed)",
            f"Goal '{gid_beta}': {desc_beta} (status: suspended/pending authorization)",
            "No active goals remaining in registry",
        ]
        all_goal_opts = [target_goal_ans] + goal_foils
        self.rng.shuffle(all_goal_opts)
        goal_opt_dict = {l: opt for l, opt in zip(letters, all_goal_opts)}
        correct_goal_letter = [l for l, opt in goal_opt_dict.items() if opt == target_goal_ans][0]

        probes.append(QuietIntervalProbe(
            probe_id=f"{ep_id}_p3_goal",
            probe_type="unresolved_goal",
            question=f"Following the arrival of the authorization token, what is the correct updated status of secondary goal '{gid_beta}'?",
            options=goal_opt_dict,
            correct_letter=correct_goal_letter,
            correct_answer=target_goal_ans,
            metadata={"domain": "unresolved_goal", "goal_id": gid_beta},
        ))

        # --- Probe 4: Stable Working Memory Retention / Invariant (4AFC) ---
        target_stable_ans = v_stable
        stable_foils = list(all_context_values - {target_stable_ans})
        self.rng.shuffle(stable_foils)
        chosen_stable_foils = stable_foils[:3]
        all_stable_opts = [target_stable_ans] + chosen_stable_foils
        self.rng.shuffle(all_stable_opts)
        stable_opt_dict = {l: val for l, val in zip(letters, all_stable_opts)}
        correct_stable_letter = [l for l, val in stable_opt_dict.items() if val == target_stable_ans][0]

        probes.append(QuietIntervalProbe(
            probe_id=f"{ep_id}_p4_stable_kv",
            probe_type="stable_kv",
            question=f"What is the value of key '{k_stable}' established at initialization?",
            options=stable_opt_dict,
            correct_letter=correct_stable_letter,
            correct_answer=target_stable_ans,
            metadata={"domain": "stable_kv", "key": k_stable, "val": v_stable},
        ))

        return QuietIntervalEpisode(
            episode_id=ep_id,
            prefix_events=prefix_events,
            continuation_events=continuation_events,
            probes=probes,
            oracle_prefix_state=oracle_prefix_state,
            oracle_terminal_state=oracle_terminal_state,
            metadata={
                "k_hop1": k_hop1,
                "v_hop2": v_hop2,
                "k_conflict": k_conflict,
                "k_stable": k_stable,
                "gid_beta": gid_beta,
            },
        )
