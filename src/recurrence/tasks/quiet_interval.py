"""Task generator and episode specifications for Sprint S07.1 (Experiment E06b: Available-Inference Null Consolidation Test).

Supports 2 matched informational regimes:
1. available_inference: Both premises (A->B and B->C) are asserted pre-null; quiet interval allows legitimate derivation of A->C.
2. missing_premise_control: Premise A->B is pre-null; Premise B->C arrives post-null. Tests resistance to premature derivation/hallucination.

Diagnostic constructs:
1. derivation_multihop: Transitive deduction (A->C)
2. goal_activation: Verification and activation of Goal Beta upon authorization
3. stable_kv: Direct working memory retention invariant
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
    probe_type: Literal["derivation_multihop", "goal_activation", "stable_kv"]
    regime: Literal["available_inference", "missing_premise_control"]
    question: str
    options: Dict[str, str]
    correct_letter: str
    correct_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuietIntervalEpisode:
    """Episode specification with split prefix, continuation, and paired probes."""
    episode_id: str
    regime: Literal["available_inference", "missing_premise_control"]
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
        regime: Literal["available_inference", "missing_premise_control"] = "available_inference",
        prefix_ticks: int = 4,
        continuation_ticks: int = 3,
        seed: Optional[int] = None,
    ) -> QuietIntervalEpisode:
        """Generate a single controlled episode with common prefix, continuation, and 3 diagnostic probes."""
        if seed is not None:
            self.rng = random.Random(seed + episode_idx * 997 + (10000 if regime == "missing_premise_control" else 0))

        reg_tag = "avail" if regime == "available_inference" else "missing"
        ep_id = f"ep_s07_{episode_idx:03d}_{reg_tag}"

        # -------------------------------------------------------------
        # 1. Generate Entity Bindings
        # -------------------------------------------------------------
        # Multi-Hop: Hop 1 (k_hop1 -> v_hop1), Hop 2 (k_hop2 -> v_hop2 where k_hop2 == key_v_hop1)
        k_hop1 = self._make_key("hop")
        v_hop1 = self._make_val("mid")
        k_hop2 = f"key_{v_hop1.replace('val_', '')}"
        v_hop2 = self._make_val("term")

        # Stable Working Memory (Direct KV)
        k_stable = self._make_key("stable")
        v_stable = self._make_val("stable")

        # Goal Management: Goal Alpha (active) and Goal Beta (pending -> active)
        gid_alpha = "goal_alpha"
        desc_alpha = "Calibrate primary diagnostic matrix"
        gid_beta = "goal_beta"
        desc_beta = "Engage secondary telemetry subsystem"
        auth_token_val = self._make_val("auth")
        k_auth = "key_telemetry_authorization"

        # In-Context Distractors
        k_dist1 = self._make_key("dist1")
        v_dist1 = self._make_val("dist1")
        k_dist2 = self._make_key("dist2")
        v_dist2 = self._make_val("dist2")
        k_dist3 = self._make_key("dist3")
        v_dist3 = self._make_val("dist3")

        all_context_values = {
            v_hop1, v_hop2, v_stable, auth_token_val,
            v_dist1, v_dist2, v_dist3,
        }

        # -------------------------------------------------------------
        # 2. Build Prefix Events (Ticks 0 ... prefix_ticks - 1)
        # -------------------------------------------------------------
        prefix_events: List[MemoryEvent] = []

        # Tick 0: Stable KV & Initial Active Goal Alpha
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

        # Tick 2: Goal Beta declared (pending) & Distractor 1
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
            content=f"Background monitoring registers {k_dist1} = {v_dist1}.",
            key_bindings={k_dist1: v_dist1},
            metadata={"domain": "distractor"},
        ))

        # Tick 3:
        # If available_inference: Premise 2 (B -> C) and Goal Beta Authorization arrive in prefix!
        # If missing_premise_control: Distractor 2 arrives in prefix; Premise 2 arrives in continuation.
        if regime == "available_inference":
            prefix_events.append(MemoryEvent(
                event_id=f"{ep_id}_ev04",
                step_index=3,
                source=EventSource.ENVIRONMENT,
                event_type="observation",
                content=f"Routing matrix updates: {k_hop2} links directly to terminal target {v_hop2}.",
                key_bindings={k_hop2: v_hop2},
                metadata={"domain": "multihop_step2"},
            ))
            prefix_events.append(MemoryEvent(
                event_id=f"{ep_id}_ev05",
                step_index=3,
                source=EventSource.EXPERIMENTER,
                event_type="goal_update",
                content=f"Experimenter provides verification token: {k_auth} = {auth_token_val}. Authorization satisfied; {gid_beta} transitioned to active.",
                key_bindings={k_auth: auth_token_val},
                metadata={"goal_id": gid_beta, "goal_description": desc_beta, "goal_status": "active"},
            ))
        else:
            prefix_events.append(MemoryEvent(
                event_id=f"{ep_id}_ev04_ctrl",
                step_index=3,
                source=EventSource.ENVIRONMENT,
                event_type="distractor",
                content=f"Sensors detect background flux: {k_dist2} = {v_dist2}.",
                key_bindings={k_dist2: v_dist2},
                metadata={"domain": "distractor"},
            ))
            prefix_events.append(MemoryEvent(
                event_id=f"{ep_id}_ev05_ctrl",
                step_index=3,
                source=EventSource.EXPERIMENTER,
                event_type="goal_update",
                content=f"Experimenter provides verification token: {k_auth} = {auth_token_val}. Authorization satisfied; {gid_beta} transitioned to active.",
                key_bindings={k_auth: auth_token_val},
                metadata={"goal_id": gid_beta, "goal_description": desc_beta, "goal_status": "active"},
            ))

        # -------------------------------------------------------------
        # 3. Build Continuation Events
        # -------------------------------------------------------------
        continuation_events: List[MemoryEvent] = []

        if regime == "available_inference":
            # All evidence was already in prefix; continuation is pure distractor interference
            continuation_events.append(MemoryEvent(
                event_id=f"{ep_id}_ev06",
                step_index=0,
                source=EventSource.ENVIRONMENT,
                event_type="distractor",
                content=f"Telemetry scan logs {k_dist2} = {v_dist2}.",
                key_bindings={k_dist2: v_dist2},
                metadata={"domain": "distractor"},
            ))
            continuation_events.append(MemoryEvent(
                event_id=f"{ep_id}_ev07",
                step_index=1,
                source=EventSource.ENVIRONMENT,
                event_type="distractor",
                content=f"Auxiliary sensor reads {k_dist3} = {v_dist3}.",
                key_bindings={k_dist3: v_dist3},
                metadata={"domain": "distractor"},
            ))
        else:
            # Missing premise control: Premise 2 (B -> C) arrives post-null
            continuation_events.append(MemoryEvent(
                event_id=f"{ep_id}_ev06",
                step_index=0,
                source=EventSource.ENVIRONMENT,
                event_type="observation",
                content=f"Routing matrix updates: {k_hop2} links directly to terminal target {v_hop2}.",
                key_bindings={k_hop2: v_hop2},
                metadata={"domain": "multihop_step2"},
            ))
            continuation_events.append(MemoryEvent(
                event_id=f"{ep_id}_ev07",
                step_index=1,
                source=EventSource.ENVIRONMENT,
                event_type="distractor",
                content=f"Auxiliary sensor reads {k_dist3} = {v_dist3}.",
                key_bindings={k_dist3: v_dist3},
                metadata={"domain": "distractor"},
            ))

        # -------------------------------------------------------------
        # 4. Compute Reference Oracle States
        # -------------------------------------------------------------
        if regime == "available_inference":
            oracle_prefix_wm = {
                k_stable: v_stable,
                k_hop1: v_hop1,
                k_dist1: v_dist1,
                k_hop2: v_hop2,
                k_auth: auth_token_val,
            }
            oracle_terminal_wm = {
                **oracle_prefix_wm,
                k_dist2: v_dist2,
                k_dist3: v_dist3,
            }
        else:
            oracle_prefix_wm = {
                k_stable: v_stable,
                k_hop1: v_hop1,
                k_dist1: v_dist1,
                k_dist2: v_dist2,
                k_auth: auth_token_val,
            }
            oracle_terminal_wm = {
                **oracle_prefix_wm,
                k_hop2: v_hop2,
                k_dist3: v_dist3,
            }

        oracle_prefix_state = StructuredSelfState(
            working_memory=oracle_prefix_wm,
            goals=[
                GoalState(goal_id=gid_alpha, description=desc_alpha, status="active", created_at_step=0, updated_at_step=0),
                GoalState(goal_id=gid_beta, description=desc_beta, status="active", created_at_step=2, updated_at_step=3),
            ],
            source_ledger={k: "environment" if "auth" not in k and "goal" not in k else "experimenter" for k in oracle_prefix_wm},
            unresolved_items=[],
            derived_inferences={},
            last_updated_step=prefix_ticks - 1,
        )

        oracle_terminal_state = StructuredSelfState(
            working_memory=oracle_terminal_wm,
            goals=[
                GoalState(goal_id=gid_alpha, description=desc_alpha, status="active", created_at_step=0, updated_at_step=0),
                GoalState(goal_id=gid_beta, description=desc_beta, status="active", created_at_step=2, updated_at_step=3),
            ],
            source_ledger={k: "environment" if "auth" not in k and "goal" not in k else "experimenter" for k in oracle_terminal_wm},
            unresolved_items=[],
            derived_inferences={k_hop1: v_hop2},
            last_updated_step=prefix_ticks + len(continuation_events) - 1,
        )

        # -------------------------------------------------------------
        # 5. Generate 3 Paired Diagnostic Probes (4AFC with In-Context Foils)
        # -------------------------------------------------------------
        letters = ["A", "B", "C", "D"]
        probes: List[QuietIntervalProbe] = []

        # --- Probe 1: Multi-Hop Transitive Binding (4AFC) ---
        target_multihop_ans = v_hop2
        multihop_foils = list(all_context_values - {target_multihop_ans})
        self.rng.shuffle(multihop_foils)
        chosen_multihop_foils = multihop_foils[:3]
        all_multihop_opts = [target_multihop_ans] + chosen_multihop_foils
        self.rng.shuffle(all_multihop_opts)
        multihop_opt_dict = {l: val for l, val in zip(letters, all_multihop_opts)}
        correct_multihop_letter = [l for l, val in multihop_opt_dict.items() if val == target_multihop_ans][0]

        probes.append(QuietIntervalProbe(
            probe_id=f"{ep_id}_p1_multihop",
            probe_type="derivation_multihop",
            regime=regime,
            question=f"Tracing all relational links starting from '{k_hop1}', what is the final terminal value reached?",
            options=multihop_opt_dict,
            correct_letter=correct_multihop_letter,
            correct_answer=target_multihop_ans,
            metadata={"domain": "derivation_multihop", "root_key": k_hop1, "terminal_val": target_multihop_ans, "regime": regime},
        ))

        # --- Probe 2: Repaired Goal State Activation (4AFC) ---
        target_goal_ans = f"Goal '{gid_beta}': {desc_beta} (status: active, authorization satisfied)"
        goal_foils = [
            f"Goal '{gid_alpha}': {desc_alpha} (status: completed)",
            f"Goal '{gid_beta}': {desc_beta} (status: pending authorization)",
            "No active goals remaining in registry",
        ]
        all_goal_opts = [target_goal_ans] + goal_foils
        self.rng.shuffle(all_goal_opts)
        goal_opt_dict = {l: opt for l, opt in zip(letters, all_goal_opts)}
        correct_goal_letter = [l for l, opt in goal_opt_dict.items() if opt == target_goal_ans][0]

        probes.append(QuietIntervalProbe(
            probe_id=f"{ep_id}_p2_goal",
            probe_type="goal_activation",
            regime=regime,
            question=f"Following the arrival of the authorization token, what is the verified current status of secondary goal '{gid_beta}'?",
            options=goal_opt_dict,
            correct_letter=correct_goal_letter,
            correct_answer=target_goal_ans,
            metadata={"domain": "goal_activation", "goal_id": gid_beta, "regime": regime},
        ))

        # --- Probe 3: Stable Working Memory Retention / Invariant (4AFC) ---
        target_stable_ans = v_stable
        stable_foils = list(all_context_values - {target_stable_ans})
        self.rng.shuffle(stable_foils)
        chosen_stable_foils = stable_foils[:3]
        all_stable_opts = [target_stable_ans] + chosen_stable_foils
        self.rng.shuffle(all_stable_opts)
        stable_opt_dict = {l: val for l, val in zip(letters, all_stable_opts)}
        correct_stable_letter = [l for l, val in stable_opt_dict.items() if val == target_stable_ans][0]

        probes.append(QuietIntervalProbe(
            probe_id=f"{ep_id}_p3_stable_kv",
            probe_type="stable_kv",
            regime=regime,
            question=f"What is the value of key '{k_stable}' established at initialization?",
            options=stable_opt_dict,
            correct_letter=correct_stable_letter,
            correct_answer=target_stable_ans,
            metadata={"domain": "stable_kv", "key": k_stable, "val": v_stable, "regime": regime},
        ))

        return QuietIntervalEpisode(
            episode_id=ep_id,
            regime=regime,
            prefix_events=prefix_events,
            continuation_events=continuation_events,
            probes=probes,
            oracle_prefix_state=oracle_prefix_state,
            oracle_terminal_state=oracle_terminal_state,
            metadata={
                "k_hop1": k_hop1,
                "v_hop1": v_hop1,
                "k_hop2": k_hop2,
                "v_hop2": v_hop2,
                "k_stable": k_stable,
                "gid_beta": gid_beta,
                "regime": regime,
            },
        )
