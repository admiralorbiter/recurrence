"""Task generator and episode specifications for Sprint S08 (Experiment E07: State x Memory Conflict & Causal Interventions).

Constructs structurally matched twin episodes (A and B) with balanced in-context vocabularies
and builds causal intervention testbeds:
1. State x Memory Conflict (M_A + S_B vs M_A + S_A)
2. Reset with Memory Preserved (M_A + S_empty vs M_A + S_A)
3. Surgical Single-Slot Inversion (M_A + S_A[k_target <- V_blue])
4. Clone, Fork, Cross-Swap & Reconvergence (Lineage / Infrastructure)
"""

from dataclasses import dataclass, field
import hashlib
import json
import random
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from recurrence.memory.schemas import (
    EventSource,
    GoalState,
    MemoryEvent,
    StructuredSelfState,
)


@dataclass
class InterventionProbe:
    """Diagnostic probe evaluating causal steering under state/memory interventions."""
    probe_id: str
    probe_type: Literal["target_key", "control_key", "goal_status", "reconvergence_target"]
    question: str
    options: Dict[str, str]
    correct_letter_congruent: str
    target_value_A: str
    target_value_B: str
    control_value: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchedTwinEpisodePair:
    """Structurally isomorphic twin episode pair (World A and World B) with balanced vocabulary."""
    pair_id: str
    twin_index: int
    episode_A_id: str
    episode_B_id: str
    prefix_events_A: List[MemoryEvent]
    prefix_events_B: List[MemoryEvent]
    oracle_state_A: StructuredSelfState
    oracle_state_B: StructuredSelfState
    probes_A: List[InterventionProbe]
    probes_B: List[InterventionProbe]
    k_target: str
    k_control: str
    val_target_A: str  # e.g. V_red
    val_target_B: str  # e.g. V_blue
    val_control: str   # e.g. V_gold
    gid_beta: str
    status_beta_A: str  # "active"
    status_beta_B: str  # "suspended"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CloneReconvergenceSpec:
    """Lineage specification for clone -> fork -> cross-swap -> reconvergence testbed."""
    spec_id: str
    twin_index: int
    prefix_events_common: List[MemoryEvent]
    fork_events_A: List[MemoryEvent]
    fork_events_B: List[MemoryEvent]
    reconvergence_events: List[MemoryEvent]
    oracle_prefix_state: StructuredSelfState
    oracle_fork_state_A: StructuredSelfState
    oracle_fork_state_B: StructuredSelfState
    oracle_reconverged_state: StructuredSelfState
    probes_fork: List[InterventionProbe]
    probes_reconverged: List[InterventionProbe]
    k_target: str
    k_control: str
    val_common: str
    val_fork_A: str
    val_fork_B: str
    val_reconverge: str
    metadata: Dict[str, Any] = field(default_factory=dict)


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


class StateInterventionGenerator:
    """Generator for matched twin episodes and causal intervention testbeds."""

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

    def generate_twin_pair(
        self,
        twin_idx: int,
        prefix_ticks: int = 4,
        seed: Optional[int] = None,
    ) -> MatchedTwinEpisodePair:
        """Generate a structurally matched twin episode pair (World A and World B).
        
        Enforces Vocabulary Balancing: Both V_red (A target) and V_blue (B target) appear
        in BOTH histories as valid in-context candidates.
        """
        if seed is not None:
            self.rng = random.Random(seed + twin_idx * 1013)

        pair_id = f"twin_s08_{twin_idx:03d}"
        ep_A_id = f"{pair_id}_world_A"
        ep_B_id = f"{pair_id}_world_B"

        # Shared Keys
        k_target = self._make_key("target")
        k_control = self._make_key("control")
        k_distractor = self._make_key("aux")

        # Values
        val_target_A = self._make_val("red")   # World A target value
        val_target_B = self._make_val("blue")  # World B target value
        val_control = self._make_val("gold")   # Shared control value
        val_foil1 = self._make_val("foil1")
        val_foil2 = self._make_val("foil2")

        # Goals
        gid_alpha = "goal_alpha"
        desc_alpha = "Calibrate primary diagnostic matrix"
        gid_beta = "goal_beta"
        desc_beta = "Engage secondary telemetry subsystem"
        status_beta_A = "active"
        status_beta_B = "suspended"

        # -------------------------------------------------------------
        # 1. Build Balanced Episodic Memory Events
        # -------------------------------------------------------------
        # In World A:
        # - Step 0: k_control = val_control, goal_alpha = active
        # - Step 1: Auxiliary distractor registers k_distractor = val_target_B (introduces V_blue to A's history!)
        # - Step 2: goal_beta declared as active
        # - Step 3: k_target = val_target_A (V_red)
        events_A: List[MemoryEvent] = [
            MemoryEvent(
                event_id=f"{ep_A_id}_ev00",
                step_index=0,
                source=EventSource.ENVIRONMENT,
                event_type="observation",
                content=f"System initialization establishes {k_control} = {val_control}.",
                key_bindings={k_control: val_control},
                metadata={"domain": "control_key"},
            ),
            MemoryEvent(
                event_id=f"{ep_A_id}_ev01",
                step_index=0,
                source=EventSource.SELF,
                event_type="goal_update",
                content=f"Subsystem initialized. Primary objective: {desc_alpha}.",
                key_bindings={},
                metadata={"goal_id": gid_alpha, "goal_description": desc_alpha, "goal_status": "active"},
            ),
            MemoryEvent(
                event_id=f"{ep_A_id}_ev02",
                step_index=1,
                source=EventSource.ENVIRONMENT,
                event_type="observation",
                content=f"Auxiliary sensor calibration logs reference constant {k_distractor} = {val_target_B}.",
                key_bindings={k_distractor: val_target_B},
                metadata={"domain": "vocabulary_balance_distractor"},
            ),
            MemoryEvent(
                event_id=f"{ep_A_id}_ev03",
                step_index=2,
                source=EventSource.SELF,
                event_type="goal_update",
                content=f"Secondary objective verified and engaged: {desc_beta}. Status: {status_beta_A}.",
                key_bindings={},
                metadata={"goal_id": gid_beta, "goal_description": desc_beta, "goal_status": status_beta_A},
            ),
            MemoryEvent(
                event_id=f"{ep_A_id}_ev04",
                step_index=3,
                source=EventSource.ENVIRONMENT,
                event_type="observation",
                content=f"Primary channel establishes target binding: {k_target} = {val_target_A}.",
                key_bindings={k_target: val_target_A},
                metadata={"domain": "target_key"},
            ),
        ]

        # In World B:
        # - Step 0: k_control = val_control, goal_alpha = active
        # - Step 1: Auxiliary distractor registers k_distractor = val_target_A (introduces V_red to B's history!)
        # - Step 2: goal_beta declared as suspended
        # - Step 3: k_target = val_target_B (V_blue)
        events_B: List[MemoryEvent] = [
            MemoryEvent(
                event_id=f"{ep_B_id}_ev00",
                step_index=0,
                source=EventSource.ENVIRONMENT,
                event_type="observation",
                content=f"System initialization establishes {k_control} = {val_control}.",
                key_bindings={k_control: val_control},
                metadata={"domain": "control_key"},
            ),
            MemoryEvent(
                event_id=f"{ep_B_id}_ev01",
                step_index=0,
                source=EventSource.SELF,
                event_type="goal_update",
                content=f"Subsystem initialized. Primary objective: {desc_alpha}.",
                key_bindings={},
                metadata={"goal_id": gid_alpha, "goal_description": desc_alpha, "goal_status": "active"},
            ),
            MemoryEvent(
                event_id=f"{ep_B_id}_ev02",
                step_index=1,
                source=EventSource.ENVIRONMENT,
                event_type="observation",
                content=f"Auxiliary sensor calibration logs reference constant {k_distractor} = {val_target_A}.",
                key_bindings={k_distractor: val_target_A},
                metadata={"domain": "vocabulary_balance_distractor"},
            ),
            MemoryEvent(
                event_id=f"{ep_B_id}_ev03",
                step_index=2,
                source=EventSource.SELF,
                event_type="goal_update",
                content=f"Secondary objective placed on hold: {desc_beta}. Status: {status_beta_B}.",
                key_bindings={},
                metadata={"goal_id": gid_beta, "goal_description": desc_beta, "goal_status": status_beta_B},
            ),
            MemoryEvent(
                event_id=f"{ep_B_id}_ev04",
                step_index=3,
                source=EventSource.ENVIRONMENT,
                event_type="observation",
                content=f"Primary channel establishes target binding: {k_target} = {val_target_B}.",
                key_bindings={k_target: val_target_B},
                metadata={"domain": "target_key"},
            ),
        ]

        # -------------------------------------------------------------
        # 2. Build Structured Oracle States
        # -------------------------------------------------------------
        oracle_state_A = StructuredSelfState(
            working_memory={
                k_control: val_control,
                k_distractor: val_target_B,
                k_target: val_target_A,
            },
            goals=[
                GoalState(goal_id=gid_alpha, description=desc_alpha, status="active", created_at_step=0, updated_at_step=0),
                GoalState(goal_id=gid_beta, description=desc_beta, status=status_beta_A, created_at_step=2, updated_at_step=2),
            ],
            source_ledger={
                k_control: "environment",
                k_distractor: "environment",
                k_target: "environment",
            },
            unresolved_items=[],
            derived_inferences={},
            last_updated_step=3,
        )

        oracle_state_B = StructuredSelfState(
            working_memory={
                k_control: val_control,
                k_distractor: val_target_A,
                k_target: val_target_B,
            },
            goals=[
                GoalState(goal_id=gid_alpha, description=desc_alpha, status="active", created_at_step=0, updated_at_step=0),
                GoalState(goal_id=gid_beta, description=desc_beta, status=status_beta_B, created_at_step=2, updated_at_step=2),
            ],
            source_ledger={
                k_control: "environment",
                k_distractor: "environment",
                k_target: "environment",
            },
            unresolved_items=[],
            derived_inferences={},
            last_updated_step=3,
        )

        # -------------------------------------------------------------
        # 3. Build Balanced 4AFC Diagnostic Probes
        # -------------------------------------------------------------
        letters = ["A", "B", "C", "D"]

        # Option pool for Target Key Probe: includes val_target_A, val_target_B, val_foil1, val_foil2
        target_options_raw = [val_target_A, val_target_B, val_foil1, val_foil2]
        self.rng.shuffle(target_options_raw)
        target_opt_dict = {l: v for l, v in zip(letters, target_options_raw)}
        corr_let_target_A = [l for l, v in target_opt_dict.items() if v == val_target_A][0]
        corr_let_target_B = [l for l, v in target_opt_dict.items() if v == val_target_B][0]

        probe_target_A = InterventionProbe(
            probe_id=f"{ep_A_id}_p_target",
            probe_type="target_key",
            question=f"What is the current value assigned to target key '{k_target}'?",
            options=target_opt_dict,
            correct_letter_congruent=corr_let_target_A,
            target_value_A=val_target_A,
            target_value_B=val_target_B,
            control_value=val_control,
            metadata={"domain": "target_key", "k_target": k_target},
        )
        probe_target_B = InterventionProbe(
            probe_id=f"{ep_B_id}_p_target",
            probe_type="target_key",
            question=f"What is the current value assigned to target key '{k_target}'?",
            options=target_opt_dict,
            correct_letter_congruent=corr_let_target_B,
            target_value_A=val_target_A,
            target_value_B=val_target_B,
            control_value=val_control,
            metadata={"domain": "target_key", "k_target": k_target},
        )

        # Option pool for Control Key Probe
        control_options_raw = [val_control, val_target_A, val_target_B, val_foil1]
        self.rng.shuffle(control_options_raw)
        control_opt_dict = {l: v for l, v in zip(letters, control_options_raw)}
        corr_let_control = [l for l, v in control_opt_dict.items() if v == val_control][0]

        probe_control_A = InterventionProbe(
            probe_id=f"{ep_A_id}_p_control",
            probe_type="control_key",
            question=f"What is the current value assigned to control key '{k_control}'?",
            options=control_opt_dict,
            correct_letter_congruent=corr_let_control,
            target_value_A=val_target_A,
            target_value_B=val_target_B,
            control_value=val_control,
            metadata={"domain": "control_key", "k_control": k_control},
        )
        probe_control_B = InterventionProbe(
            probe_id=f"{ep_B_id}_p_control",
            probe_type="control_key",
            question=f"What is the current value assigned to control key '{k_control}'?",
            options=control_opt_dict,
            correct_letter_congruent=corr_let_control,
            target_value_A=val_target_A,
            target_value_B=val_target_B,
            control_value=val_control,
            metadata={"domain": "control_key", "k_control": k_control},
        )

        # Option pool for Goal Status Probe
        goal_ans_A = f"Goal '{gid_beta}': status active"
        goal_ans_B = f"Goal '{gid_beta}': status suspended"
        goal_foils = [
            f"Goal '{gid_alpha}': status active",
            f"Goal '{gid_beta}': status completed",
        ]
        goal_options_raw = [goal_ans_A, goal_ans_B] + goal_foils
        self.rng.shuffle(goal_options_raw)
        goal_opt_dict = {l: v for l, v in zip(letters, goal_options_raw)}
        corr_let_goal_A = [l for l, v in goal_opt_dict.items() if v == goal_ans_A][0]
        corr_let_goal_B = [l for l, v in goal_opt_dict.items() if v == goal_ans_B][0]

        probe_goal_A = InterventionProbe(
            probe_id=f"{ep_A_id}_p_goal",
            probe_type="goal_status",
            question=f"What is the current status of secondary goal '{gid_beta}'?",
            options=goal_opt_dict,
            correct_letter_congruent=corr_let_goal_A,
            target_value_A=goal_ans_A,
            target_value_B=goal_ans_B,
            control_value=desc_beta,
            metadata={"domain": "goal_status", "gid_beta": gid_beta},
        )
        probe_goal_B = InterventionProbe(
            probe_id=f"{ep_B_id}_p_goal",
            probe_type="goal_status",
            question=f"What is the current status of secondary goal '{gid_beta}'?",
            options=goal_opt_dict,
            correct_letter_congruent=corr_let_goal_B,
            target_value_A=goal_ans_A,
            target_value_B=goal_ans_B,
            control_value=desc_beta,
            metadata={"domain": "goal_status", "gid_beta": gid_beta},
        )

        return MatchedTwinEpisodePair(
            pair_id=pair_id,
            twin_index=twin_idx,
            episode_A_id=ep_A_id,
            episode_B_id=ep_B_id,
            prefix_events_A=events_A,
            prefix_events_B=events_B,
            oracle_state_A=oracle_state_A,
            oracle_state_B=oracle_state_B,
            probes_A=[probe_target_A, probe_control_A, probe_goal_A],
            probes_B=[probe_target_B, probe_control_B, probe_goal_B],
            k_target=k_target,
            k_control=k_control,
            val_target_A=val_target_A,
            val_target_B=val_target_B,
            val_control=val_control,
            gid_beta=gid_beta,
            status_beta_A=status_beta_A,
            status_beta_B=status_beta_B,
            metadata={
                "k_distractor": k_distractor,
                "val_foil1": val_foil1,
                "val_foil2": val_foil2,
            },
        )

    def generate_clone_reconvergence_spec(
        self,
        twin_idx: int,
        seed: Optional[int] = None,
    ) -> CloneReconvergenceSpec:
        """Generate specification for clone -> fork -> cross-swap -> reconvergence testbed."""
        if seed is not None:
            self.rng = random.Random(seed + twin_idx * 2017)

        spec_id = f"clone_s08_{twin_idx:03d}"
        k_target = self._make_key("target")
        k_control = self._make_key("control")

        val_common = self._make_val("common")
        val_fork_A = self._make_val("forkA")
        val_fork_B = self._make_val("forkB")
        val_reconverge = self._make_val("reconv")
        val_foil = self._make_val("foil")

        # Step 0: Common Prefix
        prefix_events = [
            MemoryEvent(
                event_id=f"{spec_id}_ev00",
                step_index=0,
                source=EventSource.ENVIRONMENT,
                event_type="observation",
                content=f"Baseline initialization registers {k_control} = {val_common} and {k_target} = {val_common}.",
                key_bindings={k_control: val_common, k_target: val_common},
                metadata={"domain": "common_prefix"},
            )
        ]
        oracle_prefix = StructuredSelfState(
            working_memory={k_control: val_common, k_target: val_common},
            goals=[],
            source_ledger={k_control: "environment", k_target: "environment"},
            unresolved_items=[],
            derived_inferences={},
            last_updated_step=0,
        )

        # Step 1: Fork Branch A & Fork Branch B
        fork_A = [
            MemoryEvent(
                event_id=f"{spec_id}_ev01_A",
                step_index=1,
                source=EventSource.ENVIRONMENT,
                event_type="observation",
                content=f"Branch A routing updates {k_target} = {val_fork_A}.",
                key_bindings={k_target: val_fork_A},
                metadata={"domain": "fork_A"},
            )
        ]
        oracle_fork_A = StructuredSelfState(
            working_memory={k_control: val_common, k_target: val_fork_A},
            goals=[],
            source_ledger={k_control: "environment", k_target: "environment"},
            unresolved_items=[],
            derived_inferences={},
            last_updated_step=1,
        )

        fork_B = [
            MemoryEvent(
                event_id=f"{spec_id}_ev01_B",
                step_index=1,
                source=EventSource.ENVIRONMENT,
                event_type="observation",
                content=f"Branch B routing updates {k_target} = {val_fork_B}.",
                key_bindings={k_target: val_fork_B},
                metadata={"domain": "fork_B"},
            )
        ]
        oracle_fork_B = StructuredSelfState(
            working_memory={k_control: val_common, k_target: val_fork_B},
            goals=[],
            source_ledger={k_control: "environment", k_target: "environment"},
            unresolved_items=[],
            derived_inferences={},
            last_updated_step=1,
        )

        # Step 2: Synchronizing Event (Reconvergence)
        reconvergence_events = [
            MemoryEvent(
                event_id=f"{spec_id}_ev02_sync",
                step_index=2,
                source=EventSource.ENVIRONMENT,
                event_type="observation",
                content=f"Global synchronizing directive overwrites {k_target} = {val_reconverge}.",
                key_bindings={k_target: val_reconverge},
                metadata={"domain": "reconvergence_sync"},
            )
        ]
        oracle_reconverged = StructuredSelfState(
            working_memory={k_control: val_common, k_target: val_reconverge},
            goals=[],
            source_ledger={k_control: "environment", k_target: "environment"},
            unresolved_items=[],
            derived_inferences={},
            last_updated_step=2,
        )

        letters = ["A", "B", "C", "D"]
        fork_opts_raw = [val_fork_A, val_fork_B, val_reconverge, val_foil]
        self.rng.shuffle(fork_opts_raw)
        fork_opt_dict = {l: v for l, v in zip(letters, fork_opts_raw)}
        corr_let_fork_A = [l for l, v in fork_opt_dict.items() if v == val_fork_A][0]

        reconv_opts_raw = [val_reconverge, val_fork_A, val_fork_B, val_foil]
        self.rng.shuffle(reconv_opts_raw)
        reconv_opt_dict = {l: v for l, v in zip(letters, reconv_opts_raw)}
        corr_let_reconv = [l for l, v in reconv_opt_dict.items() if v == val_reconverge][0]

        probe_fork = InterventionProbe(
            probe_id=f"{spec_id}_p_fork",
            probe_type="target_key",
            question=f"Following branch routing, what is the current value of '{k_target}'?",
            options=fork_opt_dict,
            correct_letter_congruent=corr_let_fork_A,
            target_value_A=val_fork_A,
            target_value_B=val_fork_B,
            control_value=val_common,
            metadata={"domain": "fork_target", "k_target": k_target},
        )

        probe_reconverged = InterventionProbe(
            probe_id=f"{spec_id}_p_reconv",
            probe_type="reconvergence_target",
            question=f"Following global synchronization, what is the current value of '{k_target}'?",
            options=reconv_opt_dict,
            correct_letter_congruent=corr_let_reconv,
            target_value_A=val_reconverge,
            target_value_B=val_reconverge,
            control_value=val_common,
            metadata={"domain": "reconvergence_target", "k_target": k_target},
        )

        return CloneReconvergenceSpec(
            spec_id=spec_id,
            twin_index=twin_idx,
            prefix_events_common=prefix_events,
            fork_events_A=fork_A,
            fork_events_B=fork_B,
            reconvergence_events=reconvergence_events,
            oracle_prefix_state=oracle_prefix,
            oracle_fork_state_A=oracle_fork_A,
            oracle_fork_state_B=oracle_fork_B,
            oracle_reconverged_state=oracle_reconverged,
            probes_fork=[probe_fork],
            probes_reconverged=[probe_reconverged],
            k_target=k_target,
            k_control=k_control,
            val_common=val_common,
            val_fork_A=val_fork_A,
            val_fork_B=val_fork_B,
            val_reconverge=val_reconverge,
            metadata={},
        )
