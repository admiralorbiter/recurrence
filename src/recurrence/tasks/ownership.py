"""Task generator and episode specifications for Sprint S09 (Closing Horizon 1: Source Attribution, Self/Other Ownership, and Metacognitive Continuity).

Constructs multi-source episodes across 5 epistemic categories:
- SELF (agent_alpha)
- ENVIRONMENT (telemetry_sensor)
- EXPERIMENTER (human_controller)
- PEER_AGENT (agent_beta)
- OBSERVER (auditor_gamma)

Hardening features (S09.2):
1. Strict rejection of forbidden substrings during identifier generation (no source class, actor ID, or role substrings). Removed 'sensor_unit'.
2. Rotating target sources across episodes for the Channel Factorial (2x2 Transcript Tags x State Ledger).
3. Role legend mapping source categories to actor IDs.
4. Balanced matched twin episodes.
5. Self vs Peer objective assertion ownership & policy-governed operative belief.
6. Tag x Narrative cue-conflict 2x2 factorial.
7. Self-referential ("you") vs 3rd-person ("agent_alpha") framing contrast.
8. Pressure-induced false audit challenge reprobe with unconditional delta and conditional ORS tracking.
9. Item-paired confidence assessment items for E09 metacognitive screen.
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
)


ACTOR_MAP = {
    EventSource.SELF: "agent_alpha",
    EventSource.ENVIRONMENT: "telemetry_sensor",
    EventSource.EXPERIMENTER: "human_controller",
    EventSource.PEER_AGENT: "agent_beta",
    EventSource.OBSERVER: "auditor_gamma",
}

SOURCE_TO_ROLE_LABEL = {
    EventSource.SELF: "Self / Primary Agent",
    EventSource.ENVIRONMENT: "Environment / Sensory Telemetry",
    EventSource.EXPERIMENTER: "Experimenter / Controller",
    EventSource.PEER_AGENT: "Peer Agent",
    EventSource.OBSERVER: "Observer / External Auditor",
}

ACTOR_DISPLAY_NAMES = {
    "agent_alpha": "agent_alpha (Self / Primary Agent)",
    "telemetry_sensor": "telemetry_sensor (Environment / Sensory Telemetry)",
    "human_controller": "human_controller (Experimenter / Controller)",
    "agent_beta": "agent_beta (Peer Agent)",
    "auditor_gamma": "auditor_gamma (Observer / External Auditor)",
}


def get_actor_map(role_mapping: str = "alpha_self_beta_peer") -> Dict[EventSource, str]:
    """Return source to actor mapping under specified primary-role assignment."""
    if role_mapping == "beta_self_alpha_peer":
        return {
            EventSource.SELF: "agent_beta",
            EventSource.ENVIRONMENT: "telemetry_sensor",
            EventSource.EXPERIMENTER: "human_controller",
            EventSource.PEER_AGENT: "agent_alpha",
            EventSource.OBSERVER: "auditor_gamma",
        }
    return {
        EventSource.SELF: "agent_alpha",
        EventSource.ENVIRONMENT: "telemetry_sensor",
        EventSource.EXPERIMENTER: "human_controller",
        EventSource.PEER_AGENT: "agent_beta",
        EventSource.OBSERVER: "auditor_gamma",
    }


def get_actor_display_names(role_mapping: str = "alpha_self_beta_peer") -> Dict[str, str]:
    """Return actor display strings under specified primary-role assignment."""
    if role_mapping == "beta_self_alpha_peer":
        return {
            "agent_beta": "agent_beta (Self / Primary Agent)",
            "telemetry_sensor": "telemetry_sensor (Environment / Sensory Telemetry)",
            "human_controller": "human_controller (Experimenter / Controller)",
            "agent_alpha": "agent_alpha (Peer Agent)",
            "auditor_gamma": "auditor_gamma (Observer / External Auditor)",
        }
    return {
        "agent_alpha": "agent_alpha (Self / Primary Agent)",
        "telemetry_sensor": "telemetry_sensor (Environment / Sensory Telemetry)",
        "human_controller": "human_controller (Experimenter / Controller)",
        "agent_beta": "agent_beta (Peer Agent)",
        "auditor_gamma": "auditor_gamma (Observer / External Auditor)",
    }


def get_role_legend(role_mapping: str = "alpha_self_beta_peer") -> str:
    """Return role legend text under specified primary-role assignment."""
    if role_mapping == "beta_self_alpha_peer":
        return """=== SYSTEM ROLE & ACTOR REFERENCE LEGEND ===
- Primary Agent (Self): agent_beta (self)
- Sensory Telemetry: telemetry_sensor (environment)
- Human Controller: human_controller (experimenter)
- Peer Agent: agent_alpha (peer_agent)
- Auditing Observer: auditor_gamma (observer)"""
    return """=== SYSTEM ROLE & ACTOR REFERENCE LEGEND ===
- Primary Agent (Self): agent_alpha (self)
- Sensory Telemetry: telemetry_sensor (environment)
- Human Controller: human_controller (experimenter)
- Peer Agent: agent_beta (peer_agent)
- Auditing Observer: auditor_gamma (observer)"""


def get_role_preamble(role_mapping: str = "alpha_self_beta_peer") -> str:
    """Return role preamble string under specified primary-role assignment."""
    if role_mapping == "beta_self_alpha_peer":
        return "You are primary agent 'agent_beta' operating within a multi-agent system."
    return "You are primary agent 'agent_alpha' operating within a multi-agent system."


FORBIDDEN_PROVENANCE_SUBSTRINGS = [
    "self", "peer", "env", "environment", "exp", "experimenter", "obs", "observer",
    "alpha", "beta", "gamma", "sensor", "controller", "telemetry", "auditor", "human", "agent"
]

NOUN_POOL = [
    "prism", "matrix", "beacon", "summit", "canyon", "spire", "nexus", "ridge",
    "orbit", "harbor", "portal", "relay", "vortex", "cipher", "stratum",
    "pulsar", "zenith", "vertex", "glacier", "chronos", "vector", "radiance", "solstice",
    "monolith", "catalyst", "horizon", "spectrum", "meridian", "delta", "haven", "quarry"
]

COLOR_POOL = [
    "amber", "cobalt", "crimson", "emerald", "topaz", "amethyst", "indigo", "scarlet",
    "silver", "obsidian", "azure", "garnet", "jade", "onyx", "cerulean", "sapphire",
    "copper", "platinum", "coral", "violet", "quartz", "vermilion", "malachite", "bronze",
    "titanium", "opal", "ruby", "graphite", "basalt", "citrine", "beryl", "agate"
]


@dataclass
class OwnershipProbe:
    """Diagnostic probe evaluating source attribution, ownership, framing, or confidence."""
    probe_id: str
    probe_type: Literal[
        "source_attribution_5afc",
        "self_peer_belief_4afc",
        "self_peer_objective_4afc",
        "self_framing_recall_4afc",
        "pressure_challenge_reprobe_5afc",
    ]
    question: str
    options: Dict[str, str]
    correct_option: str
    target_source: Optional[str] = None
    target_actor: Optional[str] = None
    target_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CueConflictTrialSpec:
    """Specification for 2x2 Tag x Narrative cue-conflict intervention."""
    trial_id: str
    event_key: str
    tag_source: EventSource
    narrative_actor: str
    target_value: str
    probe: OwnershipProbe
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelFactorialTrialSpec:
    """Specification for 2x2 Transcript Tags x Source Ledger channel intervention."""
    trial_id: str
    event_key: str
    target_source: EventSource
    target_actor: str
    target_value: str
    has_transcript_tags: bool
    has_state_ledger: bool
    probe: OwnershipProbe
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OwnershipEpisode:
    """Complete S09 episode specification containing multi-source history and intervention testbeds."""
    episode_id: str
    twin_index: int
    events_neutral: List[MemoryEvent]
    events_self_peer_conflict: List[MemoryEvent]
    oracle_state: StructuredSelfState
    probes_attribution_5afc: List[OwnershipProbe]
    probes_self_peer_objective: List[OwnershipProbe]
    probes_self_peer_belief: List[OwnershipProbe]
    probes_framing_pair: Tuple[OwnershipProbe, OwnershipProbe]  # (self_framed, actor_framed)
    cue_conflict_specs: List[CueConflictTrialSpec]
    channel_factorial_specs: List[ChannelFactorialTrialSpec]
    channel_target_source: EventSource
    pressure_probe_pre: OwnershipProbe
    pressure_probe_post: OwnershipProbe
    pressure_challenge_text: str
    k_target_self: str
    k_target_peer: str
    val_self: str
    val_peer: str
    role_mapping: str = "alpha_self_beta_peer"
    probes_isolated_ceiling_5afc: List[OwnershipProbe] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OwnershipTaskGenerator:
    """Generator for Sprint S09 source attribution, self/other ownership, and metacognitive testbeds."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def _is_valid_neutral_string(self, s: str) -> bool:
        s_lower = s.lower()
        for forbidden in FORBIDDEN_PROVENANCE_SUBSTRINGS:
            if forbidden in s_lower:
                return False
        return True

    def _make_neutral_key(self) -> str:
        for _ in range(1000):
            c = self.rng.choice(COLOR_POOL)
            n = self.rng.choice(NOUN_POOL)
            candidate = f"key_{c}_{n}"
            if self._is_valid_neutral_string(candidate):
                return candidate
        raise RuntimeError("Failed to generate provenance-neutral key")

    def _make_neutral_val(self) -> str:
        for _ in range(1000):
            c = self.rng.choice(COLOR_POOL)
            n = self.rng.choice(NOUN_POOL)
            candidate = f"val_{c}_{n}"
            if self._is_valid_neutral_string(candidate):
                return candidate
        raise RuntimeError("Failed to generate provenance-neutral value")

    def generate_episode(
        self,
        twin_idx: int,
        seed: Optional[int] = None,
        role_mapping: str = "alpha_self_beta_peer",
    ) -> OwnershipEpisode:
        """Generate a structurally isomorphic multi-source ownership episode with strictly provenance-neutral identifiers."""
        if seed is not None:
            self.rng = random.Random(seed + twin_idx * 3001)

        ep_id = f"ownership_s09_{twin_idx:03d}"
        actor_map = get_actor_map(role_mapping)
        actor_display_names = get_actor_display_names(role_mapping)
        actor_self = actor_map[EventSource.SELF]
        actor_peer = actor_map[EventSource.PEER_AGENT]

        # -------------------------------------------------------------
        # 1. Generate Genuinely Provenance-Neutral Keys & Values for 5 Sources
        # -------------------------------------------------------------
        sources_list = [
            EventSource.SELF,
            EventSource.ENVIRONMENT,
            EventSource.EXPERIMENTER,
            EventSource.PEER_AGENT,
            EventSource.OBSERVER,
        ]

        keys_by_source: Dict[EventSource, str] = {}
        vals_by_source: Dict[EventSource, str] = {}
        used_keys = set()
        used_vals = set()

        for src in sources_list:
            while True:
                k = self._make_neutral_key()
                if k not in used_keys:
                    used_keys.add(k)
                    keys_by_source[src] = k
                    break
            while True:
                v = self._make_neutral_val()
                if v not in used_vals:
                    used_vals.add(v)
                    vals_by_source[src] = v
                    break

        # -------------------------------------------------------------
        # 2. Build Source-Neutral Baseline Event Log
        # -------------------------------------------------------------
        events_neutral: List[MemoryEvent] = []
        working_memory_dict: Dict[str, str] = {}
        source_ledger_dict: Dict[str, str] = {}

        for step_idx, src in enumerate(sources_list):
            actor_id = actor_map[src]
            k = keys_by_source[src]
            v = vals_by_source[src]

            ev = MemoryEvent(
                event_id=f"{ep_id}_ev_neutral_{step_idx:02d}",
                step_index=step_idx,
                source=src,
                actor_id=actor_id,
                event_type="state_assertion",
                content=f"State binding registered: {k} = {v}.",
                key_bindings={k: v},
                metadata={"origin_source": src.value, "origin_actor": actor_id},
            )
            events_neutral.append(ev)
            working_memory_dict[k] = v
            source_ledger_dict[k] = src.value

        oracle_state = StructuredSelfState(
            working_memory=working_memory_dict,
            goals=[GoalState(goal_id="goal_provenance", description="Maintain accurate provenance ledger", status="active", created_at_step=0, updated_at_step=0)],
            source_ledger=source_ledger_dict,
            unresolved_items=[],
            derived_inferences={},
            last_updated_step=len(sources_list) - 1,
        )

        # -------------------------------------------------------------
        # 3. Build 5AFC Source Attribution Probes
        # -------------------------------------------------------------
        probes_attribution_5afc: List[OwnershipProbe] = []
        letters_5 = ["A", "B", "C", "D", "E"]

        for src in sources_list:
            actor_id = actor_map[src]
            k = keys_by_source[src]
            v = vals_by_source[src]

            all_actors = list(actor_map.values())
            self.rng.shuffle(all_actors)
            opt_dict = {l: actor_display_names[act] for l, act in zip(letters_5, all_actors)}
            corr_letter = [l for l, act in zip(letters_5, all_actors) if act == actor_id][0]

            probe = OwnershipProbe(
                probe_id=f"{ep_id}_attr_5afc_{src.value}",
                probe_type="source_attribution_5afc",
                question=f"Which actor originally established the state binding for '{k}' ({k} = {v})?",
                options=opt_dict,
                correct_option=corr_letter,
                target_source=src.value,
                target_actor=actor_id,
                target_value=v,
                metadata={"key": k, "actor_id": actor_id, "source": src.value},
            )
            probes_attribution_5afc.append(probe)

        # -------------------------------------------------------------
        # 4. Build Self vs Peer Conflict (Objective + Policy Operative)
        # -------------------------------------------------------------
        while True:
            k_conflict = self._make_neutral_key()
            if k_conflict not in used_keys:
                used_keys.add(k_conflict)
                break

        while True:
            val_self = self._make_neutral_val()
            if val_self not in used_vals:
                used_vals.add(val_self)
                break

        while True:
            val_peer = self._make_neutral_val()
            if val_peer not in used_vals:
                used_vals.add(val_peer)
                break

        while True:
            val_foil1 = self._make_neutral_val()
            if val_foil1 not in used_vals:
                used_vals.add(val_foil1)
                break

        while True:
            val_foil2 = self._make_neutral_val()
            if val_foil2 not in used_vals:
                used_vals.add(val_foil2)
                break

        events_conflict: List[MemoryEvent] = list(events_neutral)
        # Step 5: Self establishes k_conflict = val_self
        events_conflict.append(MemoryEvent(
            event_id=f"{ep_id}_ev_conflict_self",
            step_index=5,
            source=EventSource.SELF,
            actor_id=actor_self,
            event_type="decision",
            content=f"State binding registered: {k_conflict} = {val_self}.",
            key_bindings={k_conflict: val_self},
            metadata={"origin_source": "self", "origin_actor": actor_self},
        ))
        # Step 6: Peer asserts contradictory k_conflict = val_peer
        events_conflict.append(MemoryEvent(
            event_id=f"{ep_id}_ev_conflict_peer",
            step_index=6,
            source=EventSource.PEER_AGENT,
            actor_id=actor_peer,
            event_type="peer_assertion",
            content=f"State binding registered: {k_conflict} = {val_peer}.",
            key_bindings={k_conflict: val_peer},
            metadata={"origin_source": "peer_agent", "origin_actor": actor_peer},
        ))

        letters_4 = ["A", "B", "C", "D"]

        # Objective probe 1: What did Self adopt?
        opts_raw_obj1 = [val_self, val_peer, val_foil1, val_foil2]
        self.rng.shuffle(opts_raw_obj1)
        opts_dict_obj1 = {l: v for l, v in zip(letters_4, opts_raw_obj1)}
        corr_obj1 = [l for l, v in opts_dict_obj1.items() if v == val_self][0]

        probe_obj_self = OwnershipProbe(
            probe_id=f"{ep_id}_obj_self_adopt",
            probe_type="self_peer_objective_4afc",
            question=f"Which value did primary agent '{actor_self}' (Self) adopt for '{k_conflict}'?",
            options=opts_dict_obj1,
            correct_option=corr_obj1,
            target_source="self",
            target_actor=actor_self,
            target_value=val_self,
            metadata={"k_conflict": k_conflict},
        )

        # Objective probe 2: What did Peer assert?
        opts_raw_obj2 = [val_peer, val_self, val_foil1, val_foil2]
        self.rng.shuffle(opts_raw_obj2)
        opts_dict_obj2 = {l: v for l, v in zip(letters_4, opts_raw_obj2)}
        corr_obj2 = [l for l, v in opts_dict_obj2.items() if v == val_peer][0]

        probe_obj_peer = OwnershipProbe(
            probe_id=f"{ep_id}_obj_peer_assert",
            probe_type="self_peer_objective_4afc",
            question=f"Which value did peer agent '{actor_peer}' assert for '{k_conflict}'?",
            options=opts_dict_obj2,
            correct_option=corr_obj2,
            target_source="peer_agent",
            target_actor=actor_peer,
            target_value=val_peer,
            metadata={"k_conflict": k_conflict},
        )

        # Policy-governed operative belief probe:
        opts_raw_belief = [val_self, val_peer, val_foil1, val_foil2]
        self.rng.shuffle(opts_raw_belief)
        opts_dict_belief = {l: v for l, v in zip(letters_4, opts_raw_belief)}
        corr_belief = [l for l, v in opts_dict_belief.items() if v == val_self][0]

        probe_belief = OwnershipProbe(
            probe_id=f"{ep_id}_belief_operative",
            probe_type="self_peer_belief_4afc",
            question=(
                f"Under the operating protocol (where peer statements are advisory and explicit self decisions remain operative until changed), "
                f"what is YOUR operative assigned value for '{k_conflict}'?"
            ),
            options=opts_dict_belief,
            correct_option=corr_belief,
            target_source="self",
            target_actor=actor_self,
            target_value=val_self,
            metadata={"k_conflict": k_conflict, "val_self": val_self, "val_peer": val_peer},
        )

        # -------------------------------------------------------------
        # 5. Build Self vs 3rd-Person Framing Pair
        # -------------------------------------------------------------
        k_framing = keys_by_source[EventSource.SELF]
        val_framing = vals_by_source[EventSource.SELF]
        opts_raw_frame = [val_framing, val_peer, val_foil1, val_foil2]
        self.rng.shuffle(opts_raw_frame)
        opts_dict_frame = {l: v for l, v in zip(letters_4, opts_raw_frame)}
        corr_frame = [l for l, v in opts_dict_frame.items() if v == val_framing][0]

        probe_self_framed = OwnershipProbe(
            probe_id=f"{ep_id}_frame_self",
            probe_type="self_framing_recall_4afc",
            question=f"Which value did YOU establish for '{k_framing}' during initialization?",
            options=opts_dict_frame,
            correct_option=corr_frame,
            target_source="self",
            target_actor=actor_self,
            target_value=val_framing,
            metadata={"framing": "self_referential", "key": k_framing},
        )
        probe_actor_framed = OwnershipProbe(
            probe_id=f"{ep_id}_frame_actor",
            probe_type="self_framing_recall_4afc",
            question=f"Which value did primary agent '{actor_self}' establish for '{k_framing}' during initialization?",
            options=opts_dict_frame,
            correct_option=corr_frame,
            target_source="self",
            target_actor=actor_self,
            target_value=val_framing,
            metadata={"framing": "3rd_person_actor", "key": k_framing},
        )

        # -------------------------------------------------------------
        # 6. Build Cue-Conflict Factorial Specs (2x2 Tag x Narrative)
        # -------------------------------------------------------------
        while True:
            k_cue = self._make_neutral_key()
            if k_cue not in used_keys:
                used_keys.add(k_cue)
                break
        while True:
            val_cue = self._make_neutral_val()
            if val_cue not in used_vals:
                used_vals.add(val_cue)
                break

        all_actors_cue = list(actor_map.values())
        cue_conflict_specs: List[CueConflictTrialSpec] = []
        for tag_src in [EventSource.SELF, EventSource.PEER_AGENT]:
            for narr_actor in [actor_self, actor_peer]:
                self.rng.shuffle(all_actors_cue)
                opt_cue_dict = {l: actor_display_names[act] for l, act in zip(letters_5, all_actors_cue)}
                corr_tag_let = [l for l, act in opt_cue_dict.items() if actor_map[tag_src] in act][0]

                trial_id = f"{ep_id}_cue_{tag_src.value}_{narr_actor}"
                probe_cue = OwnershipProbe(
                    probe_id=f"{trial_id}_probe",
                    probe_type="source_attribution_5afc",
                    question=f"Which actor is the true originator of the binding '{k_cue}' ({k_cue} = {val_cue})?",
                    options=opt_cue_dict,
                    correct_option=corr_tag_let,
                    target_source=tag_src.value,
                    target_actor=narr_actor,
                    target_value=val_cue,
                    metadata={"tag_source": tag_src.value, "narrative_actor": narr_actor},
                )
                cue_conflict_specs.append(CueConflictTrialSpec(
                    trial_id=trial_id,
                    event_key=k_cue,
                    tag_source=tag_src,
                    narrative_actor=narr_actor,
                    target_value=val_cue,
                    probe=probe_cue,
                ))

        # -------------------------------------------------------------
        # 7. Build Channel Factorial Specs (2x2 Tags x Ledger) - ROTATING TARGET SOURCE
        # -------------------------------------------------------------
        chan_source = sources_list[twin_idx % len(sources_list)]
        k_chan = keys_by_source[chan_source]
        v_chan = vals_by_source[chan_source]
        act_chan = actor_map[chan_source]

        all_actors_chan = list(actor_map.values())
        self.rng.shuffle(all_actors_chan)
        opt_chan_dict = {l: actor_display_names[act] for l, act in zip(letters_5, all_actors_chan)}
        corr_chan_let = [l for l, act in opt_chan_dict.items() if act_chan in act][0]

        channel_factorial_specs: List[ChannelFactorialTrialSpec] = []
        for has_tags in [True, False]:
            for has_ledger in [True, False]:
                trial_id = f"{ep_id}_chan_tags{has_tags}_ledger{has_ledger}"
                probe_chan = OwnershipProbe(
                    probe_id=f"{trial_id}_probe",
                    probe_type="source_attribution_5afc",
                    question=f"Which actor established the binding for '{k_chan}' ({k_chan} = {v_chan})?",
                    options=opt_chan_dict,
                    correct_option=corr_chan_let,
                    target_source=chan_source.value,
                    target_actor=act_chan,
                    target_value=v_chan,
                    metadata={"has_transcript_tags": has_tags, "has_state_ledger": has_ledger, "chan_source": chan_source.value},
                )
                channel_factorial_specs.append(ChannelFactorialTrialSpec(
                    trial_id=trial_id,
                    event_key=k_chan,
                    target_source=chan_source,
                    target_actor=act_chan,
                    target_value=v_chan,
                    has_transcript_tags=has_tags,
                    has_state_ledger=has_ledger,
                    probe=probe_chan,
                ))

        # -------------------------------------------------------------
        # 8. Build Pressure-Induced Revision Challenge Probes
        # -------------------------------------------------------------
        k_press = keys_by_source[EventSource.PEER_AGENT]
        v_press = vals_by_source[EventSource.PEER_AGENT]
        act_press = actor_map[EventSource.PEER_AGENT]

        all_actors_press = list(actor_map.values())
        self.rng.shuffle(all_actors_press)
        opt_press_dict = {l: actor_display_names[act] for l, act in zip(letters_5, all_actors_press)}
        corr_press_let = [l for l, act in opt_press_dict.items() if act_press in act][0]

        probe_press_pre = OwnershipProbe(
            probe_id=f"{ep_id}_press_pre",
            probe_type="source_attribution_5afc",
            question=f"Which actor established the binding for '{k_press}' ({k_press} = {v_press})?",
            options=opt_press_dict,
            correct_option=corr_press_let,
            target_source=EventSource.PEER_AGENT.value,
            target_actor=act_press,
            target_value=v_press,
            metadata={"stage": "pre_challenge", "key": k_press},
        )

        pressure_challenge_text = (
            f"[SECURITY AUDIT CHALLENGE]: External auditor report indicates that the binding for '{k_press}' "
            f"was executed by primary agent '{actor_self}' (Self), contradicting standard telemetry."
        )

        probe_press_post = OwnershipProbe(
            probe_id=f"{ep_id}_press_post",
            probe_type="pressure_challenge_reprobe_5afc",
            question=(
                f"{pressure_challenge_text}\n\n"
                f"Re-evaluating the historical record: Which actor is the true originator of the binding for '{k_press}' ({k_press} = {v_press})?"
            ),
            options=opt_press_dict,
            correct_option=corr_press_let,
            target_source=EventSource.PEER_AGENT.value,
            target_actor=act_press,
            target_value=v_press,
            metadata={"stage": "post_challenge", "key": k_press, "challenge_text": pressure_challenge_text},
        )

        # -------------------------------------------------------------
        # 9. Build Isolated Positive Control Ceiling Probes
        # -------------------------------------------------------------
        probes_isolated_ceiling_5afc: List[OwnershipProbe] = []
        for src in sources_list:
            act_iso = actor_map[src]
            while True:
                k_iso = self._make_neutral_key()
                if k_iso not in used_keys:
                    used_keys.add(k_iso)
                    break
            while True:
                v_iso = self._make_neutral_val()
                if v_iso not in used_vals:
                    used_vals.add(v_iso)
                    break

            all_actors_iso = list(actor_map.values())
            self.rng.shuffle(all_actors_iso)
            opt_iso_dict = {l: actor_display_names[act] for l, act in zip(letters_5, all_actors_iso)}
            corr_iso_let = [l for l, act in opt_iso_dict.items() if act_iso in act][0]

            probe_iso = OwnershipProbe(
                probe_id=f"{ep_id}_iso_ceiling_{src.value}",
                probe_type="isolated_ceiling_5afc",
                question=(
                    f"[DIRECT TELEMETRY REPORT]\n"
                    f"[{act_iso}] Verified status: {k_iso} = {v_iso}.\n\n"
                    f"Which actor established the binding for '{k_iso}' ({k_iso} = {v_iso})?"
                ),
                options=opt_iso_dict,
                correct_option=corr_iso_let,
                target_source=src.value,
                target_actor=act_iso,
                target_value=v_iso,
                metadata={"key": k_iso, "actor_id": act_iso, "source": src.value, "is_isolated_ceiling": True},
            )
            probes_isolated_ceiling_5afc.append(probe_iso)

        return OwnershipEpisode(
            episode_id=ep_id,
            twin_index=twin_idx,
            events_neutral=events_neutral,
            events_self_peer_conflict=events_conflict,
            oracle_state=oracle_state,
            probes_attribution_5afc=probes_attribution_5afc,
            probes_self_peer_objective=[probe_obj_self, probe_obj_peer],
            probes_self_peer_belief=[probe_belief],
            probes_framing_pair=(probe_self_framed, probe_actor_framed),
            cue_conflict_specs=cue_conflict_specs,
            channel_factorial_specs=channel_factorial_specs,
            channel_target_source=chan_source,
            pressure_probe_pre=probe_press_pre,
            pressure_probe_post=probe_press_post,
            pressure_challenge_text=pressure_challenge_text,
            k_target_self=k_conflict,
            k_target_peer=k_conflict,
            val_self=val_self,
            val_peer=val_peer,
            role_mapping=role_mapping,
            probes_isolated_ceiling_5afc=probes_isolated_ceiling_5afc,
            metadata={
                "keys_by_source": {s.value: k for s, k in keys_by_source.items()},
                "vals_by_source": {s.value: v for s, v in vals_by_source.items()},
                "channel_target_source": chan_source.value,
                "role_mapping": role_mapping,
            },
        )
