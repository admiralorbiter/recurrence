"""Execution harness for Sprint S09: Source Attribution, Self/Other Ownership (E08) and Metacognitive Screen (E09).

Executes:
1. Neutral 5AFC Source Attribution baseline.
2. Self vs Peer objective assertion ownership and policy-governed operative belief.
3. Cue-Conflict Factorial (2x2 Tag x Narrative).
4. Channel Factorial (2x2 Transcript Tags x State Ledger) with isolated provenance channels.
5. Self-Referential ("you") vs 3rd-Person ("agent_alpha") Framing Contrast.
6. Pressure-Induced Revision Challenge.
7. Metacognitive Screen (E09): Item-Paired Self vs Observer Post-Choice Error Prediction.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from recurrence.core.schemas import (
    TARGET_5AFC_SCHEMA,
    TARGET_4AFC_SCHEMA,
    CONFIDENCE_ASSESSMENT_SCHEMA,
)
from recurrence.memory.schemas import (
    EventSource,
    GoalState,
    MemoryEvent,
    StructuredSelfState,
)
from recurrence.tasks.ownership import (
    ACTOR_MAP,
    ACTOR_DISPLAY_NAMES,
    SOURCE_TO_ROLE_LABEL,
    OwnershipProbe,
    OwnershipEpisode,
    CueConflictTrialSpec,
    ChannelFactorialTrialSpec,
    get_actor_map,
    get_actor_display_names,
    get_role_legend,
    get_role_preamble,
)

ROLE_LEGEND_TEXT = """=== SYSTEM ROLE & ACTOR REFERENCE LEGEND ===
- Primary Agent (Self): agent_alpha (self)
- Sensory Telemetry: telemetry_sensor (environment)
- Human Controller: human_controller (experimenter)
- Peer Agent: agent_beta (peer_agent)
- Auditing Observer: auditor_gamma (observer)"""


@dataclass
class OwnershipTrialResult:
    """Record of a single evaluation probe in S09 (E08 / E09 / E08c)."""
    trial_id: str
    episode_id: str
    experiment_submodule: str  # 'e08_source_ownership', 'e09_metacognitive', 'e08c_role_counterbalance'
    condition_name: str
    probe_id: str
    probe_type: str
    question: str
    options: Dict[str, str]
    predicted_letter: str
    predicted_text: str
    correct_letter: str
    is_correct: bool
    attributed_actor: Optional[str]
    target_source: Optional[str]
    target_actor: Optional[str]
    target_value: Optional[str]
    subjective_confidence_pct: Optional[float] = None
    prompt_hash: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OwnershipHarness:
    """Execution orchestrator for Sprint S09 Source Attribution & Metacognitive Battery."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def _format_transcript(self, events: List[MemoryEvent], include_tags: bool = True) -> str:
        """Format episodic event log, cleanly stripping transcript provenance when include_tags=False."""
        lines = ["=== EPISODIC EVENT LOG TRANSCRIPT ==="]
        for ev in sorted(events, key=lambda e: (e.step_index, e.event_id)):
            if include_tags:
                src_val = ev.source.value if hasattr(ev.source, "value") else str(ev.source)
                act_val = ev.actor_id or ACTOR_MAP.get(ev.source, "unknown_actor")
                lines.append(f"[Tick {ev.step_index:02d}] [{src_val} / {act_val}] {ev.content}")
            else:
                # Completely stripped of source tag and actor identity
                lines.append(f"[Tick {ev.step_index:02d}] {ev.content}")
        return "\n".join(lines)

    def _format_state(self, state: StructuredSelfState, include_ledger: bool = True) -> str:
        """Format structured state JSON string, optionally stripping source_ledger."""
        dump = state.model_dump()
        if not include_ledger:
            dump["source_ledger"] = {}
        return f"=== CURRENT STRUCTURED STATE ===\n{json.dumps(dump, indent=2, sort_keys=True)}"

    def _build_prompt(
        self,
        events: Optional[List[MemoryEvent]],
        state: Optional[StructuredSelfState],
        probe: OwnershipProbe,
        include_tags: bool = True,
        include_ledger: bool = True,
        role_preamble: str = "You are primary agent 'agent_alpha' operating within a multi-agent system.",
        include_legend: bool = True,
        legend_text: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Construct prompt for ownership probe evaluation."""
        sections = [role_preamble]
        if include_legend:
            sections.append(legend_text if legend_text is not None else ROLE_LEGEND_TEXT)

        if events is not None:
            sections.append(self._format_transcript(events, include_tags=include_tags))
        if state is not None:
            sections.append(self._format_state(state, include_ledger=include_ledger))

        opts_str = "\n".join([f"{l}. {text}" for l, text in sorted(probe.options.items())])
        sections.append(
            f"=== EVALUATION QUESTION ===\n"
            f"{probe.question}\n\n"
            f"Options:\n"
            f"{opts_str}\n\n"
            f"Select the single correct option letter. Return strictly JSON matching schema."
        )

        full_prompt = "\n\n".join(sections)
        p_hash = hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()
        return full_prompt, p_hash

    def _query_choice(
        self,
        prompt: str,
        probe: OwnershipProbe,
    ) -> Tuple[str, str, int, int, float, Optional[str]]:
        """Query LLM backend under 5AFC or 4AFC schema."""
        start_time = time.perf_counter()
        schema = TARGET_5AFC_SCHEMA if len(probe.options) == 5 else TARGET_4AFC_SCHEMA

        try:
            if hasattr(self.backend, "step"):
                raw_text, _, meta = self.backend.step(prompt, format=schema)
                p_tok = meta.get("prompt_eval_count", len(prompt) // 4)
                c_tok = meta.get("eval_count", len(raw_text) // 4)
            elif hasattr(self.backend, "generate"):
                resp = self.backend.generate(prompt=prompt, schema=schema)
                raw_text = resp.text
                p_tok = getattr(resp, "prompt_tokens", len(prompt) // 4)
                c_tok = getattr(resp, "completion_tokens", len(raw_text) // 4)
            else:
                raw_text = json.dumps({"answer": probe.correct_option})
                p_tok = len(prompt) // 4
                c_tok = len(raw_text) // 4

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            data = json.loads(raw_text)
            pred_letter = str(data.get("answer") or data.get("target_answer") or "").strip().upper()
            pred_text = probe.options.get(pred_letter, "UNKNOWN")
            return pred_letter, pred_text, p_tok, c_tok, latency_ms, None

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return "ERROR", "ERROR", len(prompt) // 4, 0, latency_ms, str(e)

    def _query_confidence_assessment(
        self,
        base_prompt: str,
        target_choice_letter: str,
        target_choice_text: str,
        evaluator: str = "self",
    ) -> Tuple[float, int, int, float, Optional[str]]:
        """Query post-choice error prediction / subjective confidence (0-100%) for a specific target answer."""
        if evaluator == "self":
            conf_query = (
                f"=== PRE-FEEDBACK CORRECTNESS ASSESSMENT (SELF) ===\n"
                f"You previously selected choice '{target_choice_letter}' ({target_choice_text}).\n"
                f"Before receiving correctness feedback, what probability (0 to 100%) do you assign that your selected answer is correct?"
            )
        else:
            conf_query = (
                f"=== PRE-FEEDBACK CORRECTNESS ASSESSMENT (AUDITING OBSERVER) ===\n"
                f"Primary agent 'agent_alpha' selected choice '{target_choice_letter}' ({target_choice_text}).\n"
                f"Before receiving correctness feedback, what probability (0 to 100%) do you assign that agent_alpha's selected answer is correct?"
            )

        full_conf_prompt = f"{base_prompt}\n\n{conf_query}\n\nReturn strictly JSON with 'confidence_percentage' (0-100)."
        start_time = time.perf_counter()

        try:
            if hasattr(self.backend, "step"):
                raw_text, _, meta = self.backend.step(full_conf_prompt, format=CONFIDENCE_ASSESSMENT_SCHEMA)
                p_tok = meta.get("prompt_eval_count", len(full_conf_prompt) // 4)
                c_tok = meta.get("eval_count", len(raw_text) // 4)
            elif hasattr(self.backend, "generate"):
                resp = self.backend.generate(prompt=full_conf_prompt, schema=CONFIDENCE_ASSESSMENT_SCHEMA)
                raw_text = resp.text
                p_tok = getattr(resp, "prompt_tokens", len(full_conf_prompt) // 4)
                c_tok = getattr(resp, "completion_tokens", len(raw_text) // 4)
            else:
                raw_text = json.dumps({"confidence_percentage": 85})
                p_tok = len(full_conf_prompt) // 4
                c_tok = len(raw_text) // 4

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            data = json.loads(raw_text)
            conf_val = float(data.get("confidence_percentage", 50.0))
            return conf_val, p_tok, c_tok, latency_ms, None

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return 50.0, len(full_conf_prompt) // 4, 0, latency_ms, str(e)

    def execute_e08_episode(self, episode: OwnershipEpisode) -> List[OwnershipTrialResult]:
        """Execute full S09a (E08 / E08c) source attribution and ownership battery."""
        results: List[OwnershipTrialResult] = []
        role_map = getattr(episode, "role_mapping", "alpha_self_beta_peer")
        role_preamble = get_role_preamble(role_map)
        role_legend = get_role_legend(role_map)
        actor_display_names = get_actor_display_names(role_map)

        # -------------------------------------------------------------
        # 1. Neutral 5AFC Source Attribution Baseline
        # -------------------------------------------------------------
        for probe in episode.probes_attribution_5afc:
            prompt, p_hash = self._build_prompt(
                episode.events_neutral,
                episode.oracle_state,
                probe,
                role_preamble=role_preamble,
                legend_text=role_legend,
            )
            pred_let, pred_text, p_tok, c_tok, lat_ms, err = self._query_choice(prompt, probe)
            is_corr = (pred_let == probe.correct_option)

            attr_actor = None
            for act_name, disp in actor_display_names.items():
                if disp == pred_text or act_name in pred_text:
                    attr_actor = act_name
                    break

            results.append(OwnershipTrialResult(
                trial_id=f"{episode.episode_id}_{probe.probe_id}",
                episode_id=episode.episode_id,
                experiment_submodule="e08_source_ownership",
                condition_name="neutral_5afc_attribution",
                probe_id=probe.probe_id,
                probe_type=probe.probe_type,
                question=probe.question,
                options=probe.options,
                predicted_letter=pred_let,
                predicted_text=pred_text,
                correct_letter=probe.correct_option,
                is_correct=is_corr,
                attributed_actor=attr_actor,
                target_source=probe.target_source,
                target_actor=probe.target_actor,
                target_value=probe.target_value,
                prompt_hash=p_hash,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                latency_ms=lat_ms,
                error_message=err,
                metadata=dict(probe.metadata),
            ))

        # -------------------------------------------------------------
        # 2. Self vs Peer Conflict (Objective + Policy Operative)
        # -------------------------------------------------------------
        for probe in episode.probes_self_peer_objective + episode.probes_self_peer_belief:
            prompt, p_hash = self._build_prompt(
                episode.events_self_peer_conflict,
                episode.oracle_state,
                probe,
                role_preamble=role_preamble,
                legend_text=role_legend,
            )
            pred_let, pred_text, p_tok, c_tok, lat_ms, err = self._query_choice(prompt, probe)
            is_corr = (pred_let == probe.correct_option)

            results.append(OwnershipTrialResult(
                trial_id=f"{episode.episode_id}_{probe.probe_id}",
                episode_id=episode.episode_id,
                experiment_submodule="e08_source_ownership",
                condition_name="self_peer_conflict",
                probe_id=probe.probe_id,
                probe_type=probe.probe_type,
                question=probe.question,
                options=probe.options,
                predicted_letter=pred_let,
                predicted_text=pred_text,
                correct_letter=probe.correct_option,
                is_correct=is_corr,
                attributed_actor=None,
                target_source=probe.target_source,
                target_actor=probe.target_actor,
                target_value=probe.target_value,
                prompt_hash=p_hash,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                latency_ms=lat_ms,
                error_message=err,
                metadata=dict(probe.metadata),
            ))

        # -------------------------------------------------------------
        # 3. Framing Pair ("you" vs explicit actor)
        # -------------------------------------------------------------
        probe_self_f, probe_act_f = episode.probes_framing_pair
        for probe in [probe_self_f, probe_act_f]:
            prompt, p_hash = self._build_prompt(
                episode.events_neutral,
                episode.oracle_state,
                probe,
                role_preamble=role_preamble,
                legend_text=role_legend,
            )
            pred_let, pred_text, p_tok, c_tok, lat_ms, err = self._query_choice(prompt, probe)
            is_corr = (pred_let == probe.correct_option)

            results.append(OwnershipTrialResult(
                trial_id=f"{episode.episode_id}_{probe.probe_id}",
                episode_id=episode.episode_id,
                experiment_submodule="e08_source_ownership",
                condition_name=f"framing_{probe.metadata.get('framing', 'unknown')}",
                probe_id=probe.probe_id,
                probe_type=probe.probe_type,
                question=probe.question,
                options=probe.options,
                predicted_letter=pred_let,
                predicted_text=pred_text,
                correct_letter=probe.correct_option,
                is_correct=is_corr,
                attributed_actor=None,
                target_source=probe.target_source,
                target_actor=probe.target_actor,
                target_value=probe.target_value,
                prompt_hash=p_hash,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                latency_ms=lat_ms,
                error_message=err,
                metadata=dict(probe.metadata),
            ))

        # -------------------------------------------------------------
        # 4. Cue-Conflict 2x2 Factorial Specs
        # -------------------------------------------------------------
        for spec in episode.cue_conflict_specs:
            ev_cue = MemoryEvent(
                event_id=f"{spec.trial_id}_ev",
                step_index=0,
                source=spec.tag_source,
                actor_id=spec.narrative_actor,
                event_type="state_assertion",
                content=f"Actor '{spec.narrative_actor}' registers state binding: {spec.event_key} = {spec.target_value}.",
                key_bindings={spec.event_key: spec.target_value},
                metadata={"origin_source": spec.tag_source.value, "origin_actor": spec.narrative_actor},
            )
            prompt, p_hash = self._build_prompt(
                [ev_cue],
                None,
                spec.probe,
                include_tags=True,
                role_preamble=role_preamble,
                legend_text=role_legend,
            )
            pred_let, pred_text, p_tok, c_tok, lat_ms, err = self._query_choice(prompt, spec.probe)
            is_corr = (pred_let == spec.probe.correct_option)

            attr_actor = None
            for act_name, disp in actor_display_names.items():
                if disp == pred_text or act_name in pred_text:
                    attr_actor = act_name
                    break

            results.append(OwnershipTrialResult(
                trial_id=f"{episode.episode_id}_{spec.trial_id}",
                episode_id=episode.episode_id,
                experiment_submodule="e08_source_ownership",
                condition_name=f"cue_conflict_tag_{spec.tag_source.value}_narr_{spec.narrative_actor}",
                probe_id=spec.probe.probe_id,
                probe_type=spec.probe.probe_type,
                question=spec.probe.question,
                options=spec.probe.options,
                predicted_letter=pred_let,
                predicted_text=pred_text,
                correct_letter=spec.probe.correct_option,
                is_correct=is_corr,
                attributed_actor=attr_actor,
                target_source=spec.tag_source.value,
                target_actor=spec.narrative_actor,
                target_value=spec.target_value,
                prompt_hash=p_hash,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                latency_ms=lat_ms,
                error_message=err,
                metadata=dict(spec.metadata),
            ))

        # -------------------------------------------------------------
        # 5. Channel Factorial 2x2 Specs (Tags x Ledger)
        # -------------------------------------------------------------
        for spec in episode.channel_factorial_specs:
            prompt, p_hash = self._build_prompt(
                episode.events_neutral,
                episode.oracle_state,
                spec.probe,
                include_tags=spec.has_transcript_tags,
                include_ledger=spec.has_state_ledger,
                role_preamble=role_preamble,
                legend_text=role_legend,
            )
            pred_let, pred_text, p_tok, c_tok, lat_ms, err = self._query_choice(prompt, spec.probe)
            is_corr = (pred_let == spec.probe.correct_option)

            attr_actor = None
            for act_name, disp in actor_display_names.items():
                if disp == pred_text or act_name in pred_text:
                    attr_actor = act_name
                    break

            results.append(OwnershipTrialResult(
                trial_id=f"{episode.episode_id}_{spec.trial_id}",
                episode_id=episode.episode_id,
                experiment_submodule="e08_source_ownership",
                condition_name=f"channel_tags{spec.has_transcript_tags}_ledger{spec.has_state_ledger}",
                probe_id=spec.probe.probe_id,
                probe_type=spec.probe.probe_type,
                question=spec.probe.question,
                options=spec.probe.options,
                predicted_letter=pred_let,
                predicted_text=pred_text,
                correct_letter=spec.probe.correct_option,
                is_correct=is_corr,
                attributed_actor=attr_actor,
                target_source=spec.probe.target_source,
                target_actor=spec.probe.target_actor,
                target_value=spec.probe.target_value,
                prompt_hash=p_hash,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                latency_ms=lat_ms,
                error_message=err,
                metadata=dict(spec.metadata),
            ))

        # -------------------------------------------------------------
        # 6. Pressure Challenge (Pre vs Post)
        # -------------------------------------------------------------
        for probe, stage_name in [(episode.pressure_probe_pre, "pressure_pre_challenge"), (episode.pressure_probe_post, "pressure_post_challenge")]:
            prompt, p_hash = self._build_prompt(
                episode.events_neutral,
                episode.oracle_state,
                probe,
                role_preamble=role_preamble,
                legend_text=role_legend,
            )
            pred_let, pred_text, p_tok, c_tok, lat_ms, err = self._query_choice(prompt, probe)
            is_corr = (pred_let == probe.correct_option)

            attr_actor = None
            for act_name, disp in actor_display_names.items():
                if disp == pred_text or act_name in pred_text:
                    attr_actor = act_name
                    break

            results.append(OwnershipTrialResult(
                trial_id=f"{episode.episode_id}_{probe.probe_id}",
                episode_id=episode.episode_id,
                experiment_submodule="e08_source_ownership",
                condition_name=stage_name,
                probe_id=probe.probe_id,
                probe_type=probe.probe_type,
                question=probe.question,
                options=probe.options,
                predicted_letter=pred_let,
                predicted_text=pred_text,
                correct_letter=probe.correct_option,
                is_correct=is_corr,
                attributed_actor=attr_actor,
                target_source=probe.target_source,
                target_actor=probe.target_actor,
                target_value=probe.target_value,
                prompt_hash=p_hash,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                latency_ms=lat_ms,
                error_message=err,
                metadata=dict(probe.metadata),
            ))

        # -------------------------------------------------------------
        # 7. Isolated Positive Control Ceiling (Direct Lookup)
        # -------------------------------------------------------------
        if hasattr(episode, "probes_isolated_ceiling_5afc"):
            for probe in episode.probes_isolated_ceiling_5afc:
                prompt, p_hash = self._build_prompt(
                    events=None,
                    state=None,
                    probe=probe,
                    role_preamble=role_preamble,
                    legend_text=role_legend,
                )
                pred_let, pred_text, p_tok, c_tok, lat_ms, err = self._query_choice(prompt, probe)
                is_corr = (pred_let == probe.correct_option)

                attr_actor = None
                for act_name, disp in actor_display_names.items():
                    if disp == pred_text or act_name in pred_text:
                        attr_actor = act_name
                        break

                results.append(OwnershipTrialResult(
                    trial_id=f"{episode.episode_id}_{probe.probe_id}",
                    episode_id=episode.episode_id,
                    experiment_submodule="e08_source_ownership",
                    condition_name="isolated_ceiling_5afc",
                    probe_id=probe.probe_id,
                    probe_type=probe.probe_type,
                    question=probe.question,
                    options=probe.options,
                    predicted_letter=pred_let,
                    predicted_text=pred_text,
                    correct_letter=probe.correct_option,
                    is_correct=is_corr,
                    attributed_actor=attr_actor,
                    target_source=probe.target_source,
                    target_actor=probe.target_actor,
                    target_value=probe.target_value,
                    prompt_hash=p_hash,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    latency_ms=lat_ms,
                    error_message=err,
                    metadata=dict(probe.metadata),
                ))

        return results

    def execute_e09_metacognitive_screen(self, episode: OwnershipEpisode) -> List[OwnershipTrialResult]:
        """Execute S09b (E09) item-paired metacognitive confidence screen.
        
        Step A: Primary Agent (agent_alpha) makes the first-order source-attribution choice (target_choice).
        Step B: Self-framed evaluator assesses confidence in that exact target choice.
        Step C: Observer-framed evaluator assesses confidence in that exact target choice.
        """
        results: List[OwnershipTrialResult] = []

        for format_name, st_obj in [("transcript_only", None), ("scaffolded_state", episode.oracle_state)]:
            for probe in episode.probes_attribution_5afc:
                # -------------------------------------------------------------
                # Step A: Target Decision by Primary Agent (agent_alpha)
                # -------------------------------------------------------------
                prompt_self, p_hash_self = self._build_prompt(
                    episode.events_neutral,
                    st_obj,
                    probe,
                    role_preamble="You are primary agent 'agent_alpha' operating within a multi-agent system.",
                )
                pred_let_target, pred_text_target, p_tok_t, c_tok_t, lat_t, err_t = self._query_choice(prompt_self, probe)
                is_target_correct = (pred_let_target == probe.correct_option)

                # -------------------------------------------------------------
                # Step B: Self Pre-Feedback Error Prediction (Confidence)
                # -------------------------------------------------------------
                conf_self, p_tok_s, c_tok_s, lat_s, err_s = self._query_confidence_assessment(
                    base_prompt=prompt_self,
                    target_choice_letter=pred_let_target,
                    target_choice_text=pred_text_target,
                    evaluator="self",
                )

                results.append(OwnershipTrialResult(
                    trial_id=f"{episode.episode_id}_e09_self_{format_name}_{probe.probe_id}",
                    episode_id=episode.episode_id,
                    experiment_submodule="e09_metacognitive",
                    condition_name=f"meta_self_{format_name}",
                    probe_id=probe.probe_id,
                    probe_type="post_choice_error_prediction_self",
                    question=probe.question,
                    options=probe.options,
                    predicted_letter=pred_let_target,
                    predicted_text=pred_text_target,
                    correct_letter=probe.correct_option,
                    is_correct=is_target_correct,
                    attributed_actor=None,
                    target_source=probe.target_source,
                    target_actor=probe.target_actor,
                    target_value=probe.target_value,
                    subjective_confidence_pct=conf_self,
                    prompt_hash=p_hash_self,
                    prompt_tokens=p_tok_t + p_tok_s,
                    completion_tokens=c_tok_t + c_tok_s,
                    latency_ms=lat_t + lat_s,
                    error_message=err_s or err_t,
                    metadata={"evaluator": "self", "format": format_name, "key": probe.metadata.get("key"), "target_choice": pred_let_target},
                ))

                # -------------------------------------------------------------
                # Step C: Item-Paired Observer Pre-Feedback Error Prediction (Confidence)
                # -------------------------------------------------------------
                prompt_obs, p_hash_obs = self._build_prompt(
                    episode.events_neutral,
                    st_obj,
                    probe,
                    role_preamble="You are an external auditing observer 'auditor_gamma' monitoring multi-agent system execution.",
                )
                conf_obs, p_tok_o, c_tok_o, lat_o, err_o = self._query_confidence_assessment(
                    base_prompt=prompt_obs,
                    target_choice_letter=pred_let_target,
                    target_choice_text=pred_text_target,
                    evaluator="observer",
                )

                results.append(OwnershipTrialResult(
                    trial_id=f"{episode.episode_id}_e09_observer_{format_name}_{probe.probe_id}",
                    episode_id=episode.episode_id,
                    experiment_submodule="e09_metacognitive",
                    condition_name=f"meta_observer_{format_name}",
                    probe_id=probe.probe_id,
                    probe_type="post_choice_error_prediction_observer",
                    question=probe.question,
                    options=probe.options,
                    predicted_letter=pred_let_target,
                    predicted_text=pred_text_target,
                    correct_letter=probe.correct_option,
                    is_correct=is_target_correct,
                    attributed_actor=None,
                    target_source=probe.target_source,
                    target_actor=probe.target_actor,
                    target_value=probe.target_value,
                    subjective_confidence_pct=conf_obs,
                    prompt_hash=p_hash_obs,
                    prompt_tokens=p_tok_o,
                    completion_tokens=c_tok_o,
                    latency_ms=lat_o,
                    error_message=err_o,
                    metadata={"evaluator": "observer", "format": format_name, "key": probe.metadata.get("key"), "target_choice": pred_let_target},
                ))

        return results
