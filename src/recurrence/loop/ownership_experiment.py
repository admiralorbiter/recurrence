"""Execution harness for Sprint S09: Source Attribution, Self/Other Ownership (E08) and Metacognitive Continuity (E09).

Executes:
1. Neutral 5AFC Source Attribution baseline.
2. Self vs Peer objective assertion ownership and policy-governed operative belief.
3. Cue-Conflict Factorial (2x2 Tag x Narrative).
4. Channel Factorial (2x2 Transcript Tags x State Ledger).
5. Self-Referential ("you") vs 3rd-Person ("agent_alpha") Framing Contrast.
6. Pressure-Induced Revision Challenge.
7. Metacognitive Screen (E09): Paired Self vs Observer Confidence Calibration and Future-Failure Prediction.
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
    OwnershipProbe,
    OwnershipEpisode,
    CueConflictTrialSpec,
    ChannelFactorialTrialSpec,
)


@dataclass
class OwnershipTrialResult:
    """Record of a single evaluation probe in S09 (E08 / E09)."""
    trial_id: str
    episode_id: str
    experiment_submodule: str  # 'e08_source_ownership', 'e09_metacognitive'
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
        """Format episodic event log, optionally stripping explicit source tags."""
        lines = ["=== EPISODIC EVENT LOG TRANSCRIPT ==="]
        for ev in sorted(events, key=lambda e: (e.step_index, e.event_id)):
            if include_tags:
                src_val = ev.source.value if hasattr(ev.source, "value") else str(ev.source)
                lines.append(f"[Tick {ev.step_index:02d}] ({src_val} / {ev.event_type}) {ev.content}")
            else:
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
    ) -> Tuple[str, str]:
        """Construct prompt for ownership probe evaluation."""
        sections = [role_preamble]

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

    def _query_confidence(
        self,
        prompt: str,
        choice_letter: str,
    ) -> Tuple[float, int, int, float, Optional[str]]:
        """Query agent or observer for subjective confidence estimate (0-100%)."""
        conf_prompt = (
            f"{prompt}\n\n"
            f"Selected Choice: {choice_letter}\n"
            f"Please assess your subjective probability (0 to 100%) that this selection is accurate."
        )
        start_time = time.perf_counter()

        try:
            if hasattr(self.backend, "step"):
                raw_text, _, meta = self.backend.step(conf_prompt, format=CONFIDENCE_ASSESSMENT_SCHEMA)
                p_tok = meta.get("prompt_eval_count", len(conf_prompt) // 4)
                c_tok = meta.get("eval_count", len(raw_text) // 4)
            elif hasattr(self.backend, "generate"):
                resp = self.backend.generate(prompt=conf_prompt, schema=CONFIDENCE_ASSESSMENT_SCHEMA)
                raw_text = resp.text
                p_tok = getattr(resp, "prompt_tokens", len(conf_prompt) // 4)
                c_tok = getattr(resp, "completion_tokens", len(raw_text) // 4)
            else:
                raw_text = json.dumps({"confidence_percentage": 85})
                p_tok = len(conf_prompt) // 4
                c_tok = len(raw_text) // 4

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            data = json.loads(raw_text)
            conf_val = float(data.get("confidence_percentage", 50.0))
            return conf_val, p_tok, c_tok, latency_ms, None

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return 50.0, len(conf_prompt) // 4, 0, latency_ms, str(e)

    def execute_e08_episode(self, episode: OwnershipEpisode) -> List[OwnershipTrialResult]:
        """Execute full S09a (E08) source attribution and ownership battery."""
        results: List[OwnershipTrialResult] = []

        # -------------------------------------------------------------
        # 1. Neutral 5AFC Source Attribution Baseline
        # -------------------------------------------------------------
        for probe in episode.probes_attribution_5afc:
            prompt, p_hash = self._build_prompt(episode.events_neutral, episode.oracle_state, probe)
            pred_let, pred_text, p_tok, c_tok, lat_ms, err = self._query_choice(prompt, probe)
            is_corr = (pred_let == probe.correct_option)

            # Extract attributed actor
            attr_actor = None
            for act_name, disp in ACTOR_DISPLAY_NAMES.items():
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
            prompt, p_hash = self._build_prompt(episode.events_self_peer_conflict, episode.oracle_state, probe)
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
        # 3. Framing Pair ("you" vs "agent_alpha")
        # -------------------------------------------------------------
        probe_self_f, probe_act_f = episode.probes_framing_pair
        for probe in [probe_self_f, probe_act_f]:
            prompt, p_hash = self._build_prompt(episode.events_neutral, episode.oracle_state, probe)
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
            # Build manipulated single event
            ev_cue = MemoryEvent(
                event_id=f"{spec.trial_id}_ev",
                step_index=0,
                source=spec.tag_source,
                actor_id=spec.narrative_actor,
                event_type="state_assertion",
                content=f"System event log: Actor '{spec.narrative_actor}' registers state binding: {spec.event_key} = {spec.target_value}.",
                key_bindings={spec.event_key: spec.target_value},
                metadata={"origin_source": spec.tag_source.value, "origin_actor": spec.narrative_actor},
            )
            prompt, p_hash = self._build_prompt([ev_cue], None, spec.probe)
            pred_let, pred_text, p_tok, c_tok, lat_ms, err = self._query_choice(prompt, spec.probe)
            is_corr = (pred_let == spec.probe.correct_option)

            attr_actor = None
            for act_name, disp in ACTOR_DISPLAY_NAMES.items():
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
            )
            pred_let, pred_text, p_tok, c_tok, lat_ms, err = self._query_choice(prompt, spec.probe)
            is_corr = (pred_let == spec.probe.correct_option)

            attr_actor = None
            for act_name, disp in ACTOR_DISPLAY_NAMES.items():
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
            prompt, p_hash = self._build_prompt(episode.events_neutral, episode.oracle_state, probe)
            pred_let, pred_text, p_tok, c_tok, lat_ms, err = self._query_choice(prompt, probe)
            is_corr = (pred_let == probe.correct_option)

            attr_actor = None
            for act_name, disp in ACTOR_DISPLAY_NAMES.items():
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

        return results

    def execute_e09_metacognitive_screen(self, episode: OwnershipEpisode) -> List[OwnershipTrialResult]:
        """Execute S09b (E09) metacognitive confidence and future-failure screen."""
        results: List[OwnershipTrialResult] = []

        # Evaluate attribution probes under Transcript-Only and Scaffolded Persistence
        for format_name, st_obj in [("transcript_only", None), ("scaffolded_state", episode.oracle_state)]:
            for probe in episode.probes_attribution_5afc:
                # 1. Primary Agent Decision & Confidence
                prompt_self, p_hash_self = self._build_prompt(
                    episode.events_neutral,
                    st_obj,
                    probe,
                    role_preamble="You are primary agent 'agent_alpha' operating within a multi-agent system.",
                )
                pred_let_self, pred_text_self, p_tok_s, c_tok_s, lat_s, err_s = self._query_choice(prompt_self, probe)
                conf_self, _, _, _, _ = self._query_confidence(prompt_self, pred_let_self)
                is_corr_self = (pred_let_self == probe.correct_option)

                results.append(OwnershipTrialResult(
                    trial_id=f"{episode.episode_id}_e09_self_{format_name}_{probe.probe_id}",
                    episode_id=episode.episode_id,
                    experiment_submodule="e09_metacognitive",
                    condition_name=f"meta_self_{format_name}",
                    probe_id=probe.probe_id,
                    probe_type="metacognitive_self_confidence",
                    question=probe.question,
                    options=probe.options,
                    predicted_letter=pred_let_self,
                    predicted_text=pred_text_self,
                    correct_letter=probe.correct_option,
                    is_correct=is_corr_self,
                    attributed_actor=None,
                    target_source=probe.target_source,
                    target_actor=probe.target_actor,
                    target_value=probe.target_value,
                    subjective_confidence_pct=conf_self,
                    prompt_hash=p_hash_self,
                    prompt_tokens=p_tok_s,
                    completion_tokens=c_tok_s,
                    latency_ms=lat_s,
                    error_message=err_s,
                    metadata={"evaluator": "self", "format": format_name, "key": probe.metadata.get("key")},
                ))

                # 2. External Observer Decision & Confidence
                prompt_obs, p_hash_obs = self._build_prompt(
                    episode.events_neutral,
                    st_obj,
                    probe,
                    role_preamble="You are an external auditing observer 'auditor_gamma' monitoring multi-agent system execution.",
                )
                pred_let_obs, pred_text_obs, p_tok_o, c_tok_o, lat_o, err_o = self._query_choice(prompt_obs, probe)
                conf_obs, _, _, _, _ = self._query_confidence(prompt_obs, pred_let_obs)
                is_corr_obs = (pred_let_obs == probe.correct_option)

                results.append(OwnershipTrialResult(
                    trial_id=f"{episode.episode_id}_e09_observer_{format_name}_{probe.probe_id}",
                    episode_id=episode.episode_id,
                    experiment_submodule="e09_metacognitive",
                    condition_name=f"meta_observer_{format_name}",
                    probe_id=probe.probe_id,
                    probe_type="metacognitive_observer_confidence",
                    question=probe.question,
                    options=probe.options,
                    predicted_letter=pred_let_obs,
                    predicted_text=pred_text_obs,
                    correct_letter=probe.correct_option,
                    is_correct=is_corr_obs,
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
                    metadata={"evaluator": "observer", "format": format_name, "key": probe.metadata.get("key")},
                ))

        return results
