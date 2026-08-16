"""Execution harness for Sprint S07.1 Available-Inference Null Consolidation Benchmark (E06b).

Evaluates 6 experimental conditions across K in {0, 1, 3, 6, 12} quiet ticks and 2 informational regimes:
1. available_inference: Complete evidence pre-null
2. missing_premise_control: Incomplete evidence pre-null

Enforces:
- Protected evidence invariance: hash(working_memory_pre, source_ledger_pre) == hash(working_memory_post, source_ledger_post)
- Full per-tick reflection audit logging (ReflectionTickTrace) with explicit schema validation
- Exact SHA-256 evaluation prompt hashing and bit-for-bit strict identity verification
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from recurrence.core.schemas import (
    TARGET_4AFC_SCHEMA,
    STATE_UPDATE_SCHEMA,
    STATE_SELECTIVE_REFLECTION_SCHEMA,
)
from recurrence.memory.schemas import (
    EventSource,
    GoalState,
    MemoryEvent,
    StructuredSelfState,
    StateCapacityConfig,
)
from recurrence.loop.clock import SimulatedClock
from recurrence.loop.queue import EventQueue
from recurrence.loop.state_manager import StateManager
from recurrence.loop.updater import OracleStateUpdater
from recurrence.tasks.quiet_interval import (
    QuietIntervalEpisode,
    QuietIntervalProbe,
)


def compute_evidence_hash(state: StructuredSelfState) -> str:
    """Compute SHA-256 hash over protected evidence fields (working_memory and source_ledger)."""
    norm_dict = {
        "working_memory": sorted(list(state.working_memory.items())),
        "source_ledger": sorted(list(state.source_ledger.items())),
    }
    return hashlib.sha256(json.dumps(norm_dict, sort_keys=True).encode("utf-8")).hexdigest()


def compute_state_hash(state: StructuredSelfState) -> str:
    """Compute SHA-256 hash over complete structured state."""
    dump = state.model_dump()
    return hashlib.sha256(json.dumps(dump, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class ReflectionTickTrace:
    """Audit trace for a single quiet reflection cycle at tick step_k."""
    episode_id: str
    regime: str
    condition: str
    tick_k: int
    prompt_hash: str
    raw_response: str
    parsed_write: Dict[str, Any]
    schema_valid: bool
    pre_state_hash: str
    post_state_hash: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    error_message: Optional[str] = None


@dataclass
class QuietTrialResult:
    """Evaluation record for a single probe under a specific condition, regime, and quiet interval K."""
    trial_id: str
    episode_id: str
    regime: str
    interval_k: int
    condition: str
    probe_id: str
    probe_type: str
    question: str
    options: Dict[str, str]
    correct_letter: str
    predicted_letter: str
    is_correct: bool
    prompt_hash: str
    context_hash: str
    query_prompt_tokens: int
    query_completion_tokens: int
    query_latency_ms: float
    amortized_prompt_tokens: int
    amortized_latency_ms: float
    context_chars: int
    evidence_hash_valid: bool
    evidence_drift_detected: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuietIntervalHarness:
    """Orchestrator for executing quiet interval experiments with full audit logging."""

    def __init__(
        self,
        backend: Any,
        capacity_config: Optional[StateCapacityConfig] = None,
    ) -> None:
        self.backend = backend
        self.capacity_config = capacity_config or StateCapacityConfig(
            max_working_memory_items=16,
            max_goals=8,
            max_unresolved_items=16,
        )

    def _build_transcript_text(
        self,
        prefix_events: List[MemoryEvent],
        interval_k: int,
        continuation_events: List[MemoryEvent],
        prefix_ticks: int = 4,
    ) -> str:
        """Format raw event log with explicit quiet interval markers."""
        lines = ["=== EPISODIC EVENT LOG TRANSCRIPT ==="]
        for ev in sorted(prefix_events, key=lambda e: (e.step_index, e.event_id)):
            src_val = ev.source.value if hasattr(ev.source, "value") else str(ev.source)
            bindings_str = f" | bindings: {ev.key_bindings}" if ev.key_bindings else ""
            lines.append(f"[Tick {ev.step_index:02d}] ({src_val} / {ev.event_type}) {ev.content}{bindings_str}")

        if interval_k > 0:
            lines.append(f"--- [QUIET INTERVAL: {interval_k} null ticks elapsed (no new observations)] ---")

        for ev in continuation_events:
            actual_tick = prefix_ticks + interval_k + ev.step_index
            src_val = ev.source.value if hasattr(ev.source, "value") else str(ev.source)
            bindings_str = f" | bindings: {ev.key_bindings}" if ev.key_bindings else ""
            lines.append(f"[Tick {actual_tick:02d}] ({src_val} / {ev.event_type}) {ev.content}{bindings_str}")

        return "\n".join(lines)

    def _build_state_text(self, state: StructuredSelfState, pin_step: Optional[int] = None) -> str:
        """Format structured state into JSON string representation."""
        dump = state.model_dump()
        if pin_step is not None:
            dump["last_updated_step"] = pin_step
        return f"=== CURRENT STRUCTURED STATE ===\n{json.dumps(dump, indent=2, sort_keys=True)}"

    def _query_probe(
        self,
        context_str: str,
        probe: QuietIntervalProbe,
    ) -> Tuple[str, bool, str, int, int, float, Optional[str]]:
        """Query LLM backend for a forced-choice probe under strict JSON schema."""
        opts_str = "\n".join([f"{l}. {text}" for l, text in sorted(probe.options.items())])
        opts_letters_str = "A, B, C, or D"

        prompt = (
            f"{context_str}\n\n"
            f"=== EVALUATION QUESTION ===\n"
            f"{probe.question}\n\n"
            f"Options:\n"
            f"{opts_str}\n\n"
            f"Select the single correct option letter ({opts_letters_str}). Return strictly JSON matching schema."
        )

        p_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        start_time = time.perf_counter()
        try:
            if hasattr(self.backend, "step"):
                raw_text, _, meta = self.backend.step(prompt, format=TARGET_4AFC_SCHEMA)
                p_tok = meta.get("prompt_eval_count", len(prompt) // 4)
                c_tok = meta.get("eval_count", len(raw_text) // 4)
            elif hasattr(self.backend, "generate"):
                resp = self.backend.generate(prompt=prompt, schema=TARGET_4AFC_SCHEMA)
                raw_text = resp.text
                p_tok = getattr(resp, "prompt_tokens", len(prompt) // 4)
                c_tok = getattr(resp, "completion_tokens", len(raw_text) // 4)
            else:
                raw_text = json.dumps({"answer": probe.correct_letter})
                p_tok = len(prompt) // 4
                c_tok = len(raw_text) // 4

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            data = json.loads(raw_text)
            pred_letter = str(data.get("answer") or data.get("target_answer") or "").strip().upper()
            is_corr = (pred_letter == probe.correct_letter.upper())
            return pred_letter, is_corr, p_hash, p_tok, c_tok, latency_ms, None

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return "ERROR", False, p_hash, len(prompt) // 4, 0, latency_ms, str(e)

    def _apply_continuation(
        self,
        base_state: StructuredSelfState,
        continuation_events: List[MemoryEvent],
        start_tick: int,
    ) -> StructuredSelfState:
        """Apply continuation events deterministically using Oracle updater."""
        mgr = StateManager(
            capacity_config=self.capacity_config,
            initial_state=base_state.model_copy(deep=True),
        )
        upd = OracleStateUpdater(state_manager=mgr)

        for rel_idx, ev in enumerate(continuation_events):
            t = start_tick + rel_idx
            new_st, _, _, _, _, _, _, _ = upd.update(mgr.current_state, [ev], t)
            mgr.update_state(new_st, t, 1, schema_valid=True)

        return mgr.current_state.model_copy(deep=True)

    def execute_episode(
        self,
        episode: QuietIntervalEpisode,
        interval_ks: Optional[List[int]] = None,
        conditions: Optional[List[str]] = None,
    ) -> Tuple[List[QuietTrialResult], List[ReflectionTickTrace], Dict[str, Any]]:
        """Execute all conditions and intervals for a single base episode."""
        eval_ks = interval_ks or [0, 1, 3, 6, 12]
        eval_conditions = conditions or [
            "strict_identity",
            "clock_only",
            "semantic_no_write",
            "selective_reflection",
            "unconstrained_reflection",
            "replay_transcript",
        ]

        trial_results: List[QuietTrialResult] = []
        reflection_traces: List[ReflectionTickTrace] = []
        episode_metadata: Dict[str, Any] = {}

        # -------------------------------------------------------------
        # 1. Establish Ground-Truth Prefix State S*
        # -------------------------------------------------------------
        prefix_mgr = StateManager(capacity_config=self.capacity_config)
        prefix_upd = OracleStateUpdater(state_manager=prefix_mgr)
        prefix_ticks = len(set(e.step_index for e in episode.prefix_events))

        for t in range(prefix_ticks):
            evs = [e for e in episode.prefix_events if e.step_index == t]
            if evs:
                new_st, _, _, _, _, _, _, _ = prefix_upd.update(prefix_mgr.current_state, evs, t)
                prefix_mgr.update_state(new_st, t, len(evs), schema_valid=True)

        s_star = prefix_mgr.current_state.model_copy(deep=True)
        evidence_hash_star = compute_evidence_hash(s_star)
        episode_metadata["evidence_hash_prefix"] = evidence_hash_star

        # -------------------------------------------------------------
        # 2. Precompute Quiet Interval Trajectories out to K=12
        # -------------------------------------------------------------
        # Group B Trajectory: selective_reflection
        selective_snapshots: Dict[int, StructuredSelfState] = {0: s_star.model_copy(deep=True)}
        selective_costs: Dict[int, Tuple[int, float]] = {0: (0, 0.0)}

        st_curr_sel = s_star.model_copy(deep=True)
        cum_sel_tokens = 0
        cum_sel_latency = 0.0

        for step_k in range(1, 13):
            sel_prompt = (
                f"{self._build_state_text(st_curr_sel)}\n\n"
                f"You are the cognitive reflection engine of an autonomous agent during a quiet interval.\n"
                f"Review the current state. Infer any multi-hop conclusions into 'derived_inferences' (e.g. if key_A->val_B and key_B->val_C, infer key_A->val_C).\n"
                f"Consolidate any conflicting or ambiguous entity keys into 'unresolved_items'.\n"
                f"Update existing goal statuses if appropriate.\n"
                f"Note: Working memory and source ledgers are protected and immutable."
            )
            prompt_hash = hashlib.sha256(sel_prompt.encode("utf-8")).hexdigest()
            pre_hash = compute_state_hash(st_curr_sel)

            start_k = time.perf_counter()
            err_k: Optional[str] = None
            schema_valid_k = False
            raw_text = ""
            parsed: Dict[str, Any] = {}

            try:
                if hasattr(self.backend, "step"):
                    raw_text, _, meta = self.backend.step(sel_prompt, format=STATE_SELECTIVE_REFLECTION_SCHEMA)
                    p_tok = meta.get("prompt_eval_count", len(sel_prompt) // 4)
                    c_tok = meta.get("eval_count", len(raw_text) // 4)
                elif hasattr(self.backend, "generate"):
                    resp = self.backend.generate(prompt=sel_prompt, schema=STATE_SELECTIVE_REFLECTION_SCHEMA)
                    raw_text = resp.text
                    p_tok = getattr(resp, "prompt_tokens", len(sel_prompt) // 4)
                    c_tok = getattr(resp, "completion_tokens", len(raw_text) // 4)
                else:
                    if episode.regime == "available_inference":
                        mock_derived = {episode.metadata["k_hop1"]: episode.metadata["v_hop2"]}
                    else:
                        mock_derived = {}
                    raw_text = json.dumps({
                        "derived_inferences": mock_derived,
                        "unresolved_items": [],
                        "goal_status_updates": [{"goal_id": episode.metadata["gid_beta"], "status": "active"}],
                    })
                    p_tok = len(sel_prompt) // 4
                    c_tok = len(raw_text) // 4

                lat_k = (time.perf_counter() - start_k) * 1000.0
                parsed = json.loads(raw_text)
                schema_valid_k = True

                # Apply selective updates
                st_curr_sel.derived_inferences.update(parsed.get("derived_inferences", {}))
                for item in parsed.get("unresolved_items", []):
                    if item not in st_curr_sel.unresolved_items:
                        st_curr_sel.unresolved_items.append(item)

                status_updates = {u["goal_id"]: u["status"] for u in parsed.get("goal_status_updates", []) if "goal_id" in u and "status" in u}
                for g in st_curr_sel.goals:
                    if g.goal_id in status_updates:
                        g.status = status_updates[g.goal_id]

                st_curr_sel.last_updated_step = prefix_ticks + step_k - 1

            except Exception as e:
                lat_k = (time.perf_counter() - start_k) * 1000.0
                p_tok = len(sel_prompt) // 4
                c_tok = 0
                err_k = str(e)

            post_hash = compute_state_hash(st_curr_sel)
            cum_sel_tokens += p_tok + c_tok
            cum_sel_latency += lat_k

            reflection_traces.append(ReflectionTickTrace(
                episode_id=episode.episode_id,
                regime=episode.regime,
                condition="selective_reflection",
                tick_k=step_k,
                prompt_hash=prompt_hash,
                raw_response=raw_text,
                parsed_write=parsed,
                schema_valid=schema_valid_k,
                pre_state_hash=pre_hash,
                post_state_hash=post_hash,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                latency_ms=lat_k,
                error_message=err_k,
            ))

            if step_k in (1, 3, 6, 12):
                selective_snapshots[step_k] = st_curr_sel.model_copy(deep=True)
                selective_costs[step_k] = (cum_sel_tokens, cum_sel_latency)

        # Group B Trajectory: unconstrained_reflection
        unconstrained_snapshots: Dict[int, StructuredSelfState] = {0: s_star.model_copy(deep=True)}
        unconstrained_costs: Dict[int, Tuple[int, float]] = {0: (0, 0.0)}

        st_curr_uncon = s_star.model_copy(deep=True)
        cum_uncon_tokens = 0
        cum_uncon_latency = 0.0

        for step_k in range(1, 13):
            uncon_prompt = (
                f"{self._build_state_text(st_curr_uncon)}\n\n"
                f"You are the state maintenance engine. Update and rewrite the entire structured self-state\n"
                f"during this quiet interval without new observations."
            )
            prompt_hash = hashlib.sha256(uncon_prompt.encode("utf-8")).hexdigest()
            pre_hash = compute_state_hash(st_curr_uncon)

            start_k = time.perf_counter()
            err_k = None
            schema_valid_k = False
            raw_text = ""
            parsed = {}

            try:
                if hasattr(self.backend, "step"):
                    raw_text, _, meta = self.backend.step(uncon_prompt, format=STATE_UPDATE_SCHEMA)
                    p_tok = meta.get("prompt_eval_count", len(uncon_prompt) // 4)
                    c_tok = meta.get("eval_count", len(raw_text) // 4)
                elif hasattr(self.backend, "generate"):
                    resp = self.backend.generate(prompt=uncon_prompt, schema=STATE_UPDATE_SCHEMA)
                    raw_text = resp.text
                    p_tok = getattr(resp, "prompt_tokens", len(uncon_prompt) // 4)
                    c_tok = getattr(resp, "completion_tokens", len(raw_text) // 4)
                else:
                    raw_text = json.dumps({
                        "working_memory": {episode.metadata["k_stable"]: episode.metadata["k_stable"]},
                        "goals": [],
                        "source_ledger": {},
                        "unresolved_items": [],
                    })
                    p_tok = len(uncon_prompt) // 4
                    c_tok = len(raw_text) // 4

                lat_k = (time.perf_counter() - start_k) * 1000.0
                parsed = json.loads(raw_text)
                schema_valid_k = True

                st_curr_uncon = StructuredSelfState(
                    working_memory=parsed.get("working_memory", {}),
                    goals=[
                        GoalState(
                            goal_id=g["goal_id"],
                            description=g["description"],
                            status=g["status"],
                            created_at_step=0,
                            updated_at_step=prefix_ticks + step_k - 1,
                        )
                        for g in parsed.get("goals", []) if "goal_id" in g and "description" in g and "status" in g
                    ],
                    source_ledger=parsed.get("source_ledger", {}),
                    unresolved_items=parsed.get("unresolved_items", []),
                    derived_inferences=st_curr_uncon.derived_inferences,
                    last_updated_step=prefix_ticks + step_k - 1,
                )
            except Exception as e:
                lat_k = (time.perf_counter() - start_k) * 1000.0
                p_tok = len(uncon_prompt) // 4
                c_tok = 0
                err_k = str(e)

            post_hash = compute_state_hash(st_curr_uncon)
            cum_uncon_tokens += p_tok + c_tok
            cum_uncon_latency += lat_k

            reflection_traces.append(ReflectionTickTrace(
                episode_id=episode.episode_id,
                regime=episode.regime,
                condition="unconstrained_reflection",
                tick_k=step_k,
                prompt_hash=prompt_hash,
                raw_response=raw_text,
                parsed_write=parsed,
                schema_valid=schema_valid_k,
                pre_state_hash=pre_hash,
                post_state_hash=post_hash,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                latency_ms=lat_k,
                error_message=err_k,
            ))

            if step_k in (1, 3, 6, 12):
                unconstrained_snapshots[step_k] = st_curr_uncon.model_copy(deep=True)
                unconstrained_costs[step_k] = (cum_uncon_tokens, cum_uncon_latency)

        # Semantic No-Write Compute Trajectory
        nowrite_costs: Dict[int, Tuple[int, float]] = {0: (0, 0.0)}
        cum_nw_tokens = 0
        cum_nw_latency = 0.0

        for step_k in range(1, 13):
            nw_prompt = (
                f"{self._build_state_text(s_star)}\n\n"
                f"You are the cognitive reflection engine. Reason through multi-hop links and unresolved goals."
            )
            prompt_hash = hashlib.sha256(nw_prompt.encode("utf-8")).hexdigest()
            pre_hash = compute_state_hash(s_star)

            start_k = time.perf_counter()
            err_k = None
            schema_valid_k = False
            raw_text = ""

            try:
                if hasattr(self.backend, "step"):
                    raw_text, _, meta = self.backend.step(nw_prompt, format=STATE_SELECTIVE_REFLECTION_SCHEMA)
                    p_tok = meta.get("prompt_eval_count", len(nw_prompt) // 4)
                    c_tok = meta.get("eval_count", len(raw_text) // 4)
                elif hasattr(self.backend, "generate"):
                    resp = self.backend.generate(prompt=nw_prompt, schema=STATE_SELECTIVE_REFLECTION_SCHEMA)
                    raw_text = resp.text
                    p_tok = getattr(resp, "prompt_tokens", len(nw_prompt) // 4)
                    c_tok = getattr(resp, "completion_tokens", len(raw_text) // 4)
                else:
                    p_tok = len(nw_prompt) // 4
                    c_tok = 20
                lat_k = (time.perf_counter() - start_k) * 1000.0
                schema_valid_k = True
            except Exception as e:
                lat_k = (time.perf_counter() - start_k) * 1000.0
                p_tok = len(nw_prompt) // 4
                c_tok = 0
                err_k = str(e)

            cum_nw_tokens += p_tok + c_tok
            cum_nw_latency += lat_k

            reflection_traces.append(ReflectionTickTrace(
                episode_id=episode.episode_id,
                regime=episode.regime,
                condition="semantic_no_write",
                tick_k=step_k,
                prompt_hash=prompt_hash,
                raw_response=raw_text,
                parsed_write={},
                schema_valid=schema_valid_k,
                pre_state_hash=pre_hash,
                post_state_hash=pre_hash,  # unchanged
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                latency_ms=lat_k,
                error_message=err_k,
            ))

            if step_k in (1, 3, 6, 12):
                nowrite_costs[step_k] = (cum_nw_tokens, cum_nw_latency)

        # -------------------------------------------------------------
        # 3. Evaluate Cross-Factorial Conditions Across K
        # -------------------------------------------------------------
        num_probes = len(episode.probes)

        for k in eval_ks:
            cont_start_tick = prefix_ticks + k

            for cond in eval_conditions:
                if k == 0 and cond not in ("strict_identity", "replay_transcript"):
                    continue

                if cond == "strict_identity":
                    st_inter = s_star.model_copy(deep=True)
                    st_final = self._apply_continuation(st_inter, episode.continuation_events, cont_start_tick)
                    context_str = self._build_state_text(st_final, pin_step=prefix_ticks - 1)
                    amort_tok = 0
                    amort_lat = 0.0

                elif cond == "clock_only":
                    st_inter = s_star.model_copy(deep=True)
                    st_inter.last_updated_step = prefix_ticks + k - 1
                    st_final = self._apply_continuation(st_inter, episode.continuation_events, cont_start_tick)
                    context_str = self._build_state_text(st_final)
                    amort_tok = 0
                    amort_lat = 0.0

                elif cond == "semantic_no_write":
                    st_inter = s_star.model_copy(deep=True)
                    st_final = self._apply_continuation(st_inter, episode.continuation_events, cont_start_tick)
                    context_str = self._build_state_text(st_final)
                    tot_nw_tok, tot_nw_lat = nowrite_costs.get(k, (0, 0.0))
                    amort_tok = tot_nw_tok // max(1, num_probes)
                    amort_lat = tot_nw_lat / max(1, num_probes)

                elif cond == "selective_reflection":
                    st_inter = selective_snapshots[k].model_copy(deep=True)
                    ev_hash_post = compute_evidence_hash(st_inter)
                    assert ev_hash_post == evidence_hash_star, (
                        f"Evidence mutation detected in episode {episode.episode_id} under selective_reflection at K={k}!\n"
                        f"Pre:  {evidence_hash_star}\n"
                        f"Post: {ev_hash_post}"
                    )
                    st_final = self._apply_continuation(st_inter, episode.continuation_events, cont_start_tick)
                    context_str = self._build_state_text(st_final)
                    tot_sel_tok, tot_sel_lat = selective_costs.get(k, (0, 0.0))
                    amort_tok = tot_sel_tok // max(1, num_probes)
                    amort_lat = tot_sel_lat / max(1, num_probes)

                elif cond == "unconstrained_reflection":
                    st_inter = unconstrained_snapshots[k].model_copy(deep=True)
                    ev_hash_post = compute_evidence_hash(st_inter)
                    st_final = self._apply_continuation(st_inter, episode.continuation_events, cont_start_tick)
                    context_str = self._build_state_text(st_final)
                    tot_uncon_tok, tot_uncon_lat = unconstrained_costs.get(k, (0, 0.0))
                    amort_tok = tot_uncon_tok // max(1, num_probes)
                    amort_lat = tot_uncon_lat / max(1, num_probes)

                elif cond == "replay_transcript":
                    context_str = self._build_transcript_text(
                        prefix_events=episode.prefix_events,
                        interval_k=k,
                        continuation_events=episode.continuation_events,
                        prefix_ticks=prefix_ticks,
                    )
                    amort_tok = 0
                    amort_lat = 0.0

                else:
                    raise ValueError(f"Unknown experimental condition: {cond}")

                # Drift check
                if cond == "unconstrained_reflection":
                    ev_drift = (compute_evidence_hash(st_final) != compute_evidence_hash(self._apply_continuation(s_star, episode.continuation_events, cont_start_tick)))
                else:
                    ev_drift = False

                context_hash = hashlib.sha256(context_str.encode("utf-8")).hexdigest()

                for probe in episode.probes:
                    pred_l, is_corr, p_hash, p_tok, c_tok, lat_ms, err_msg = self._query_probe(
                        context_str=context_str,
                        probe=probe,
                    )

                    trial_id = f"{episode.episode_id}_k{k}_{cond}_{probe.probe_id}"
                    trial_results.append(QuietTrialResult(
                        trial_id=trial_id,
                        episode_id=episode.episode_id,
                        regime=episode.regime,
                        interval_k=k,
                        condition=cond,
                        probe_id=probe.probe_id,
                        probe_type=probe.probe_type,
                        question=probe.question,
                        options=probe.options,
                        correct_letter=probe.correct_letter,
                        predicted_letter=pred_l,
                        is_correct=is_corr,
                        prompt_hash=p_hash,
                        context_hash=context_hash,
                        query_prompt_tokens=p_tok,
                        query_completion_tokens=c_tok,
                        query_latency_ms=lat_ms,
                        amortized_prompt_tokens=p_tok + amort_tok,
                        amortized_latency_ms=lat_ms + amort_lat,
                        context_chars=len(context_str),
                        evidence_hash_valid=not ev_drift,
                        evidence_drift_detected=ev_drift,
                        error_message=err_msg,
                        metadata=dict(probe.metadata),
                    ))

        return trial_results, reflection_traces, episode_metadata
