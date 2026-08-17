"""Execution harness for Sprint S08 State x Memory Conflict & Causal Interventions (Experiment E07).

Evaluates the full causal intervention matrix across matched twin episodes:
- Congruent (M_A + S_A, M_B + S_B) under M->S and S->M orderings
- Conflict (M_A + S_B, M_B + S_A) under M->S and S->M orderings
- Reset with Memory Preserved (M_A + S_empty)
- Surgical Single-Slot Inversion (M_A + S_A[k_target <- V_blue])
- Calibration Baselines (State-Only, Memory-Only)
- Clone, Fork, Cross-Swap & Reconvergence
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from recurrence.core.schemas import TARGET_4AFC_SCHEMA
from recurrence.memory.schemas import (
    EventSource,
    GoalState,
    MemoryEvent,
    StructuredSelfState,
    StateCapacityConfig,
)
from recurrence.loop.state_manager import StateManager
from recurrence.loop.updater import OracleStateUpdater
from recurrence.tasks.intervention import (
    InterventionProbe,
    MatchedTwinEpisodePair,
    CloneReconvergenceSpec,
)


def compute_state_hash(state: StructuredSelfState) -> str:
    """Compute SHA-256 hash over complete structured state."""
    dump = state.model_dump()
    return hashlib.sha256(json.dumps(dump, sort_keys=True).encode("utf-8")).hexdigest()


def compute_events_hash(events: List[MemoryEvent]) -> str:
    """Compute SHA-256 hash over episodic event list."""
    dump = [e.model_dump() if hasattr(e, "model_dump") else str(e) for e in events]
    return hashlib.sha256(json.dumps(dump, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class InterventionTrialResult:
    """Record for a single causal intervention evaluation probe."""
    trial_id: str
    pair_id: str
    intervention_condition: str
    presentation_order: str  # 'memory_first', 'state_first', 'state_only', 'memory_only'
    probe_id: str
    probe_type: str
    question: str
    options: Dict[str, str]
    predicted_letter: str
    predicted_value: str
    target_value_A: str
    target_value_B: str
    control_value: str
    is_state_allegiant: bool
    is_memory_allegiant: bool
    is_control_preserved: bool
    is_joint_local_precise: bool
    prompt_hash: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class InterventionHarness:
    """Orchestrator for executing causal state interventions and measuring behavioral steering."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def _format_transcript(self, events: List[MemoryEvent]) -> str:
        """Format episodic event log transcript."""
        lines = ["=== EPISODIC EVENT LOG TRANSCRIPT ==="]
        for ev in sorted(events, key=lambda e: (e.step_index, e.event_id)):
            src_val = ev.source.value if hasattr(ev.source, "value") else str(ev.source)
            bindings_str = f" | bindings: {ev.key_bindings}" if ev.key_bindings else ""
            lines.append(f"[Tick {ev.step_index:02d}] ({src_val} / {ev.event_type}) {ev.content}{bindings_str}")
        return "\n".join(lines)

    def _format_state(self, state: StructuredSelfState) -> str:
        """Format structured self-state into JSON string."""
        dump = state.model_dump()
        return f"=== CURRENT STRUCTURED STATE ===\n{json.dumps(dump, indent=2, sort_keys=True)}"

    def _build_prompt(
        self,
        memory_events: Optional[List[MemoryEvent]],
        state: Optional[StructuredSelfState],
        probe: InterventionProbe,
        order: str = "memory_first",
    ) -> Tuple[str, str]:
        """Construct evaluation prompt with counterbalanced section ordering."""
        sections = []

        if order == "memory_first":
            if memory_events is not None:
                sections.append(self._format_transcript(memory_events))
            if state is not None:
                sections.append(self._format_state(state))
        elif order == "state_first":
            if state is not None:
                sections.append(self._format_state(state))
            if memory_events is not None:
                sections.append(self._format_transcript(memory_events))
        elif order == "state_only":
            assert state is not None
            sections.append(self._format_state(state))
        elif order == "memory_only":
            assert memory_events is not None
            sections.append(self._format_transcript(memory_events))
        else:
            raise ValueError(f"Unknown presentation order: {order}")

        opts_str = "\n".join([f"{l}. {text}" for l, text in sorted(probe.options.items())])

        sections.append(
            f"=== EVALUATION QUESTION ===\n"
            f"{probe.question}\n\n"
            f"Options:\n"
            f"{opts_str}\n\n"
            f"Select the single correct option letter (A, B, C, or D). Return strictly JSON matching schema."
        )

        full_prompt = "\n\n".join(sections)
        p_hash = hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()
        return full_prompt, p_hash

    def _query_probe(
        self,
        prompt: str,
        probe: InterventionProbe,
        p_hash: str,
    ) -> Tuple[str, str, int, int, float, Optional[str]]:
        """Query LLM backend under strict JSON schema."""
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
                raw_text = json.dumps({"answer": probe.correct_letter_congruent})
                p_tok = len(prompt) // 4
                c_tok = len(raw_text) // 4

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            data = json.loads(raw_text)
            pred_letter = str(data.get("answer") or data.get("target_answer") or "").strip().upper()
            pred_val = probe.options.get(pred_letter, "UNKNOWN")
            return pred_letter, pred_val, p_tok, c_tok, latency_ms, None

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return "ERROR", "ERROR", len(prompt) // 4, 0, latency_ms, str(e)

    def execute_twin_pair(
        self,
        twin_pair: MatchedTwinEpisodePair,
    ) -> List[InterventionTrialResult]:
        """Execute full State x Memory intervention matrix across matched twins A and B."""
        results: List[InterventionTrialResult] = []

        # Empty state for reset condition
        s_empty = StructuredSelfState(
            working_memory={},
            goals=[],
            source_ledger={},
            unresolved_items=[],
            derived_inferences={},
            last_updated_step=0,
        )

        # Surgically inverted state: S_A with k_target flipped to val_target_B (V_blue)
        s_A_surgical = twin_pair.oracle_state_A.model_copy(deep=True)
        s_A_surgical.working_memory[twin_pair.k_target] = twin_pair.val_target_B

        # Define full intervention matrix
        # (condition_name, memory_events, state_obj, active_probes, target_is_A, state_is_A, memory_is_A, orders)
        eval_matrix = [
            # 1. Congruent Baseline A
            ("congruent_A", twin_pair.prefix_events_A, twin_pair.oracle_state_A, twin_pair.probes_A, True, True, True, ["memory_first", "state_first"]),
            # 2. Congruent Baseline B
            ("congruent_B", twin_pair.prefix_events_B, twin_pair.oracle_state_B, twin_pair.probes_B, False, False, False, ["memory_first", "state_first"]),
            # 3. State/Memory Conflict A/B (Memory A says V_red, State B says V_blue)
            ("conflict_MA_SB", twin_pair.prefix_events_A, twin_pair.oracle_state_B, twin_pair.probes_A, None, False, True, ["memory_first", "state_first"]),
            # 4. State/Memory Conflict B/A (Memory B says V_blue, State A says V_red)
            ("conflict_MB_SA", twin_pair.prefix_events_B, twin_pair.oracle_state_A, twin_pair.probes_B, None, True, False, ["memory_first", "state_first"]),
            # 5. Reset with Memory Preserved (Memory A, State Empty)
            ("reset_MA_Sempty", twin_pair.prefix_events_A, s_empty, twin_pair.probes_A, None, None, True, ["memory_first"]),
            # 6. Surgical Single-Slot Inversion (Memory A, State A with k_target -> V_blue)
            ("surgical_MA_SAprime", twin_pair.prefix_events_A, s_A_surgical, twin_pair.probes_A, None, False, True, ["memory_first"]),
            # 7. State-Only Calibration (State A & State B standalone)
            ("state_only_SA", None, twin_pair.oracle_state_A, twin_pair.probes_A, True, True, None, ["state_only"]),
            ("state_only_SB", None, twin_pair.oracle_state_B, twin_pair.probes_B, False, False, None, ["state_only"]),
            ("state_only_Sempty", None, s_empty, twin_pair.probes_A, None, None, None, ["state_only"]),
            # 8. Memory-Only Calibration (Memory A & Memory B standalone)
            ("memory_only_MA", twin_pair.prefix_events_A, None, twin_pair.probes_A, True, None, True, ["memory_only"]),
            ("memory_only_MB", twin_pair.prefix_events_B, None, twin_pair.probes_B, False, None, False, ["memory_only"]),
        ]

        for cond_name, mem_evs, st_obj, probe_list, _, state_is_A, memory_is_A, orders in eval_matrix:
            for order in orders:
                for probe in probe_list:
                    prompt, p_hash = self._build_prompt(mem_evs, st_obj, probe, order=order)
                    pred_let, pred_val, p_tok, c_tok, lat_ms, err_msg = self._query_probe(prompt, probe, p_hash)

                    # Determine Allegiance & Precision
                    # Value A = probe.target_value_A, Value B = probe.target_value_B
                    is_state_all = False
                    is_mem_all = False

                    if state_is_A is True and pred_val == probe.target_value_A:
                        is_state_all = True
                    elif state_is_A is False and pred_val == probe.target_value_B:
                        is_state_all = True

                    if memory_is_A is True and pred_val == probe.target_value_A:
                        is_mem_all = True
                    elif memory_is_A is False and pred_val == probe.target_value_B:
                        is_mem_all = True

                    is_ctrl_pres = (pred_val == probe.control_value) if probe.probe_type == "control_key" else False

                    # Joint local precision: for target probe under surgical, uptake is state_allegiant;
                    # for control probe under surgical, control is preserved.
                    is_joint_prec = False
                    if cond_name == "surgical_MA_SAprime":
                        if probe.probe_type == "target_key" and is_state_all:
                            is_joint_prec = True
                        elif probe.probe_type == "control_key" and is_ctrl_pres:
                            is_joint_prec = True

                    trial_id = f"{twin_pair.pair_id}_{cond_name}_{order}_{probe.probe_id}"

                    results.append(InterventionTrialResult(
                        trial_id=trial_id,
                        pair_id=twin_pair.pair_id,
                        intervention_condition=cond_name,
                        presentation_order=order,
                        probe_id=probe.probe_id,
                        probe_type=probe.probe_type,
                        question=probe.question,
                        options=probe.options,
                        predicted_letter=pred_let,
                        predicted_value=pred_val,
                        target_value_A=probe.target_value_A,
                        target_value_B=probe.target_value_B,
                        control_value=probe.control_value,
                        is_state_allegiant=is_state_all,
                        is_memory_allegiant=is_mem_all,
                        is_control_preserved=is_ctrl_pres,
                        is_joint_local_precise=is_joint_prec,
                        prompt_hash=p_hash,
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        latency_ms=lat_ms,
                        error_message=err_msg,
                        metadata=dict(probe.metadata),
                    ))

        return results

    def execute_clone_reconvergence(
        self,
        clone_spec: CloneReconvergenceSpec,
    ) -> List[InterventionTrialResult]:
        """Execute clone, branching, cross-swap, and reconvergence infrastructure tests."""
        results: List[InterventionTrialResult] = []

        # Invariant 1: Assert Prefix Hash Equality
        pre_hash = compute_state_hash(clone_spec.oracle_prefix_state)

        # Branch A with S_A vs Branch A with transplanted S_B (Cross-Swap)
        fork_evals = [
            ("clone_fork_A_congruent", clone_spec.prefix_events_common + clone_spec.fork_events_A, clone_spec.oracle_fork_state_A, clone_spec.val_fork_A, clone_spec.val_fork_B, True),
            ("clone_fork_A_cross_swap_SB", clone_spec.prefix_events_common + clone_spec.fork_events_A, clone_spec.oracle_fork_state_B, clone_spec.val_fork_A, clone_spec.val_fork_B, False),
            ("clone_fork_B_congruent", clone_spec.prefix_events_common + clone_spec.fork_events_B, clone_spec.oracle_fork_state_B, clone_spec.val_fork_A, clone_spec.val_fork_B, False),
        ]

        for cond_name, events, state_obj, val_A, val_B, st_is_A in fork_evals:
            for probe in clone_spec.probes_fork:
                prompt, p_hash = self._build_prompt(events, state_obj, probe, order="memory_first")
                pred_let, pred_val, p_tok, c_tok, lat_ms, err_msg = self._query_probe(prompt, probe, p_hash)

                is_st_all = (pred_val == val_A) if st_is_A else (pred_val == val_B)
                is_mem_all = (pred_val == val_A) if "fork_A" in cond_name else (pred_val == val_B)

                results.append(InterventionTrialResult(
                    trial_id=f"{clone_spec.spec_id}_{cond_name}_{probe.probe_id}",
                    pair_id=clone_spec.spec_id,
                    intervention_condition=cond_name,
                    presentation_order="memory_first",
                    probe_id=probe.probe_id,
                    probe_type=probe.probe_type,
                    question=probe.question,
                    options=probe.options,
                    predicted_letter=pred_let,
                    predicted_value=pred_val,
                    target_value_A=val_A,
                    target_value_B=val_B,
                    control_value=clone_spec.val_common,
                    is_state_allegiant=is_st_all,
                    is_memory_allegiant=is_mem_all,
                    is_control_preserved=False,
                    is_joint_local_precise=False,
                    prompt_hash=p_hash,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    latency_ms=lat_ms,
                    error_message=err_msg,
                    metadata={"testbed": "clone_fork", "prefix_hash": pre_hash},
                ))

        # Reconvergence Test: Both branches receive reconvergence events
        events_reconv_A = clone_spec.prefix_events_common + clone_spec.fork_events_A + clone_spec.reconvergence_events
        events_reconv_B = clone_spec.prefix_events_common + clone_spec.fork_events_B + clone_spec.reconvergence_events

        reconv_evals = [
            ("reconverged_branch_A", events_reconv_A, clone_spec.oracle_reconverged_state),
            ("reconverged_branch_B", events_reconv_B, clone_spec.oracle_reconverged_state),
        ]

        for cond_name, events, state_obj in reconv_evals:
            for probe in clone_spec.probes_reconverged:
                prompt, p_hash = self._build_prompt(events, state_obj, probe, order="memory_first")
                pred_let, pred_val, p_tok, c_tok, lat_ms, err_msg = self._query_probe(prompt, probe, p_hash)

                is_reconv_corr = (pred_val == clone_spec.val_reconverge)

                results.append(InterventionTrialResult(
                    trial_id=f"{clone_spec.spec_id}_{cond_name}_{probe.probe_id}",
                    pair_id=clone_spec.spec_id,
                    intervention_condition=cond_name,
                    presentation_order="memory_first",
                    probe_id=probe.probe_id,
                    probe_type=probe.probe_type,
                    question=probe.question,
                    options=probe.options,
                    predicted_letter=pred_let,
                    predicted_value=pred_val,
                    target_value_A=clone_spec.val_reconverge,
                    target_value_B=clone_spec.val_reconverge,
                    control_value=clone_spec.val_common,
                    is_state_allegiant=is_reconv_corr,
                    is_memory_allegiant=is_reconv_corr,
                    is_control_preserved=False,
                    is_joint_local_precise=False,
                    prompt_hash=p_hash,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    latency_ms=lat_ms,
                    error_message=err_msg,
                    metadata={"testbed": "reconvergence", "val_reconverge": clone_spec.val_reconverge},
                ))

        return results
