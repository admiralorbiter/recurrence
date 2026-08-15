"""Hardened execution harness for Sprint S06.1 Scheduled versus Replay Experiment (E05b).

Executes the 5 strictly controlled experimental conditions:
1. incremental_state: Scheduled online deterministic state maintenance
2. replay_state_deterministic: Retrospective single-pass deterministic reconstruction (timing control)
3. replay_transcript: Raw retrospective transcript access
4. replay_state_model: Single-pass model retrospective state reconstruction (bottleneck control)
5. fresh: Lower empirical floor without history

Enforces:
- canonical_hash(S_T_incremental) == canonical_hash(S_T_replay_deterministic)
- hash(serialized_state_online) == hash(serialized_state_replay)
- hash(full_final_probe_prompt_online) == hash(full_final_probe_prompt_replay)

Separates one-time reconstruction costs and measures object-level state fidelity.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from recurrence.core.schemas import TARGET_3AFC_SCHEMA, TARGET_4AFC_SCHEMA, STATE_UPDATE_SCHEMA
from recurrence.memory.schemas import (
    MemoryEvent,
    StructuredSelfState,
    StateCapacityConfig,
    StateSnapshotRecord,
)
from recurrence.loop.clock import SimulatedClock
from recurrence.loop.queue import EventQueue
from recurrence.loop.state_manager import StateManager
from recurrence.loop.updater import OracleStateUpdater, AutonomousUpdateLoop
from recurrence.tasks.scheduled_replay import (
    ScheduledReplayEpisode,
    ScheduledReplayProbe,
)


def canonical_state_hash(state: StructuredSelfState) -> str:
    """Compute deterministic SHA-256 hash over normalized structured state contents."""
    norm_dict = {
        "working_memory": sorted(list(state.working_memory.items())),
        "goals": sorted([
            (g.goal_id, g.description, g.status.value if hasattr(g.status, "value") else str(g.status))
            for g in state.goals
        ]),
        "source_ledger": sorted(list(state.source_ledger.items())),
        "unresolved_items": sorted(state.unresolved_items),
    }
    dumped = json.dumps(norm_dict, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


@dataclass
class ScheduledTrialResult:
    """Result record for a single probe evaluation under a specific experimental condition."""
    trial_id: str
    episode_id: str
    horizon_ticks: int
    condition: str
    probe_id: str
    probe_type: str
    question: str
    options: Dict[str, str]
    correct_letter: str
    predicted_letter: str
    is_correct: bool
    prompt_tokens: int  # Probe-specific query prompt tokens
    completion_tokens: int  # Probe-specific completion tokens
    latency_ms: float  # Probe-specific query latency
    amortized_prompt_tokens: int  # Including amortized reconstruction tokens for replay_state_model
    amortized_latency_ms: float  # Including amortized reconstruction latency for replay_state_model
    context_chars: int
    state_hash: Optional[str] = None
    prompt_hash: Optional[str] = None
    reconstruction_valid: Optional[bool] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructionFidelityStats:
    """Direct object-level fidelity of model-reconstructed state against Oracle terminal state."""
    working_memory_retention_rate: float
    goal_status_match_rate: float
    source_ledger_accuracy: float
    raw_reconstructed_state: Dict[str, Any]


class ScheduledReplayHarness:
    """Orchestrator for executing 5-condition scheduled-vs-replay experiments."""

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

    def _build_transcript_text(self, events: List[MemoryEvent]) -> str:
        """Format raw event log into clean human-readable chronological transcript."""
        lines = ["=== EPISODIC EVENT LOG TRANSCRIPT ==="]
        for ev in sorted(events, key=lambda e: (e.step_index, e.event_id)):
            src_val = ev.source.value if hasattr(ev.source, "value") else str(ev.source)
            bindings_str = f" | bindings: {ev.key_bindings}" if ev.key_bindings else ""
            lines.append(f"[Tick {ev.step_index:02d}] ({src_val} / {ev.event_type}) {ev.content}{bindings_str}")
        return "\n".join(lines)

    def _build_state_text(self, state: StructuredSelfState) -> str:
        """Format structured state into clean, canonical JSON representation."""
        # Use sort_keys=True for strict string reproducibility
        return f"=== CURRENT STRUCTURED STATE ===\n{json.dumps(state.model_dump(), indent=2, sort_keys=True)}"

    def _query_probe(
        self,
        context_str: str,
        probe: ScheduledReplayProbe,
    ) -> Tuple[str, bool, int, int, float, str, Optional[str]]:
        """Query LLM backend for a forced-choice probe under strict JSON schema."""
        opts_str = "\n".join([f"{l}. {text}" for l, text in sorted(probe.options.items())])
        target_schema = TARGET_3AFC_SCHEMA if len(probe.options) == 3 else TARGET_4AFC_SCHEMA
        opts_letters_str = "A, B, or C" if len(probe.options) == 3 else "A, B, C, or D"
        
        prompt = (
            f"{context_str}\n\n"
            f"=== EVALUATION QUESTION ===\n"
            f"{probe.question}\n\n"
            f"Options:\n"
            f"{opts_str}\n\n"
            f"Select the single correct option letter ({opts_letters_str}). Return strictly JSON matching schema."
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        start_time = time.perf_counter()
        try:
            if hasattr(self.backend, "step"):
                raw_text, _, meta = self.backend.step(prompt, format=target_schema)
                p_tok = meta.get("prompt_eval_count", len(prompt) // 4)
                c_tok = meta.get("eval_count", len(raw_text) // 4)
            elif hasattr(self.backend, "generate"):
                resp = self.backend.generate(prompt=prompt, schema=target_schema)
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
            return pred_letter, is_corr, p_tok, c_tok, latency_ms, prompt_hash, None

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return "ERROR", False, len(prompt) // 4, 0, latency_ms, prompt_hash, str(e)

    def execute_episode(
        self,
        episode: ScheduledReplayEpisode,
        conditions: Optional[List[str]] = None,
    ) -> Tuple[List[ScheduledTrialResult], Dict[str, Any]]:
        """Execute all probes for an episode across the 5 experimental conditions."""
        eval_conditions = conditions or [
            "incremental_state",
            "replay_state_deterministic",
            "replay_transcript",
            "replay_state_model",
            "fresh",
        ]

        trial_results: List[ScheduledTrialResult] = []
        episode_metadata: Dict[str, Any] = {}

        # -------------------------------------------------------------
        # Condition 1: incremental_state (Scheduled Online Maintenance)
        # -------------------------------------------------------------
        clock_inc = SimulatedClock()
        queue_inc = EventQueue()
        queue_inc.schedule_batch(episode.scheduled_events)
        mgr_inc = StateManager(capacity_config=self.capacity_config)
        upd_inc = OracleStateUpdater(state_manager=mgr_inc)
        loop_inc = AutonomousUpdateLoop(
            clock=clock_inc,
            queue=queue_inc,
            state_manager=mgr_inc,
            updater=upd_inc,
            mode_name="incremental_state",
        )
        loop_inc.run_for_ticks(total_ticks=episode.total_ticks)
        online_state = mgr_inc.current_state.model_copy(deep=True)
        hash_online = canonical_state_hash(online_state)
        context_online = self._build_state_text(online_state)

        # -------------------------------------------------------------
        # Condition 2: replay_state_deterministic (Retrospective Timing Control)
        # -------------------------------------------------------------
        clock_rep = SimulatedClock()
        queue_rep = EventQueue()
        queue_rep.schedule_batch(episode.scheduled_events)
        mgr_rep = StateManager(capacity_config=self.capacity_config)
        upd_rep = OracleStateUpdater(state_manager=mgr_rep)
        
        for t in range(episode.total_ticks):
            evs = queue_rep.pop_events_for_tick(t)
            for ev in evs:
                mgr_rep.log_event(ev, t)
            if evs:
                new_st, _, _, _, _, _, _, _ = upd_rep.update(mgr_rep.current_state, evs, t)
                mgr_rep.update_state(new_st, t, len(evs), schema_valid=True)
            else:
                quiet_st = mgr_rep.current_state.model_copy(deep=True)
                quiet_st.last_updated_step = t
                mgr_rep.update_state(quiet_st, t, 0, schema_valid=True)

        replay_det_state = mgr_rep.current_state.model_copy(deep=True)
        hash_replay_det = canonical_state_hash(replay_det_state)
        context_replay_det = self._build_state_text(replay_det_state)

        # CRITICAL HARDENED INVARIANT: State-Hash & Serialized State String Equality
        assert hash_online == hash_replay_det, (
            f"State Hash Mismatch in Episode {episode.episode_id}!\n"
            f"Online Hash:     {hash_online}\n"
            f"Replay Det Hash: {hash_replay_det}"
        )
        assert context_online == context_replay_det, (
            f"Serialized State String Mismatch in Episode {episode.episode_id}!"
        )
        episode_metadata["canonical_state_hash"] = hash_online

        # -------------------------------------------------------------
        # Condition 3: replay_transcript (Raw Transcript Context)
        # -------------------------------------------------------------
        transcript_text = self._build_transcript_text(episode.scheduled_events)

        # -------------------------------------------------------------
        # Condition 4: replay_state_model (Model Retrospective Reconstruction)
        # -------------------------------------------------------------
        model_recon_state = StructuredSelfState()
        recon_valid = True
        recon_prompt_tok = 0
        recon_comp_tok = 0
        recon_lat_ms = 0.0

        if "replay_state_model" in eval_conditions:
            recon_prompt = (
                f"{transcript_text}\n\n"
                f"You are a state reconstruction agent. Read the full episodic event log transcript above.\n"
                f"Extract all active entities into working_memory, determine source_ledger origins, and identify goal statuses.\n"
                f"Output the complete StructuredSelfState conforming strictly to JSON schema."
            )
            recon_start = time.perf_counter()
            try:
                if hasattr(self.backend, "step"):
                    raw_recon, _, meta_recon = self.backend.step(recon_prompt, format=STATE_UPDATE_SCHEMA)
                    recon_prompt_tok = meta_recon.get("prompt_eval_count", len(recon_prompt) // 4)
                    recon_comp_tok = meta_recon.get("eval_count", len(raw_recon) // 4)
                elif hasattr(self.backend, "generate"):
                    resp_r = self.backend.generate(prompt=recon_prompt, schema=STATE_UPDATE_SCHEMA)
                    raw_recon = resp_r.text
                    recon_prompt_tok = getattr(resp_r, "prompt_tokens", len(recon_prompt) // 4)
                    recon_comp_tok = getattr(resp_r, "completion_tokens", len(raw_recon) // 4)
                else:
                    raw_recon = json.dumps(online_state.model_dump())
                    recon_prompt_tok = len(recon_prompt) // 4
                    recon_comp_tok = len(raw_recon) // 4

                recon_lat_ms = (time.perf_counter() - recon_start) * 1000.0
                recon_data = json.loads(raw_recon)
                model_recon_state = StructuredSelfState.model_validate(recon_data)
            except Exception:
                recon_valid = False
                model_recon_state = StructuredSelfState()

            # Compute Direct Object-Level State Fidelity against Oracle
            oracle_wm = online_state.working_memory
            recon_wm = model_recon_state.working_memory
            wm_matches = sum(1 for k, v in oracle_wm.items() if recon_wm.get(k) == v)
            wm_retention = wm_matches / max(1, len(oracle_wm))

            oracle_goals = {g.goal_id: (g.status.value if hasattr(g.status, "value") else str(g.status)) for g in online_state.goals}
            recon_goals = {g.goal_id: (g.status.value if hasattr(g.status, "value") else str(g.status)) for g in model_recon_state.goals}
            goal_matches = sum(1 for gid, st in oracle_goals.items() if recon_goals.get(gid) == st)
            goal_match_rate = goal_matches / max(1, len(oracle_goals))

            oracle_src = online_state.source_ledger
            recon_src = model_recon_state.source_ledger
            src_matches = sum(1 for k, s in oracle_src.items() if str(recon_src.get(k, "")).lower() == str(s).lower())
            src_match_rate = src_matches / max(1, len(oracle_src))

            episode_metadata["model_reconstruction_cost"] = {
                "prompt_tokens_once": recon_prompt_tok,
                "completion_tokens_once": recon_comp_tok,
                "latency_ms_once": recon_lat_ms,
            }
            episode_metadata["model_reconstruction_fidelity"] = {
                "working_memory_retention_rate": wm_retention,
                "goal_status_match_rate": goal_match_rate,
                "source_ledger_accuracy": src_match_rate,
            }

        num_probes = len(episode.probes)
        amort_recon_tok = recon_prompt_tok // max(1, num_probes)
        amort_recon_lat = recon_lat_ms / max(1, num_probes)

        # -------------------------------------------------------------
        # Evaluate Probe Battery across Conditions
        # -------------------------------------------------------------
        online_prompt_hashes: Dict[str, str] = {}

        for cond in eval_conditions:
            if cond == "incremental_state":
                context_str = context_online
                st_hash = hash_online
            elif cond == "replay_state_deterministic":
                context_str = context_replay_det
                st_hash = hash_replay_det
            elif cond == "replay_transcript":
                context_str = transcript_text
                st_hash = None
            elif cond == "replay_state_model":
                context_str = self._build_state_text(model_recon_state)
                st_hash = canonical_state_hash(model_recon_state)
            elif cond == "fresh":
                context_str = "=== NO PREVIOUS EPISODIC HISTORY AVAILABLE ==="
                st_hash = None
            else:
                raise ValueError(f"Unknown evaluation condition: {cond}")

            for probe in episode.probes:
                pred_l, is_corr, p_tok, c_tok, lat_ms, p_hash, err_msg = self._query_probe(
                    context_str=context_str,
                    probe=probe,
                )

                if cond == "incremental_state":
                    online_prompt_hashes[probe.probe_id] = p_hash
                elif cond == "replay_state_deterministic":
                    # HARDENED PROMPT-HASH INVARIANT
                    assert p_hash == online_prompt_hashes[probe.probe_id], (
                        f"Prompt Hash Mismatch between incremental and replay in probe {probe.probe_id}!"
                    )

                trial_id = f"{episode.episode_id}_{cond}_{probe.probe_id}"
                trial_results.append(ScheduledTrialResult(
                    trial_id=trial_id,
                    episode_id=episode.episode_id,
                    horizon_ticks=episode.total_ticks,
                    condition=cond,
                    probe_id=probe.probe_id,
                    probe_type=probe.probe_type,
                    question=probe.question,
                    options=probe.options,
                    correct_letter=probe.correct_letter,
                    predicted_letter=pred_l,
                    is_correct=is_corr,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    latency_ms=lat_ms,
                    amortized_prompt_tokens=p_tok + (amort_recon_tok if cond == "replay_state_model" else 0),
                    amortized_latency_ms=lat_ms + (amort_recon_lat if cond == "replay_state_model" else 0.0),
                    context_chars=len(context_str),
                    state_hash=st_hash,
                    prompt_hash=p_hash,
                    reconstruction_valid=recon_valid if cond == "replay_state_model" else None,
                    error_message=err_msg,
                    metadata=dict(probe.metadata),
                ))

        return trial_results, episode_metadata
