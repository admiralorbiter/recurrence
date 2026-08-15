"""State updater implementations (Oracle, Full State, Delta Model) and AutonomousUpdateLoop engine."""

import json
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple
from recurrence.core.schemas import STATE_UPDATE_SCHEMA, STATE_DELTA_SCHEMA
from recurrence.loop.clock import SimulatedClock
from recurrence.loop.queue import EventQueue
from recurrence.loop.state_manager import StateManager
from recurrence.memory.schemas import (
    MemoryEvent,
    GoalState,
    StructuredSelfState,
    StateSnapshotRecord,
)


class StateUpdaterProtocol(Protocol):
    """Protocol for state update strategies."""

    def update(
        self,
        prev_state: StructuredSelfState,
        events: List[MemoryEvent],
        tick: int,
    ) -> Tuple[StructuredSelfState, bool, int, int, float, Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """Compute updated state given previous state and newly arrived events.
        
        Returns:
            (new_state, schema_valid, prompt_tokens, completion_tokens, latency_ms, error_message, raw_response, parsed_data)
        """
        ...


class OracleStateUpdater:
    """Deterministic, programmatic state updater deriving ground truth transitions from event metadata."""

    def __init__(self, state_manager: Optional[StateManager] = None) -> None:
        self.state_manager = state_manager or StateManager()

    def update(
        self,
        prev_state: StructuredSelfState,
        events: List[MemoryEvent],
        tick: int,
    ) -> Tuple[StructuredSelfState, bool, int, int, float, Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        start_time = time.perf_counter()
        
        # Build programmatic delta
        wm_upserts: Dict[str, str] = {}
        src_upserts: Dict[str, str] = {}
        goal_updates: List[Dict[str, Any]] = []

        for ev in events:
            if ev.key_bindings:
                for k, v in ev.key_bindings.items():
                    wm_upserts[k] = v
                    src_upserts[k] = ev.source.value

            if ev.event_type == "goal_update" or "goal_id" in ev.metadata:
                gid = ev.metadata.get("goal_id", f"goal_{ev.event_id}")
                desc = ev.metadata.get("goal_description", ev.content)
                status = ev.metadata.get("goal_status", "active")
                goal_updates.append({
                    "goal_id": gid,
                    "description": desc,
                    "status": status,
                })

        delta_payload = {
            "working_memory_upserts": wm_upserts,
            "working_memory_deletions": [],
            "source_upserts": src_upserts,
            "goal_updates": goal_updates,
            "unresolved_items_add": [],
            "unresolved_items_remove": [],
        }

        new_state, warnings = self.state_manager.apply_delta(prev_state, delta_payload, tick)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        err_msg = "; ".join(warnings) if warnings else None

        return new_state, True, 0, 0, latency_ms, err_msg, json.dumps(delta_payload), delta_payload


class FullModelStateUpdater:
    """Naïve LLM state updater attempting full world-state regeneration at every tick (E04a Scout)."""

    def __init__(
        self,
        backend: Any,
        system_instructions: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.system_instructions = system_instructions or (
            "You are the internal state maintenance engine of an autonomous system.\n"
            "At each discrete time tick, update the system's structured self-state to incorporate new events.\n"
            "Maintain active entity key-values in working_memory, register all goals and their current statuses,\n"
            "record entity sources in source_ledger ('environment', 'self', 'experimenter'),\n"
            "and list pending/suspended goal IDs in unresolved_items.\n"
            "Do NOT hallucinate or drop active entities."
        )

    def _build_update_prompt(
        self,
        prev_state: StructuredSelfState,
        events: List[MemoryEvent],
        tick: int,
    ) -> str:
        state_json = json.dumps(prev_state.model_dump(), indent=2)
        events_str = ""
        for i, ev in enumerate(events, 1):
            bindings_str = f" | Bindings: {ev.key_bindings}" if ev.key_bindings else ""
            events_str += f"- [{ev.source.value.upper()} | {ev.event_type}] {ev.content}{bindings_str}\n"

        prompt = (
            f"{self.system_instructions}\n\n"
            f"=== CURRENT LOGICAL TICK: {tick} ===\n\n"
            f"--- PREVIOUS STRUCTURED STATE ---\n"
            f"{state_json}\n\n"
            f"--- NEWLY ARRIVED EVENTS (TICK {tick}) ---\n"
            f"{events_str if events_str else '(No new events)'}\n\n"
            f"Output the complete updated JSON structured state conforming to schema."
        )
        return prompt

    def update(
        self,
        prev_state: StructuredSelfState,
        events: List[MemoryEvent],
        tick: int,
    ) -> Tuple[StructuredSelfState, bool, int, int, float, Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        start_time = time.perf_counter()
        prompt = self._build_update_prompt(prev_state, events, tick)

        try:
            if hasattr(self.backend, "step"):
                raw_text, _, meta = self.backend.step(prompt, format=STATE_UPDATE_SCHEMA)
                prompt_tokens = meta.get("prompt_eval_count", len(prompt) // 4)
                completion_tokens = meta.get("eval_count", len(raw_text) // 4)
            elif hasattr(self.backend, "generate"):
                resp = self.backend.generate(
                    prompt=prompt,
                    schema=STATE_UPDATE_SCHEMA,
                    temperature=0.0,
                )
                raw_text = resp.text
                prompt_tokens = resp.prompt_tokens or (len(prompt) // 4)
                completion_tokens = resp.completion_tokens or (len(resp.text) // 4)
            elif hasattr(self.backend, "chat"):
                raw_text, meta = self.backend.chat(
                    messages=[{"role": "user", "content": prompt}],
                    format=STATE_UPDATE_SCHEMA,
                    temperature=0.0,
                )
                prompt_tokens = meta.get("prompt_eval_count", len(prompt) // 4)
                completion_tokens = meta.get("eval_count", len(raw_text) // 4)
            else:
                raise TypeError(f"Backend {type(self.backend)} does not support step(), generate(), or chat()")

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            raw_text = raw_text.strip()
            parsed_json = json.loads(raw_text)

            goals_list: List[GoalState] = []
            for g_dict in parsed_json.get("goals", []):
                prev_match = [g for g in prev_state.goals if g.goal_id == g_dict.get("goal_id")]
                created_step = prev_match[0].created_at_step if prev_match else tick
                goals_list.append(GoalState(
                    goal_id=str(g_dict.get("goal_id", "")),
                    description=str(g_dict.get("description", "")),
                    status=g_dict.get("status", "active"),
                    created_at_step=created_step,
                    updated_at_step=tick,
                ))

            updated_state = StructuredSelfState(
                working_memory={str(k): str(v) for k, v in parsed_json.get("working_memory", {}).items()},
                goals=goals_list,
                source_ledger={str(k): str(v) for k, v in parsed_json.get("source_ledger", {}).items()},
                unresolved_items=[str(item) for item in parsed_json.get("unresolved_items", [])],
                last_updated_step=tick,
            )
            return updated_state, True, prompt_tokens, completion_tokens, latency_ms, None, raw_text, parsed_json

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            fallback_state = prev_state.model_copy(deep=True)
            fallback_state.last_updated_step = tick
            return fallback_state, False, len(prompt) // 4, 0, latency_ms, str(e), None, None


class DeltaModelStateUpdater:
    """Scaffolded autonomous LLM state updater emitting structured deltas merged deterministically (S05.1)."""

    def __init__(
        self,
        backend: Any,
        state_manager: Optional[StateManager] = None,
        system_instructions: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.state_manager = state_manager or StateManager()
        self.system_instructions = system_instructions or (
            "You are the delta state updater for an autonomous recurrence agent.\n"
            "At each discrete tick, examine the current structured state and newly arrived events.\n"
            "Output ONLY the state changes (delta) to apply:\n"
            "- working_memory_upserts: map of entity key to value string for newly asserted or updated entities\n"
            "- working_memory_deletions: list of keys to remove if an entity is deleted\n"
            "- source_upserts: map of updated keys to their event source ('environment', 'self', 'experimenter')\n"
            "- goal_updates: list of goal status updates (goal_id, description, status in 'pending'/'active'/'suspended'/'completed')\n"
            "- unresolved_items_add: list of unresolved task descriptions or IDs to track\n"
            "- unresolved_items_remove: list of unresolved items to clear upon resolution\n"
            "Do NOT hallucinate keys not present in the incoming events. Preserve prior state implicitly."
        )

    def _build_delta_prompt(
        self,
        prev_state: StructuredSelfState,
        events: List[MemoryEvent],
        tick: int,
    ) -> str:
        state_json = json.dumps(prev_state.model_dump(), indent=2)
        events_json = json.dumps([
            {
                "event_id": ev.event_id,
                "source": ev.source.value,
                "event_type": ev.event_type,
                "content": ev.content,
                "key_bindings": ev.key_bindings,
                "metadata": ev.metadata,
            }
            for ev in events
        ], indent=2)

        prompt = (
            f"{self.system_instructions}\n\n"
            f"=== CURRENT LOGICAL TICK: {tick} ===\n\n"
            f"--- PREVIOUS STRUCTURED STATE ---\n"
            f"{state_json}\n\n"
            f"--- NEWLY ARRIVED STRUCTURED EVENTS (TICK {tick}) ---\n"
            f"{events_json if events else '[]'}\n\n"
            f"Output the JSON state delta conforming to schema."
        )
        return prompt

    def update(
        self,
        prev_state: StructuredSelfState,
        events: List[MemoryEvent],
        tick: int,
    ) -> Tuple[StructuredSelfState, bool, int, int, float, Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        start_time = time.perf_counter()
        prompt = self._build_delta_prompt(prev_state, events, tick)

        try:
            if hasattr(self.backend, "step"):
                raw_text, _, meta = self.backend.step(prompt, format=STATE_DELTA_SCHEMA)
                prompt_tokens = meta.get("prompt_eval_count", len(prompt) // 4)
                completion_tokens = meta.get("eval_count", len(raw_text) // 4)
            elif hasattr(self.backend, "generate"):
                resp = self.backend.generate(
                    prompt=prompt,
                    schema=STATE_DELTA_SCHEMA,
                    temperature=0.0,
                )
                raw_text = resp.text
                prompt_tokens = resp.prompt_tokens or (len(prompt) // 4)
                completion_tokens = resp.completion_tokens or (len(resp.text) // 4)
            elif hasattr(self.backend, "chat"):
                raw_text, meta = self.backend.chat(
                    messages=[{"role": "user", "content": prompt}],
                    format=STATE_DELTA_SCHEMA,
                    temperature=0.0,
                )
                prompt_tokens = meta.get("prompt_eval_count", len(prompt) // 4)
                completion_tokens = meta.get("eval_count", len(raw_text) // 4)
            else:
                raise TypeError(f"Backend {type(self.backend)} does not support step(), generate(), or chat()")

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            raw_text = raw_text.strip()
            parsed_delta = json.loads(raw_text)

            # Apply deterministic delta merge via StateManager
            updated_state, warnings = self.state_manager.apply_delta(prev_state, parsed_delta, tick)
            err_msg = "; ".join(warnings) if warnings else None

            return updated_state, True, prompt_tokens, completion_tokens, latency_ms, err_msg, raw_text, parsed_delta

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            fallback_state = prev_state.model_copy(deep=True)
            fallback_state.last_updated_step = tick
            return fallback_state, False, len(prompt) // 4, 0, latency_ms, str(e), None, None


# Backwards compatibility alias
ModelStateUpdater = DeltaModelStateUpdater


class AutonomousUpdateLoop:
    """Main loop engine advancing clock, dispatching events, and invoking state updater."""

    def __init__(
        self,
        clock: SimulatedClock,
        queue: EventQueue,
        state_manager: StateManager,
        updater: Any,
        mode_name: str = "oracle",
    ) -> None:
        self.clock = clock
        self.queue = queue
        self.state_manager = state_manager
        self.updater = updater
        self.mode_name = mode_name
        self.state_traces: List[Dict[str, Any]] = []

    def step(self) -> StateSnapshotRecord:
        """Execute a single discrete logical tick step, handling both event-arrival and quiet ticks."""
        current_tick = self.clock.current_tick
        
        # 1. Pop all events scheduled for <= current tick
        events = self.queue.pop_events_for_tick(current_tick)
        
        # 2. Log events to audit log
        for ev in events:
            self.state_manager.log_event(ev, current_tick)

        prev_state = self.state_manager.current_state.model_copy(deep=True)

        # 3. If events arrived, compute updated state via updater
        if events:
            (
                new_state,
                schema_valid,
                prompt_tok,
                comp_tok,
                latency_ms,
                err_msg,
                raw_resp,
                parsed_data,
            ) = self.updater.update(prev_state, events, current_tick)

            explicit_keys = list(new_state.working_memory.keys()) if hasattr(self.updater, "backend") else None
            snapshot = self.state_manager.update_state(
                new_state=new_state,
                tick=current_tick,
                incoming_event_count=len(events),
                schema_valid=schema_valid,
                prompt_tokens=prompt_tok,
                completion_tokens=comp_tok,
                latency_ms=latency_ms,
                updater_mode=self.mode_name,
                error_message=err_msg,
                explicit_written_keys=explicit_keys,
            )
            raw_response_str = raw_resp
            parsed_data_obj = parsed_data
        else:
            # 4. Quiet Tick: Identity / No-Op state preservation
            quiet_state = prev_state.model_copy(deep=True)
            quiet_state.last_updated_step = current_tick
            snapshot = self.state_manager.update_state(
                new_state=quiet_state,
                tick=current_tick,
                incoming_event_count=0,
                schema_valid=True,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0.0,
                updater_mode=self.mode_name,
                error_message=None,
            )
            schema_valid = True
            err_msg = None
            raw_response_str = "(quiet_tick_identity_noop)"
            parsed_data_obj = {}

        # 5. Record full state trace for auditing
        self.state_traces.append({
            "tick": current_tick,
            "updater_mode": self.mode_name,
            "previous_state": prev_state.model_dump(),
            "incoming_events": [ev.model_dump() for ev in events],
            "raw_model_response": raw_response_str,
            "parsed_delta_or_state": parsed_data_obj,
            "resulting_state": self.state_manager.current_state.model_dump(),
            "schema_valid": schema_valid,
            "error_message": err_msg,
        })

        # 6. Advance clock by 1 tick
        self.clock.advance(1)
        return snapshot

    def run_for_ticks(self, total_ticks: int) -> List[StateSnapshotRecord]:
        """Run update loop for a fixed number of logical ticks, evaluating every tick (including quiet ticks)."""
        snapshots: List[StateSnapshotRecord] = []
        for _ in range(total_ticks):
            snap = self.step()
            snapshots.append(snap)
        return snapshots

    def run_until_complete(self, max_ticks: int = 1000) -> List[StateSnapshotRecord]:
        """Run loop until queue is empty or max_ticks is reached."""
        snapshots: List[StateSnapshotRecord] = []
        ticks_executed = 0

        while self.queue.has_pending_events() and ticks_executed < max_ticks:
            snap = self.step()
            snapshots.append(snap)
            ticks_executed += 1

        return snapshots
