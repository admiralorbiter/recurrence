"""State updater implementations (Oracle, Model, Replay) and the AutonomousUpdateLoop engine."""

import json
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple
from recurrence.core.schemas import STATE_UPDATE_SCHEMA
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
    ) -> Tuple[StructuredSelfState, bool, int, int, float, Optional[str]]:
        """Compute updated state given previous state and newly arrived events.
        
        Returns:
            (new_state, schema_valid, prompt_tokens, completion_tokens, latency_ms, error_message)
        """
        ...


class OracleStateUpdater:
    """Deterministic, programmatic state updater deriving ground truth transitions from event metadata."""

    def update(
        self,
        prev_state: StructuredSelfState,
        events: List[MemoryEvent],
        tick: int,
    ) -> Tuple[StructuredSelfState, bool, int, int, float, Optional[str]]:
        start_time = time.perf_counter()
        # Deep copy previous state
        state = prev_state.model_copy(deep=True)
        
        for ev in events:
            # 1. Apply key-value bindings to working memory and source ledger
            if ev.key_bindings:
                for k, v in ev.key_bindings.items():
                    state.working_memory[k] = v
                    state.source_ledger[k] = ev.source.value

            # 2. Goal tracking
            if ev.event_type == "goal_update" or "goal_id" in ev.metadata:
                gid = ev.metadata.get("goal_id", f"goal_{ev.event_id}")
                desc = ev.metadata.get("goal_description", ev.content)
                status = ev.metadata.get("goal_status", "active")
                
                # Check if goal already exists
                existing = [g for g in state.goals if g.goal_id == gid]
                if existing:
                    existing[0].status = status
                    existing[0].updated_at_step = tick
                    if "goal_description" in ev.metadata:
                        existing[0].description = desc
                else:
                    state.goals.append(GoalState(
                        goal_id=gid,
                        description=desc,
                        status=status,
                        created_at_step=tick,
                        updated_at_step=tick,
                    ))

        # 3. Update unresolved items from goal list
        state.unresolved_items = [
            g.goal_id for g in state.goals if g.status in ("pending", "suspended")
        ]
        state.last_updated_step = tick
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return state, True, 0, 0, latency_ms, None


class ModelStateUpdater:
    """Autonomous LLM state updater using native JSON Schema decoding."""

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
    ) -> Tuple[StructuredSelfState, bool, int, int, float, Optional[str]]:
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

            # Parse JSON
            raw_text = raw_text.strip()
            parsed_json = json.loads(raw_text)

            # Reconstruct goals with timestamps
            goals_list: List[GoalState] = []
            for g_dict in parsed_json.get("goals", []):
                # Find matching prev goal for created_at_step
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
            return updated_state, True, prompt_tokens, completion_tokens, latency_ms, None

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            # Safe Fallback on schema parse or network error: retain prev state
            fallback_state = prev_state.model_copy(deep=True)
            fallback_state.last_updated_step = tick
            return fallback_state, False, len(prompt) // 4, 0, latency_ms, str(e)


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

    def step(self) -> Optional[StateSnapshotRecord]:
        """Execute a single discrete logical tick step."""
        current_tick = self.clock.current_tick
        
        # 1. Pop all events scheduled for <= current tick
        events = self.queue.pop_events_for_tick(current_tick)
        
        # 2. Log events to audit log
        for ev in events:
            self.state_manager.log_event(ev, current_tick)

        # 3. If events arrived, compute updated state
        if events:
            prev_state = self.state_manager.current_state
            (
                new_state,
                schema_valid,
                prompt_tok,
                comp_tok,
                latency_ms,
                err_msg,
            ) = self.updater.update(prev_state, events, current_tick)

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
            )
        else:
            snapshot = None

        # 4. Advance clock by 1 tick
        self.clock.advance(1)
        return snapshot

    def run_until_complete(self, max_ticks: int = 1000) -> List[StateSnapshotRecord]:
        """Run loop until queue is empty or max_ticks is reached."""
        snapshots: List[StateSnapshotRecord] = []
        ticks_executed = 0

        while self.queue.has_pending_events() and ticks_executed < max_ticks:
            snap = self.step()
            if snap:
                snapshots.append(snap)
            ticks_executed += 1

        return snapshots
