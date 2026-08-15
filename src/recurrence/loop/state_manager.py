"""State manager, immutable audit event log, and versioned snapshot history for S05."""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple
from recurrence.memory.schemas import (
    MemoryEvent,
    GoalState,
    StructuredSelfState,
    StateCapacityConfig,
    StateSnapshotRecord,
)


class ImmutableEventLog:
    """Tamper-evident append-only event log with SHA-256 cryptographic hash chaining."""

    def __init__(self, genesis_hash: str = "0" * 64) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._latest_hash: str = genesis_hash

    @property
    def latest_hash(self) -> str:
        """Current tip SHA-256 digest of the event chain."""
        return self._latest_hash

    @property
    def entry_count(self) -> int:
        """Number of appended events in the log."""
        return len(self._entries)

    def append(self, event: MemoryEvent, tick: int) -> str:
        """Append an event with link to previous hash, return new entry hash."""
        event_dict = event.model_dump()
        payload = {
            "index": len(self._entries),
            "tick": tick,
            "prev_hash": self._latest_hash,
            "event": event_dict,
        }
        canonical_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        entry_hash = hashlib.sha256(canonical_bytes).hexdigest()
        
        self._entries.append({
            "payload": payload,
            "hash": entry_hash,
        })
        self._latest_hash = entry_hash
        return entry_hash

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """Verify the cryptographic hash chain from genesis to tip."""
        expected_prev = "0" * 64
        for idx, entry in enumerate(self._entries):
            payload = entry["payload"]
            recorded_hash = entry["hash"]
            if payload["prev_hash"] != expected_prev:
                return False, f"Broken chain link at index {idx}: expected {expected_prev}, got {payload['prev_hash']}"
            
            canonical_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
            calculated_hash = hashlib.sha256(canonical_bytes).hexdigest()
            if calculated_hash != recorded_hash:
                return False, f"Tampered entry at index {idx}: calculated {calculated_hash} != recorded {recorded_hash}"
            
            expected_prev = recorded_hash

        return True, None

    def get_raw_events(self) -> List[MemoryEvent]:
        """Return list of MemoryEvents in chronological order."""
        return [MemoryEvent(**e["payload"]["event"]) for e in self._entries]


class StateManager:
    """Manages explicit StructuredSelfState, capacity bounding, and versioned snapshots."""

    VALID_GOAL_TRANSITIONS = {
        "pending": {"active", "suspended", "completed"},
        "active": {"suspended", "completed"},
        "suspended": {"active", "completed"},
        "completed": set(),  # Terminal state
    }

    def __init__(
        self,
        capacity_config: Optional[StateCapacityConfig] = None,
        initial_state: Optional[StructuredSelfState] = None,
    ) -> None:
        self.capacity = capacity_config or StateCapacityConfig()
        self._current_state = initial_state or StructuredSelfState()
        self._event_log = ImmutableEventLog()
        self._snapshots: List[StateSnapshotRecord] = []
        self._access_order: List[str] = list(self._current_state.working_memory.keys())

    @property
    def current_state(self) -> StructuredSelfState:
        """Get the current structured self-state."""
        return self._current_state

    @property
    def event_log(self) -> ImmutableEventLog:
        """Audit log of all events processed by the state manager."""
        return self._event_log

    @property
    def snapshots(self) -> List[StateSnapshotRecord]:
        """History of tick-by-tick state snapshots."""
        return self._snapshots

    def log_event(self, event: MemoryEvent, tick: int) -> str:
        """Append an incoming event to the immutable log."""
        return self._event_log.append(event, tick)

    def apply_capacity_bounds(self, state: StructuredSelfState) -> StructuredSelfState:
        """Enforce maximum item limits on working memory, goals, and unresolved items."""
        # 1. Bounded Working Memory (LRU eviction based on access order)
        if len(state.working_memory) > self.capacity.max_working_memory_items:
            # Keep the most recently accessed items
            keys_to_keep = set(self._access_order[-self.capacity.max_working_memory_items:])
            # If access order doesn't cover all keys, keep arbitrary trailing keys
            if len(keys_to_keep) < self.capacity.max_working_memory_items:
                remaining = [k for k in state.working_memory if k not in keys_to_keep]
                for k in remaining:
                    if len(keys_to_keep) >= self.capacity.max_working_memory_items:
                        break
                    keys_to_keep.add(k)
            
            state.working_memory = {
                k: v for k, v in state.working_memory.items() if k in keys_to_keep
            }
            # Also clean source ledger for evicted keys
            state.source_ledger = {
                k: v for k, v in state.source_ledger.items() if k in keys_to_keep
            }

        # 2. Bounded Goals (Keep active/suspended goals preferentially over completed)
        if len(state.goals) > self.capacity.max_goals:
            active_goals = [g for g in state.goals if g.status in ("active", "suspended")]
            other_goals = [g for g in state.goals if g.status not in ("active", "suspended")]
            
            combined = (active_goals + other_goals)[:self.capacity.max_goals]
            state.goals = combined

        # 3. Bounded Unresolved Items
        if len(state.unresolved_items) > self.capacity.max_unresolved_items:
            state.unresolved_items = state.unresolved_items[-self.capacity.max_unresolved_items:]

        return state

    def update_state(
        self,
        new_state: StructuredSelfState,
        tick: int,
        incoming_event_count: int = 0,
        schema_valid: bool = True,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        updater_mode: str = "oracle",
        error_message: Optional[str] = None,
    ) -> StateSnapshotRecord:
        """Apply new state, enforce capacity bounds, touch access order, and record snapshot."""
        # Update internal access tracking
        for k in new_state.working_memory.keys():
            if k in self._access_order:
                self._access_order.remove(k)
            self._access_order.append(k)

        # Enforce capacity bounds
        bounded_state = self.apply_capacity_bounds(new_state)
        bounded_state.last_updated_step = tick
        self._current_state = bounded_state

        snapshot = StateSnapshotRecord(
            tick=tick,
            state=bounded_state.model_copy(deep=True),
            incoming_event_count=incoming_event_count,
            schema_valid=schema_valid,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            updater_mode=updater_mode,
            error_message=error_message,
        )
        self._snapshots.append(snapshot)
        return snapshot

    def get_snapshot_at_tick(self, tick: int) -> Optional[StateSnapshotRecord]:
        """Retrieve state snapshot recorded at a specific tick."""
        for snap in self._snapshots:
            if snap.tick == tick:
                return snap
        return None
