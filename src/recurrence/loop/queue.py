"""Event queue and scheduling engine for the autonomous update loop."""

import heapq
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
from recurrence.memory.schemas import MemoryEvent


@dataclass(order=True)
class ScheduledEvent:
    """An event scheduled to be dispatched at a discrete logical tick."""
    tick: int
    priority: int = field(default=0)  # Lower number = higher priority
    counter: int = field(default=0)   # Tie-breaker for stable ordering
    event: MemoryEvent = field(compare=False, default=None)  # type: ignore


class EventQueue:
    """Priority event queue dispatching events at scheduled discrete clock ticks."""

    def __init__(self) -> None:
        self._heap: List[ScheduledEvent] = []
        self._counter: int = 0

    def schedule(self, event: MemoryEvent, tick: int, priority: int = 0) -> None:
        """Schedule a MemoryEvent to be delivered at a specific clock tick."""
        if tick < 0:
            raise ValueError(f"Cannot schedule event at negative tick: {tick}")
        self._counter += 1
        scheduled = ScheduledEvent(
            tick=tick,
            priority=priority,
            counter=self._counter,
            event=event,
        )
        heapq.heappush(self._heap, scheduled)

    def schedule_batch(self, items: List[Any]) -> None:
        """Schedule multiple MemoryEvent objects or (event, tick, priority) tuples."""
        for item in items:
            if isinstance(item, MemoryEvent):
                self.schedule(item, tick=item.step_index, priority=0)
            elif isinstance(item, (tuple, list)):
                event = item[0]
                tick = item[1]
                priority = item[2] if len(item) > 2 else 0
                self.schedule(event, tick, priority)
            else:
                raise TypeError(f"Unsupported event item type: {type(item)}")

    def pop_events_for_tick(self, tick: int) -> List[MemoryEvent]:
        """Pop and return all events scheduled for <= current tick in priority order."""
        dispatched: List[MemoryEvent] = []
        while self._heap and self._heap[0].tick <= tick:
            scheduled = heapq.heappop(self._heap)
            dispatched.append(scheduled.event)
        return dispatched

    def peek_next_tick(self) -> Optional[int]:
        """Return the tick of the earliest scheduled event without popping, or None."""
        if not self._heap:
            return None
        return self._heap[0].tick

    def has_pending_events(self) -> bool:
        """Return True if any events remain in the queue."""
        return len(self._heap) > 0

    @property
    def pending_count(self) -> int:
        """Total number of events pending delivery in the queue."""
        return len(self._heap)

    def clear(self) -> None:
        """Clear all pending events."""
        self._heap.clear()
        self._counter = 0

    def __repr__(self) -> str:
        return f"<EventQueue pending_count={len(self._heap)} next_tick={self.peek_next_tick()}>"
