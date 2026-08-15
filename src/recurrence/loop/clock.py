"""Simulated discrete logical clock for autonomous update loop execution."""

from datetime import datetime, timezone, timedelta
from typing import Optional


class SimulatedClock:
    """Discrete logical tick clock providing deterministic temporal pacing for the loop."""

    def __init__(
        self,
        start_tick: int = 0,
        tick_duration_ms: int = 1000,
        base_datetime: Optional[datetime] = None,
    ):
        self._current_tick = start_tick
        self._tick_duration_ms = tick_duration_ms
        self._base_datetime = base_datetime or datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    @property
    def current_tick(self) -> int:
        """Get the current discrete logical tick."""
        return self._current_tick

    @property
    def tick_duration_ms(self) -> int:
        """Duration represented by one logical tick in milliseconds."""
        return self._tick_duration_ms

    def advance(self, ticks: int = 1) -> int:
        """Advance clock by N ticks (default 1) and return updated tick."""
        if ticks < 0:
            raise ValueError(f"Cannot advance clock by negative ticks: {ticks}")
        self._current_tick += ticks
        return self._current_tick

    def reset(self, start_tick: int = 0) -> None:
        """Reset logical clock to start_tick."""
        self._current_tick = start_tick

    def get_simulated_time(self) -> datetime:
        """Return the calculated datetime corresponding to the current tick."""
        delta = timedelta(milliseconds=self._current_tick * self._tick_duration_ms)
        return self._base_datetime + delta

    def get_timestamp_str(self) -> str:
        """Return formatted ISO timestamp string for current tick."""
        return self.get_simulated_time().isoformat()

    def __repr__(self) -> str:
        return f"<SimulatedClock tick={self._current_tick} ({self.get_timestamp_str()})>"
