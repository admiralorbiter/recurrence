"""Autonomous update loop primitives for Horizon 1 (Sprint S05)."""

from recurrence.loop.clock import SimulatedClock
from recurrence.loop.queue import EventQueue, ScheduledEvent
from recurrence.loop.state_manager import (
    ImmutableEventLog,
    StateManager,
)
from recurrence.loop.updater import (
    OracleStateUpdater,
    ModelStateUpdater,
    AutonomousUpdateLoop,
)

__all__ = [
    "SimulatedClock",
    "EventQueue",
    "ScheduledEvent",
    "ImmutableEventLog",
    "StateManager",
    "OracleStateUpdater",
    "ModelStateUpdater",
    "AutonomousUpdateLoop",
]
