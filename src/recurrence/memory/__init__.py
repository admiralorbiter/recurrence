"""Level 1 explicit memory module."""

from recurrence.memory.schemas import (
    ConsolidationRecord,
    EventSource,
    GoalState,
    MemoryEvent,
    MemoryFormat,
    StructuredSelfState,
)
from recurrence.memory.base import BaseMemoryAdapter
from recurrence.memory.adapters import (
    CombinedStateAdapter,
    DeterministicSummaryAdapter,
    FreshAdapter,
    ModelSummaryAdapter,
    StructuredStateAdapter,
    TranscriptAdapter,
    get_memory_adapter,
)

__all__ = [
    "ConsolidationRecord",
    "EventSource",
    "GoalState",
    "MemoryEvent",
    "MemoryFormat",
    "StructuredSelfState",
    "BaseMemoryAdapter",
    "CombinedStateAdapter",
    "DeterministicSummaryAdapter",
    "FreshAdapter",
    "ModelSummaryAdapter",
    "StructuredStateAdapter",
    "TranscriptAdapter",
    "get_memory_adapter",
]
