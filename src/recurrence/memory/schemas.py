"""Pydantic schemas for Level 1 explicit memory and structured state representation."""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class MemoryFormat(str, Enum):
    """The 6 standard Level 1 explicit memory conditions."""
    FRESH = "fresh"
    TRANSCRIPT = "transcript"
    DETERMINISTIC_SUMMARY = "deterministic_summary"
    MODEL_SUMMARY = "model_summary"
    STRUCTURED_STATE = "structured_state"
    COMBINED = "combined"


class EventSource(str, Enum):
    """Source origin of an event in the episodic stream."""
    ENVIRONMENT = "environment"
    SELF = "self"
    EXPERIMENTER = "experimenter"


class MemoryEvent(BaseModel):
    """A single discrete event in the episodic stream."""
    event_id: str
    step_index: int
    source: EventSource
    event_type: str = Field(description="e.g. observation, action, statement, goal_update, distractor")
    content: str
    key_bindings: Dict[str, str] = Field(default_factory=dict, description="Structured key-value bindings asserted in this event")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GoalState(BaseModel):
    """Explicit goal tracking representation."""
    goal_id: str
    description: str
    status: Literal["pending", "active", "completed", "suspended"] = "pending"
    created_at_step: int
    updated_at_step: int


class StructuredSelfState(BaseModel):
    """Typed Level 1 explicit self-state schema."""
    working_memory: Dict[str, str] = Field(
        default_factory=dict,
        description="Key-value entity bindings currently active in working memory"
    )
    goals: List[GoalState] = Field(
        default_factory=list,
        description="Explicit goal registry"
    )
    source_ledger: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from entity/assertion key to origin source (environment, self, experimenter)"
    )
    unresolved_items: List[str] = Field(
        default_factory=list,
        description="List of suspended or pending task IDs"
    )
    derived_inferences: Dict[str, str] = Field(
        default_factory=dict,
        description="Consolidated multi-hop deductions and derived hypotheses"
    )
    last_updated_step: int = 0


class ConsolidationRecord(BaseModel):
    """Record of an offline LLM consolidation step."""
    source_event_count: int
    raw_event_digest: str
    model_name: str
    summary_text: str
    prompt_tokens: int
    completion_tokens: int
    created_at: str


class StateCapacityConfig(BaseModel):
    """Configuration limits on explicit self-state capacity to prevent unbounded expansion."""
    max_working_memory_items: int = 16
    max_goals: int = 8
    max_unresolved_items: int = 16


class StateSnapshotRecord(BaseModel):
    """Record of a state snapshot at discrete tick t."""
    tick: int
    state: StructuredSelfState
    incoming_event_count: int
    schema_valid: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    updater_mode: str = "oracle"
    error_message: Optional[str] = None

