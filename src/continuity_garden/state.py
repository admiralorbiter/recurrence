"""Ground Truth State and Environment Snapshot dataclasses for Continuity Garden."""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass
class GroundTruthState:
    """Environment-internal ground truth state. NEVER exposed directly to the agent."""
    step_idx: int
    hidden_mode: int             # z in {0, 1}
    current_phase: str           # "cue", "distractor", "query", "terminal"
    query_bit: Optional[int]     # x_t in {0, 1} during query phase
    target_action: Optional[int] # a_t* = x_t ^ z
    true_source: int             # 0: environment, 1: self
    resource_integrity: float    # Internal viability variable
    is_terminal: bool


@dataclass
class EnvironmentSnapshot:
    """Complete serialized state of the environment for exact mid-life freezing and branching."""
    step_idx: int
    hidden_mode: int
    current_phase: str
    query_bit: Optional[int]
    target_action: Optional[int]
    true_source: int
    resource_integrity: float
    is_terminal: bool
    min_delay: int
    max_delay: int
    num_queries: int
    delay_len: int
    query_idx: int
    query_sequence: List[int]
    last_action: Optional[int]
    pending_target_action: Optional[int]
    rng_state: Tuple[Any, ...]
