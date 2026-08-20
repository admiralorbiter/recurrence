"""Ground Truth State dataclasses for Continuity Garden."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GroundTruthState:
    """Environment-internal ground truth state. NEVER exposed directly to the agent."""
    step_idx: int
    hidden_mode: int             # z in {0, 1}
    current_phase: str           # "cue", "distractor", "query"
    query_bit: Optional[int]     # x_t in {0, 1} during query phase
    target_action: Optional[int] # a_t* = x_t ^ z
    true_source: int             # 0: environment, 1: self
    resource_integrity: float    # Internal viability variable
    is_terminal: bool
