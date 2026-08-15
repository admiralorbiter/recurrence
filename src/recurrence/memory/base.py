"""Base interface for Level 1 explicit memory adapters."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from recurrence.memory.schemas import MemoryEvent, MemoryFormat, StructuredSelfState


class BaseMemoryAdapter(ABC):
    """Abstract base class for all Level 1 memory representation adapters."""

    format_name: MemoryFormat

    @abstractmethod
    def build_context_prompt(
        self,
        events: List[MemoryEvent],
        structured_state: Optional[StructuredSelfState] = None,
        cached_summary: Optional[str] = None,
    ) -> str:
        """Render the memory representation as context text to be prepended to the task probe prompt."""
        pass

    def compute_context_stats(
        self,
        context_text: str,
    ) -> Dict[str, int]:
        """Compute basic text footprint metrics (character count, estimated token count, byte length)."""
        # Conservative token estimation: ~4 chars per token for English text
        est_tokens = max(1, len(context_text) // 4) if context_text else 0
        return {
            "char_count": len(context_text),
            "byte_count": len(context_text.encode("utf-8")),
            "estimated_tokens": est_tokens,
        }
