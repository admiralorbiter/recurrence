"""Concrete Level 1 memory representation adapters for the 6 standard memory conditions."""

import json
from typing import Dict, List, Optional
from recurrence.memory.base import BaseMemoryAdapter
from recurrence.memory.schemas import MemoryEvent, MemoryFormat, StructuredSelfState


class FreshAdapter(BaseMemoryAdapter):
    """Condition 0: Fresh (No Memory).

    Returns no context. The model receives only the isolated current query probe.
    """
    format_name = MemoryFormat.FRESH

    def build_context_prompt(
        self,
        events: List[MemoryEvent],
        structured_state: Optional[StructuredSelfState] = None,
        cached_summary: Optional[str] = None,
    ) -> str:
        return ""


class TranscriptAdapter(BaseMemoryAdapter):
    """Condition 1: Full Transcript (Verbatim History).

    Presents the complete chronological sequence of raw events in context.
    """
    format_name = MemoryFormat.TRANSCRIPT

    def build_context_prompt(
        self,
        events: List[MemoryEvent],
        structured_state: Optional[StructuredSelfState] = None,
        cached_summary: Optional[str] = None,
    ) -> str:
        if not events:
            return ""

        lines = ["=== FULL EVENT TRANSCRIPT ==="]
        for ev in events:
            source_label = ev.source.value.upper()
            lines.append(f"[Step {ev.step_index:02d} | Source: {source_label} | Type: {ev.event_type}] {ev.content}")
        lines.append("=== END TRANSCRIPT ===\n")
        return "\n".join(lines)


class DeterministicSummaryAdapter(BaseMemoryAdapter):
    """Condition 2: Deterministic Summary (Rule-Based Lossless Extraction).

    Programmatically extracts all asserted key-value associations and source records
    without LLM neural compression or hallucination.
    """
    format_name = MemoryFormat.DETERMINISTIC_SUMMARY

    def build_context_prompt(
        self,
        events: List[MemoryEvent],
        structured_state: Optional[StructuredSelfState] = None,
        cached_summary: Optional[str] = None,
    ) -> str:
        if not events:
            return ""

        # Programmatically aggregate all key bindings and sources
        bindings: Dict[str, str] = {}
        sources: Dict[str, str] = {}
        active_goals: List[str] = []

        for ev in events:
            for k, v in ev.key_bindings.items():
                bindings[k] = v
                sources[k] = ev.source.value

            if ev.event_type == "goal_assertion":
                active_goals.append(ev.content)

        lines = ["=== DETERMINISTIC FACTUAL SUMMARY ==="]
        if bindings:
            lines.append("Known Key-Value Bindings:")
            for k, v in bindings.items():
                src = sources.get(k, "unknown")
                lines.append(f"- {k}: {v} (Source: {src})")
        if active_goals:
            lines.append("\nActive Goals:")
            for g in active_goals:
                lines.append(f"- {g}")
        lines.append("=== END FACTUAL SUMMARY ===\n")
        return "\n".join(lines)


class ModelSummaryAdapter(BaseMemoryAdapter):
    """Condition 3: Model-Written Summary (Autobiographical LLM Narrative).

    Presents pre-consolidated narrative text produced by an earlier LLM consolidation step.
    """
    format_name = MemoryFormat.MODEL_SUMMARY

    def build_context_prompt(
        self,
        events: List[MemoryEvent],
        structured_state: Optional[StructuredSelfState] = None,
        cached_summary: Optional[str] = None,
    ) -> str:
        summary_text = (cached_summary or "").strip()
        if not summary_text:
            return ""

        lines = [
            "=== AUTOBIOGRAPHICAL MODEL MEMORY SUMMARY ===",
            summary_text,
            "=== END MEMORY SUMMARY ===\n"
        ]
        return "\n".join(lines)


class StructuredStateAdapter(BaseMemoryAdapter):
    """Condition 4: Structured Self-State (Typed State Object).

    Presents working memory, goal registry, source ledger, and unresolved items as a typed JSON/YAML object.
    """
    format_name = MemoryFormat.STRUCTURED_STATE

    def build_context_prompt(
        self,
        events: List[MemoryEvent],
        structured_state: Optional[StructuredSelfState] = None,
        cached_summary: Optional[str] = None,
    ) -> str:
        if structured_state is None:
            # Construct state from events if not explicitly passed
            bindings: Dict[str, str] = {}
            sources: Dict[str, str] = {}
            for ev in events:
                for k, v in ev.key_bindings.items():
                    bindings[k] = v
                    sources[k] = ev.source.value
            structured_state = StructuredSelfState(
                working_memory=bindings,
                source_ledger=sources,
                last_updated_step=len(events)
            )

        state_dict = structured_state.model_dump()
        state_json = json.dumps(state_dict, indent=2)

        lines = [
            "=== STRUCTURED SELF-STATE ===",
            state_json,
            "=== END STRUCTURED STATE ===\n"
        ]
        return "\n".join(lines)


class CombinedStateAdapter(BaseMemoryAdapter):
    """Condition 5: Combined State (Structured State + Model Narrative).

    Jointly presents typed structured state alongside autobiographical narrative memory.
    """
    format_name = MemoryFormat.COMBINED

    def __init__(self):
        self._state_adapter = StructuredStateAdapter()
        self._summary_adapter = ModelSummaryAdapter()

    def build_context_prompt(
        self,
        events: List[MemoryEvent],
        structured_state: Optional[StructuredSelfState] = None,
        cached_summary: Optional[str] = None,
    ) -> str:
        state_part = self._state_adapter.build_context_prompt(events, structured_state=structured_state)
        summary_part = self._summary_adapter.build_context_prompt(events, cached_summary=cached_summary)

        parts = [p for p in [state_part, summary_part] if p]
        return "\n".join(parts) if parts else ""


def get_memory_adapter(format_type: MemoryFormat) -> BaseMemoryAdapter:
    """Factory helper to obtain the adapter for a given memory format."""
    adapters = {
        MemoryFormat.FRESH: FreshAdapter(),
        MemoryFormat.TRANSCRIPT: TranscriptAdapter(),
        MemoryFormat.DETERMINISTIC_SUMMARY: DeterministicSummaryAdapter(),
        MemoryFormat.MODEL_SUMMARY: ModelSummaryAdapter(),
        MemoryFormat.STRUCTURED_STATE: StructuredStateAdapter(),
        MemoryFormat.COMBINED: CombinedStateAdapter(),
    }
    if format_type not in adapters:
        raise ValueError(f"Unknown memory format: {format_type}")
    return adapters[format_type]
