"""Fidelity, distortion, and cost metrics for Level 1 memory representations."""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from recurrence.memory.schemas import MemoryEvent, MemoryFormat, StructuredSelfState


class DistortionMetrics(BaseModel):
    """Fidelity and distortion metrics for consolidated memory summaries."""
    total_target_facts: int = 0
    retained_target_facts: int = 0
    omitted_target_facts: int = 0
    omission_rate: float = 0.0
    mutated_facts: int = 0
    mutation_rate: float = 0.0
    unsupported_intrusions: int = 0
    intrusion_rate: float = 0.0


class MemoryFormatSummary(BaseModel):
    """Summary statistics for a single memory representation condition."""
    memory_format: MemoryFormat
    trial_count: int
    overall_accuracy: float
    accuracy_by_probe_type: Dict[str, float] = Field(default_factory=dict)
    accuracy_by_position: Dict[str, float] = Field(default_factory=dict)
    mean_prompt_chars: float = 0.0
    mean_estimated_tokens: float = 0.0
    mean_byte_count: float = 0.0
    accuracy_per_1k_tokens: float = 0.0


def compute_summary_distortion(
    events: List[MemoryEvent],
    target_bindings: Dict[str, str],
    summary_text: str,
) -> DistortionMetrics:
    """Quantify omission, mutation, and intrusion rates of a consolidated text summary against ground truth."""
    if not target_bindings:
        return DistortionMetrics()

    total_targets = len(target_bindings)
    retained = 0
    mutated = 0
    omitted = 0

    norm_summary = summary_text.lower()

    for k, v in target_bindings.items():
        k_norm = k.lower()
        v_norm = v.lower()

        # Check key presence
        if k_norm in norm_summary:
            # Check if correct value is associated
            if v_norm in norm_summary:
                retained += 1
            else:
                # Key present but value wrong -> mutation
                mutated += 1
        else:
            omitted += 1

    omission_rate = omitted / total_targets if total_targets > 0 else 0.0
    mutation_rate = mutated / total_targets if total_targets > 0 else 0.0

    return DistortionMetrics(
        total_target_facts=total_targets,
        retained_target_facts=retained,
        omitted_target_facts=omitted,
        omission_rate=omission_rate,
        mutated_facts=mutated,
        mutation_rate=mutation_rate,
        unsupported_intrusions=0,
        intrusion_rate=0.0,
    )


def compute_memory_format_benchmarks(
    records: List[Dict[str, Any]],
) -> Dict[str, MemoryFormatSummary]:
    """Aggregate benchmark performance, cost, and positional breakdown across all memory formats."""
    by_format: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        fmt = r["memory_format"]
        by_format.setdefault(fmt, []).append(r)

    summaries: Dict[str, MemoryFormatSummary] = {}

    for fmt_str, f_records in by_format.items():
        total = len(f_records)
        if total == 0:
            continue

        correct_count = sum(1 for r in f_records if r.get("correct") is True)
        overall_acc = correct_count / total

        # Breakdown by probe type
        by_probe: Dict[str, List[bool]] = {}
        by_pos: Dict[str, List[bool]] = {}

        total_chars = 0
        total_tokens = 0
        total_bytes = 0

        for r in f_records:
            p_type = r.get("probe_type", "unknown")
            is_c = bool(r.get("correct", False))
            by_probe.setdefault(p_type, []).append(is_c)

            pos = r.get("position_stratum")
            if pos:
                by_pos.setdefault(pos, []).append(is_c)

            total_chars += r.get("prompt_chars", 0)
            total_tokens += r.get("estimated_tokens", 0)
            total_bytes += r.get("byte_count", 0)

        probe_acc = {p: (sum(vals) / len(vals)) for p, vals in by_probe.items()}
        pos_acc = {p: (sum(vals) / len(vals)) for p, vals in by_pos.items()}

        mean_chars = total_chars / total
        mean_tokens = total_tokens / total
        mean_bytes = total_bytes / total

        acc_per_1k = (overall_acc / (mean_tokens / 1000.0)) if mean_tokens > 0 else overall_acc * 1000.0

        summaries[fmt_str] = MemoryFormatSummary(
            memory_format=MemoryFormat(fmt_str),
            trial_count=total,
            overall_accuracy=overall_acc,
            accuracy_by_probe_type=probe_acc,
            accuracy_by_position=pos_acc,
            mean_prompt_chars=mean_chars,
            mean_estimated_tokens=mean_tokens,
            mean_byte_count=mean_bytes,
            accuracy_per_1k_tokens=acc_per_1k,
        )

    return summaries
