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
    overall_accuracy: float  # Micro-average across all trials
    macro_accuracy: float = 0.0  # Equal macro-average across (KV, Source, Goal)
    accuracy_by_probe_type: Dict[str, float] = Field(default_factory=dict)
    accuracy_by_position: Dict[str, float] = Field(default_factory=dict)
    delayed_kv_accuracy_by_position: Dict[str, float] = Field(default_factory=dict)
    mean_prompt_chars: float = 0.0
    mean_estimated_tokens: float = 0.0
    mean_byte_count: float = 0.0
    accuracy_per_1k_tokens: float = 0.0
    is_pareto_optimal: bool = False
    compliance_rate: float = 1.0


def compute_summary_distortion(
    events: List[MemoryEvent],
    target_bindings: Dict[str, str],
    summary_text: str,
) -> DistortionMetrics:
    """Quantify omission, mutation, and intrusion rates of a consolidated text summary against ground truth.
    
    Guarantees the partition invariant: retained + mutated + omitted == total_targets.
    """
    if not target_bindings:
        return DistortionMetrics()

    total_targets = len(target_bindings)
    retained = 0
    mutated = 0
    omitted = 0

    norm_summary = summary_text.lower()
    # Split into sentences or clauses for association checking
    clauses = [c.strip() for c in re.split(r"[\n\.\;\!]", norm_summary) if c.strip()]

    for k, v in target_bindings.items():
        k_norm = k.lower()
        v_norm = v.lower()

        # Find clauses containing the key
        matching_clauses = [c for c in clauses if k_norm in c]

        if not matching_clauses:
            omitted += 1
        else:
            # Check if any matching clause contains the correct value association
            has_correct_association = any(v_norm in c for c in matching_clauses)
            if has_correct_association:
                retained += 1
            else:
                mutated += 1

    # Check for hallucinated/unsupported entity intrusions
    all_valid_values = set()
    for ev in events:
        if ev.key_bindings:
            for val in ev.key_bindings.values():
                all_valid_values.add(val.lower())

    summary_vals = set(re.findall(r"val_[a-z0-9_]+", norm_summary))
    unsupported = summary_vals - all_valid_values
    unsupported_count = len(unsupported)

    # Enforce strict partition invariant
    assert retained + mutated + omitted == total_targets, (
        f"Partition invariant failed: {retained} + {mutated} + {omitted} != {total_targets}"
    )

    omission_rate = omitted / total_targets if total_targets > 0 else 0.0
    mutation_rate = mutated / total_targets if total_targets > 0 else 0.0
    intrusion_rate = unsupported_count / max(1, len(summary_vals)) if summary_vals else 0.0

    return DistortionMetrics(
        total_target_facts=total_targets,
        retained_target_facts=retained,
        omitted_target_facts=omitted,
        omission_rate=omission_rate,
        mutated_facts=mutated,
        mutation_rate=mutation_rate,
        unsupported_intrusions=unsupported_count,
        intrusion_rate=intrusion_rate,
    )


def compute_memory_format_benchmarks(
    records: List[Dict[str, Any]],
) -> Dict[str, MemoryFormatSummary]:
    """Aggregate benchmark performance, cost, macro-accuracy, and isolated positional breakdown."""
    by_format: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        fmt = r["memory_format"]
        by_format.setdefault(fmt, []).append(r)

    summaries: Dict[str, MemoryFormatSummary] = {}

    for fmt_str, f_records in by_format.items():
        total = len(f_records)
        if total == 0:
            continue

        valid_count = sum(1 for r in f_records if r.get("schema_valid", True))
        compliance_rate = valid_count / total

        correct_count = sum(1 for r in f_records if r.get("correct") is True)
        overall_acc = correct_count / total

        # Breakdown by probe type
        by_probe: Dict[str, List[bool]] = {}
        by_pos_all: Dict[str, List[bool]] = {}
        by_pos_delayed_kv: Dict[str, List[bool]] = {}

        total_chars = 0
        total_tokens = 0
        total_bytes = 0

        for r in f_records:
            p_type = r.get("probe_type", "unknown")
            is_c = bool(r.get("correct", False))
            by_probe.setdefault(p_type, []).append(is_c)

            pos = r.get("position_stratum")
            if pos:
                by_pos_all.setdefault(pos, []).append(is_c)
                if p_type == "delayed_kv":
                    by_pos_delayed_kv.setdefault(pos, []).append(is_c)

            total_chars += r.get("prompt_chars", 0)
            total_tokens += r.get("estimated_tokens", 0)
            total_bytes += r.get("byte_count", 0)

        probe_acc = {p: (sum(vals) / len(vals)) for p, vals in by_probe.items()}
        pos_acc = {p: (sum(vals) / len(vals)) for p, vals in by_pos_all.items()}
        delayed_kv_pos_acc = {p: (sum(vals) / len(vals)) for p, vals in by_pos_delayed_kv.items()}

        # Compute Macro-Average (equal 1/3 weight across the 3 probe tasks)
        kv_score = probe_acc.get("delayed_kv", overall_acc)
        src_score = probe_acc.get("source_attribution", overall_acc)
        goal_score = probe_acc.get("goal_resumption", overall_acc)
        macro_acc = (kv_score + src_score + goal_score) / 3.0

        mean_chars = total_chars / total
        mean_tokens = total_tokens / total
        mean_bytes = total_bytes / total

        acc_per_1k = (overall_acc / (mean_tokens / 1000.0)) if mean_tokens > 0 else overall_acc * 1000.0

        summaries[fmt_str] = MemoryFormatSummary(
            memory_format=MemoryFormat(fmt_str),
            trial_count=total,
            overall_accuracy=overall_acc,
            macro_accuracy=macro_acc,
            accuracy_by_probe_type=probe_acc,
            accuracy_by_position=pos_acc,
            delayed_kv_accuracy_by_position=delayed_kv_pos_acc,
            mean_prompt_chars=mean_chars,
            mean_estimated_tokens=mean_tokens,
            mean_byte_count=mean_bytes,
            accuracy_per_1k_tokens=acc_per_1k,
            compliance_rate=compliance_rate,
        )

    # Determine true Pareto optimality
    for fmt_a, sum_a in summaries.items():
        is_dominated = False
        for fmt_b, sum_b in summaries.items():
            if fmt_a == fmt_b:
                continue
            # Format B dominates Format A if it has higher or equal accuracy AND fewer or equal tokens (with at least one strict inequality)
            if (sum_b.overall_accuracy >= sum_a.overall_accuracy and sum_b.mean_estimated_tokens <= sum_a.mean_estimated_tokens) and (
                sum_b.overall_accuracy > sum_a.overall_accuracy or sum_b.mean_estimated_tokens < sum_a.mean_estimated_tokens
            ):
                is_dominated = True
                break
        sum_a.is_pareto_optimal = not is_dominated

    return summaries
