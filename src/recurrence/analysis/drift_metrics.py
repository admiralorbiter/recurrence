"""Quantitative metrics for autonomous state maintenance, drift, mutation, and stability."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from recurrence.memory.schemas import StateSnapshotRecord, StructuredSelfState
from recurrence.tasks.stream_scenarios import StreamScenario


@dataclass
class TickStabilityMetric:
    """Quantitative stability metrics evaluated at a single tick t."""
    tick: int
    schema_valid: bool
    ground_truth_key_count: int
    retained_keys_count: int
    mutated_keys_count: int
    omitted_keys_count: int
    phantom_keys_count: int
    phantom_keys_list: List[str]
    never_seen_keys_count: int = 0
    never_seen_keys_list: List[str] = field(default_factory=list)
    stale_evicted_keys_count: int = 0
    stale_evicted_keys_list: List[str] = field(default_factory=list)
    retention_fidelity: float = 1.0
    omission_rate: float = 0.0
    mutation_rate: float = 0.0
    source_attribution_accuracy: float = 1.0
    goal_coherence: float = 1.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    state_size_chars: int = 0
    is_active_inference: bool = False
    failure_categories: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class ScenarioStabilitySummary:
    """Aggregate stability metrics for an update loop execution across a scenario."""
    scenario_id: str
    updater_mode: str
    total_ticks: int
    schema_compliance_rate: float
    mean_retention_fidelity: float
    terminal_retention_fidelity: float
    mean_omission_rate: float
    mean_mutation_rate: float
    phantom_key_tick_count: int
    unique_phantom_keys_count: int
    never_seen_key_tick_count: int = 0
    unique_never_seen_keys_count: int = 0
    stale_evicted_key_tick_count: int = 0
    unique_stale_evicted_keys_count: int = 0
    mean_goal_coherence: float = 1.0
    terminal_goal_coherence: float = 1.0
    is_ossified: bool = False
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    active_inference_count: int = 0
    valid_active_inferences: int = 0
    mean_prompt_tokens_per_active_inference: float = 0.0
    failure_category_counts: Dict[str, int] = field(default_factory=dict)
    tick_metrics: List[TickStabilityMetric] = field(default_factory=list)


def evaluate_tick_state(
    tick: int,
    evaluated_state: StructuredSelfState,
    oracle_state: StructuredSelfState,
    schema_valid: bool,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    is_active_inference: bool = False,
    seen_keys_so_far: Optional[Set[str]] = None,
    error_message: Optional[str] = None,
) -> TickStabilityMetric:
    """Compare an evaluated state against the ground-truth oracle state at tick t."""
    gt_wm = oracle_state.working_memory
    eval_wm = evaluated_state.working_memory
    
    total_gt = len(gt_wm)
    retained = 0
    mutated = 0
    omitted = 0
    phantom_keys: List[str] = []
    never_seen_keys: List[str] = []
    stale_evicted_keys: List[str] = []

    if total_gt == 0:
        retention_fid = 1.0
        omission_r = 0.0
        mutation_r = 0.0
    else:
        for k, true_v in gt_wm.items():
            if k in eval_wm:
                if eval_wm[k] == true_v:
                    retained += 1
                else:
                    mutated += 1
            else:
                omitted += 1

        assert retained + mutated + omitted == total_gt
        retention_fid = retained / total_gt
        omission_r = omitted / total_gt
        mutation_r = mutated / total_gt

    # Phantom keys and distinction between never-seen vs stale/evicted
    seen = seen_keys_so_far if seen_keys_so_far is not None else set(gt_wm.keys())
    for k in eval_wm.keys():
        if k not in gt_wm:
            phantom_keys.append(k)
            if k in seen:
                stale_evicted_keys.append(k)
            else:
                never_seen_keys.append(k)

    phantom_count = len(phantom_keys)
    never_seen_count = len(never_seen_keys)
    stale_evicted_count = len(stale_evicted_keys)

    # Source attribution accuracy on present keys
    src_matches = 0
    gt_src = oracle_state.source_ledger
    eval_src = evaluated_state.source_ledger
    present_keys = [k for k in gt_wm.keys() if k in eval_wm]
    if present_keys:
        for k in present_keys:
            if eval_src.get(k) == gt_src.get(k):
                src_matches += 1
        source_acc = src_matches / len(present_keys)
    else:
        source_acc = 1.0 if total_gt == 0 else 0.0

    # Goal coherence
    gt_goals = {g.goal_id: g.status for g in oracle_state.goals}
    eval_goals = {g.goal_id: g.status for g in evaluated_state.goals}
    if gt_goals:
        goal_matches = sum(1 for gid, st in gt_goals.items() if eval_goals.get(gid) == st)
        goal_coh = goal_matches / len(gt_goals)
    else:
        goal_coh = 1.0

    state_chars = len(evaluated_state.model_dump_json())

    # Independent Non-Exclusive Failure Categories
    failures: List[str] = []
    if not schema_valid:
        failures.append("Schema Violation")
    if omitted > 0:
        failures.append("Exact KV Omission")
    if mutated > 0:
        failures.append("Exact Association Mutation")
    if never_seen_count > 0:
        failures.append("Phantom Intrusion")
    if stale_evicted_count > 0:
        failures.append("Stale Key Retention")
    if goal_coh < 1.0:
        failures.append("Goal Desynchronization")

    return TickStabilityMetric(
        tick=tick,
        schema_valid=schema_valid,
        ground_truth_key_count=total_gt,
        retained_keys_count=retained,
        mutated_keys_count=mutated,
        omitted_keys_count=omitted,
        phantom_keys_count=phantom_count,
        phantom_keys_list=phantom_keys,
        never_seen_keys_count=never_seen_count,
        never_seen_keys_list=never_seen_keys,
        stale_evicted_keys_count=stale_evicted_count,
        stale_evicted_keys_list=stale_evicted_keys,
        retention_fidelity=retention_fid,
        omission_rate=omission_r,
        mutation_rate=mutation_r,
        source_attribution_accuracy=source_acc,
        goal_coherence=goal_coh,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        state_size_chars=state_chars,
        is_active_inference=is_active_inference,
        failure_categories=failures,
        error_message=error_message,
    )


def compute_scenario_stability(
    scenario: StreamScenario,
    snapshots: List[StateSnapshotRecord],
    updater_mode: str = "oracle",
) -> ScenarioStabilitySummary:
    """Compute comprehensive stability summary across all snapshots for a scenario."""
    tick_metrics: List[TickStabilityMetric] = []
    unique_phantoms: Set[str] = set()
    unique_never_seen: Set[str] = set()
    unique_stale_evicted: Set[str] = set()

    failure_counts: Dict[str, int] = {
        "Schema Violation": 0,
        "Exact KV Omission": 0,
        "Exact Association Mutation": 0,
        "Phantom Intrusion": 0,
        "Stale Key Retention": 0,
        "Goal Desynchronization": 0,
    }
    
    state_changed_at_least_once = False
    prev_wm: Optional[Dict[str, str]] = None
    seen_keys_so_far: Set[str] = set()

    # Track all asserted keys up to each tick
    scheduled_by_tick: Dict[int, List[Any]] = {}
    for ev in scenario.scheduled_events:
        scheduled_by_tick.setdefault(ev.step_index, []).append(ev)

    for snap in snapshots:
        t = snap.tick
        for ev in scheduled_by_tick.get(t, []):
            for k in ev.key_bindings.keys():
                seen_keys_so_far.add(k)

        oracle_st = scenario.oracle_states.get(t, StructuredSelfState())
        is_active = (snap.incoming_event_count > 0)
        
        metric = evaluate_tick_state(
            tick=t,
            evaluated_state=snap.state,
            oracle_state=oracle_st,
            schema_valid=snap.schema_valid,
            prompt_tokens=snap.prompt_tokens,
            completion_tokens=snap.completion_tokens,
            is_active_inference=is_active,
            seen_keys_so_far=set(seen_keys_so_far),
            error_message=snap.error_message,
        )
        tick_metrics.append(metric)

        for p_key in metric.phantom_keys_list:
            unique_phantoms.add(p_key)
        for ns_key in metric.never_seen_keys_list:
            unique_never_seen.add(ns_key)
        for st_key in metric.stale_evicted_keys_list:
            unique_stale_evicted.add(st_key)

        for f_cat in metric.failure_categories:
            failure_counts[f_cat] = failure_counts.get(f_cat, 0) + 1

        if prev_wm is not None and snap.state.working_memory != prev_wm:
            state_changed_at_least_once = True
        prev_wm = dict(snap.state.working_memory)

    total_ticks = len(tick_metrics)
    if total_ticks == 0:
        return ScenarioStabilitySummary(
            scenario_id=scenario.scenario_id,
            updater_mode=updater_mode,
            total_ticks=0,
            schema_compliance_rate=1.0,
            mean_retention_fidelity=1.0,
            terminal_retention_fidelity=1.0,
            mean_omission_rate=0.0,
            mean_mutation_rate=0.0,
            phantom_key_tick_count=0,
            unique_phantom_keys_count=0,
            never_seen_key_tick_count=0,
            unique_never_seen_keys_count=0,
            stale_evicted_key_tick_count=0,
            unique_stale_evicted_keys_count=0,
            mean_goal_coherence=1.0,
            terminal_goal_coherence=1.0,
            is_ossified=False,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            active_inference_count=0,
            valid_active_inferences=0,
            mean_prompt_tokens_per_active_inference=0.0,
            failure_category_counts=failure_counts,
        )

    valid_count = sum(1 for m in tick_metrics if m.schema_valid)
    compliance_rate = valid_count / total_ticks
    
    mean_retention = sum(m.retention_fidelity for m in tick_metrics) / total_ticks
    terminal_retention = tick_metrics[-1].retention_fidelity
    
    mean_omission = sum(m.omission_rate for m in tick_metrics) / total_ticks
    mean_mutation = sum(m.mutation_rate for m in tick_metrics) / total_ticks
    
    total_phantoms = sum(m.phantom_keys_count for m in tick_metrics)
    total_never_seen = sum(m.never_seen_keys_count for m in tick_metrics)
    total_stale = sum(m.stale_evicted_keys_count for m in tick_metrics)

    mean_goal_coh = sum(m.goal_coherence for m in tick_metrics) / total_ticks
    terminal_goal_coh = tick_metrics[-1].goal_coherence

    total_prompt_tok = sum(m.prompt_tokens for m in tick_metrics)
    total_comp_tok = sum(m.completion_tokens for m in tick_metrics)

    active_infs = sum(1 for m in tick_metrics if m.is_active_inference)
    valid_active_infs = sum(1 for m in tick_metrics if m.is_active_inference and m.schema_valid)
    p_tok_per_active = (total_prompt_tok / max(1, active_infs)) if active_infs > 0 else 0.0

    is_ossified = (total_ticks > 3 and not state_changed_at_least_once)

    return ScenarioStabilitySummary(
        scenario_id=scenario.scenario_id,
        updater_mode=updater_mode,
        total_ticks=total_ticks,
        schema_compliance_rate=compliance_rate,
        mean_retention_fidelity=mean_retention,
        terminal_retention_fidelity=terminal_retention,
        mean_omission_rate=mean_omission,
        mean_mutation_rate=mean_mutation,
        phantom_key_tick_count=total_phantoms,
        unique_phantom_keys_count=len(unique_phantoms),
        never_seen_key_tick_count=total_never_seen,
        unique_never_seen_keys_count=len(unique_never_seen),
        stale_evicted_key_tick_count=total_stale,
        unique_stale_evicted_keys_count=len(unique_stale_evicted),
        mean_goal_coherence=mean_goal_coh,
        terminal_goal_coherence=terminal_goal_coh,
        is_ossified=is_ossified,
        total_prompt_tokens=total_prompt_tok,
        total_completion_tokens=total_comp_tok,
        active_inference_count=active_infs,
        valid_active_inferences=valid_active_infs,
        mean_prompt_tokens_per_active_inference=p_tok_per_active,
        failure_category_counts=failure_counts,
        tick_metrics=tick_metrics,
    )
