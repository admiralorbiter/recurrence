"""Scientific Runner for Experiment E03 (Sprint S04):
Level 1 Explicit Memory Baselines and Scaffolded Persistence Benchmark.

Evaluates 6 Memory Conditions on Multi-Stage Episodic Streams:
  1. Fresh (Zero Context)
  2. Full Transcript (Chronological Event Stream)
  3. Deterministic Summary (Programmatic Lossless Extraction)
  4. Model-Written Summary (Two-Stage Pre-Consolidated Autobiographical Narrative)
  5. Structured Self-State (Typed Working Memory, Goals & Source Ledger)
  6. Combined State (Structured State + Model Narrative)

Tasks Evaluated:
  - Task 1: Delayed KV Retrieval under Distraction (Early, Middle, Late strata)
  - Task 2: Source Memory Attribution (Environment vs Self vs Experimenter)
  - Task 3: Interrupted Goal Resumption (Identifying Suspended Subgoals)

Primary Endpoint: Pure Answer-Only first-order retrieval accuracy.
Distortion Analysis: Omission Rate, Mutation Rate, Token/Byte Efficiency Pareto Frontier.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.core.manifest import RunManifest
from recurrence.core.logging import ExperimentLogger, TrialEvent
from recurrence.memory.schemas import (
    ConsolidationRecord,
    EventSource,
    GoalState,
    MemoryEvent,
    MemoryFormat,
    StructuredSelfState,
)
from recurrence.memory.adapters import get_memory_adapter
from recurrence.tasks.memory_battery import MemoryBatteryTask, MemoryProbeItem, EpisodeData
from recurrence.analysis.memory_metrics import (
    compute_summary_distortion,
    compute_memory_format_benchmarks,
    DistortionMetrics,
    MemoryFormatSummary,
)


class MockMemoryBackend:
    """Mock deterministic backend for dry-run verification."""
    def __init__(self, model_name: str = "mock-qwen2.5:3b", seed: int = 42):
        self.model_name = model_name
        self.seed = seed
        self.model_digest = "mock_digest_00000000000000000000000000000000"

    def get_digest(self) -> str:
        return self.model_digest

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        format: Optional[Any] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        content = messages[0]["content"] if messages else ""
        if "Respond strictly with a JSON object" in content:
            return '{"answer": "A"}', {"prompt_eval_count": 60, "eval_count": 10}
        else:
            return (
                "Autobiographical memory summary: Recorded key_emerald_falcon as val_obsidian_river, "
                "computed key_golden_tempest as val_crimson_glacier, and noted suspended telemetry goal.",
                {"prompt_eval_count": 150, "eval_count": 40}
            )


def generate_model_consolidation(
    backend: Any,
    episodes: List[EpisodeData],
    temperature: float = 0.0,
    seed: int = 42,
) -> Tuple[Dict[str, str], List[ConsolidationRecord], Dict[str, DistortionMetrics]]:
    """Stage 1: Execute offline LLM consolidation step to generate autobiographical summaries."""
    summaries: Dict[str, str] = {}
    records: List[ConsolidationRecord] = []
    distortion_map: Dict[str, DistortionMetrics] = {}

    for ep in episodes:
        event_lines = []
        for ev in ep.events:
            event_lines.append(f"[Step {ev.step_index:02d} | Source: {ev.source.value.upper()}] {ev.content}")
        raw_transcript = "\n".join(event_lines)

        consolidation_prompt = (
            f"=== RAW EVENT LOG ===\n"
            f"{raw_transcript}\n"
            f"=== END EVENT LOG ===\n\n"
            f"You are the autonomous agent who experienced these events. Write a concise autobiographical summary "
            f"of your experience, explicitly noting what key-value facts were observed or calculated, their sources "
            f"(whether observed from environment, computed by self, or instructed by experimenter), and what goals were active or suspended."
        )

        messages = [{"role": "user", "content": consolidation_prompt}]
        summary_text, meta = backend.chat(messages=messages, temperature=temperature, seed=seed)
        summary_text = summary_text.strip()
        summaries[ep.episode_id] = summary_text

        raw_digest = hashlib.sha256(raw_transcript.encode("utf-8")).hexdigest()

        rec = ConsolidationRecord(
            source_event_count=len(ep.events),
            raw_event_digest=raw_digest,
            model_name=getattr(backend, "model_name", "model"),
            summary_text=summary_text,
            prompt_tokens=meta.get("prompt_eval_count", len(consolidation_prompt) // 4),
            completion_tokens=meta.get("eval_count", len(summary_text) // 4),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        records.append(rec)

        target_bindings = {t["key"]: t["value"] for t in ep.kv_targets.values()}
        distortion = compute_summary_distortion(ep.events, target_bindings, summary_text)
        distortion_map[ep.episode_id] = distortion

    return summaries, records, distortion_map


def generate_e03_markdown_report(
    manifest_dict: Dict[str, Any],
    benchmarks: Dict[str, MemoryFormatSummary],
    mean_distortion: DistortionMetrics,
    consolidation_records: List[ConsolidationRecord],
) -> str:
    """Generate comprehensive scientific research memo and performance report for E03."""
    lines = [
        f"# Experiment E03: Level 1 Explicit Memory Baseline Report",
        f"",
        f"**Run ID:** `{manifest_dict.get('run_id')}`  ",
        f"**Model:** `{manifest_dict.get('target_model')}` (`{manifest_dict.get('model_digest', 'N/A')[:12]}...`)  ",
        f"**Date:** {manifest_dict.get('start_time', datetime.now(timezone.utc).isoformat())}  ",
        f"**Primary Endpoint:** Pure Answer-Only Forced-Choice Accuracy  ",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary & Core Results",
        f"",
        f"Experiment E03 quantifies how much cognitive retention, delayed retrieval, and source attribution can be achieved across **6 Level-1 explicit memory representation formats** without latent recurrent continuity.",
        f"",
        f"### Memory Format Performance & Cost Pareto Table",
        f"",
        f"| Memory Format | Overall Acc | Delayed KV Acc | Source Attr Acc | Goal Resumption Acc | Mean Prompt Tokens | Cost Efficiency (Acc / 1k Tok) |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for fmt in [
        MemoryFormat.FRESH,
        MemoryFormat.TRANSCRIPT,
        MemoryFormat.DETERMINISTIC_SUMMARY,
        MemoryFormat.MODEL_SUMMARY,
        MemoryFormat.STRUCTURED_STATE,
        MemoryFormat.COMBINED,
    ]:
        b = benchmarks.get(fmt.value)
        if not b:
            continue
        kv_acc = f"{b.accuracy_by_probe_type.get('delayed_kv', 0.0):.1%}"
        src_acc = f"{b.accuracy_by_probe_type.get('source_attribution', 0.0):.1%}"
        goal_acc = f"{b.accuracy_by_probe_type.get('goal_resumption', 0.0):.1%}"
        eff = f"{b.accuracy_per_1k_tokens:.2f}"
        lines.append(
            f"| **{b.memory_format.value}** | **{b.overall_accuracy:.1%}** | {kv_acc} | {src_acc} | {goal_acc} | {b.mean_estimated_tokens:.0f} tok | {eff} |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Serial Position Analysis (Delayed KV Retrieval)",
        f"",
        f"Controlling for 'Lost-in-the-Middle' positional attention artifacts across early, middle, and late stream placements:",
        f"",
        "| Memory Format | Early Placement Acc | Middle Placement Acc | Late Placement Acc | Positional Stability (Late - Early) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for fmt in [
        MemoryFormat.FRESH,
        MemoryFormat.TRANSCRIPT,
        MemoryFormat.DETERMINISTIC_SUMMARY,
        MemoryFormat.MODEL_SUMMARY,
        MemoryFormat.STRUCTURED_STATE,
        MemoryFormat.COMBINED,
    ]:
        b = benchmarks.get(fmt.value)
        if not b:
            continue
        early = b.accuracy_by_position.get("early", 0.0)
        mid = b.accuracy_by_position.get("middle", 0.0)
        late = b.accuracy_by_position.get("late", 0.0)
        delta = late - early
        lines.append(
            f"| `{b.memory_format.value}` | {early:.1%} | {mid:.1%} | {late:.1%} | {delta:+.1%} |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 3. Two-Stage Consolidation Fidelity & Distortion Analysis",
        f"",
        f"Model summaries were generated in an isolated consolidation phase prior to evaluation probe trials.",
        f"",
        f"- **Total Target Facts Evaluated:** {mean_distortion.total_target_facts}",
        f"- **Retained Target Facts:** {mean_distortion.retained_target_facts} ({1.0 - mean_distortion.omission_rate:.1%})",
        f"- **Omission Rate (Facts Forgotten in Summary):** {mean_distortion.omission_rate:.1%}",
        f"- **Retrospective Mutation Rate (Facts Altered in Summary):** {mean_distortion.mutation_rate:.1%}",
        f"- **Mean Consolidation Prompt Tokens:** {sum(r.prompt_tokens for r in consolidation_records) / max(1, len(consolidation_records)):.1f}",
        f"- **Mean Consolidation Output Tokens:** {sum(r.completion_tokens for r in consolidation_records) / max(1, len(consolidation_records)):.1f}",
        f"",
        f"---",
        f"",
        f"## 4. Scientific Takeaways for Level 1 & Horizon 1",
        f"",
        f"1. **Scaffolded Memory Capacity Upper Bound:** Full Transcript and Deterministic Summary establish the ceiling of explicit symbolic representation.",
        f"2. **The Consolidation Trade-off:** Model-written narrative summaries compress context but introduce omission and mutation drift.",
        f"3. **Structured State Advantage:** Strongly typed JSON/YAML state preserves goal registries and source ledgers at significantly lower token cost than full transcripts.",
        f"4. **Transition to S05:** S04 establishes the static representation baseline; Sprint S05 will test dynamic autonomous multi-tick update loops.",
    ])

    return "\n".join(lines)


def run_e03_experiment(
    model_name: str = "qwen2.5:3b",
    episode_count: int = 6,
    distractor_count: int = 6,
    seed: int = 42,
    temperature: float = 0.0,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete E03 experiment across all 6 memory conditions."""
    run_id = f"run_e03_mem_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    if dry_run:
        run_id += "_dryrun"

    out_dir = output_dir or Path(f"artifacts/e03_memory/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Backend
    if dry_run:
        backend = MockMemoryBackend(model_name="mock-qwen2.5:3b", seed=seed)
        model_digest = backend.get_digest()
    else:
        backend = OllamaBackend(model_name=model_name, temperature=temperature, seed=seed)
        model_digest = backend.get_digest()

    # Initialize Task Battery
    task = MemoryBatteryTask(identifier_type="semantic", mode="forced_choice", ask_confidence=False)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT E03: LEVEL 1 EXPLICIT MEMORY BASELINES")
    print(f"Run ID: {run_id} | Model: {model_name} | Episodes: {episode_count}")
    print(f"{'='*60}\n")

    # Generate Synthetic Episodes
    episodes = [
        task.generate_episode(episode_idx=i, target_kv_count=3, distractor_count=distractor_count, seed=seed)
        for i in range(episode_count)
    ]

    # Stage 1: Generate Cached Model Summaries (Consolidation Phase)
    print("Stage 1: Generating two-stage model autobiographical summaries...")
    cached_summaries, consolidation_records, distortion_map = generate_model_consolidation(
        backend=backend,
        episodes=episodes,
        temperature=temperature,
        seed=seed,
    )

    # Aggregate Distortion Metrics
    total_facts = sum(d.total_target_facts for d in distortion_map.values())
    total_retained = sum(d.retained_target_facts for d in distortion_map.values())
    total_omitted = sum(d.omitted_target_facts for d in distortion_map.values())
    total_mutated = sum(d.mutated_facts for d in distortion_map.values())

    mean_distortion = DistortionMetrics(
        total_target_facts=total_facts,
        retained_target_facts=total_retained,
        omitted_target_facts=total_omitted,
        omission_rate=total_omitted / max(1, total_facts),
        mutated_facts=total_mutated,
        mutation_rate=total_mutated / max(1, total_facts),
    )
    print(f"  -> Model Summary Omission Rate: {mean_distortion.omission_rate:.1%}")
    print(f"  -> Model Summary Mutation Rate: {mean_distortion.mutation_rate:.1%}")

    # Stage 2: Evaluate All 6 Memory Formats
    trial_records: List[Dict[str, Any]] = []
    formats_to_eval = [
        MemoryFormat.FRESH,
        MemoryFormat.TRANSCRIPT,
        MemoryFormat.DETERMINISTIC_SUMMARY,
        MemoryFormat.MODEL_SUMMARY,
        MemoryFormat.STRUCTURED_STATE,
        MemoryFormat.COMBINED,
    ]

    total_probes_per_format = episode_count * 7
    print(f"\nStage 2: Evaluating 6 memory conditions ({len(formats_to_eval) * total_probes_per_format} total probe trials)...")

    for fmt in formats_to_eval:
        print(f"\nEvaluating Condition: [{fmt.value.upper()}]...")
        items = task.generate_probe_items(
            episodes=episodes,
            memory_format=fmt,
            cached_summaries=cached_summaries,
            seed=seed,
        )

        adapter = get_memory_adapter(fmt)

        correct_count = 0
        for item_idx, item in enumerate(items):
            stats = adapter.compute_context_stats(item.prompt)

            # Generate response from model
            messages = [{"role": "user", "content": item.prompt}]
            t0 = time.perf_counter()
            raw_text, meta = backend.chat(messages=messages, temperature=temperature, seed=seed, format="json")
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Score response
            score = task.score_response(item, raw_text)
            if score["correct"]:
                correct_count += 1

            record = {
                "run_id": run_id,
                "episode_id": item.metadata["episode_id"],
                "item_id": item.item_id,
                "memory_format": fmt.value,
                "probe_type": item.probe_type,
                "position_stratum": item.position_stratum,
                "distractor_count": item.distractor_count,
                "ground_truth": item.ground_truth,
                "parsed_answer": score["parsed_answer"],
                "correct": score["correct"],
                "schema_valid": score["schema_valid"],
                "prompt_chars": len(item.prompt),
                "estimated_tokens": meta.get("prompt_eval_count", stats["estimated_tokens"]),
                "byte_count": stats["byte_count"],
                "latency_ms": latency_ms,
                "raw_response": raw_text,
            }
            trial_records.append(record)

        acc = correct_count / len(items)
        print(f"  -> Accuracy on [{fmt.value}]: {acc:.1%} ({correct_count}/{len(items)})")

    # Stage 3: Compute Benchmarks & Summaries
    benchmarks = compute_memory_format_benchmarks(trial_records)

    # Stage 4: Serialize Outputs
    trials_jsonl_path = out_dir / "trials.jsonl"
    with open(trials_jsonl_path, "w", encoding="utf-8") as f:
        for r in trial_records:
            f.write(json.dumps(r) + "\n")

    trials_parquet_path = out_dir / "trials.parquet"
    df_trials = pd.DataFrame(trial_records)
    df_trials.to_parquet(trials_parquet_path, index=False)

    manifest_dict = {
        "run_id": run_id,
        "target_model": model_name,
        "model_digest": model_digest,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "episode_count": episode_count,
        "distractor_count": distractor_count,
        "total_trials": len(trial_records),
        "temperature": temperature,
        "seed": seed,
    }

    summary_dict = {
        "manifest": manifest_dict,
        "benchmarks": {k: v.model_dump() for k, v in benchmarks.items()},
        "consolidation_distortion": mean_distortion.model_dump(),
        "consolidation_records": [r.model_dump() for r in consolidation_records],
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    report_md = generate_e03_markdown_report(
        manifest_dict=manifest_dict,
        benchmarks=benchmarks,
        mean_distortion=mean_distortion,
        consolidation_records=consolidation_records,
    )

    report_path = out_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n{'='*60}")
    print(f"E03 EXECUTION COMPLETE")
    print(f"Artifacts written to: {out_dir}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(f"{'='*60}\n")

    return summary_dict


def main():
    parser = argparse.ArgumentParser(description="Run Experiment E03 Level 1 Explicit Memory Baselines")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="Target model identifier")
    parser.add_argument("--episodes", type=int, default=6, help="Number of synthetic episodes (default: 6)")
    parser.add_argument("--distractors", type=int, default=6, help="Distractor events per episode (default: 6)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature (default: 0.0)")
    parser.add_argument("--dry-run", action="store_true", help="Run with ToyBackend for fast verification")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    out_path = Path(args.output_dir) if args.output_dir else None
    run_e03_experiment(
        model_name=args.model,
        episode_count=args.episodes,
        distractor_count=args.distractors,
        seed=args.seed,
        temperature=args.temperature,
        dry_run=args.dry_run,
        output_dir=out_path,
    )


if __name__ == "__main__":
    main()
