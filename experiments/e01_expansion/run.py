"""E01 Expansion: Paired Factorial Matrix, Hardened Directory Isolation, and Auto-Emitted Artifacts."""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
from recurrence.core.manifest import RunManifest
from recurrence.core.logging import ExperimentLogger, TrialEvent
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.tasks.kv_retrieval import KVRetrievalTask
from recurrence.tasks.context_tracking import ContextTrackingTask
from recurrence.analysis.calibration import (
    compute_post_decision_discrimination_from_pairs,
)


def run_e01_expansion(
    model_name: str = "qwen2.5:3b",
    use_ollama: bool = True,
    items_per_condition: int = 20,
    seed: int = 42,
    base_output_dir: str = "artifacts/e01_expansion",
    results_dir: str = "results/e01_expansion",
    run_id: str = "run_e01_exp_002",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Execute E01 expansion battery with atomic run directories, paired items, and auto-generated result tables."""
    run_dir = Path(base_output_dir) / run_id
    res_dir = Path(results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize collision-safe logger (fails if directory exists and overwrite=False)
    logger = ExperimentLogger(output_dir=run_dir, run_id=run_id, overwrite=overwrite)

    # 2. Initialize backend with strict verification
    backend: Union[OllamaBackend, ToyBackend]
    if use_ollama:
        backend = OllamaBackend(model_name=model_name, temperature=0.0, seed=seed)
        model_tag = f"ollama-{model_name}"
        model_digest = backend.get_digest()
    else:
        backend = ToyBackend(seed=seed)
        model_tag = "toy-backend"
        model_digest = "toy-digest-sha256"

    # 3. Preregistered Manifest saved through logger
    manifest = RunManifest(
        experiment_id="E01_Expansion_Hardened",
        run_id=run_id,
        seed=seed,
        model_tag=model_tag,
        model_digest=model_digest,
        parameters={
            "items_per_condition": items_per_condition,
            "use_ollama": use_ollama,
            "model_name": model_name,
            "design": "paired_2x2_factorial_with_confidence_control_and_interleaved_context",
        },
    )
    manifest.compute_environment_hash()
    logger.save_manifest(manifest)

    # 4. Generate PAIRED underlying KV instances for true within-item factorial comparison
    raw_semantic_pairs = KVRetrievalTask.generate_raw_pairs(
        count=items_per_condition, distractor_count=5, identifier_type="semantic", seed=seed
    )
    raw_opaque_pairs = KVRetrievalTask.generate_raw_pairs(
        count=items_per_condition, distractor_count=5, identifier_type="opaque", seed=seed
    )

    # Instantiate paired task conditions
    task_semantic_fc = KVRetrievalTask(identifier_type="semantic", mode="forced_choice", ask_confidence=True)
    task_semantic_fg = KVRetrievalTask(identifier_type="semantic", mode="free_generation", ask_confidence=True)
    task_opaque_fc = KVRetrievalTask(identifier_type="opaque", mode="forced_choice", ask_confidence=True)
    task_opaque_fg = KVRetrievalTask(identifier_type="opaque", mode="free_generation", ask_confidence=True)
    # Confidence intervention control: Opaque Free Gen WITHOUT asking for confidence
    task_opaque_fg_noconf = KVRetrievalTask(identifier_type="opaque", mode="free_generation", ask_confidence=False)

    # Interleaved Context Tracking Task
    task_context_interleaved = ContextTrackingTask(num_objects=3, total_transitions=6)

    # Map tasks to their specific items
    benchmark_plan = [
        (task_semantic_fc, task_semantic_fc.generate_items_from_raw(raw_semantic_pairs, seed=seed)),
        (task_semantic_fg, task_semantic_fg.generate_items_from_raw(raw_semantic_pairs, seed=seed)),
        (task_opaque_fc, task_opaque_fc.generate_items_from_raw(raw_opaque_pairs, seed=seed)),
        (task_opaque_fg, task_opaque_fg.generate_items_from_raw(raw_opaque_pairs, seed=seed)),
        (task_opaque_fg_noconf, task_opaque_fg_noconf.generate_items_from_raw(raw_opaque_pairs, seed=seed)),
        (task_context_interleaved, task_context_interleaved.generate_items(count=items_per_condition, seed=seed)),
    ]

    step = 0
    condition_summaries: Dict[str, Dict[str, Any]] = {}
    all_confidence_pairs: List[Tuple[Optional[int], bool]] = []
    all_correct: List[bool] = []
    context_substitution_counts: Dict[str, int] = {}
    context_lag_performance: Dict[str, Dict[str, int]] = {}
    trial_records: List[Dict[str, Any]] = []

    # Track paired outcomes for confidence reactivity check
    opaque_fg_with_conf_correct: List[bool] = []
    opaque_fg_without_conf_correct: List[bool] = []

    for task, items in benchmark_plan:
        task_correct = 0
        task_pairs: List[Tuple[Optional[int], bool]] = []
        failure_counts: Dict[str, int] = {}

        for item in items:
            step += 1
            if isinstance(backend, OllamaBackend):
                messages = [{"role": "user", "content": item.prompt}]
                response_text, meta = backend.chat(messages=messages, temperature=0.0, seed=seed)
                state_hash = meta.get("digest", "none")[:16]
            else:
                response_text, state_hash, meta = backend.step(item.prompt)

            score_res = task.score_response(item, response_text)
            is_corr = score_res["correct"]
            conf = score_res.get("confidence")

            if is_corr:
                task_correct += 1
            else:
                fail_type = score_res.get("failure_type") or score_res.get("substitution_category") or "unresolved"
                failure_counts[fail_type] = failure_counts.get(fail_type, 0) + 1
                if isinstance(task, ContextTrackingTask) and score_res.get("substitution_category"):
                    sub_cat = score_res["substitution_category"]
                    context_substitution_counts[sub_cat] = context_substitution_counts.get(sub_cat, 0) + 1

            if isinstance(task, ContextTrackingTask):
                lag_key = f"lag_k_{score_res.get('lag_k')}"
                if lag_key not in context_lag_performance:
                    context_lag_performance[lag_key] = {"total": 0, "correct": 0}
                context_lag_performance[lag_key]["total"] += 1
                if is_corr:
                    context_lag_performance[lag_key]["correct"] += 1

            if task.name == "kv_opaque_free_generation_conf":
                opaque_fg_with_conf_correct.append(is_corr)
            elif task.name == "kv_opaque_free_generation_noconf":
                opaque_fg_without_conf_correct.append(is_corr)

            task_pairs.append((conf, is_corr))
            if conf is not None:
                all_confidence_pairs.append((conf, is_corr))
            all_correct.append(is_corr)

            meta.update(score_res)
            meta["task_name"] = task.name
            meta["item_id"] = item.item_id

            event = TrialEvent(
                run_id=run_id,
                step=step,
                event_type=f"{task.name}_step",
                observation=item.prompt,
                action=response_text,
                reward=score_res["score"],
                latent_state_hash=state_hash,
                metadata=meta,
            )
            logger.log_event(event)

            # Record compact structured trial record for first-class commit
            trial_records.append({
                "run_id": run_id,
                "step": step,
                "task_name": task.name,
                "item_id": item.item_id,
                "target_key": item.metadata.get("target_key"),
                "target_object": item.metadata.get("target_object"),
                "target_option_letter": item.metadata.get("target_option_letter"),
                "ground_truth": item.ground_truth,
                "parsed_answer": score_res.get("parsed_answer"),
                "normalized_answer": score_res.get("normalized_answer"),
                "correct": is_corr,
                "confidence": conf,
                "lag_k": score_res.get("lag_k"),
                "failure_type": score_res.get("failure_type") or score_res.get("substitution_category"),
            })

        task_acc = task_correct / len(items) if items else 0.0
        cal_metrics = compute_post_decision_discrimination_from_pairs(task_pairs)
        condition_summaries[task.name] = {
            "total_items": len(items),
            "correct": task_correct,
            "accuracy": task_acc,
            "failure_counts": failure_counts,
            "discrimination": cal_metrics,
        }

    overall_accuracy = sum(all_correct) / len(all_correct) if all_correct else 0.0
    overall_discrimination = compute_post_decision_discrimination_from_pairs(all_confidence_pairs)
    checksum = logger.compute_stream_checksum()
    parquet_path = logger.export_parquet()

    # Compute paired reactivity contingency table
    both_correct = sum(w and wo for w, wo in zip(opaque_fg_with_conf_correct, opaque_fg_without_conf_correct))
    conf_only = sum(w and not wo for w, wo in zip(opaque_fg_with_conf_correct, opaque_fg_without_conf_correct))
    noconf_only = sum(not w and wo for w, wo in zip(opaque_fg_with_conf_correct, opaque_fg_without_conf_correct))
    both_wrong = sum(not w and not wo for w, wo in zip(opaque_fg_with_conf_correct, opaque_fg_without_conf_correct))

    results_summary = {
        "experiment_id": "E01_Expansion_Hardened",
        "sprint": "S02",
        "run_id": run_id,
        "model_name": model_name,
        "model_digest": model_digest,
        "seed": seed,
        "manifest_path": str(logger.manifest_path),
        "jsonl_path": str(logger.jsonl_path),
        "parquet_path": str(parquet_path),
        "total_items": len(all_correct),
        "overall_accuracy": overall_accuracy,
        "paired_factorial_2x2_matrix": {
            "semantic_forced_choice": condition_summaries.get("kv_semantic_forced_choice_conf", {}).get("accuracy"),
            "semantic_free_generation": condition_summaries.get("kv_semantic_free_generation_conf", {}).get("accuracy"),
            "opaque_forced_choice": condition_summaries.get("kv_opaque_forced_choice_conf", {}).get("accuracy"),
            "opaque_free_generation": condition_summaries.get("kv_opaque_free_generation_conf", {}).get("accuracy"),
        },
        "main_effects": {
            "forced_choice_mean": (
                (condition_summaries.get("kv_semantic_forced_choice_conf", {}).get("accuracy", 0.0) +
                 condition_summaries.get("kv_opaque_forced_choice_conf", {}).get("accuracy", 0.0)) / 2.0
            ),
            "free_generation_mean": (
                (condition_summaries.get("kv_semantic_free_generation_conf", {}).get("accuracy", 0.0) +
                 condition_summaries.get("kv_opaque_free_generation_conf", {}).get("accuracy", 0.0)) / 2.0
            ),
            "semantic_mean": (
                (condition_summaries.get("kv_semantic_forced_choice_conf", {}).get("accuracy", 0.0) +
                 condition_summaries.get("kv_semantic_free_generation_conf", {}).get("accuracy", 0.0)) / 2.0
            ),
            "opaque_mean": (
                (condition_summaries.get("kv_opaque_forced_choice_conf", {}).get("accuracy", 0.0) +
                 condition_summaries.get("kv_opaque_free_generation_conf", {}).get("accuracy", 0.0)) / 2.0
            ),
        },
        "confidence_elicitation_paired_contingency": {
            "with_confidence_accuracy": condition_summaries.get("kv_opaque_free_generation_conf", {}).get("accuracy"),
            "without_confidence_accuracy": condition_summaries.get("kv_opaque_free_generation_noconf", {}).get("accuracy"),
            "both_correct": both_correct,
            "conf_only_correct": conf_only,
            "noconf_only_correct": noconf_only,
            "both_wrong": both_wrong,
        },
        "context_tracking_interleaved": {
            "overall_accuracy": condition_summaries.get(task_context_interleaved.name, {}).get("accuracy"),
            "lag_performance": {
                k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
                for k, v in sorted(context_lag_performance.items())
            },
            "error_distribution": context_substitution_counts,
            "metacognitive_discrimination": condition_summaries.get(task_context_interleaved.name, {}).get("discrimination"),
        },
        "overall_post_decision_discrimination": overall_discrimination,
        "condition_details": condition_summaries,
        "checksum": checksum,
        "environment_hash": manifest.environment_hash,
    }

    # Automatically write canonical result files to results_dir
    trials_file = res_dir / "trials.jsonl"
    with open(trials_file, "w", encoding="utf-8") as f:
        for rec in trial_records:
            f.write(json.dumps(rec) + "\n")

    summary_file = res_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(results_summary, indent=2))

    return results_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run E01 Expansion benchmark.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing run directory.")
    parser.add_argument("--items", type=int, default=20, help="Items per condition.")
    parser.add_argument("--run-id", type=str, default="run_e01_exp_002", help="Run ID.")
    args = parser.parse_args()

    res = run_e01_expansion(items_per_condition=args.items, run_id=args.run_id, overwrite=args.overwrite)
    print("E01 Expansion Hardened Run Completed!")
    print(json.dumps(res, indent=2))
