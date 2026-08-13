"""E01 Expansion: Paired Factorial Matrix, Interleaved Context Tracking, and Confidence Controls."""

import json
from pathlib import Path
from typing import Dict, Any, List, Union
from recurrence.core.manifest import RunManifest
from recurrence.core.logging import ExperimentLogger, TrialEvent
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.tasks.kv_retrieval import KVRetrievalTask
from recurrence.tasks.context_tracking import ContextTrackingTask
from recurrence.analysis.calibration import compute_calibration_metrics


def run_e01_expansion(
    model_name: str = "qwen2.5:3b",
    use_ollama: bool = True,
    items_per_condition: int = 20,
    seed: int = 42,
    output_dir: str = "artifacts/e01_expansion",
    run_id: str = "run_e01_exp_002",
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Execute E01 expansion battery with paired factorial items, interleaved tracking, and confidence controls."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Initialize backend with strict verification (fail fast on scientific runs)
    backend: Union[OllamaBackend, ToyBackend]
    if use_ollama:
        backend = OllamaBackend(model_name=model_name, temperature=0.0, seed=seed)
        model_tag = f"ollama-{model_name}"
        model_digest = backend.get_digest()
    else:
        backend = ToyBackend(seed=seed)
        model_tag = "toy-backend"
        model_digest = "toy-digest-sha256"

    # 2. Preregistered Manifest
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

    manifest_file = out_path / f"{run_id}_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    # 3. Generate PAIRED underlying KV instances for true within-item factorial comparison
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

    logger = ExperimentLogger(output_dir=out_path, run_id=run_id, overwrite=overwrite)
    step = 0
    condition_summaries: Dict[str, Dict[str, Any]] = {}
    all_confidences: List[int] = []
    all_correct: List[bool] = []
    context_substitution_counts: Dict[str, int] = {}
    context_lag_performance: Dict[str, Dict[str, int]] = {}

    for task, items in benchmark_plan:
        task_correct = 0
        task_confidences = []
        task_correct_flags = []
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

            task_correct_flags.append(is_corr)
            all_correct.append(is_corr)
            conf = score_res.get("confidence")
            task_confidences.append(conf)
            if conf is not None:
                all_confidences.append(conf)

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

        task_acc = task_correct / len(items) if items else 0.0
        cal_metrics = compute_calibration_metrics(task_confidences, task_correct_flags)
        condition_summaries[task.name] = {
            "total_items": len(items),
            "correct": task_correct,
            "accuracy": task_acc,
            "failure_counts": failure_counts,
            "calibration": cal_metrics,
        }

    overall_accuracy = sum(all_correct) / len(all_correct) if all_correct else 0.0
    overall_calibration = compute_calibration_metrics(all_confidences, all_correct[:len(all_confidences)])
    checksum = logger.compute_stream_checksum()
    parquet_path = logger.export_parquet()

    results_summary = {
        "manifest_path": str(manifest_file),
        "jsonl_path": str(logger.jsonl_path),
        "parquet_path": str(parquet_path),
        "total_items": len(all_correct),
        "overall_accuracy": overall_accuracy,
        "factorial_2x2_matrix": {
            "semantic_forced_choice": condition_summaries.get("kv_semantic_forced_choice_conf", {}).get("accuracy"),
            "semantic_free_generation": condition_summaries.get("kv_semantic_free_generation_conf", {}).get("accuracy"),
            "opaque_forced_choice": condition_summaries.get("kv_opaque_forced_choice_conf", {}).get("accuracy"),
            "opaque_free_generation": condition_summaries.get("kv_opaque_free_generation_conf", {}).get("accuracy"),
        },
        "confidence_elicitation_intervention_check": {
            "opaque_free_generation_with_conf": condition_summaries.get("kv_opaque_free_generation_conf", {}).get("accuracy"),
            "opaque_free_generation_without_conf": condition_summaries.get("kv_opaque_free_generation_noconf", {}).get("accuracy"),
        },
        "context_tracking": {
            "overall_accuracy": condition_summaries.get(task_context_interleaved.name, {}).get("accuracy"),
            "lag_performance": {
                k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
                for k, v in context_lag_performance.items()
            },
            "error_distribution": context_substitution_counts,
            "metacognitive_discrimination": condition_summaries.get(task_context_interleaved.name, {}).get("calibration"),
        },
        "condition_details": condition_summaries,
        "checksum": checksum,
        "environment_hash": manifest.environment_hash,
    }

    return results_summary


if __name__ == "__main__":
    res = run_e01_expansion(items_per_condition=20, run_id="run_e01_exp_002", overwrite=True)
    print("E01 Expansion Hardened Run Completed!")
    print(json.dumps(res, indent=2))
