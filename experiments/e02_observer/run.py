"""E02 Observer and Reconstruction Controls Benchmark Runner."""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional
from recurrence.core.manifest import RunManifest
from recurrence.core.logging import ExperimentLogger, TrialEvent
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.tasks.kv_retrieval import KVRetrievalTask
from recurrence.observers.visible import VisibleEvidenceObserver
from recurrence.observers.reconstruction import ReconstructionObserver
from recurrence.observers.ablated import InputOnlyObserver, OutputOnlyObserver
from recurrence.analysis.privileged_access import (
    compute_privileged_access_index,
    compute_brier_score_from_predictions,
)
from recurrence.analysis.calibration import (
    compute_post_decision_discrimination_from_pairs,
)


def run_e02_observer(
    model_name: str = "qwen2.5:3b",
    use_ollama: bool = True,
    items_per_stratum: int = 20,
    seed: int = 42,
    base_output_dir: str = "artifacts/e02_observer",
    results_base_dir: str = "results/e02_observer",
    run_id: str = "run_e02_obs_001",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Execute E02 Observer Ladder battery across counterbalanced Forced Choice KV retrieval."""
    run_dir = Path(base_output_dir) / run_id
    res_base = Path(results_base_dir)
    res_run_dir = res_base / run_id
    res_run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize atomic collision-guarded logger
    logger = ExperimentLogger(output_dir=run_dir, run_id=run_id, overwrite=overwrite)

    # 2. Initialize backends (target and observers)
    target_backend: Union[OllamaBackend, ToyBackend]
    observer_backend: Union[OllamaBackend, ToyBackend]
    if use_ollama:
        target_backend = OllamaBackend(model_name=model_name, temperature=0.0, seed=seed)
        observer_backend = OllamaBackend(model_name=model_name, temperature=0.0, seed=seed)
        model_tag = f"ollama-{model_name}"
        model_digest = target_backend.get_digest()
    else:
        target_backend = ToyBackend(seed=seed)
        observer_backend = ToyBackend(seed=seed)
        model_tag = "toy-backend"
        model_digest = "toy-digest-sha256"

    # 3. Preregistered Manifest
    manifest = RunManifest(
        experiment_id="E02_Observer_Baseline",
        run_id=run_id,
        seed=seed,
        model_tag=model_tag,
        model_digest=model_digest,
        parameters={
            "items_per_stratum": items_per_stratum,
            "use_ollama": use_ollama,
            "model_name": model_name,
            "design": "4_rung_observer_ladder_over_counterbalanced_fc_kv_retrieval",
            "rungs": [
                "self_post_decision",
                "observer_visible_evidence",
                "observer_reconstruction",
                "observer_input_only",
                "observer_output_only",
            ],
        },
    )
    manifest.compute_environment_hash()
    logger.save_manifest(manifest)

    # 4. Generate Stimuli: 20 Semantic FC + 20 Opaque FC (counterbalanced 5 targets per letter A/B/C/D)
    raw_semantic = KVRetrievalTask.generate_raw_pairs(
        count=items_per_stratum, distractor_count=5, identifier_type="semantic", seed=seed
    )
    raw_opaque = KVRetrievalTask.generate_raw_pairs(
        count=items_per_stratum, distractor_count=5, identifier_type="opaque", seed=seed
    )

    task_semantic_fc = KVRetrievalTask(identifier_type="semantic", mode="forced_choice", ask_confidence=True)
    task_opaque_fc = KVRetrievalTask(identifier_type="opaque", mode="forced_choice", ask_confidence=True)

    items_semantic = task_semantic_fc.generate_items_from_raw(raw_semantic, seed=seed)
    items_opaque = task_opaque_fc.generate_items_from_raw(raw_opaque, seed=seed)

    all_test_items = [
        (task_semantic_fc, item) for item in items_semantic
    ] + [
        (task_opaque_fc, item) for item in items_opaque
    ]

    # Instantiate Observers
    obs_visible = VisibleEvidenceObserver(backend=observer_backend, name="observer_visible")
    obs_recon = ReconstructionObserver(backend=observer_backend, name="observer_reconstruction")
    obs_input = InputOnlyObserver(backend=observer_backend, name="observer_input_only")
    obs_output = OutputOnlyObserver(backend=observer_backend, name="observer_output_only")

    trial_records: List[Dict[str, Any]] = []
    self_confidence_pairs: List[Tuple[Optional[int], bool]] = []
    obs_visible_pairs: List[Tuple[Optional[int], bool]] = []
    obs_recon_pairs: List[Tuple[Optional[int], bool]] = []
    obs_input_pairs: List[Tuple[Optional[int], bool]] = []
    obs_output_pairs: List[Tuple[Optional[int], bool]] = []

    obs_visible_forecasts: List[Tuple[Optional[bool], bool]] = []
    obs_recon_forecasts: List[Tuple[Optional[bool], bool]] = []
    obs_input_forecasts: List[Tuple[Optional[bool], bool]] = []
    obs_output_forecasts: List[Tuple[Optional[bool], bool]] = []

    semantic_target_correct = 0
    opaque_target_correct = 0
    step = 0

    for task, item in all_test_items:
        step += 1
        # Step A: Target model solves the item
        if isinstance(target_backend, OllamaBackend):
            messages = [{"role": "user", "content": item.prompt}]
            target_response, meta = target_backend.chat(messages=messages, temperature=0.0, seed=seed)
            state_hash = meta.get("digest", "none")[:16]
        else:
            target_response, state_hash, meta = target_backend.step(item.prompt)

        score_res = task.score_response(item, target_response)
        is_corr = score_res["correct"]
        self_conf = score_res.get("confidence")

        if is_corr:
            if "semantic" in task.name:
                semantic_target_correct += 1
            else:
                opaque_target_correct += 1

        self_confidence_pairs.append((self_conf, is_corr))

        # Step B: Observers evaluate target trial
        eval_vis = obs_visible.evaluate(item.prompt, target_response, item_metadata=item.metadata, seed=seed)
        eval_rec = obs_recon.evaluate(item.prompt, target_response, item_metadata=item.metadata, seed=seed)
        eval_inp = obs_input.evaluate(item.prompt, target_response, item_metadata=item.metadata, seed=seed)
        eval_out = obs_output.evaluate(item.prompt, target_response, item_metadata=item.metadata, seed=seed)

        obs_visible_pairs.append((eval_vis.observer_confidence, is_corr))
        obs_recon_pairs.append((eval_rec.observer_confidence, is_corr))
        obs_input_pairs.append((eval_inp.observer_confidence, is_corr))
        obs_output_pairs.append((eval_out.observer_confidence, is_corr))

        obs_visible_forecasts.append((eval_vis.predicted_correct, is_corr))
        obs_recon_forecasts.append((eval_rec.predicted_correct, is_corr))
        obs_input_forecasts.append((eval_inp.predicted_correct, is_corr))
        obs_output_forecasts.append((eval_out.predicted_correct, is_corr))

        # Log trial event
        meta.update(score_res)
        meta["task_name"] = task.name
        meta["item_id"] = item.item_id
        meta["eval_visible"] = eval_vis.model_dump()
        meta["eval_reconstruction"] = eval_rec.model_dump()
        meta["eval_input_only"] = eval_inp.model_dump()
        meta["eval_output_only"] = eval_out.model_dump()

        event = TrialEvent(
            run_id=run_id,
            step=step,
            event_type=f"{task.name}_observer_trial",
            observation=item.prompt,
            action=target_response,
            reward=score_res["score"],
            latent_state_hash=state_hash,
            metadata=meta,
        )
        logger.log_event(event)

        # Committed Trial Record
        trial_records.append({
            "run_id": run_id,
            "step": step,
            "task_name": task.name,
            "item_id": item.item_id,
            "target_key": item.metadata.get("target_key"),
            "target_option_letter": item.metadata.get("target_option_letter"),
            "ground_truth": item.ground_truth,
            "target_response": target_response,
            "target_parsed_answer": score_res.get("parsed_answer"),
            "target_correct": is_corr,
            "target_confidence": self_conf,
            "obs_visible_pred_correct": eval_vis.predicted_correct,
            "obs_visible_confidence": eval_vis.observer_confidence,
            "obs_recon_pred_correct": eval_rec.predicted_correct,
            "obs_recon_confidence": eval_rec.observer_confidence,
            "obs_recon_answer": eval_rec.reconstructed_answer,
            "obs_input_pred_correct": eval_inp.predicted_correct,
            "obs_input_confidence": eval_inp.observer_confidence,
            "obs_output_pred_correct": eval_out.predicted_correct,
            "obs_output_confidence": eval_out.observer_confidence,
        })

    total_items = len(all_test_items)
    checksum = logger.compute_stream_checksum()
    parquet_path = logger.export_parquet()

    # 5. Compute Privileged Access Index and Discrimination Analytics
    observer_dict = {
        "observer_visible": obs_visible_pairs,
        "observer_reconstruction": obs_recon_pairs,
        "observer_input_only": obs_input_pairs,
        "observer_output_only": obs_output_pairs,
    }
    pai_analysis = compute_privileged_access_index(
        self_pairs=self_confidence_pairs,
        observer_dict=observer_dict,
        n_bootstraps=1000,
        seed=seed,
    )

    # 6. Compute Observer Forecast Calibration (Brier Scores)
    brier_scores = {
        "observer_visible": compute_brier_score_from_predictions(obs_visible_forecasts),
        "observer_reconstruction": compute_brier_score_from_predictions(obs_recon_forecasts),
        "observer_input_only": compute_brier_score_from_predictions(obs_input_forecasts),
        "observer_output_only": compute_brier_score_from_predictions(obs_output_forecasts),
    }

    # Observer Accuracy (percentage of times observer correctly predicted target correctness)
    def compute_obs_accuracy(forecasts: List[Tuple[Optional[bool], bool]]) -> Optional[float]:
        valid = [f for f in forecasts if f[0] is not None]
        if not valid:
            return None
        return float(sum(1 for pred, actual in valid if pred == actual) / len(valid))

    observer_accuracies = {
        "observer_visible": compute_obs_accuracy(obs_visible_forecasts),
        "observer_reconstruction": compute_obs_accuracy(obs_recon_forecasts),
        "observer_input_only": compute_obs_accuracy(obs_input_forecasts),
        "observer_output_only": compute_obs_accuracy(obs_output_forecasts),
    }

    # Summary dictionary
    results_summary = {
        "experiment_id": "E02_Observer_Baseline",
        "sprint": "S03",
        "run_id": run_id,
        "model_name": model_name,
        "model_digest": model_digest,
        "seed": seed,
        "total_items": total_items,
        "target_task_performance": {
            "semantic_fc_accuracy": semantic_target_correct / items_per_stratum if items_per_stratum else 0.0,
            "opaque_fc_accuracy": opaque_target_correct / items_per_stratum if items_per_stratum else 0.0,
            "overall_accuracy": (semantic_target_correct + opaque_target_correct) / total_items if total_items else 0.0,
        },
        "privileged_access_index": pai_analysis,
        "observer_brier_scores": brier_scores,
        "observer_prediction_accuracies": observer_accuracies,
        "manifest_path": str(logger.manifest_path),
        "jsonl_path": str(logger.jsonl_path),
        "parquet_path": str(parquet_path),
        "checksum": checksum,
        "environment_hash": manifest.environment_hash,
    }

    # Automatically write canonical result files to res_run_dir
    trials_file = res_run_dir / "trials.jsonl"
    with open(trials_file, "w", encoding="utf-8") as f:
        for rec in trial_records:
            f.write(json.dumps(rec) + "\n")

    summary_file = res_run_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(results_summary, indent=2))

    # Update latest pointer in res_base
    latest_pointer_file = res_base / "latest.json"
    with open(latest_pointer_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": "E02_Observer_Baseline",
                "promoted_run_id": run_id,
                "summary_path": str(summary_file.relative_to(res_base.parent.parent)),
                "trials_path": str(trials_file.relative_to(res_base.parent.parent)),
            },
            f,
            indent=2,
        )

    return results_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run E02 Observer & Reconstruction benchmark.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing run directory.")
    parser.add_argument("--items", type=int, default=20, help="Items per condition stratum.")
    parser.add_argument("--run-id", type=str, default="run_e02_obs_001", help="Run ID.")
    args = parser.parse_args()

    res = run_e02_observer(items_per_stratum=args.items, run_id=args.run_id, overwrite=args.overwrite)
    print("E02 Observer Benchmark Completed!")
    print(json.dumps(res, indent=2))
