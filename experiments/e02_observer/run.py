"""E02 Hardened Observer & Reconstruction Benchmark Runner (S03.1)."""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional
from recurrence.core.manifest import RunManifest
from recurrence.core.logging import ExperimentLogger, TrialEvent
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.tasks.kv_retrieval import KVRetrievalTask
from recurrence.observers.visible import (
    VisibleAnswerOnlyObserver,
    VisibleFullTranscriptObserver,
)
from recurrence.observers.reconstruction import ReconstructionObserver
from recurrence.observers.ablated import (
    EqualComputeReviewObserver,
    InputOnlyObserver,
    OutputOnlyObserver,
)
from recurrence.analysis.privileged_access import (
    compute_continuous_brier_score,
    compute_item_paired_contrasts,
)


def run_e02_observer(
    model_name: str = "qwen2.5:3b",
    use_ollama: bool = True,
    items_per_stratum: int = 20,
    seed: int = 42,
    base_output_dir: str = "artifacts/e02_observer",
    results_base_dir: str = "results/e02_observer",
    run_id: str = "run_e02_obs_002",
    overwrite: bool = False,
    sesoi: float = 0.10,
) -> Dict[str, Any]:
    """Execute E02 Hardened Observer Battery across counterbalanced Forced Choice KV retrieval."""
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
        experiment_id="E02_Observer_Hardened",
        sprint="S03.1",
        run_id=run_id,
        seed=seed,
        model_tag=model_tag,
        model_digest=model_digest,
        parameters={
            "items_per_stratum": items_per_stratum,
            "use_ollama": use_ollama,
            "model_name": model_name,
            "metric": "probability_0_to_100_percent",
            "sesoi_margin": sesoi,
            "design": "6_condition_observer_battery_with_strict_paired_intersections",
            "conditions": [
                "self_immediate",
                "self_review_equal_compute",
                "observer_review_other",
                "observer_visible_answer_only",
                "observer_visible_full_transcript",
                "observer_reconstruction",
                "observer_input_only",
                "observer_output_only",
            ],
        },
    )
    manifest.compute_environment_hash()
    logger.save_manifest(manifest)

    # 4. Generate Stimuli: Counterbalanced 4-way FC KV Retrieval (5 targets per letter A/B/C/D)
    raw_semantic = KVRetrievalTask.generate_raw_pairs(
        count=items_per_stratum, distractor_count=5, identifier_type="semantic", seed=seed
    )
    raw_opaque = KVRetrievalTask.generate_raw_pairs(
        count=items_per_stratum, distractor_count=5, identifier_type="opaque", seed=seed
    )

    task_semantic_fc = KVRetrievalTask(
        identifier_type="semantic", mode="forced_choice", ask_confidence=True, confidence_format="probability"
    )
    task_opaque_fc = KVRetrievalTask(
        identifier_type="opaque", mode="forced_choice", ask_confidence=True, confidence_format="probability"
    )

    items_semantic = task_semantic_fc.generate_items_from_raw(raw_semantic, seed=seed)
    items_opaque = task_opaque_fc.generate_items_from_raw(raw_opaque, seed=seed)

    all_test_items = [
        (task_semantic_fc, item) for item in items_semantic
    ] + [
        (task_opaque_fc, item) for item in items_opaque
    ]

    # Instantiate Observers
    obs_self_review = EqualComputeReviewObserver(backend=target_backend, framing="self", name="self_review_equal_compute")
    obs_other_review = EqualComputeReviewObserver(backend=observer_backend, framing="other", name="observer_review_other")
    obs_vis_ans = VisibleAnswerOnlyObserver(backend=observer_backend, name="observer_visible_answer_only")
    obs_vis_full = VisibleFullTranscriptObserver(backend=observer_backend, name="observer_visible_full_transcript")
    obs_recon = ReconstructionObserver(backend=observer_backend, name="observer_reconstruction")
    obs_input = InputOnlyObserver(backend=observer_backend, name="observer_input_only")
    obs_output = OutputOnlyObserver(backend=observer_backend, name="observer_output_only")

    trial_records: List[Dict[str, Any]] = []
    
    # Item-level maps for strict pairwise intersection analysis
    self_item_map: Dict[str, Tuple[Optional[float], bool]] = {}
    observer_item_maps: Dict[str, Dict[str, Tuple[Optional[float], bool]]] = {
        "self_review_equal_compute": {},
        "observer_review_other": {},
        "observer_visible_answer_only": {},
        "observer_visible_full_transcript": {},
        "observer_reconstruction": {},
        "observer_input_only": {},
        "observer_output_only": {},
    }

    semantic_target_correct = 0
    opaque_target_correct = 0
    step = 0

    for task, item in all_test_items:
        step += 1
        item_id = item.item_id

        # Step A: Target model solves the item
        if isinstance(target_backend, OllamaBackend):
            messages = [{"role": "user", "content": item.prompt}]
            target_response, meta = target_backend.chat(messages=messages, temperature=0.0, seed=seed)
            state_hash = meta.get("digest", "none")[:16]
        else:
            target_response, state_hash, meta = target_backend.step(item.prompt)

        score_res = task.score_response(item, target_response)
        is_corr = score_res["correct"]
        self_prob = score_res.get("probability")

        if is_corr:
            if "semantic" in task.name:
                semantic_target_correct += 1
            else:
                opaque_target_correct += 1

        self_item_map[item_id] = (self_prob, is_corr)

        # Step B: Observers evaluate target trial
        eval_self_rev = obs_self_review.evaluate(item.prompt, target_response, item_metadata=item.metadata, seed=seed)
        eval_oth_rev = obs_other_review.evaluate(item.prompt, target_response, item_metadata=item.metadata, seed=seed)
        eval_vis_ans = obs_vis_ans.evaluate(item.prompt, target_response, item_metadata=item.metadata, seed=seed)
        eval_vis_full = obs_vis_full.evaluate(item.prompt, target_response, item_metadata=item.metadata, seed=seed)
        eval_recon = obs_recon.evaluate(item.prompt, target_response, item_metadata=item.metadata, seed=seed)
        eval_inp = obs_input.evaluate(item.prompt, target_response, item_metadata=item.metadata, seed=seed)
        eval_out = obs_output.evaluate(item.prompt, target_response, item_metadata=item.metadata, seed=seed)

        observer_item_maps["self_review_equal_compute"][item_id] = (eval_self_rev.predicted_probability, is_corr)
        observer_item_maps["observer_review_other"][item_id] = (eval_oth_rev.predicted_probability, is_corr)
        observer_item_maps["observer_visible_answer_only"][item_id] = (eval_vis_ans.predicted_probability, is_corr)
        observer_item_maps["observer_visible_full_transcript"][item_id] = (eval_vis_full.predicted_probability, is_corr)
        observer_item_maps["observer_reconstruction"][item_id] = (eval_recon.predicted_probability, is_corr)
        observer_item_maps["observer_input_only"][item_id] = (eval_inp.predicted_probability, is_corr)
        observer_item_maps["observer_output_only"][item_id] = (eval_out.predicted_probability, is_corr)

        # Log trial event
        meta.update(score_res)
        meta["task_name"] = task.name
        meta["item_id"] = item.item_id
        meta["eval_self_review"] = eval_self_rev.model_dump()
        meta["eval_other_review"] = eval_oth_rev.model_dump()
        meta["eval_visible_answer_only"] = eval_vis_ans.model_dump()
        meta["eval_visible_full_transcript"] = eval_vis_full.model_dump()
        meta["eval_reconstruction"] = eval_recon.model_dump()
        meta["eval_input_only"] = eval_inp.model_dump()
        meta["eval_output_only"] = eval_out.model_dump()

        event = TrialEvent(
            run_id=run_id,
            step=step,
            event_type=f"{task.name}_observer_hardened_trial",
            observation=item.prompt,
            action=target_response,
            reward=score_res["score"],
            latent_state_hash=state_hash,
            metadata=meta,
        )
        logger.log_event(event)

        # Committed Item-Level Record
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
            "self_probability": self_prob,
            "self_review_prob": eval_self_rev.predicted_probability,
            "observer_review_other_prob": eval_oth_rev.predicted_probability,
            "obs_vis_ans_prob": eval_vis_ans.predicted_probability,
            "obs_vis_full_prob": eval_vis_full.predicted_probability,
            "obs_recon_prob": eval_recon.predicted_probability,
            "obs_recon_answer": eval_recon.reconstructed_answer,
            "obs_input_prob": eval_inp.predicted_probability,
            "obs_output_prob": eval_out.predicted_probability,
        })

    total_items = len(all_test_items)
    checksum = logger.compute_stream_checksum()
    parquet_path = logger.export_parquet()

    # 5. Compute Strict Item-Paired Intersection Contrasts & PAI
    contrast_analysis = compute_item_paired_contrasts(
        self_item_map=self_item_map,
        observer_item_maps=observer_item_maps,
        sesoi=sesoi,
        n_bootstraps=1000,
        seed=seed,
    )

    # 6. Overall Compliance Metrics
    self_valid_count = sum(1 for p, y in self_item_map.values() if p is not None)
    observer_valid_counts = {
        name: sum(1 for p, y in mp.values() if p is not None)
        for name, mp in observer_item_maps.items()
    }

    # Brier Scores
    brier_scores = {
        "self_immediate": compute_continuous_brier_score(list(self_item_map.values())),
    }
    for name, mp in observer_item_maps.items():
        brier_scores[name] = compute_continuous_brier_score(list(mp.values()))

    # Summary dictionary
    results_summary = {
        "experiment_id": "E02_Observer_Hardened",
        "sprint": "S03.1",
        "run_id": run_id,
        "model_name": model_name,
        "model_digest": model_digest,
        "seed": seed,
        "total_items": total_items,
        "compliance_rates": {
            "self_valid_count": self_valid_count,
            "self_compliance_rate": self_valid_count / total_items if total_items else 0.0,
            "observer_valid_counts": observer_valid_counts,
        },
        "target_task_performance": {
            "semantic_fc_accuracy": semantic_target_correct / items_per_stratum if items_per_stratum else 0.0,
            "opaque_fc_accuracy": opaque_target_correct / items_per_stratum if items_per_stratum else 0.0,
            "overall_accuracy": (semantic_target_correct + opaque_target_correct) / total_items if total_items else 0.0,
        },
        "observer_brier_scores": brier_scores,
        "paired_intersection_contrasts": contrast_analysis["contrasts"],
        "joint_pai_summary": contrast_analysis["joint_pai_summary"],
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
                "experiment_id": "E02_Observer_Hardened",
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
    parser = argparse.ArgumentParser(description="Run E02 Hardened Observer & Reconstruction benchmark.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing run directory.")
    parser.add_argument("--items", type=int, default=20, help="Items per condition stratum.")
    parser.add_argument("--run-id", type=str, default="run_e02_obs_002", help="Run ID.")
    parser.add_argument("--sesoi", type=float, default=0.10, help="Smallest Effect Size of Interest margin.")
    args = parser.parse_args()

    res = run_e02_observer(items_per_stratum=args.items, run_id=args.run_id, overwrite=args.overwrite, sesoi=args.sesoi)
    print("E02 Hardened Observer Benchmark Completed!")
    print(json.dumps(res, indent=2))
