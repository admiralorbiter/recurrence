"""Scientific Runner for Experiment E02_Observer_Hardened (Sprint S03.2):
Definitive Level-0 Privileged Access & Observer Ladder Benchmark.

Standardizes all evaluative measurements on P(Target Correct) in [0, 100%].
Evaluates 8 Total Conditions:
  1 Target Model (Immediate self-confidence)
  + 7 Evaluator Conditions:
    1. Equal-Compute Self-Review (fresh 2nd invocation, self-framing)
    2. Equal-Compute Other-Review (fresh 2nd invocation, other-framing)
    3. Visible Observer: Answer-Only (confidence stripped)
    4. Visible Observer: Full-Transcript (confidence included)
    5. Counterfactual Reconstruction Observer (4-way distribution lookup)
    6. Input-Only Observer (difficulty prior)
    7. Output-Only Observer (fluency prior)
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.logging.manifest import ExperimentManifest
from recurrence.logging.structured import ExperimentLogger, TrialEvent
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
    compute_direct_pairwise_contrast,
)


def run_e02_observer(
    items_per_stratum: int = 20,
    run_id: str = "run_e02_obs_003",
    overwrite: bool = False,
    sesoi: float = 0.10,
    seed: int = 42,
    use_toy: bool = False,
) -> Dict[str, Any]:
    """Execute the definitive Level-0 Privileged Access Benchmark under strict paired intersections."""
    # 1. Initialize Backends
    if use_toy:
        target_backend = ToyBackend(name="toy_target")
        obs_backend = ToyBackend(name="toy_observer")
        model_name = "toy_model"
        model_digest = "toy_digest_deterministic"
    else:
        target_backend = OllamaBackend(model_name="qwen2.5:3b", seed=seed)
        obs_backend = OllamaBackend(model_name="qwen2.5:3b", seed=seed)
        model_name = target_backend.model_name
        model_digest = target_backend.get_digest()

    # 2. Setup Safe Result and Artifact Directories
    artifacts_dir = Path("artifacts") / "e02_observer" / run_id
    res_base = Path("results") / "e02_observer"
    res_run_dir = res_base / run_id

    if (artifacts_dir.exists() or res_run_dir.exists()) and not overwrite:
        raise FileExistsError(
            f"Run directory already exists. Use overwrite=True or a new run_id. Paths: {artifacts_dir}, {res_run_dir}"
        )

    if overwrite:
        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir)
        if res_run_dir.exists():
            shutil.rmtree(res_run_dir)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    res_run_dir.mkdir(parents=True, exist_ok=True)

    # 3. Create Manifest and Logger
    manifest = ExperimentManifest.create(
        experiment_id="E02_Observer_Hardened",
        protocol_version="1.3.2",
        parameters={
            "sprint": "S03.2",
            "model_name": model_name,
            "model_digest": model_digest,
            "seed": seed,
            "items_per_stratum": items_per_stratum,
            "total_items": items_per_stratum * 2,
            "evaluator_conditions": [
                "self_immediate",
                "self_review_equal_compute",
                "observer_review_other",
                "observer_visible_answer_only",
                "observer_visible_full_transcript",
                "observer_reconstruction",
                "observer_input_only",
                "observer_output_only",
            ],
            "sesoi": sesoi,
            "confidence_format": "probability",
            "decoding": "greedy_temperature_0.0",
        },
        tags=["e02_observer", "s03_2", "level_0", "hardened", "stratified_bootstrap"],
    )

    logger = ExperimentLogger(
        manifest=manifest,
        run_id=run_id,
        output_dir=artifacts_dir,
        overwrite=overwrite,
    )

    # 4. Initialize Tasks and Observers
    task_semantic = KVRetrievalTask(
        mode="forced_choice",
        identifier_type="semantic",
        ask_confidence=True,
        confidence_format="probability",
    )
    task_opaque = KVRetrievalTask(
        mode="forced_choice",
        identifier_type="opaque",
        ask_confidence=True,
        confidence_format="probability",
    )

    raw_semantic = task_semantic.generate_raw_pairs(count=items_per_stratum, seed=seed)
    items_semantic = task_semantic.generate_items_from_raw(raw_semantic, seed=seed)

    raw_opaque = task_opaque.generate_raw_pairs(count=items_per_stratum, seed=seed + 1000)
    items_opaque = task_opaque.generate_items_from_raw(raw_opaque, seed=seed + 1000)

    # 7 Evaluator conditions
    obs_self_review = EqualComputeReviewObserver(backend=obs_backend, framing="self")
    obs_other_review = EqualComputeReviewObserver(backend=obs_backend, framing="other")
    obs_vis_ans = VisibleAnswerOnlyObserver(backend=obs_backend)
    obs_vis_full = VisibleFullTranscriptObserver(backend=obs_backend)
    obs_recon = ReconstructionObserver(backend=obs_backend)
    obs_input = InputOnlyObserver(backend=obs_backend)
    obs_output = OutputOnlyObserver(backend=obs_backend)

    # Data structures for strict paired intersection
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

    trial_records: List[Dict[str, Any]] = []
    semantic_target_correct = 0
    opaque_target_correct = 0

    all_test_items = [(task_semantic, it) for it in items_semantic] + [(task_opaque, it) for it in items_opaque]
    step = 0

    for task, item in all_test_items:
        step += 1
        item_id = item.item_id

        # Step A: Target model solves the item with structured JSON decoding
        if isinstance(target_backend, OllamaBackend):
            messages = [{"role": "user", "content": item.prompt}]
            target_response, meta = target_backend.chat(messages=messages, temperature=0.0, seed=seed, format="json")
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

    # 5. Compute Strict Item-Paired Intersection Contrasts & PAI (Stratified Bootstrap)
    contrast_analysis = compute_item_paired_contrasts(
        self_item_map=self_item_map,
        observer_item_maps=observer_item_maps,
        sesoi=sesoi,
        n_bootstraps=1000,
        seed=seed,
    )

    # 6. Direct Pairwise Contrasts
    # A. Self-Review vs Other-Review (Equal Compute framing test)
    framing_contrast = compute_direct_pairwise_contrast(
        map_a=observer_item_maps["self_review_equal_compute"],
        map_b=observer_item_maps["observer_review_other"],
        name_a="self_review_equal_compute",
        name_b="observer_review_other",
        sesoi=sesoi,
        n_bootstraps=1000,
        seed=seed,
    )

    # B. Visible Answer-Only vs Visible Full-Transcript (Public channel effect test)
    channel_contrast = compute_direct_pairwise_contrast(
        map_a=observer_item_maps["observer_visible_answer_only"],
        map_b=observer_item_maps["observer_visible_full_transcript"],
        name_a="observer_visible_answer_only",
        name_b="observer_visible_full_transcript",
        sesoi=sesoi,
        n_bootstraps=1000,
        seed=seed,
    )

    # 7. Compliance Metrics & Gate Verification
    self_valid_count = sum(1 for p, y in self_item_map.values() if p is not None)
    observer_valid_counts = {
        name: sum(1 for p, y in mp.values() if p is not None)
        for name, mp in observer_item_maps.items()
    }
    core_conditions = [
        "self_review_equal_compute",
        "observer_review_other",
        "observer_visible_answer_only",
        "observer_visible_full_transcript",
        "observer_reconstruction",
    ]
    core_valid_total = self_valid_count + sum(observer_valid_counts[c] for c in core_conditions)
    core_possible_total = total_items * (1 + len(core_conditions))
    core_compliance_rate = core_valid_total / core_possible_total if core_possible_total else 0.0

    # Brier Scores
    brier_scores = {
        "self_immediate": compute_continuous_brier_score(list(self_item_map.values())),
    }
    for name, mp in observer_item_maps.items():
        brier_scores[name] = compute_continuous_brier_score(list(mp.values()))

    # Summary dictionary
    results_summary = {
        "experiment_id": "E02_Observer_Hardened",
        "sprint": "S03.2",
        "run_id": run_id,
        "model_name": model_name,
        "model_digest": model_digest,
        "seed": seed,
        "total_items": total_items,
        "compliance_rates": {
            "self_valid_count": self_valid_count,
            "self_compliance_rate": self_valid_count / total_items if total_items else 0.0,
            "observer_valid_counts": observer_valid_counts,
            "core_compliance_rate": core_compliance_rate,
            "compliance_gate_passed": bool(core_compliance_rate >= 0.90),
        },
        "target_task_performance": {
            "semantic_fc_accuracy": semantic_target_correct / items_per_stratum if items_per_stratum else 0.0,
            "opaque_fc_accuracy": opaque_target_correct / items_per_stratum if items_per_stratum else 0.0,
            "overall_accuracy": (semantic_target_correct + opaque_target_correct) / total_items if total_items else 0.0,
        },
        "observer_brier_scores": brier_scores,
        "paired_intersection_contrasts": contrast_analysis["contrasts"],
        "direct_pairwise_contrasts": {
            "framing_self_vs_other_review": framing_contrast,
            "channel_answer_only_vs_full_transcript": channel_contrast,
        },
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
    parser.add_argument("--run-id", type=str, default="run_e02_obs_003", help="Run ID.")
    parser.add_argument("--sesoi", type=float, default=0.10, help="Smallest Effect Size of Interest margin.")
    args = parser.parse_args()

    res = run_e02_observer(items_per_stratum=args.items, run_id=args.run_id, overwrite=args.overwrite, sesoi=args.sesoi)
    print("E02 Hardened Observer Benchmark Completed!")
    print(json.dumps(res, indent=2))
