"""E01 Experiment: Baseline Task Evaluation Harness (KV Retrieval & Context Tracking)."""

import json
from pathlib import Path
from typing import Dict, Any, Union
from recurrence.core.manifest import RunManifest
from recurrence.core.logging import ExperimentLogger, TrialEvent
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.tasks.kv_retrieval import KVRetrievalTask
from recurrence.tasks.context_tracking import ContextTrackingTask


def run_e01_baseline(
    model_name: str = "qwen2.5:3b",
    use_ollama: bool = True,
    items_per_task: int = 5,
    seed: int = 42,
    output_dir: str = "artifacts/e01_baseline",
    run_id: str = "run_e01_001",
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Execute E01 baseline evaluation across benchmark tasks."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Initialize backend
    backend: Union[OllamaBackend, ToyBackend]
    if use_ollama:
        try:
            backend = OllamaBackend(model_name=model_name, temperature=0.0, seed=seed)
            model_tag = f"ollama-{model_name}"
        except Exception as e:
            print(f"Ollama connection failed ({e}), falling back to ToyBackend.")
            backend = ToyBackend(seed=seed)
            model_tag = "toy-backend-fallback"
    else:
        backend = ToyBackend(seed=seed)
        model_tag = "toy-backend"

    # 2. Create manifest
    manifest = RunManifest(
        experiment_id="E01",
        run_id=run_id,
        seed=seed,
        model_tag=model_tag,
        parameters={"items_per_task": items_per_task, "use_ollama": use_ollama, "model_name": model_name},
    )
    manifest.compute_environment_hash()

    manifest_file = out_path / f"{run_id}_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    # 3. Instantiate logger & tasks
    logger = ExperimentLogger(output_dir=out_path, run_id=run_id, overwrite=overwrite)
    tasks = [
        KVRetrievalTask(distractor_count=5),
        ContextTrackingTask(num_events=5),
    ]

    total_correct = 0
    total_items = 0
    step = 0

    for task in tasks:
        items = task.generate_items(count=items_per_task, seed=seed)
        for item in items:
            step += 1
            if isinstance(backend, OllamaBackend):
                messages = [{"role": "user", "content": item.prompt}]
                response_text, meta = backend.chat(messages=messages, temperature=0.0, seed=seed)
                state_hash = meta.get("digest", "none")[:16]
            else:
                response_text, state_hash, meta = backend.step(item.prompt)

            score_res = task.score_response(item, response_text)
            is_correct = score_res["correct"]
            if is_correct:
                total_correct += 1
            total_items += 1

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

    accuracy = total_correct / total_items if total_items > 0 else 0.0
    checksum = logger.compute_stream_checksum()
    parquet_path = logger.export_parquet()

    return {
        "manifest_path": str(manifest_file),
        "jsonl_path": str(logger.jsonl_path),
        "parquet_path": str(parquet_path),
        "total_items": total_items,
        "accuracy": accuracy,
        "checksum": checksum,
        "environment_hash": manifest.environment_hash,
    }


if __name__ == "__main__":
    results = run_e01_baseline()
    print("E01 Baseline Evaluation completed successfully!")
    print(json.dumps(results, indent=2))
