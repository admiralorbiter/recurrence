"""E00 Experiment: Replay & Determinism Smoke Test Harness."""

import json
from pathlib import Path
from typing import Dict, Any
from recurrence.core.manifest import RunManifest
from recurrence.core.logging import ExperimentLogger, TrialEvent
from recurrence.backends.toy import ToyBackend


def run_e00_trial(
    seed: int = 42,
    num_steps: int = 5,
    output_dir: str = "artifacts/e00_replay",
    run_id: str = "run_001",
) -> Dict[str, Any]:
    """Execute an E00 trial run and return metadata summary."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Create & hash manifest
    manifest = RunManifest(
        experiment_id="E00",
        run_id=run_id,
        seed=seed,
        model_tag="toy-backend-v1",
        parameters={"num_steps": num_steps, "seed": seed},
    )
    manifest.compute_environment_hash()

    # Save manifest JSON
    manifest_file = out_path / f"{run_id}_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    # 2. Instantiate backend & logger
    backend = ToyBackend(seed=seed)
    logger = ExperimentLogger(output_dir=out_path, run_id=run_id)

    # 3. Step through test trial
    observations = [
        "Initialize task environment",
        "Observe cue A",
        "Receive null observation",
        "Observe cue B",
        "Final query",
    ]

    for step, obs in enumerate(observations[:num_steps]):
        action, state_hash, meta = backend.step(obs)
        event = TrialEvent(
            run_id=run_id,
            step=step,
            event_type="observation_step",
            observation=obs,
            action=action,
            reward=1.0 if step == num_steps - 1 else 0.0,
            latent_state_hash=state_hash,
            metadata=meta,
        )
        logger.log_event(event)

    # 4. Compute stream checksum & export Parquet
    checksum = logger.compute_stream_checksum()
    parquet_path = logger.export_parquet()

    return {
        "manifest_path": str(manifest_file),
        "jsonl_path": str(logger.jsonl_path),
        "parquet_path": str(parquet_path),
        "checksum": checksum,
        "environment_hash": manifest.environment_hash,
    }


if __name__ == "__main__":
    results = run_e00_trial()
    print("E00 Trial completed successfully!")
    print(json.dumps(results, indent=2))
