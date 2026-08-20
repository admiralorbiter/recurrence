"""Tests for Standardized Experiment Manifest validation and serialization."""

import json
from pathlib import Path
from src.recurrence.experiment_manifest import (
    ExecutionEnvironment,
    ExperimentCondition,
    ExperimentManifest,
    LineageMetadata,
)


def test_experiment_manifest_serialization(tmp_path: Path):
    manifest = ExperimentManifest(
        experiment_id="Q04_switchboard_scout",
        gate="GATE_B",
        seed=42,
        lineage=LineageMetadata(lineage_id="org_A0", fork_step=0),
        execution=ExecutionEnvironment(device="cpu", precision="fp32", batch_size=32),
        condition=ExperimentCondition(name="gru_delay16", manipulation_type="baseline"),
        metrics={"accuracy": 0.94, "oracle": 1.0},
    )

    manifest_file = tmp_path / "manifest.json"
    manifest.save(manifest_file)
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["experiment_id"] == "Q04_switchboard_scout"
    assert loaded["gate"] == "GATE_B"
    assert loaded["seed"] == 42
    assert loaded["lineage"]["lineage_id"] == "org_A0"
    assert loaded["metrics"]["accuracy"] == 0.94
