"""Replay smoke tests verifying deterministic reproducibility for E00 harness."""

import shutil
from pathlib import Path
import pytest
from experiments.e00_replay.run import run_e00_trial


@pytest.fixture
def tmp_artifact_dir(tmp_path):
    """Temporary artifact directory for test runs."""
    d = tmp_path / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_e00_deterministic_replay(tmp_artifact_dir):
    """Verify that two runs with the same seed yield identical checksums and hashes."""
    run1 = run_e00_trial(seed=42, num_steps=5, output_dir=tmp_artifact_dir / "run1", run_id="test_run")
    run2 = run_e00_trial(seed=42, num_steps=5, output_dir=tmp_artifact_dir / "run2", run_id="test_run")

    # Environment and event stream hashes must match exactly
    assert run1["environment_hash"] == run2["environment_hash"]
    assert run1["checksum"] == run2["checksum"]

    # Verify output files exist
    assert Path(run1["manifest_path"]).exists()
    assert Path(run1["jsonl_path"]).exists()
    assert Path(run1["parquet_path"]).exists()


def test_e00_seed_sensitivity(tmp_artifact_dir):
    """Verify that different seeds produce distinct trial event streams."""
    run_seed42 = run_e00_trial(seed=42, num_steps=5, output_dir=tmp_artifact_dir / "seed42", run_id="run_42")
    run_seed99 = run_e00_trial(seed=99, num_steps=5, output_dir=tmp_artifact_dir / "seed99", run_id="run_99")

    # Checksums must differ when seed changes
    assert run_seed42["checksum"] != run_seed99["checksum"]
