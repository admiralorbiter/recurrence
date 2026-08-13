"""Unit and integration tests for OllamaBackend and E01 baseline evaluation."""

import shutil
from pathlib import Path
import pytest
from recurrence.backends.ollama import OllamaBackend
from experiments.e01_baseline.run import run_e01_baseline


@pytest.fixture
def tmp_artifact_dir(tmp_path):
    """Temporary artifact directory for E01 tests."""
    d = tmp_path / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_ollama_backend_metadata():
    """Test metadata and connection status for OllamaBackend."""
    backend = OllamaBackend(model_name="qwen2.5:3b")
    digest = backend.get_digest()
    assert isinstance(digest, str)
    assert len(digest) > 0


def test_e01_baseline_execution_toy_fallback(tmp_artifact_dir):
    """Test E01 execution using toy fallback."""
    results = run_e01_baseline(
        use_ollama=False,
        items_per_task=2,
        seed=42,
        output_dir=tmp_artifact_dir / "e01_toy",
        run_id="run_toy_test",
    )

    assert results["total_items"] == 4
    assert Path(results["manifest_path"]).exists()
    assert Path(results["jsonl_path"]).exists()
    assert Path(results["parquet_path"]).exists()


@pytest.mark.integration
def test_e01_baseline_execution_ollama(tmp_artifact_dir):
    """Integration test: run E01 baseline using local Ollama model qwen2.5:3b."""
    results = run_e01_baseline(
        model_name="qwen2.5:3b",
        use_ollama=True,
        items_per_task=2,
        seed=42,
        output_dir=tmp_artifact_dir / "e01_ollama",
        run_id="run_ollama_test",
    )

    assert results["total_items"] == 4
    assert Path(results["manifest_path"]).exists()
    assert Path(results["jsonl_path"]).exists()
    assert Path(results["parquet_path"]).exists()
    assert results["accuracy"] >= 0.0
