"""Automated tests enforcing scientific provenance, evidence modes, and manifest invariants."""

import pytest
from src.recurrence.experiment_manifest import (
    EvidenceMode,
    ExecutionEnvironment,
    ExperimentCondition,
    ExperimentManifest,
    ProvenanceMetadata,
)


def test_provenance_rejects_confirmatory_simulation():
    """Scientific Provenance Invariant: CONFIRMATORY_* status is strictly forbidden on SIMULATION runs."""
    manifest = ExperimentManifest(
        experiment_id="fake_simulation",
        gate="GATE_A",
        evidence_mode=EvidenceMode.SIMULATION,
        status="CONFIRMATORY_GATE_PASS",
    )
    with pytest.raises(ValueError, match="Scientific Provenance Violation"):
        manifest.validate()


def test_provenance_rejects_confirmatory_unexecuted_scaffold():
    """Scientific Provenance Invariant: CONFIRMATORY_* status is forbidden on UNEXECUTED_SCAFFOLD runs."""
    manifest = ExperimentManifest(
        experiment_id="fake_scaffold",
        gate="GATE_A",
        evidence_mode=EvidenceMode.UNEXECUTED_SCAFFOLD,
        status="CONFIRMATORY_CLEAN_NULL_EXIT",
    )
    with pytest.raises(ValueError, match="Scientific Provenance Violation"):
        manifest.validate()


def test_provenance_requires_training_steps_for_trained_model():
    manifest = ExperimentManifest(
        experiment_id="trained_without_steps",
        gate="GATE_B",
        evidence_mode=EvidenceMode.TRAINED_MODEL,
        status="CONFIRMATORY_GATE_PASS",
        provenance=ProvenanceMetadata(training_steps=0),
    )
    with pytest.raises(ValueError, match="TRAINED_MODEL requires non-zero training_steps"):
        manifest.validate()


def test_scout_status_allowed_for_scaffold_and_trained():
    """Validates that SCOUT or BASELINE statuses are legal with appropriate evidence modes."""
    scaffold = ExperimentManifest(
        experiment_id="q01_scaffold",
        gate="GATE_A",
        evidence_mode=EvidenceMode.UNEXECUTED_SCAFFOLD,
        status="UNEXECUTED_PIPELINE_SCAFFOLD",
    )
    scaffold.validate() # Should not raise

    trained = ExperimentManifest(
        experiment_id="q04_scout",
        gate="GATE_B",
        evidence_mode=EvidenceMode.TRAINED_MODEL,
        status="SCOUT_GATE_PASS",
        provenance=ProvenanceMetadata(training_steps=500),
    )
    trained.validate() # Should not raise
