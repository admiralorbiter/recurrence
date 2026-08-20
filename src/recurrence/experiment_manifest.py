"""Standardized Experiment Manifest dataclasses, provenance tracking, and validation for Recurrence."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Dict, List, Optional
import uuid


def get_git_sha(repo_root: Optional[Path] = None) -> str:
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        cwd = str(repo_root) if repo_root else None
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_DIRTY"


class EvidenceMode(str, Enum):
    LIVE_MODEL = "LIVE_MODEL"                     # Live model forward/generation calls
    TRAINED_MODEL = "TRAINED_MODEL"               # Live end-to-end trained model (e.g. Garden GRU)
    OFFLINE_ANALYSIS = "OFFLINE_ANALYSIS"         # Analysis on verified frozen raw cache
    SIMULATION = "SIMULATION"                     # Analytical/algebraic mock or dry-run
    UNEXECUTED_SCAFFOLD = "UNEXECUTED_SCAFFOLD"   # Code pipeline without empirical execution


@dataclass
class LineageMetadata:
    lineage_id: str
    parent_lineage_id: Optional[str] = None
    fork_step: int = 0
    event_hash: Optional[str] = None


@dataclass
class ExecutionEnvironment:
    device: str = "cpu"
    precision: str = "fp32"
    batch_size: int = 1
    torch_version: Optional[str] = None
    python_version: str = field(default_factory=lambda: platform.python_version())
    platform: str = field(default_factory=lambda: platform.platform())


@dataclass
class ExperimentCondition:
    name: str
    manipulation_type: str
    intervention_target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceMetadata:
    forward_calls: int = 0
    training_steps: int = 0
    raw_record_count: int = 0
    activation_cache_hash: Optional[str] = None
    raw_results_hash: Optional[str] = None
    source_run_ids: List[str] = field(default_factory=list)


@dataclass
class ExperimentManifest:
    experiment_id: str
    gate: str
    run_id: str = field(default_factory=lambda: f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}")
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_sha: str = field(default_factory=get_git_sha)
    evidence_mode: EvidenceMode = EvidenceMode.LIVE_MODEL
    status: str = "SCOUT"
    model_revision: str = "v0.1.0"
    environment_revision: str = "v0.1.0"
    seed: int = 42
    lineage: Optional[LineageMetadata] = None
    execution: ExecutionEnvironment = field(default_factory=ExecutionEnvironment)
    condition: Optional[ExperimentCondition] = None
    provenance: ProvenanceMetadata = field(default_factory=ProvenanceMetadata)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Enforces scientific provenance and status invariants."""
        # 1. Confirmatory status cannot be applied to simulated or unexecuted runs
        if self.status.startswith("CONFIRMATORY_"):
            if self.evidence_mode in [EvidenceMode.SIMULATION, EvidenceMode.UNEXECUTED_SCAFFOLD]:
                raise ValueError(
                    f"Scientific Provenance Violation: Status '{self.status}' is illegal for evidence_mode '{self.evidence_mode}'."
                )
            if self.evidence_mode == EvidenceMode.TRAINED_MODEL and self.provenance.training_steps == 0:
                raise ValueError("TRAINED_MODEL requires non-zero training_steps in provenance.")
            if self.evidence_mode == EvidenceMode.LIVE_MODEL and self.provenance.forward_calls == 0:
                raise ValueError("LIVE_MODEL requires non-zero forward_calls in provenance.")

    def compute_and_set_results_hash(self, results_data: Any) -> str:
        """Computes SHA256 of results data and sets it in provenance."""
        data_str = json.dumps(results_data, sort_keys=True)
        h = hashlib.sha256(data_str.encode("utf-8")).hexdigest()
        self.provenance.raw_results_hash = h
        return h

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        d = asdict(self)
        d["evidence_mode"] = self.evidence_mode.value if isinstance(self.evidence_mode, EvidenceMode) else self.evidence_mode
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
