"""Standardized Experiment Manifest dataclasses, provenance tracking, and validation for Recurrence (Provenance v1.2)."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Dict, List, Optional, Tuple
import uuid


def get_git_state(repo_root: Optional[Path] = None) -> Tuple[str, bool]:
    try:
        cmd_sha = ["git", "rev-parse", "HEAD"]
        cmd_status = ["git", "status", "--porcelain"]
        cwd = str(repo_root) if repo_root else None
        res_sha = subprocess.run(cmd_sha, cwd=cwd, capture_output=True, text=True, check=True)
        res_status = subprocess.run(cmd_status, cwd=cwd, capture_output=True, text=True, check=True)
        sha = res_sha.stdout.strip()
        is_dirty = len(res_status.stdout.strip()) > 0
        return sha, is_dirty
    except Exception:
        return "UNKNOWN_COMMIT", True


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
    forward_calls: int = 0          # Number of neural model forward passes
    training_steps: int = 0         # Number of optimizer update steps
    raw_record_count: int = 0       # Number of evaluated episode/trial records
    raw_trial_table_path: Optional[str] = None
    raw_trial_table_sha256: Optional[str] = None
    activation_cache_hash: Optional[str] = None
    raw_results_hash: Optional[str] = None
    source_run_ids: List[str] = field(default_factory=list)


@dataclass
class ExperimentManifest:
    experiment_id: str
    gate: str
    run_id: str = field(default_factory=lambda: f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}")
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_sha: str = field(default_factory=lambda: get_git_state()[0])
    worktree_dirty: bool = field(default_factory=lambda: get_git_state()[1])
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
        """Enforces scientific provenance and status invariants across all evidence modes."""
        # 1. UNEXECUTED_SCAFFOLD compatibility
        if self.evidence_mode == EvidenceMode.UNEXECUTED_SCAFFOLD:
            if not (self.status.startswith("UNEXECUTED_") or self.status == "PIPELINE_READY"):
                raise ValueError(
                    f"Scientific Provenance Violation: UNEXECUTED_SCAFFOLD cannot have status '{self.status}'. "
                    f"Allowed: 'UNEXECUTED_PIPELINE_SCAFFOLD', 'PIPELINE_READY'."
                )

        # 2. SIMULATION compatibility
        if self.evidence_mode == EvidenceMode.SIMULATION:
            if not (self.status.startswith("SIMULATION_") or self.status in ["METHOD_CHECK", "DRY_RUN"]):
                raise ValueError(
                    f"Scientific Provenance Violation: SIMULATION cannot have status '{self.status}'. "
                    f"Allowed: 'SIMULATION_COMPLETE', 'METHOD_CHECK', 'DRY_RUN'."
                )

        # 3. Confirmatory / Promotion requirements for live/trained runs
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

    def save_trial_records_jsonl(self, path: Path, records: List[Dict[str, Any]]) -> str:
        """Saves individual trial records to JSONL, computes hash, and updates provenance."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        hasher = hashlib.sha256()
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                line = json.dumps(rec, sort_keys=True) + "\n"
                f.write(line)
                hasher.update(line.encode("utf-8"))
        
        file_hash = hasher.hexdigest()
        self.provenance.raw_trial_table_path = str(path)
        self.provenance.raw_trial_table_sha256 = file_hash
        self.provenance.raw_record_count = len(records)
        return file_hash

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
