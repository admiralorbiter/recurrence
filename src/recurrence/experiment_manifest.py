"""Standardized Experiment Manifest dataclasses and serialization for Recurrence."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Dict, Optional
import uuid


def get_git_sha(repo_root: Optional[Path] = None) -> str:
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        cwd = str(repo_root) if repo_root else None
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_DIRTY"


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
class ExperimentManifest:
    experiment_id: str
    gate: str
    run_id: str = field(default_factory=lambda: f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}")
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_sha: str = field(default_factory=get_git_sha)
    model_revision: str = "v0.1.0"
    environment_revision: str = "v0.1.0"
    seed: int = 42
    lineage: Optional[LineageMetadata] = None
    execution: ExecutionEnvironment = field(default_factory=ExecutionEnvironment)
    condition: Optional[ExperimentCondition] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    status: str = "SCOUT"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
