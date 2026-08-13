"""Typed run configuration and environment manifest tracking."""

import hashlib
import json
import platform
import sys
import time
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import torch


class HardwareInfo(BaseModel):
    """Snapshot of execution hardware, OS, and PyTorch environment."""
    pytorch_version: str = Field(default_factory=lambda: torch.__version__)
    cuda_available: bool = Field(default_factory=lambda: torch.cuda.is_available())
    device_name: str = Field(
        default_factory=lambda: torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    python_version: str = Field(default_factory=lambda: platform.python_version())
    os_platform: str = Field(default_factory=lambda: f"{platform.system()}-{platform.release()}")
    git_commit: str = Field(default_factory=lambda: _get_git_commit())


def _get_git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"


class RunManifest(BaseModel):
    """Preregistered execution manifest capturing complete parameters and environment hash."""
    experiment_id: str
    run_id: str
    seed: int
    model_tag: str
    model_digest: str = "unknown"
    hardware: HardwareInfo = Field(default_factory=HardwareInfo)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    environment_hash: str = ""

    def compute_environment_hash(self) -> str:
        """Compute SHA256 digest of complete environment configuration."""
        raw_str = (
            f"{self.experiment_id}:{self.seed}:{self.model_tag}:{self.model_digest}:"
            f"{self.hardware.python_version}:{self.hardware.os_platform}:"
            f"{self.hardware.pytorch_version}:{self.hardware.device_name}:"
            f"{self.hardware.git_commit}:{json.dumps(self.parameters, sort_keys=True)}"
        )
        digest = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
        self.environment_hash = digest
        return digest
