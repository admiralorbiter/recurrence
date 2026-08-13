"""Typed run configuration and environment manifest tracking."""

import hashlib
import platform
import sys
import time
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import torch


class HardwareInfo(BaseModel):
    """System hardware and CUDA environment metadata."""
    platform: str = Field(default_factory=lambda: platform.platform())
    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    pytorch_version: str = Field(default_factory=lambda: torch.__version__)
    cuda_available: bool = Field(default_factory=lambda: torch.cuda.is_available())
    device_name: str = Field(
        default_factory=lambda: (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )
    )


class RunManifest(BaseModel):
    """Complete immutable run manifest recording experiment parameters and environment."""
    experiment_id: str
    run_id: str
    timestamp: float = Field(default_factory=time.time)
    seed: int
    model_tag: str
    hardware: HardwareInfo = Field(default_factory=HardwareInfo)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    environment_hash: str = ""

    def compute_environment_hash(self) -> str:
        """Compute SHA256 digest of core environment configuration."""
        raw_str = f"{self.experiment_id}:{self.seed}:{self.model_tag}:{self.hardware.pytorch_version}:{self.hardware.device_name}"
        digest = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
        self.environment_hash = digest
        return digest
