"""Event stream logging, JSONL persistence, checksumming, and Parquet export."""

import json
import os
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import duckdb
from pydantic import BaseModel, Field


class TrialEvent(BaseModel):
    """Single trial event step within an experiment run."""
    run_id: str
    step: int
    event_type: str
    observation: Any = None
    action: Any = None
    reward: float = 0.0
    latent_state_hash: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentLogger:
    """JSONL event stream logger with collision guards, checksum calculation, and Parquet export."""

    def __init__(self, output_dir: Union[str, Path], run_id: str, overwrite: bool = False):
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.overwrite = overwrite
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / f"{run_id}_events.jsonl"
        self.events: List[TrialEvent] = []

        if self.jsonl_path.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Run ID log already exists at {self.jsonl_path}. "
                    "Aborting to prevent accidental event stream corruption. "
                    "Provide a unique run_id or pass overwrite=True."
                )
            else:
                self.jsonl_path.unlink()

    def log_event(self, event: TrialEvent) -> None:
        """Append a trial event to memory and JSONL stream."""
        self.events.append(event)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

    def compute_stream_checksum(self) -> str:
        """Calculate SHA256 checksum over the raw JSONL event stream."""
        hasher = hashlib.sha256()
        if self.jsonl_path.exists():
            with open(self.jsonl_path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
        return hasher.hexdigest()

    def export_parquet(self, parquet_filename: Optional[str] = None) -> Path:
        """Export logged JSONL events to a columnar Parquet file using DuckDB."""
        if parquet_filename is None:
            parquet_filename = f"{self.run_id}_events.parquet"
        parquet_path = self.output_dir / parquet_filename

        conn = duckdb.connect(database=":memory:")
        jsonl_str = str(self.jsonl_path).replace("\\", "/")
        parquet_str = str(parquet_path).replace("\\", "/")

        conn.execute(
            f"COPY (SELECT * FROM read_json_auto('{jsonl_str}')) TO '{parquet_str}' (FORMAT PARQUET)"
        )
        conn.close()
        return parquet_path
