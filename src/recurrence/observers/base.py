"""Base classes and data models for Observer and Reconstruction agents."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend


class ObserverEvaluation(BaseModel):
    """Result of an observer evaluation over a target trial with standardized probability semantics."""
    observer_name: str
    predicted_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Estimated probability (0.0 to 1.0) that target answer is correct")
    predicted_correct: Optional[bool] = Field(None, description="Binary prediction (True if p >= 0.5)")
    reconstructed_answer: Optional[str] = None
    raw_response: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseObserver(ABC):
    """Abstract base observer interface."""

    def __init__(self, backend: Union[OllamaBackend, ToyBackend], name: str = "base_observer"):
        self.backend = backend
        self.name = name

    @abstractmethod
    def evaluate(
        self,
        task_prompt: str,
        target_answer: str,
        item_metadata: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> ObserverEvaluation:
        """Evaluate target response and return standardized probability P(target_correct)."""
        pass
