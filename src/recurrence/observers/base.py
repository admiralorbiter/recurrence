"""Base classes and data models for Observer and Reconstruction agents."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend


class ObserverEvaluation(BaseModel):
    """Result of an observer evaluation over a target trial."""
    observer_name: str
    predicted_correct: Optional[bool] = None
    observer_confidence: Optional[int] = Field(None, ge=1, le=5)
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
        """Evaluate target response and predict correctness and confidence."""
        pass
