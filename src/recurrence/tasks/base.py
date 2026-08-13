"""Base classes and schemas for research tasks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskItem(BaseModel):
    """Single benchmark item with prompt, ground truth, and scoring parameters."""
    item_id: str
    prompt: str
    ground_truth: str
    distractors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseTask(ABC):
    """Abstract base class for benchmark tasks."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def generate_items(self, count: int = 10, seed: int = 42) -> List[TaskItem]:
        """Generate a reproducible set of benchmark items."""
        pass

    @abstractmethod
    def score_response(self, item: TaskItem, response: str) -> Dict[str, Any]:
        """Evaluate model response against ground truth.

        Returns:
            Dict containing 'correct' (bool), 'score' (float 0..1), and detailed metrics.
        """
        pass
