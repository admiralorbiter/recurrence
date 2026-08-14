"""Visible-Evidence Observer: Evaluates task prompt and target answer transcript."""

import re
from typing import Dict, Any, Optional, Union
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.observers.base import BaseObserver, ObserverEvaluation


class VisibleEvidenceObserver(BaseObserver):
    """Observer that reads the visible task prompt and target answer transcript."""

    def __init__(self, backend: Union[OllamaBackend, ToyBackend], name: str = "observer_visible"):
        super().__init__(backend=backend, name=name)

    def format_prompt(self, task_prompt: str, target_answer: str) -> str:
        return (
            "You are an external observer evaluating whether another AI agent's answer to a task is correct or incorrect.\n\n"
            f"[Task Prompt]\n{task_prompt.strip()}\n\n"
            f"[Agent's Proposed Answer]\n{target_answer.strip()}\n\n"
            "Based strictly on the prompt and the agent's answer, evaluate whether the proposed answer is correct.\n"
            "Respond in the exact format:\n"
            "Evaluation: [CORRECT or INCORRECT]\n"
            "Confidence: [1-5]"
        )

    def evaluate(
        self,
        task_prompt: str,
        target_answer: str,
        item_metadata: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> ObserverEvaluation:
        prompt = self.format_prompt(task_prompt, target_answer)
        if isinstance(self.backend, OllamaBackend):
            messages = [{"role": "user", "content": prompt}]
            raw_response, meta = self.backend.chat(messages=messages, temperature=0.0, seed=seed)
        else:
            raw_response, state_hash, meta = self.backend.step(prompt)

        # Parse evaluation and confidence
        eval_match = re.search(r"Evaluation:\s*(CORRECT|INCORRECT)", raw_response, re.IGNORECASE)
        conf_match = re.search(r"Confidence:\s*([1-5])", raw_response)

        pred_correct: Optional[bool] = None
        if eval_match:
            pred_correct = eval_match.group(1).upper() == "CORRECT"
        elif "action_" in raw_response:
            # Deterministic fallback for ToyBackend test simulation
            pred_correct = True

        conf: Optional[int] = None
        if conf_match:
            try:
                conf = int(conf_match.group(1))
            except ValueError:
                conf = None
        elif "action_" in raw_response:
            conf = 3

        return ObserverEvaluation(
            observer_name=self.name,
            predicted_correct=pred_correct,
            observer_confidence=conf,
            raw_response=raw_response,
            metadata={
                "task_prompt_len": len(task_prompt),
                "target_answer": target_answer,
                "parsed_eval": eval_match.group(1).upper() if eval_match else None,
            },
        )
