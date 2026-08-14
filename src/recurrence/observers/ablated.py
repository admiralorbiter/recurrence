"""Ablated Observers: Input-only and Output-only baseline evaluators."""

import re
from typing import Dict, Any, Optional, Union
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.observers.base import BaseObserver, ObserverEvaluation


class InputOnlyObserver(BaseObserver):
    """Observer that sees only the task prompt (not the target answer) to predict difficulty."""

    def __init__(self, backend: Union[OllamaBackend, ToyBackend], name: str = "observer_input_only"):
        super().__init__(backend=backend, name=name)

    def evaluate(
        self,
        task_prompt: str,
        target_answer: str,
        item_metadata: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> ObserverEvaluation:
        prompt = (
            "You are an evaluator assessing the difficulty of the following task prompt for an AI agent.\n\n"
            f"[Task Prompt]\n{task_prompt.strip()}\n\n"
            "Will a standard language model likely answer this prompt correctly or incorrectly?\n"
            "Respond in the format:\n"
            "Evaluation: [CORRECT or INCORRECT]\n"
            "Confidence: [1-5]"
        )
        if isinstance(self.backend, OllamaBackend):
            messages = [{"role": "user", "content": prompt}]
            raw_response, meta = self.backend.chat(messages=messages, temperature=0.0, seed=seed)
        else:
            raw_response, state_hash, meta = self.backend.step(prompt)

        eval_match = re.search(r"Evaluation:\s*(CORRECT|INCORRECT)", raw_response, re.IGNORECASE)
        conf_match = re.search(r"Confidence:\s*([1-5])", raw_response)

        pred_correct = eval_match.group(1).upper() == "CORRECT" if eval_match else True
        conf = int(conf_match.group(1)) if conf_match else 3

        return ObserverEvaluation(
            observer_name=self.name,
            predicted_correct=pred_correct,
            observer_confidence=conf,
            raw_response=raw_response,
            metadata={"parsed_eval": eval_match.group(1).upper() if eval_match else None},
        )


class OutputOnlyObserver(BaseObserver):
    """Observer that sees only the target's answer string to score stylistic/fluency heuristics."""

    def __init__(self, backend: Union[OllamaBackend, ToyBackend], name: str = "observer_output_only"):
        super().__init__(backend=backend, name=name)

    def evaluate(
        self,
        task_prompt: str,
        target_answer: str,
        item_metadata: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> ObserverEvaluation:
        prompt = (
            "You are an evaluator assessing whether an AI agent's answer format appears valid and correct.\n\n"
            f"[Agent's Answer]\n{target_answer.strip()}\n\n"
            "Based strictly on formatting, clarity, and certainty, evaluate if this answer appears correct:\n"
            "Evaluation: [CORRECT or INCORRECT]\n"
            "Confidence: [1-5]"
        )
        if isinstance(self.backend, OllamaBackend):
            messages = [{"role": "user", "content": prompt}]
            raw_response, meta = self.backend.chat(messages=messages, temperature=0.0, seed=seed)
        else:
            raw_response, state_hash, meta = self.backend.step(prompt)

        eval_match = re.search(r"Evaluation:\s*(CORRECT|INCORRECT)", raw_response, re.IGNORECASE)
        conf_match = re.search(r"Confidence:\s*([1-5])", raw_response)

        pred_correct = eval_match.group(1).upper() == "CORRECT" if eval_match else True
        conf = int(conf_match.group(1)) if conf_match else 3

        return ObserverEvaluation(
            observer_name=self.name,
            predicted_correct=pred_correct,
            observer_confidence=conf,
            raw_response=raw_response,
            metadata={"target_answer": target_answer},
        )
