"""Visible-Evidence Observers: Evaluates visible transcript with standardized probability semantics."""

import re
from typing import Dict, Any, Optional, Union
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.observers.base import BaseObserver, ObserverEvaluation


def _parse_probability_from_text(raw_response: str) -> Optional[float]:
    """Extract probability in [0.0, 1.0] from model response text."""
    prob_match = re.search(r"(?:Probability\s*(?:correct)?|Prob):\s*<?([0-9]+(?:\.[0-9]+)?)\s*\%?>?", raw_response, re.IGNORECASE)
    if prob_match:
        try:
            val = float(prob_match.group(1))
            if val <= 1.0 and ("." in prob_match.group(1) or val == 1.0 or val == 0.0):
                return float(val)
            elif 0.0 <= val <= 100.0:
                return float(val / 100.0)
            else:
                return max(0.0, min(1.0, float(val / 100.0)))
        except ValueError:
            pass

    # Likert fallback
    conf_match = re.search(r"(?:Confidence|Confident):\s*<?([1-5])>?", raw_response, re.IGNORECASE)
    if conf_match:
        try:
            return float(int(conf_match.group(1)) / 5.0)
        except ValueError:
            pass

    # Unstructured toy fallback
    if "action_" in raw_response:
        return 0.5

    return None


class VisibleAnswerOnlyObserver(BaseObserver):
    """Observer that reads task prompt and target's answer with confidence report stripped."""

    def __init__(self, backend: Union[OllamaBackend, ToyBackend], name: str = "observer_visible_answer_only"):
        super().__init__(backend=backend, name=name)

    def format_prompt(self, task_prompt: str, target_answer_only: str) -> str:
        return (
            "You are an external observer evaluating whether another AI agent's answer to a task is correct or incorrect.\n\n"
            f"[Task Prompt]\n{task_prompt.strip()}\n\n"
            f"[Agent's Proposed Answer]\n{target_answer_only.strip()}\n\n"
            "Estimate the probability (0 to 100%) that the proposed answer is correct.\n"
            "Respond strictly in the format:\n"
            "Probability correct: <0 to 100, where 0 is definitely incorrect and 100 is definitely correct>"
        )

    def evaluate(
        self,
        task_prompt: str,
        target_answer: str,
        item_metadata: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> ObserverEvaluation:
        # Strip any confidence report from target_answer
        ans_match = re.search(
            r"Answer:\s*(?:<[^>]+>:\s*)?<?([a-zA-Z0-9_\s]+?)>?(?:\s*(?:Probability|Confidence|Confident)|\%|;|\n|\r|$)",
            target_answer,
            re.IGNORECASE,
        )
        clean_target_ans = f"Answer: {ans_match.group(1).strip()}" if ans_match else target_answer.strip()

        prompt = self.format_prompt(task_prompt, clean_target_ans)
        if isinstance(self.backend, OllamaBackend):
            messages = [{"role": "user", "content": prompt}]
            raw_response, meta = self.backend.chat(messages=messages, temperature=0.0, seed=seed)
        else:
            raw_response, state_hash, meta = self.backend.step(prompt)

        prob = _parse_probability_from_text(raw_response)
        pred_correct = (prob >= 0.5) if prob is not None else None

        return ObserverEvaluation(
            observer_name=self.name,
            predicted_probability=prob,
            predicted_correct=pred_correct,
            raw_response=raw_response,
            metadata={
                "task_prompt_len": len(task_prompt),
                "target_answer_stripped": clean_target_ans,
            },
        )


class VisibleFullTranscriptObserver(BaseObserver):
    """Observer that reads task prompt and target's full response including its stated confidence report."""

    def __init__(self, backend: Union[OllamaBackend, ToyBackend], name: str = "observer_visible_full_transcript"):
        super().__init__(backend=backend, name=name)

    def format_prompt(self, task_prompt: str, full_target_response: str) -> str:
        return (
            "You are an external observer evaluating whether another AI agent's response to a task is correct or incorrect.\n\n"
            f"[Task Prompt]\n{task_prompt.strip()}\n\n"
            f"[Agent's Full Response]\n{full_target_response.strip()}\n\n"
            "Estimate the probability (0 to 100%) that the proposed answer is correct.\n"
            "Respond strictly in the format:\n"
            "Probability correct: <0 to 100, where 0 is definitely incorrect and 100 is definitely correct>"
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

        prob = _parse_probability_from_text(raw_response)
        pred_correct = (prob >= 0.5) if prob is not None else None

        return ObserverEvaluation(
            observer_name=self.name,
            predicted_probability=prob,
            predicted_correct=pred_correct,
            raw_response=raw_response,
            metadata={
                "task_prompt_len": len(task_prompt),
                "full_target_response": target_answer,
            },
        )


# Backward compatibility alias
VisibleEvidenceObserver = VisibleAnswerOnlyObserver
