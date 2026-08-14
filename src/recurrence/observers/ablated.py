"""Ablated & Equal-Compute Review Observers with standardized probability semantics."""

import json
import re
from typing import Dict, Any, Optional, Union
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.observers.base import BaseObserver, ObserverEvaluation
from recurrence.observers.visible import _parse_probability_from_text


class EqualComputeReviewObserver(BaseObserver):
    """Observer matching compute budget (fresh 2nd invocation), contrasting self vs other framing."""

    def __init__(
        self,
        backend: Union[OllamaBackend, ToyBackend],
        framing: str = "other",  # 'self' or 'other'
        name: Optional[str] = None,
    ):
        assigned_name = name or (f"observer_review_{framing}" if framing == "other" else "self_review_equal_compute")
        super().__init__(backend=backend, name=assigned_name)
        self.framing = framing

    def format_prompt(self, task_prompt: str, target_answer: str) -> str:
        base_prompt = re.sub(r"Respond strictly with a JSON object.*", "", task_prompt, flags=re.DOTALL).strip()
        base_prompt = re.sub(r"Format your response strictly as:.*", "", base_prompt, flags=re.DOTALL).strip()

        # Clean answer
        clean_ans = target_answer.strip()
        json_match = re.search(r"\{[^{}]*\}", clean_ans)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if "answer" in data:
                    clean_ans = f"Answer: {data['answer']}"
            except Exception:
                pass

        if clean_ans == target_answer.strip():
            ans_match = re.search(
                r"Answer:\s*(?:<[^>]+>:\s*)?<?([a-zA-Z0-9_\s]+?)>?(?:\s*(?:Probability|Confidence|Confident)|\%|;|\n|\r|$)",
                target_answer,
                re.IGNORECASE,
            )
            clean_ans = f"Answer: {ans_match.group(1).strip()}" if ans_match else target_answer.strip()

        if self.framing == "self":
            return (
                "You are reviewing your own previous answer to a task.\n\n"
                f"[Task Prompt]\n{base_prompt}\n\n"
                f"[Your Previous Answer]\n{clean_ans}\n\n"
                "Estimate the probability (0 to 100%) that your answer is correct.\n"
                "Respond strictly with a JSON object in this format with no other text:\n"
                '{\n  "probability": <0 to 100, where 0 is definitely incorrect and 100 is definitely correct>\n}'
            )
        else:
            return (
                "You are reviewing an answer produced by another AI model for a task.\n\n"
                f"[Task Prompt]\n{base_prompt}\n\n"
                f"[Other Model's Answer]\n{clean_ans}\n\n"
                "Estimate the probability (0 to 100%) that the proposed answer is correct.\n"
                "Respond strictly with a JSON object in this format with no other text:\n"
                '{\n  "probability": <0 to 100, where 0 is definitely incorrect and 100 is definitely correct>\n}'
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
            raw_response, meta = self.backend.chat(messages=messages, temperature=0.0, seed=seed, format="json")
        else:
            raw_response, state_hash, meta = self.backend.step(prompt)

        prob = _parse_probability_from_text(raw_response)
        pred_correct = (prob >= 0.5) if prob is not None else None

        return ObserverEvaluation(
            observer_name=self.name,
            predicted_probability=prob,
            predicted_correct=pred_correct,
            raw_response=raw_response,
            metadata={"framing": self.framing},
        )


class InputOnlyObserver(BaseObserver):
    """Observer that sees only task prompt (difficulty prior) to estimate P(Correct)."""

    def __init__(self, backend: Union[OllamaBackend, ToyBackend], name: str = "observer_input_only"):
        super().__init__(backend=backend, name=name)

    def evaluate(
        self,
        task_prompt: str,
        target_answer: str,
        item_metadata: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> ObserverEvaluation:
        base_prompt = re.sub(r"Respond strictly with a JSON object.*", "", task_prompt, flags=re.DOTALL).strip()
        base_prompt = re.sub(r"Format your response strictly as:.*", "", base_prompt, flags=re.DOTALL).strip()

        prompt = (
            "You are an evaluator assessing the difficulty of the following task prompt for an AI agent.\n\n"
            f"[Task Prompt]\n{base_prompt}\n\n"
            "Estimate the probability (0 to 100%) that a standard AI model will solve this prompt correctly.\n"
            "Respond strictly with a JSON object in this format with no other text:\n"
            '{\n  "probability": <0 to 100, where 0 is definitely incorrect and 100 is definitely correct>\n}'
        )
        if isinstance(self.backend, OllamaBackend):
            messages = [{"role": "user", "content": prompt}]
            raw_response, meta = self.backend.chat(messages=messages, temperature=0.0, seed=seed, format="json")
        else:
            raw_response, state_hash, meta = self.backend.step(prompt)

        prob = _parse_probability_from_text(raw_response)
        pred_correct = (prob >= 0.5) if prob is not None else None

        return ObserverEvaluation(
            observer_name=self.name,
            predicted_probability=prob,
            predicted_correct=pred_correct,
            raw_response=raw_response,
            metadata={"parsed_prob": prob},
        )


class OutputOnlyObserver(BaseObserver):
    """Observer that sees only the target answer string to score stylistic/fluency priors."""

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
            "You are an evaluator assessing whether an AI agent's answer format appears valid and plausible.\n\n"
            f"[Agent's Answer]\n{target_answer.strip()}\n\n"
            "Estimate the probability (0 to 100%) that this answer is valid and correct.\n"
            "Respond strictly with a JSON object in this format with no other text:\n"
            '{\n  "probability": <0 to 100, where 0 is definitely incorrect and 100 is definitely correct>\n}'
        )
        if isinstance(self.backend, OllamaBackend):
            messages = [{"role": "user", "content": prompt}]
            raw_response, meta = self.backend.chat(messages=messages, temperature=0.0, seed=seed, format="json")
        else:
            raw_response, state_hash, meta = self.backend.step(prompt)

        prob = _parse_probability_from_text(raw_response)
        pred_correct = (prob >= 0.5) if prob is not None else None

        return ObserverEvaluation(
            observer_name=self.name,
            predicted_probability=prob,
            predicted_correct=pred_correct,
            raw_response=raw_response,
            metadata={"target_answer": target_answer},
        )
