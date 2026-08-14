"""Visible-Evidence Observers: Evaluates visible transcript with standardized probability semantics."""

import json
import math
import re
from typing import Dict, Any, Optional, Union
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.core.schemas import PROBABILITY_ONLY_SCHEMA
from recurrence.observers.base import BaseObserver, ObserverEvaluation


def _parse_probability_from_text(raw_response: str) -> Optional[float]:
    """Extract probability in [0.0, 1.0] from structured JSON or text response.
    Strictly enforces 0-100 percentage scale and rejects out-of-range (<0 or >100) or non-finite values.
    """
    cleaned = raw_response.strip()

    # 1. Try structured JSON extraction first
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict):
                clean_dict = {re.sub(r"[^a-zA-Z0-9_]", "", str(k)).lower(): v for k, v in data.items()}
                raw_val = None
                for k in ["probability", "prob", "probabilityprobability", "probabilitycorrect", "what", "p"]:
                    if k in clean_dict:
                        raw_val = clean_dict[k]
                        break
                if raw_val is None:
                    # check nested dicts
                    for v in data.values():
                        if isinstance(v, dict):
                            inner = {re.sub(r"[^a-zA-Z0-9_]", "", str(ik)).lower(): iv for ik, iv in v.items()}
                            for k in ["probability", "prob", "probabilityprobability", "probabilitycorrect", "p"]:
                                if k in inner:
                                    raw_val = inner[k]
                                    break
                            if raw_val is not None:
                                break

                if raw_val is not None:
                    if isinstance(raw_val, dict) and "probability" in raw_val:
                        raw_val = raw_val["probability"]
                    if isinstance(raw_val, (int, float, str)):
                        try:
                            if isinstance(raw_val, str):
                                cleaned_str = raw_val.strip()
                                if cleaned_str.lower() in ["nan", "inf", "-inf", "infinity", "-infinity"]:
                                    return None
                                num_m = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)", cleaned_str)
                                if not num_m:
                                    return None
                                val = float(num_m.group(1))
                            else:
                                val = float(raw_val)
                            # Strict 0-100 scale contract: Reject out-of-range or non-finite
                            if not math.isfinite(val) or val < 0.0 or val > 100.0:
                                return None
                            return float(val / 100.0)
                        except (ValueError, TypeError):
                            pass
        except Exception:
            pass

    # 2. Strict Probability regex
    prob_match = re.search(r"(?:Probability\s*(?:correct)?|Prob|p):\s*<?(-?[0-9]+(?:\.[0-9]+)?)\s*\%?>?", cleaned, re.IGNORECASE)
    if prob_match:
        try:
            val = float(prob_match.group(1))
            if math.isfinite(val) and 0.0 <= val <= 100.0:
                return float(val / 100.0)
        except ValueError:
            pass

    # 3. Unstructured toy fallback
    if "action_" in cleaned:
        return 0.5

    return None


class VisibleAnswerOnlyObserver(BaseObserver):
    """Observer that reads task prompt and target's answer with confidence report stripped."""

    def __init__(self, backend: Union[OllamaBackend, ToyBackend], name: str = "observer_visible_answer_only"):
        super().__init__(backend=backend, name=name)

    def format_prompt(self, task_prompt: str, target_answer_only: str) -> str:
        base_prompt = re.sub(r"Respond strictly with a JSON object.*", "", task_prompt, flags=re.DOTALL).strip()
        base_prompt = re.sub(r"Format your response strictly as:.*", "", base_prompt, flags=re.DOTALL).strip()

        return (
            "You are an external observer evaluating whether another AI agent's answer to a task is correct or incorrect.\n\n"
            f"[Task Prompt]\n{base_prompt}\n\n"
            f"[Agent's Proposed Answer]\n{target_answer_only.strip()}\n\n"
            "Estimate the probability as an integer from 0 to 100 (where 0 is definitely incorrect and 100 is definitely correct).\n"
            "Respond strictly with a JSON object in this format with no other text:\n"
            '{\n  "probability": <integer between 0 and 100>\n}'
        )

    def evaluate(
        self,
        task_prompt: str,
        target_answer: str,
        item_metadata: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> ObserverEvaluation:
        # Strip any confidence report from target_answer
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

        prompt = self.format_prompt(task_prompt, clean_ans)
        if isinstance(self.backend, OllamaBackend):
            messages = [{"role": "user", "content": prompt}]
            raw_response, meta = self.backend.chat(messages=messages, temperature=0.0, seed=seed, format=PROBABILITY_ONLY_SCHEMA)
        else:
            raw_response, state_hash, meta = self.backend.step(prompt, format=PROBABILITY_ONLY_SCHEMA)

        prob = _parse_probability_from_text(raw_response)
        pred_correct = (prob >= 0.5) if prob is not None else None

        return ObserverEvaluation(
            observer_name=self.name,
            predicted_probability=prob,
            predicted_correct=pred_correct,
            raw_response=raw_response,
            metadata={
                "task_prompt_len": len(task_prompt),
                "target_answer_stripped": clean_ans,
            },
        )


class VisibleFullTranscriptObserver(BaseObserver):
    """Observer that reads task prompt and target's full response including its stated confidence report."""

    def __init__(self, backend: Union[OllamaBackend, ToyBackend], name: str = "observer_visible_full_transcript"):
        super().__init__(backend=backend, name=name)

    def format_prompt(self, task_prompt: str, full_target_response: str) -> str:
        base_prompt = re.sub(r"Respond strictly with a JSON object.*", "", task_prompt, flags=re.DOTALL).strip()
        base_prompt = re.sub(r"Format your response strictly as:.*", "", base_prompt, flags=re.DOTALL).strip()

        return (
            "You are an external observer evaluating whether another AI agent's response to a task is correct or incorrect.\n\n"
            f"[Task Prompt]\n{base_prompt}\n\n"
            f"[Agent's Full Response]\n{full_target_response.strip()}\n\n"
            "Estimate the probability as an integer from 0 to 100 (where 0 is definitely incorrect and 100 is definitely correct).\n"
            "Respond strictly with a JSON object in this format with no other text:\n"
            '{\n  "probability": <integer between 0 and 100>\n}'
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
            raw_response, meta = self.backend.chat(messages=messages, temperature=0.0, seed=seed, format=PROBABILITY_ONLY_SCHEMA)
        else:
            raw_response, state_hash, meta = self.backend.step(prompt, format=PROBABILITY_ONLY_SCHEMA)

        prob = _parse_probability_from_text(raw_response)
        pred_correct = (prob >= 0.5) if prob is not None else None

        return ObserverEvaluation(
            observer_name=self.name,
            predicted_probability=prob,
            predicted_correct=pred_correct,
            raw_response=raw_response,
            metadata={
                "task_prompt_len": len(task_prompt),
                "full_target_response": target_answer.strip(),
            },
        )


# Backward compatibility alias
VisibleEvidenceObserver = VisibleAnswerOnlyObserver
