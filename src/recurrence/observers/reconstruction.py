"""Reconstruction Observer: Recomputes the task independently and looks up P(Target Selected Option) from complete 4-way distribution."""

import json
import math
import re
from typing import Dict, Any, Optional, Union
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.core.schemas import RECONSTRUCTION_DISTRIBUTION_SCHEMA
from recurrence.observers.base import BaseObserver, ObserverEvaluation


def _extract_target_letter(target_answer: str) -> Optional[str]:
    """Extract clean target option letter A, B, C, or D from structured JSON, nested dict, or text."""
    cleaned = target_answer.strip()
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict):
                clean_dict = {re.sub(r"[^a-zA-Z0-9_]", "", str(k)).lower(): v for k, v in data.items()}
                for k in ["answer", "ans", "choice", "option", "letter", "in", "selected"]:
                    if k in clean_dict:
                        v = clean_dict[k]
                        if isinstance(v, dict):
                            # nested dict: e.g. {"option": "B"} or {"letter": "C"}
                            inner = {re.sub(r"[^a-zA-Z0-9_]", "", str(ik)).lower(): iv for ik, iv in v.items()}
                            for ik in ["option", "letter", "choice", "ans", "answer"]:
                                if ik in inner:
                                    m = re.search(r"\b([A-D])\b", str(inner[ik]), re.IGNORECASE)
                                    if m:
                                        return m.group(1).upper()
                            for iv in v.values():
                                m = re.search(r"\b([A-D])\b", str(iv), re.IGNORECASE)
                                if m:
                                    return m.group(1).upper()
                        elif isinstance(v, (str, int)):
                            m = re.search(r"\b([A-D])\b", str(v), re.IGNORECASE)
                            if m:
                                return m.group(1).upper()
        except Exception:
            pass

    # Regex fallback
    m = re.search(r"(?:Answer|Option|Choice|Letter)?:\s*<?([A-D])>?", cleaned, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m2 = re.search(r"\b([A-D])\b", cleaned)
    if m2:
        return m2.group(1).upper()

    return None


class ReconstructionObserver(BaseObserver):
    """Observer that independently solves the task prompt without viewing the target answer,
    generates a 4-option probability distribution, and looks up P(target_selected_option).
    Requires a complete, valid 4-option distribution without manufactured zero-fill.
    """

    def __init__(self, backend: Union[OllamaBackend, ToyBackend], name: str = "observer_reconstruction"):
        super().__init__(backend=backend, name=name)

    def format_prompt(self, task_prompt: str) -> str:
        # Base task stimulus
        base_prompt = re.sub(r"Respond strictly with a JSON object.*", "", task_prompt, flags=re.DOTALL).strip()
        base_prompt = re.sub(r"Format your response strictly as:.*", "", base_prompt, flags=re.DOTALL).strip()

        return (
            f"{base_prompt}\n\n"
            "Independently solve this task. Estimate the probability (0 to 100) for each option (A, B, C, D). "
            "Assign a positive probability (1 to 100) to at least your best choice.\n"
            "Respond strictly with a JSON object matching the required schema:\n"
            '{\n  "A": <0 to 100>,\n  "B": <0 to 100>,\n  "C": <0 to 100>,\n  "D": <0 to 100>\n}'
        )

    def evaluate(
        self,
        task_prompt: str,
        target_answer: str,
        item_metadata: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> ObserverEvaluation:
        prompt = self.format_prompt(task_prompt)

        # 1. Independently execute task prompt on a fresh invocation with grammar-constrained JSON schema
        if isinstance(self.backend, OllamaBackend):
            messages = [{"role": "user", "content": prompt}]
            raw_response, meta = self.backend.chat(messages=messages, temperature=0.0, seed=seed, format=RECONSTRUCTION_DISTRIBUTION_SCHEMA)
        else:
            raw_response, state_hash, meta = self.backend.step(prompt, format=RECONSTRUCTION_DISTRIBUTION_SCHEMA)

        # 2. Parse 4-way probability distribution
        dist: Dict[str, float] = {}
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict):
                    for k, v in data.items():
                        m = re.search(r"\b([A-D])\b", str(k), re.IGNORECASE)
                        if not m:
                            m = re.search(r"([A-D])", str(k), re.IGNORECASE)
                        if m:
                            opt = m.group(1).upper()
                            try:
                                if isinstance(v, (int, float)):
                                    dist[opt] = float(v)
                                elif isinstance(v, str):
                                    num_m = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)", v.strip())
                                    if num_m:
                                        dist[opt] = float(num_m.group(1))
                            except (ValueError, TypeError):
                                pass
            except Exception:
                pass

        # Regex fallback for distribution if JSON parsing failed or incomplete
        for opt in ["A", "B", "C", "D"]:
            if opt not in dist:
                m = re.search(rf'["\']?{opt}["\']?\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)', raw_response, re.IGNORECASE)
                if m:
                    try:
                        dist[opt] = float(m.group(1))
                    except ValueError:
                        pass

        # Toy backend simulation fallback
        if not dist and "action_" in raw_response:
            dist = {"A": 10.0, "B": 70.0, "C": 10.0, "D": 10.0}

        target_prob: Optional[float] = None
        recon_answer: Optional[str] = None
        normalized_dist: Dict[str, float] = {}

        # STRICT VALIDATION: Require all 4 options (A, B, C, D) to be present, finite, within [0, 100], and sum > 0
        has_all_4 = all(
            opt in dist and isinstance(dist[opt], (int, float)) and math.isfinite(dist[opt]) and 0.0 <= dist[opt] <= 100.0
            for opt in ["A", "B", "C", "D"]
        )
        total_mass = sum(dist[opt] for opt in ["A", "B", "C", "D"]) if has_all_4 else 0.0

        if has_all_4 and total_mass > 0.0:
            normalized_dist = {opt: float(dist[opt] / total_mass) for opt in ["A", "B", "C", "D"]}
            recon_answer = max(normalized_dist, key=lambda k: normalized_dist[k])

            # Extract target choice strictly from target's actual response (NO ground-truth metadata fallback)
            target_letter = _extract_target_letter(target_answer)
            if target_letter is not None and target_letter in normalized_dist:
                target_prob = normalized_dist[target_letter]

        pred_correct = (target_prob >= 0.5) if target_prob is not None else None

        return ObserverEvaluation(
            observer_name=self.name,
            predicted_probability=target_prob,
            predicted_correct=pred_correct,
            reconstructed_answer=recon_answer,
            raw_response=raw_response,
            metadata={
                "distribution_raw": dist,
                "distribution_normalized": normalized_dist,
                "target_answer_parsed": target_letter if 'target_letter' in locals() else None,
                "reconstructed_top_choice": recon_answer,
                "distribution_complete": bool(has_all_4 and total_mass > 0.0),
            },
        )
