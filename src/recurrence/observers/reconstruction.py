"""Reconstruction Observer: Recomputes the task independently and looks up P(Target Selected Option) from 4-way distribution."""

import json
import re
from typing import Dict, Any, Optional, Union
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.observers.base import BaseObserver, ObserverEvaluation


class ReconstructionObserver(BaseObserver):
    """Observer that independently solves the task prompt without viewing the target answer,
    generates a 4-option probability distribution, and looks up P(target_selected_option).
    """

    def __init__(self, backend: Union[OllamaBackend, ToyBackend], name: str = "observer_reconstruction"):
        super().__init__(backend=backend, name=name)

    def format_prompt(self, task_prompt: str) -> str:
        # Base task stimulus
        base_prompt = re.sub(r"Respond strictly with a JSON object.*", "", task_prompt, flags=re.DOTALL).strip()
        base_prompt = re.sub(r"Format your response strictly as:.*", "", base_prompt, flags=re.DOTALL).strip()

        return (
            f"{base_prompt}\n\n"
            "Independently solve this task. Estimate the probability (0 to 100) that each option (A, B, C, D) is the correct answer.\n"
            "Respond strictly with a JSON object in this format with no other text:\n"
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

        # 1. Independently execute task prompt on a fresh invocation with grammar-constrained JSON
        if isinstance(self.backend, OllamaBackend):
            messages = [{"role": "user", "content": prompt}]
            raw_response, meta = self.backend.chat(messages=messages, temperature=0.0, seed=seed, format="json")
        else:
            raw_response, state_hash, meta = self.backend.step(prompt)

        # 2. Parse 4-way probability distribution
        dist: Dict[str, float] = {}
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict):
                    for k, v in data.items():
                        m = re.search(r"([A-D])", str(k), re.IGNORECASE)
                        if m:
                            opt = m.group(1).upper()
                            try:
                                dist[opt] = float(v)
                            except (ValueError, TypeError):
                                pass
            except Exception:
                pass

        # Regex fallback for distribution if JSON parsing failed or incomplete
        for opt in ["A", "B", "C", "D"]:
            if opt not in dist:
                m = re.search(rf'["\']?{opt}["\']?\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw_response, re.IGNORECASE)
                if m:
                    try:
                        dist[opt] = float(m.group(1))
                    except ValueError:
                        pass

        # Toy backend simulation fallback
        if not dist and "action_" in raw_response:
            dist = {"A": 10.0, "B": 70.0, "C": 10.0, "D": 10.0}

        # If at least one option was found, fill remaining with 0.0
        if len(dist) >= 1:
            for opt in ["A", "B", "C", "D"]:
                if opt not in dist:
                    dist[opt] = 0.0

        target_prob: Optional[float] = None
        recon_answer: Optional[str] = None
        normalized_dist: Dict[str, float] = {}

        if len(dist) >= 4:
            total_mass = sum(dist[opt] for opt in ["A", "B", "C", "D"])
            if total_mass > 0:
                normalized_dist = {opt: float(dist[opt] / total_mass) for opt in ["A", "B", "C", "D"]}
            else:
                normalized_dist = {opt: 0.25 for opt in ["A", "B", "C", "D"]}

            recon_answer = max(normalized_dist, key=lambda k: normalized_dist[k])

            # 3. Extract clean target option letter
            clean_target = target_answer.strip()
            t_json = re.search(r"\{.*\}", clean_target, re.DOTALL)
            if t_json:
                try:
                    t_data = json.loads(t_json.group(0))
                    if isinstance(t_data, dict):
                        clean_dict = {re.sub(r"[^a-zA-Z0-9_]", "", str(k)).lower(): v for k, v in t_data.items()}
                        for k in ["answer", "ans", "choice", "option", "in", "selected", "answeranswer"]:
                            if k in clean_dict:
                                ans_val = clean_dict[k]
                                if isinstance(ans_val, dict):
                                    ans_val = list(ans_val.keys())[0] if ans_val else ""
                                clean_target = str(ans_val).strip()
                                break
                except Exception:
                    pass

            target_letter_match = re.search(r"\b([A-D])\b", clean_target, re.IGNORECASE)
            if target_letter_match:
                target_letter = target_letter_match.group(1).upper()
            else:
                target_letter = clean_target.strip().upper()

            if target_letter in ["A", "B", "C", "D"]:
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
                "target_answer_parsed": clean_target if 'clean_target' in locals() else target_answer,
                "reconstructed_top_choice": recon_answer,
            },
        )
