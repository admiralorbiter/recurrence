"""Reconstruction Observer: Recomputes the task independently and maps counterfactual agreement to P(Target Correct)."""

import re
from typing import Dict, Any, Optional, Union
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.observers.base import BaseObserver, ObserverEvaluation
from recurrence.observers.visible import _parse_probability_from_text


class ReconstructionObserver(BaseObserver):
    """Observer that independently solves the task prompt and maps counterfactual agreement onto P(Target Correct)."""

    def __init__(self, backend: Union[OllamaBackend, ToyBackend], name: str = "observer_reconstruction"):
        super().__init__(backend=backend, name=name)

    def evaluate(
        self,
        task_prompt: str,
        target_answer: str,
        item_metadata: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> ObserverEvaluation:
        # 1. Independently execute task prompt on a fresh invocation
        if isinstance(self.backend, OllamaBackend):
            messages = [{"role": "user", "content": task_prompt}]
            raw_response, meta = self.backend.chat(messages=messages, temperature=0.0, seed=seed)
        else:
            raw_response, state_hash, meta = self.backend.step(task_prompt)

        # 2. Parse reconstructed answer and reconstructed confidence/probability
        ans_match = re.search(
            r"Answer:\s*(?:<[^>]+>:\s*)?<?([a-zA-Z0-9_\s]+?)>?(?:\s*(?:Probability|Confidence|Confident)|\%|;|\n|\r|$)",
            raw_response,
            re.IGNORECASE,
        )
        recon_answer = ans_match.group(1).strip() if ans_match else raw_response.strip()

        recon_prob = _parse_probability_from_text(raw_response)
        if recon_prob is None:
            recon_prob = 0.70  # Standard default if not explicitly returned

        # 3. Normalize answers for comparison
        clean_target = target_answer.strip()
        target_match = re.search(
            r"Answer:\s*(?:<[^>]+>:\s*)?<?([a-zA-Z0-9_\s]+?)>?(?:\s*(?:Probability|Confidence|Confident)|\%|;|\n|\r|$)",
            clean_target,
            re.IGNORECASE,
        )
        if target_match:
            clean_target = target_match.group(1).strip()

        clean_recon = recon_answer.strip()
        
        # Compare (case-insensitive stripped)
        agrees = (clean_target.upper() == clean_recon.upper())

        # 4. Strict Directionality Mapping to P(Target Answer Correct)
        # If agreement: P(Target Correct) = recon_prob
        # If disagreement: P(Target Correct) = 1.0 - recon_prob
        if agrees:
            target_prob = recon_prob
        else:
            target_prob = float(max(0.0, min(1.0, 1.0 - recon_prob)))

        pred_correct = (target_prob >= 0.5)

        return ObserverEvaluation(
            observer_name=self.name,
            predicted_probability=target_prob,
            predicted_correct=pred_correct,
            reconstructed_answer=clean_recon,
            raw_response=raw_response,
            metadata={
                "agrees": agrees,
                "clean_target": clean_target,
                "clean_recon": clean_recon,
                "reconstructed_probability": recon_prob,
                "target_probability": target_prob,
            },
        )
