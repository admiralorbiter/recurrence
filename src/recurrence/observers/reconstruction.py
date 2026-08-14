"""Reconstruction Observer: Recomputes the task independently and compares counterfactual answers."""

import re
from typing import Dict, Any, Optional, Union
from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.observers.base import BaseObserver, ObserverEvaluation


class ReconstructionObserver(BaseObserver):
    """Observer that independently solves the task prompt and compares its counterfactual answer against the target."""

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

        # 2. Parse reconstructed answer and confidence
        ans_match = re.search(r"Answer:\s*([^\n\r]+)", raw_response, re.IGNORECASE)
        conf_match = re.search(r"Confidence:\s*([1-5])", raw_response)

        recon_answer = ans_match.group(1).strip() if ans_match else raw_response.strip()
        recon_conf: Optional[int] = None
        if conf_match:
            try:
                recon_conf = int(conf_match.group(1))
            except ValueError:
                recon_conf = None
        else:
            recon_conf = 3  # Neutral default if not parsed

        # 3. Normalize answers for comparison
        clean_target = target_answer.strip()
        target_match = re.search(r"Answer:\s*([^\n\r]+)", clean_target, re.IGNORECASE)
        if target_match:
            clean_target = target_match.group(1).strip()

        clean_recon = recon_answer.strip()
        
        # Compare (case-insensitive stripped)
        agrees = (clean_target.upper() == clean_recon.upper())

        # If they agree, observer predicts target is CORRECT with confidence = recon_conf
        # If they disagree, observer predicts target is INCORRECT with confidence = recon_conf
        pred_correct = agrees

        return ObserverEvaluation(
            observer_name=self.name,
            predicted_correct=pred_correct,
            observer_confidence=recon_conf,
            reconstructed_answer=clean_recon,
            raw_response=raw_response,
            metadata={
                "agrees": agrees,
                "clean_target": clean_target,
                "clean_recon": clean_recon,
                "reconstructed_confidence": recon_conf,
            },
        )
