"""Agent Observation and Sensor Transforms for Continuity Garden (Q06 Construct Separation)."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .state import GroundTruthState


@dataclass(frozen=True)
class AgentObservation:
    """Observation exposed to the organism. Strictly free of ground-truth target labels."""
    symbol: int                          # Discrete observation symbol in {0, 1, 2, 3, 4}
    action_feedback: Optional[int] = None # Consequence of previous action on step t-1
    noisy_sensor: Optional[float] = None  # Optional noisy reading (NOT raw ground truth)


class SensorTransform:
    """Transforms environment-internal GroundTruthState into an AgentObservation."""

    def __init__(self, sensor_noise_std: float = 0.0, seed: int = 42):
        self.sensor_noise_std = sensor_noise_std
        self.rng = np.random.RandomState(seed)

    def transform(
        self,
        ground_truth: GroundTruthState,
        last_action: Optional[int] = None
    ) -> AgentObservation:
        """Constructs observation according to environment phase."""
        if ground_truth.current_phase == "cue":
            # Cue step: symbol 1 for z=0, symbol 2 for z=1
            symbol = ground_truth.hidden_mode + 1
        elif ground_truth.current_phase == "distractor":
            # Distractor step: symbol 0 (blank/neutral token)
            symbol = 0
        elif ground_truth.current_phase == "query":
            # Query step: symbol 3 for x_t=0, symbol 4 for x_t=1
            assert ground_truth.query_bit is not None
            symbol = ground_truth.query_bit + 3
        else:
            symbol = 0

        noisy_sensor = None
        if self.sensor_noise_std > 0:
            noise = float(self.rng.normal(0, self.sensor_noise_std))
            noisy_sensor = float(np.clip(ground_truth.resource_integrity + noise, 0.0, 1.0))

        return AgentObservation(
            symbol=symbol,
            action_feedback=last_action,
            noisy_sensor=noisy_sensor,
        )
