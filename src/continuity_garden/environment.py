"""The Hidden Switchboard POMDP Environment (Continuity Garden v0)."""

from typing import List, Optional, Tuple
import numpy as np

from .state import GroundTruthState
from .observation import AgentObservation, SensorTransform


class HiddenSwitchboardEnv:
    """A minimal partially observable environment with delayed dependency."""

    def __init__(
        self,
        min_delay: int = 8,
        max_delay: int = 16,
        num_queries: int = 5,
        sensor_noise_std: float = 0.0,
        seed: int = 42
    ):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.num_queries = num_queries
        self.sensor_noise_std = sensor_noise_std
        self.rng = np.random.RandomState(seed)
        self.sensor_transform = SensorTransform(sensor_noise_std=sensor_noise_std, seed=seed)

        self._ground_truth: Optional[GroundTruthState] = None
        self._delay_len: int = 0
        self._step_idx: int = 0
        self._query_idx: int = 0
        self._query_sequence: List[int] = []
        self._last_action: Optional[int] = None
        self._pending_target_action: Optional[int] = None

    def reset(self, explicit_mode: Optional[int] = None, explicit_delay: Optional[int] = None) -> Tuple[AgentObservation, GroundTruthState]:
        self._step_idx = 0
        self._query_idx = 0
        self._last_action = None
        self._pending_target_action = None

        # Sample hidden mode z in {0, 1}
        hidden_mode = explicit_mode if explicit_mode is not None else int(self.rng.randint(0, 2))
        
        # Sample delay length
        self._delay_len = explicit_delay if explicit_delay is not None else int(self.rng.randint(self.min_delay, self.max_delay + 1))
        
        # Pre-sample query bits x_t in {0, 1}
        self._query_sequence = [int(self.rng.randint(0, 2)) for _ in range(self.num_queries)]

        self._ground_truth = GroundTruthState(
            step_idx=0,
            hidden_mode=hidden_mode,
            current_phase="cue",
            query_bit=None,
            target_action=None,
            true_source=0,
            resource_integrity=1.0,
            is_terminal=False,
        )

        obs = self.sensor_transform.transform(self._ground_truth, last_action=None)
        return obs, self._ground_truth

    def step(self, action: int) -> Tuple[AgentObservation, float, bool, GroundTruthState]:
        assert self._ground_truth is not None, "Must call reset() before step()"
        self._step_idx += 1
        self._last_action = action
        reward = 0.0

        # Score action against pending target action from previous step if any
        if self._pending_target_action is not None:
            if action == self._pending_target_action:
                reward = 1.0
            self._pending_target_action = None

        if self._step_idx <= self._delay_len:
            # Distractor phase
            current_phase = "distractor"
            query_bit = None
            target_action = None
            is_terminal = False
        else:
            # Query phase
            if self._query_idx < self.num_queries:
                current_phase = "query"
                query_bit = self._query_sequence[self._query_idx]
                target_action = query_bit ^ self._ground_truth.hidden_mode
                self._pending_target_action = target_action
                self._query_idx += 1
                is_terminal = False
            else:
                # All queries completed
                current_phase = "terminal"
                query_bit = None
                target_action = None
                is_terminal = True

        self._ground_truth = GroundTruthState(
            step_idx=self._step_idx,
            hidden_mode=self._ground_truth.hidden_mode,
            current_phase=current_phase,
            query_bit=query_bit,
            target_action=target_action,
            true_source=0,
            resource_integrity=max(0.0, self._ground_truth.resource_integrity - 0.01),
            is_terminal=is_terminal,
        )

        obs = self.sensor_transform.transform(self._ground_truth, last_action=self._last_action)
        return obs, reward, is_terminal, self._ground_truth
