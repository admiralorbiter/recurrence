"""Continuity Garden v1: Yoked Controllability POMDP Environment.

Protocol:
  Distinguishes controllable worlds (W_ctrl) from yoked uncontrollable worlds (W_yoked)
  through sensorimotor contingency without explicit agency labels.

Phases:
  1. Exploration Phase (t = 0 .. T_exp):
     - Organism acts (a_t in {0, 1}) and observes environmental effect (E in {0, 1}).
     - In W_ctrl: P(E = a_t) = 0.90, P(E = 1 - a_t) = 0.10.
     - In W_yoked: E is drawn from matched marginal P(E), independent of a_t.
  2. Exploitation Phase (t = T_exp + 1):
     - Goal E* in {0, 1} is revealed.
     - Organism chooses: TRY_0 (0), TRY_1 (1), or ABSTAIN (2).
     - Payoff: Success (+0.90), Failure (-1.10), Abstain (0.00).
"""

import copy
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import numpy as np


@dataclass
class GroundTruthStateV1:
    step_idx: int
    world_type: str              # "ctrl" or "yoked"
    current_phase: str           # "exploration", "exploitation", "terminal"
    target_goal_effect: Optional[int]
    last_effect: Optional[int]
    last_action_executed: Optional[int]
    last_action_intended: Optional[int]
    is_terminal: bool


@dataclass
class ObservationV1:
    symbol: int                  # 0: Blank/Init, 1: Effect_0, 2: Effect_1, 3: Goal_0, 4: Goal_1
    action_executed: int         # 0: Action 0, 1: Action 1, 2: No action / Null
    action_intended: int         # 0: Action 0, 1: Action 1, 2: No action / Null


@dataclass
class EnvironmentSnapshotV1:
    step_idx: int
    world_type: str
    current_phase: str
    target_goal_effect: Optional[int]
    last_effect: Optional[int]
    last_action_executed: Optional[int]
    last_action_intended: Optional[int]
    is_terminal: bool
    exploration_len: int
    contingency_p: float
    cost_try: float
    reward_success: float
    penalty_failure: float
    yoked_effect_sequence: List[int]
    rng_state: Tuple[Any, ...]


class ControllabilityArenaEnv:
    """Yoked Controllability Arena POMDP."""

    def __init__(
        self,
        min_exploration_steps: int = 6,
        max_exploration_steps: int = 8,
        contingency_p: float = 0.90,
        cost_try: float = 0.10,
        reward_success: float = 1.00,
        penalty_failure: float = -1.00,
        seed: int = 42,
    ):
        self.min_exp = min_exploration_steps
        self.max_exp = max_exploration_steps
        self.contingency_p = contingency_p
        self.cost_try = cost_try
        self.reward_success = reward_success
        self.penalty_failure = penalty_failure
        self.rng = np.random.RandomState(seed)

        self._step_idx = 0
        self._exploration_len = 6
        self._world_type = "ctrl"
        self._target_goal = None
        self._last_effect = None
        self._last_executed = 2
        self._last_intended = 2
        self._yoked_effect_seq = []
        self._ground_truth: Optional[GroundTruthStateV1] = None

    def reset(
        self,
        explicit_world_type: Optional[str] = None,
        explicit_exploration_len: Optional[int] = None,
        explicit_goal: Optional[int] = None,
    ) -> Tuple[ObservationV1, GroundTruthStateV1]:
        self._step_idx = 0
        self._world_type = explicit_world_type if explicit_world_type in ["ctrl", "yoked"] else (
            "ctrl" if self.rng.rand() < 0.5 else "yoked"
        )
        self._exploration_len = explicit_exploration_len or int(self.rng.randint(self.min_exp, self.max_exp + 1))
        self._target_goal = explicit_goal if explicit_goal in [0, 1] else int(self.rng.randint(0, 2))
        self._last_effect = None
        self._last_executed = 2
        self._last_intended = 2

        # Pre-generate matched yoked effect sequence with balanced 50/50 marginal
        self._yoked_effect_seq = [int(self.rng.randint(0, 2)) for _ in range(self._exploration_len)]

        self._ground_truth = GroundTruthStateV1(
            step_idx=0,
            world_type=self._world_type,
            current_phase="exploration",
            target_goal_effect=None,
            last_effect=None,
            last_action_executed=2,
            last_action_intended=2,
            is_terminal=False,
        )

        obs = ObservationV1(symbol=0, action_executed=2, action_intended=2)
        return obs, self._ground_truth

    def step(
        self,
        action: int,
        forced_override_action: Optional[int] = None,
    ) -> Tuple[ObservationV1, float, bool, GroundTruthStateV1]:
        """
        Actions during exploration:
          action in {0, 1}: motor command
        Actions during exploitation:
          action == 0: TRY_0
          action == 1: TRY_1
          action == 2: ABSTAIN
        """
        assert self._ground_truth is not None, "Call reset() before step()."
        self._step_idx += 1
        reward = 0.0
        is_terminal = False

        intended_act = action
        executed_act = forced_override_action if forced_override_action is not None else action
        self._last_intended = intended_act
        self._last_executed = executed_act

        if self._step_idx <= self._exploration_len:
            # Exploration phase
            if self._world_type == "ctrl":
                if self.rng.rand() < self.contingency_p:
                    effect = executed_act
                else:
                    effect = 1 - executed_act
            else:
                # Yoked: outcome drawn from matched sequence, independent of action
                effect = self._yoked_effect_seq[self._step_idx - 1]

            self._last_effect = effect
            # Symbol: 1 for Effect_0, 2 for Effect_1
            obs_symbol = effect + 1
            phase = "exploration" if self._step_idx < self._exploration_len else "exploitation"

            # On the transition step to exploitation, reveal the goal
            if phase == "exploitation":
                # Reveal goal in symbol: 3 for Goal_0, 4 for Goal_1
                obs_symbol = self._target_goal + 3

        else:
            # Exploitation step evaluation
            phase = "terminal"
            is_terminal = True
            exploit_choice = action # 0: TRY_0, 1: TRY_1, 2: ABSTAIN

            if exploit_choice == 2:
                # ABSTAIN
                reward = 0.00
            elif exploit_choice in [0, 1]:
                # Attempt control with chosen motor action
                if self._world_type == "ctrl":
                    produced_effect = exploit_choice if self.rng.rand() < self.contingency_p else (1 - exploit_choice)
                else:
                    produced_effect = int(self.rng.randint(0, 2))

                if produced_effect == self._target_goal:
                    reward = self.reward_success - self.cost_try # +0.90
                else:
                    reward = self.penalty_failure - self.cost_try # -1.10
            else:
                reward = self.penalty_failure - self.cost_try

            obs_symbol = 0 # Terminal blank

        self._ground_truth = GroundTruthStateV1(
            step_idx=self._step_idx,
            world_type=self._world_type,
            current_phase=phase,
            target_goal_effect=self._target_goal,
            last_effect=self._last_effect,
            last_action_executed=self._last_executed,
            last_action_intended=self._last_intended,
            is_terminal=is_terminal,
        )

        obs = ObservationV1(
            symbol=obs_symbol,
            action_executed=self._last_executed,
            action_intended=self._last_intended,
        )
        return obs, reward, is_terminal, self._ground_truth

    def snapshot(self) -> EnvironmentSnapshotV1:
        assert self._ground_truth is not None, "Cannot snapshot uninitialized environment."
        return EnvironmentSnapshotV1(
            step_idx=self._step_idx,
            world_type=self._world_type,
            current_phase=self._ground_truth.current_phase,
            target_goal_effect=self._target_goal,
            last_effect=self._last_effect,
            last_action_executed=self._last_executed,
            last_action_intended=self._last_intended,
            is_terminal=self._ground_truth.is_terminal,
            exploration_len=self._exploration_len,
            contingency_p=self.contingency_p,
            cost_try=self.cost_try,
            reward_success=self.reward_success,
            penalty_failure=self.penalty_failure,
            yoked_effect_sequence=copy.deepcopy(self._yoked_effect_seq),
            rng_state=self.rng.get_state(),
        )

    def restore(self, snap: EnvironmentSnapshotV1) -> None:
        self._step_idx = snap.step_idx
        self._world_type = snap.world_type
        self._target_goal = snap.target_goal_effect
        self._last_effect = snap.last_effect
        self._last_executed = snap.last_action_executed
        self._last_intended = snap.last_action_intended
        self._exploration_len = snap.exploration_len
        self.contingency_p = snap.contingency_p
        self.cost_try = snap.cost_try
        self.reward_success = snap.reward_success
        self.penalty_failure = snap.penalty_failure
        self._yoked_effect_seq = copy.deepcopy(snap.yoked_effect_sequence)
        self.rng.set_state(snap.rng_state)

        self._ground_truth = GroundTruthStateV1(
            step_idx=snap.step_idx,
            world_type=snap.world_type,
            current_phase=snap.current_phase,
            target_goal_effect=snap.target_goal_effect,
            last_effect=snap.last_effect,
            last_action_executed=snap.last_action_executed,
            last_action_intended=snap.last_action_intended,
            is_terminal=snap.is_terminal,
        )

    def clone(self) -> "ControllabilityArenaEnv":
        cloned = ControllabilityArenaEnv(
            cost_try=self.cost_try,
            reward_success=self.reward_success,
            penalty_failure=self.penalty_failure,
            contingency_p=self.contingency_p,
            min_exploration_steps=self.min_exp,
            max_exploration_steps=self.max_exp,
        )
        if self._ground_truth is not None:
            cloned.restore(self.snapshot())
        return cloned
