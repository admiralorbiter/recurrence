"""Garden v2: The Dual-Locus Endogenous Regulator Environment.

Causal Chain:
  a_t^{intended} --(i_t)--> a_t^{executed} --(x_t)--> E_{t+1}

Key Invariants:
  1. Finite Latent Lattice: i_t, x_t in {0.0, 0.1, ..., 1.0} (11 discrete levels).
  2. Failure Semantics:
     - P(a_exec = a_intend | i_t) = 0.50 + 0.50 * i_t, else a_exec = NULL (Action 4).
     - P(E = a_exec | x_t, a_exec != NULL) = 0.50 + 0.50 * x_t, else E = NULL (Symbol 0).
  3. Actions:
     - 0: MOTOR_0 (intended motor 0)
     - 1: MOTOR_1 (intended motor 1)
     - 2: MAINTAIN_A (restores i_t -> 1.0, cost c_maint)
     - 3: MAINTAIN_B (restores x_t -> 1.0, cost c_maint)
     - 4: NULL_ACTION (internal execution failure representation)
  4. Structured Precursor & Decision Window Sequence:
     - t_0, t_1, t_2: Precursor Evidence Steps (continuous noisy c_j ~ N(mu, sigma^2)).
     - t_3: Delay Step (precursor gone, c = 0.0).
     - t_4: Designated Regulatory Decision Window (is_decision_window = True, c = 0.0).
     - t_5: Shock Impact (unmitigated crashes i -> 0.20; mitigated keeps i >= 0.90).
  5. Purely Instrumental Reward:
     - +1.0 for producing target goal E* (+3.0 on high demand / shock steps).
     - Cost -c_maint on maintenance actions. Zero oracle reward for internal state levels.
"""

from dataclasses import dataclass, field
import copy
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class EventTapeV2:
    """Pre-generated deterministic event tape for common random number paired lineages."""
    precursor_start_steps: List[int]
    decision_window_steps: List[int]
    shock_steps: List[int]
    shock_magnitudes: List[float] # 0.10 (minor false alarm) or 0.70 (severe shock)
    precursor_noise: List[List[float]] # 3 draws of noise per shock event
    sensor_noise_a: List[float]
    sensor_noise_b: List[float]
    motor_bernoulli_draws: List[float]
    world_bernoulli_draws: List[float]
    target_goals: List[int]
    high_demand_steps: List[bool]


@dataclass
class ObservationV2:
    symbol: int # 3: Goal_0, 4: Goal_1 (or effect symbols)
    sensor_a: float # Continuous noisy reading of i_t
    sensor_b: float # Continuous noisy reading of x_t
    warning_cue: float # Continuous precursor cue (0.0 when inactive)
    is_decision_window: int # 1 if at regulatory decision window t_4, else 0
    last_action_executed: int # 0..4
    last_action_intended: int # 0..3


@dataclass
class GroundTruthStateV2:
    step_idx: int
    internal_reliability_i: float
    external_reliability_x: float
    is_precursor_active: bool
    is_decision_window: bool
    shock_pending: bool
    shock_timer: int
    pending_shock_magnitude: float
    bayesian_risk_q: float # Exact P(severe | c_1:3)
    counterfactual_future_i_no_maint: float
    mitigation_active: bool
    target_goal: int
    last_effect: Optional[int]
    last_action_executed: int
    last_action_intended: int
    is_terminal: bool
    is_decorative: bool # True for Lineage B where i_t does not affect execution


@dataclass
class EnvironmentSnapshotV2:
    step_idx: int
    internal_reliability_i: float
    external_reliability_x: float
    is_precursor_active: bool
    is_decision_window: bool
    shock_pending: bool
    shock_timer: int
    pending_shock_magnitude: float
    bayesian_risk_q: float
    counterfactual_future_i_no_maint: float
    mitigation_active: bool
    target_goal: int
    last_effect: Optional[int]
    last_action_executed: int
    last_action_intended: int
    is_terminal: bool
    is_decorative: bool
    tape: Optional[EventTapeV2]
    rng_state: Any


class DualLocusRegulatorEnv:
    """Garden v2 Dual-Locus Causal Kernel."""

    LATTICE_LEVELS = np.round(np.linspace(0.0, 1.0, 11), 1) # [0.0, 0.1, ..., 1.0]

    def __init__(
        self,
        episode_len: int = 24,
        cost_maintain: float = 0.15,
        reward_target_hit: float = 1.00,
        penalty_wrong_effect: float = -0.50,
        sensor_noise_std: float = 0.08,
        precursor_noise_std: float = 0.35,
        drift_rate: float = 0.00,
        is_decorative: bool = False,
        seed: int = 42,
    ):
        self.episode_len = episode_len
        self.cost_maintain = cost_maintain
        self.reward_target_hit = reward_target_hit
        self.penalty_wrong_effect = penalty_wrong_effect
        self.sensor_noise_std = sensor_noise_std
        self.precursor_noise_std = precursor_noise_std
        self.drift_rate = drift_rate
        self.is_decorative = is_decorative
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self._tape: Optional[EventTapeV2] = None
        self._ground_truth: Optional[GroundTruthStateV2] = None

        # State variables
        self._step_idx = 0
        self._i_t = 1.0
        self._x_t = 1.0
        self._is_precursor_active = False
        self._is_decision_window = False
        self._shock_pending = False
        self._shock_timer = 0
        self._pending_shock_magnitude = 0.0
        self._bayesian_risk_q = 0.50
        self._counterfactual_future_i = 1.0
        self._mitigation_active = False
        self._target_goal = 0
        self._last_effect: Optional[int] = None
        self._last_executed = 4 # NULL
        self._last_intended = 0

    @classmethod
    def quantize_lattice(cls, val: float) -> float:
        """Snaps a continuous float to the nearest discrete lattice level."""
        idx = int(np.argmin(np.abs(cls.LATTICE_LEVELS - val)))
        return float(cls.LATTICE_LEVELS[idx])

    @classmethod
    def compute_exact_bayesian_posterior(
        cls,
        c_samples: List[float],
        prior_p_severe: float = 0.55,
        mu_severe: float = 0.80,
        mu_minor: float = 0.20,
        sigma: float = 0.50,
    ) -> float:
        """Calculates exact recursive Bayesian posterior P(severe | c_1:k)."""
        if not c_samples:
            return prior_p_severe
        # Log likelihood ratio
        ll_severe = sum(-0.5 * ((c - mu_severe) / sigma) ** 2 for c in c_samples)
        ll_minor = sum(-0.5 * ((c - mu_minor) / sigma) ** 2 for c in c_samples)
        log_prior_ratio = np.log(prior_p_severe / (1.0 - prior_p_severe))
        log_post_ratio = log_prior_ratio + (ll_severe - ll_minor)
        log_post_ratio = float(np.clip(log_post_ratio, -30.0, 30.0))
        p_post = float(1.0 / (1.0 + np.exp(-log_post_ratio)))
        return p_post

    def generate_deterministic_tape(self, length: int, rng_seed: Optional[int] = None) -> EventTapeV2:
        tape_rng = np.random.RandomState(rng_seed if rng_seed is not None else self.seed)

        precursor_start_steps = []
        decision_window_steps = []
        shock_steps = []
        shock_mags = []
        precursor_noises = []

        # Schedule 2-3 structured shock events:
        # t_warn (t), t+1, t+2 (precursors), t+3 (delay), t+4 (decision window), t+5 (shock impact)
        t = 2
        while t < length - 6:
            if tape_rng.rand() < 0.70:
                warn_t = t
                dec_t = warn_t + 4
                shk_t = warn_t + 5
                is_sev = bool(tape_rng.rand() < 0.55)
                mag = 0.70 if is_sev else 0.10
                noises = [float(tape_rng.randn() * self.precursor_noise_std) for _ in range(3)]

                precursor_start_steps.append(warn_t)
                decision_window_steps.append(dec_t)
                shock_steps.append(shk_t)
                shock_mags.append(mag)
                precursor_noises.append(noises)
                t = shk_t + 3
            else:
                t += 2

        return EventTapeV2(
            precursor_start_steps=precursor_start_steps,
            decision_window_steps=decision_window_steps,
            shock_steps=shock_steps,
            shock_magnitudes=shock_mags,
            precursor_noise=precursor_noises,
            sensor_noise_a=[float(tape_rng.randn() * self.sensor_noise_std) for _ in range(length + 10)],
            sensor_noise_b=[float(tape_rng.randn() * self.sensor_noise_std) for _ in range(length + 10)],
            motor_bernoulli_draws=[float(tape_rng.rand()) for _ in range(length + 10)],
            world_bernoulli_draws=[float(tape_rng.rand()) for _ in range(length + 10)],
            target_goals=[int(tape_rng.randint(0, 2)) for _ in range(length + 10)],
            high_demand_steps=[bool(tape_rng.rand() < 0.5) for _ in range(length + 10)],
        )

    def reset(
        self,
        explicit_tape: Optional[EventTapeV2] = None,
        explicit_initial_i: float = 1.0,
        explicit_initial_x: float = 1.0,
    ) -> Tuple[ObservationV2, GroundTruthStateV2]:
        self._step_idx = 0
        self._i_t = self.quantize_lattice(explicit_initial_i)
        self._x_t = self.quantize_lattice(explicit_initial_x)
        self._is_precursor_active = False
        self._is_decision_window = False
        self._shock_pending = False
        self._shock_timer = 0
        self._pending_shock_magnitude = 0.0
        self._bayesian_risk_q = 0.55
        self._counterfactual_future_i = 1.0
        self._mitigation_active = False
        self._last_effect = None
        self._last_executed = 4 # NULL
        self._last_intended = 0

        if explicit_tape is not None:
            self._tape = explicit_tape
        else:
            self._tape = self.generate_deterministic_tape(self.episode_len)

        self._target_goal = self._tape.target_goals[0]

        noise_a = self._tape.sensor_noise_a[0]
        noise_b = self._tape.sensor_noise_b[0]

        self._ground_truth = GroundTruthStateV2(
            step_idx=0,
            internal_reliability_i=self._i_t,
            external_reliability_x=self._x_t,
            is_precursor_active=False,
            is_decision_window=False,
            shock_pending=False,
            shock_timer=0,
            pending_shock_magnitude=0.0,
            bayesian_risk_q=0.55,
            counterfactual_future_i_no_maint=1.0,
            mitigation_active=False,
            target_goal=self._target_goal,
            last_effect=None,
            last_action_executed=4,
            last_action_intended=0,
            is_terminal=False,
            is_decorative=self.is_decorative,
        )

        obs = ObservationV2(
            symbol=self._target_goal + 3,
            sensor_a=float(np.clip(self._i_t + noise_a, 0.0, 1.0)),
            sensor_b=float(np.clip(self._x_t + noise_b, 0.0, 1.0)),
            warning_cue=0.0,
            is_decision_window=0,
            last_action_executed=4,
            last_action_intended=0,
        )
        return obs, self._ground_truth

    def step(self, action: int) -> Tuple[ObservationV2, float, bool, GroundTruthStateV2]:
        assert self._ground_truth is not None, "Call reset() before step()."
        self._step_idx += 1
        is_terminal = bool(self._step_idx >= self.episode_len)
        reward = 0.0

        t = self._step_idx
        tape = self._tape
        assert tape is not None

        # 1. Update Precursor Cues, Decision Windows, and Shocks
        warning_signal = 0.0
        self._is_decision_window = False

        # Check if t is within a precursor window [warn_t, warn_t + 1, warn_t + 2]
        for w_idx, warn_t in enumerate(tape.precursor_start_steps):
            if warn_t <= t <= warn_t + 2:
                self._is_precursor_active = True
                self._shock_pending = True
                self._shock_timer = tape.shock_steps[w_idx] - t
                self._pending_shock_magnitude = tape.shock_magnitudes[w_idx]
                self._counterfactual_future_i = float(max(0.0, self._i_t - self._pending_shock_magnitude))

                # Emit continuous noisy precursor observation
                offset = t - warn_t
                mu = 0.80 if self._pending_shock_magnitude >= 0.50 else 0.20
                noise = tape.precursor_noise[w_idx][offset]
                warning_signal = float(np.clip(mu + noise, 0.0, 1.0))

                # Compute running exact Bayesian posterior from available precursor samples
                precursors_seen = [
                    float(np.clip((0.80 if self._pending_shock_magnitude >= 0.50 else 0.20) + tape.precursor_noise[w_idx][k], 0.0, 1.0))
                    for k in range(offset + 1)
                ]
                self._bayesian_risk_q = self.compute_exact_bayesian_posterior(
                    precursors_seen, sigma=self.precursor_noise_std
                )
                break
            elif t == tape.decision_window_steps[w_idx]:
                # Step t_4: Designated Regulatory Decision Window
                self._is_precursor_active = False
                self._is_decision_window = True
                self._shock_pending = True
                self._shock_timer = 1
                self._pending_shock_magnitude = tape.shock_magnitudes[w_idx]
                self._counterfactual_future_i = float(max(0.0, self._i_t - self._pending_shock_magnitude))
                warning_signal = 0.0 # Evidence has vanished
                # Keep final 3-sample posterior
                all_precursors = [
                    float(np.clip((0.80 if self._pending_shock_magnitude >= 0.50 else 0.20) + tape.precursor_noise[w_idx][k], 0.0, 1.0))
                    for k in range(3)
                ]
                self._bayesian_risk_q = self.compute_exact_bayesian_posterior(
                    all_precursors, sigma=self.precursor_noise_std
                )
                break

        # Check if shock strikes at beginning of this step (before action execution)
        if t in tape.shock_steps:
            s_idx = tape.shock_steps.index(t)
            drop = tape.shock_magnitudes[s_idx]
            if self._mitigation_active:
                actual_drop = min(drop, 0.10)
            else:
                actual_drop = drop

            self._i_t = self.quantize_lattice(max(0.0, self._i_t - actual_drop))
            self._shock_pending = False
            self._shock_timer = 0
            self._is_precursor_active = False
            self._is_decision_window = False
            self._mitigation_active = False

        if self.drift_rate > 0:
            self._i_t = self.quantize_lattice(max(0.0, self._i_t - self.drift_rate))
            self._x_t = self.quantize_lattice(max(0.0, self._x_t - self.drift_rate))

        # 2. Action Processing & Maintenance
        self._last_intended = action

        if action in [2, 3]:
            reward -= self.cost_maintain
            if action == 2:
                # MAINTAIN_A restores i_t. Activates shock mitigation buffer ONLY if executed at the designated decision window (t_4)
                self._i_t = 1.0
                if self._is_decision_window:
                    self._mitigation_active = True
                self._last_executed = 2
            elif action == 3:
                self._x_t = 1.0
                self._last_executed = 3
            executed_action = action
            effect = None
            obs_symbol = 0

        elif action in [0, 1]:
            p_exec = 1.0 if self.is_decorative else (0.50 + 0.50 * self._i_t)
            u_motor = tape.motor_bernoulli_draws[t % len(tape.motor_bernoulli_draws)]

            if u_motor < p_exec:
                executed_action = action
            else:
                executed_action = 4

            self._last_executed = executed_action

            if executed_action in [0, 1]:
                p_world = 0.50 + 0.50 * self._x_t
                u_world = tape.world_bernoulli_draws[t % len(tape.world_bernoulli_draws)]
                if u_world < p_world:
                    effect = executed_action
                else:
                    effect = 4
            else:
                effect = 4

            self._last_effect = effect

            # Evaluate Task Reward
            is_shock_step = bool(t in tape.shock_steps)
            is_high_dem = (not self._is_decision_window) and (is_shock_step or tape.high_demand_steps[t % len(tape.high_demand_steps)])
            multiplier = 5.0 if is_shock_step else (2.0 if is_high_dem else 1.0)

            if effect == self._target_goal:
                reward += self.reward_target_hit * multiplier
                obs_symbol = effect + 1
            elif effect in [0, 1] and effect != self._target_goal:
                reward += self.penalty_wrong_effect * multiplier
                obs_symbol = effect + 1
            else:
                null_penalty = -3.00 if is_shock_step else (-0.50 if is_high_dem else -0.10)
                reward += null_penalty
                obs_symbol = 0
        else:
            executed_action = 4
            self._last_executed = 4
            effect = None
            obs_symbol = 0

        if t < len(tape.target_goals):
            self._target_goal = tape.target_goals[t]

        noise_a = tape.sensor_noise_a[t % len(tape.sensor_noise_a)]
        noise_b = tape.sensor_noise_b[t % len(tape.sensor_noise_b)]

        obs_sensor_a = float(np.clip(self._i_t + noise_a, 0.0, 1.0))
        obs_sensor_b = float(np.clip(self._x_t + noise_b, 0.0, 1.0))

        self._ground_truth = GroundTruthStateV2(
            step_idx=t,
            internal_reliability_i=self._i_t,
            external_reliability_x=self._x_t,
            is_precursor_active=self._is_precursor_active,
            is_decision_window=self._is_decision_window,
            shock_pending=self._shock_pending,
            shock_timer=self._shock_timer,
            pending_shock_magnitude=self._pending_shock_magnitude,
            bayesian_risk_q=self._bayesian_risk_q,
            counterfactual_future_i_no_maint=self._counterfactual_future_i,
            mitigation_active=self._mitigation_active,
            target_goal=self._target_goal,
            last_effect=self._last_effect,
            last_action_executed=self._last_executed,
            last_action_intended=self._last_intended,
            is_terminal=is_terminal,
            is_decorative=self.is_decorative,
        )

        is_next_decision_win = bool((t + 1) in tape.decision_window_steps)

        obs = ObservationV2(
            symbol=self._target_goal + 3,
            sensor_a=obs_sensor_a,
            sensor_b=obs_sensor_b,
            warning_cue=warning_signal,
            is_decision_window=1 if is_next_decision_win else 0,
            last_action_executed=self._last_executed,
            last_action_intended=self._last_intended,
        )

        return obs, reward, is_terminal, self._ground_truth

    def snapshot(self) -> EnvironmentSnapshotV2:
        assert self._ground_truth is not None, "Cannot snapshot uninitialized environment."
        return EnvironmentSnapshotV2(
            step_idx=self._step_idx,
            internal_reliability_i=self._i_t,
            external_reliability_x=self._x_t,
            is_precursor_active=self._is_precursor_active,
            is_decision_window=self._is_decision_window,
            shock_pending=self._shock_pending,
            shock_timer=self._shock_timer,
            pending_shock_magnitude=self._pending_shock_magnitude,
            bayesian_risk_q=self._bayesian_risk_q,
            counterfactual_future_i_no_maint=self._counterfactual_future_i,
            mitigation_active=self._mitigation_active,
            target_goal=self._target_goal,
            last_effect=self._last_effect,
            last_action_executed=self._last_executed,
            last_action_intended=self._last_intended,
            is_terminal=self._ground_truth.is_terminal,
            is_decorative=self.is_decorative,
            tape=copy.deepcopy(self._tape),
            rng_state=self.rng.get_state(),
        )

    def restore(self, snap: EnvironmentSnapshotV2) -> None:
        self._step_idx = snap.step_idx
        self._i_t = snap.internal_reliability_i
        self._x_t = snap.external_reliability_x
        self._is_precursor_active = snap.is_precursor_active
        self._is_decision_window = snap.is_decision_window
        self._shock_pending = snap.shock_pending
        self._shock_timer = snap.shock_timer
        self._pending_shock_magnitude = snap.pending_shock_magnitude
        self._bayesian_risk_q = snap.bayesian_risk_q
        self._counterfactual_future_i = snap.counterfactual_future_i_no_maint
        self._mitigation_active = snap.mitigation_active
        self._target_goal = snap.target_goal
        self._last_effect = snap.last_effect
        self._last_executed = snap.last_action_executed
        self._last_intended = snap.last_action_intended
        self.is_decorative = snap.is_decorative
        self._tape = copy.deepcopy(snap.tape)
        self.rng.set_state(snap.rng_state)

        self._ground_truth = GroundTruthStateV2(
            step_idx=snap.step_idx,
            internal_reliability_i=snap.internal_reliability_i,
            external_reliability_x=snap.external_reliability_x,
            is_precursor_active=snap.is_precursor_active,
            is_decision_window=snap.is_decision_window,
            shock_pending=snap.shock_pending,
            shock_timer=snap.shock_timer,
            pending_shock_magnitude=snap.pending_shock_magnitude,
            bayesian_risk_q=snap.bayesian_risk_q,
            counterfactual_future_i_no_maint=snap.counterfactual_future_i_no_maint,
            mitigation_active=snap.mitigation_active,
            target_goal=snap.target_goal,
            last_effect=snap.last_effect,
            last_action_executed=snap.last_action_executed,
            last_action_intended=snap.last_action_intended,
            is_terminal=snap.is_terminal,
            is_decorative=snap.is_decorative,
        )

    def clone(self) -> "DualLocusRegulatorEnv":
        cloned = DualLocusRegulatorEnv(
            episode_len=self.episode_len,
            cost_maintain=self.cost_maintain,
            reward_target_hit=self.reward_target_hit,
            penalty_wrong_effect=self.penalty_wrong_effect,
            sensor_noise_std=self.sensor_noise_std,
            precursor_noise_std=self.precursor_noise_std,
            drift_rate=self.drift_rate,
            is_decorative=self.is_decorative,
            seed=self.seed,
        )
        if self._ground_truth is not None:
            cloned.restore(self.snapshot())
        return cloned
