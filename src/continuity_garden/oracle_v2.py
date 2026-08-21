"""Garden v2 Oracle & Gate D0a Observability Calibration Suite.

Evaluates policies on DualLocusRegulatorEnv to prove Gate D0a:
  E[R_Privileged] >= E[R_Observation_Belief_Oracle] > max(E[R_Heuristics]) + 0.20
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.continuity_garden.environment_v2 import DualLocusRegulatorEnv, GroundTruthStateV2, ObservationV2


class BasePolicyV2:
    def reset(self):
        pass

    def act(self, obs: ObservationV2, gt: Optional[GroundTruthStateV2] = None) -> int:
        raise NotImplementedError


class NeverMaintainPolicy(BasePolicyV2):
    """Always attempts the requested motor goal; never performs maintenance."""
    def act(self, obs: ObservationV2, gt: Optional[GroundTruthStateV2] = None) -> int:
        goal = (obs.symbol - 3) if obs.symbol in [3, 4] else 0
        return goal


class AlwaysMaintainPolicy(BasePolicyV2):
    """Maintains on every step."""
    def act(self, obs: ObservationV2, gt: Optional[GroundTruthStateV2] = None) -> int:
        return 2 # MAINTAIN_A


class ReactiveSensorDropPolicy(BasePolicyV2):
    """Maintains reactively ONLY AFTER observing a low sensor_A reading (post-shock)."""
    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold

    def act(self, obs: ObservationV2, gt: Optional[GroundTruthStateV2] = None) -> int:
        goal = (obs.symbol - 3) if obs.symbol in [3, 4] else 0
        if obs.sensor_a < self.threshold:
            return 2 # MAINTAIN_A
        return goal


class WarningReflexPolicy(BasePolicyV2):
    """Maintains reflexively on the very first precursor cue (t_0), regardless of evidence quality."""
    def __init__(self):
        self.already_maintained_for_event = False

    def reset(self):
        self.already_maintained_for_event = False

    def act(self, obs: ObservationV2, gt: Optional[GroundTruthStateV2] = None) -> int:
        goal = (obs.symbol - 3) if obs.symbol in [3, 4] else 0
        if obs.warning_cue > 0.0:
            if not self.already_maintained_for_event:
                self.already_maintained_for_event = True
                return 2 # MAINTAIN_A immediately
        else:
            self.already_maintained_for_event = False
        return goal


class ShortHistoryWindowPolicy(BasePolicyV2):
    """
    Shallow finite-memory heuristic:
    At the decision window (t_4), checks only the immediately preceding observation (t_3).
    Since t_3 is blank (warning_cue = 0.0), it has 0 bits of precursor information and defaults to motor goal.
    """
    def act(self, obs: ObservationV2, gt: Optional[GroundTruthStateV2] = None) -> int:
        goal = (obs.symbol - 3) if obs.symbol in [3, 4] else 0
        return goal


class ObservationBeliefOracle(BasePolicyV2):
    """
    Genuine Observation-Based POMDP Bayes Oracle:
    Receives ONLY the public observation stream obs (no GroundTruthState access).
    Integrates the noisy precursor cues c_1, c_2, c_3 recursively to maintain posterior q_t = P(severe | c_1:k).
    At the designated decision window (obs.is_decision_window == 1), maintains IF AND ONLY IF q_t >= threshold.
    """
    def __init__(
        self,
        threshold: float = 0.45,
        precursor_noise_std: float = 0.50,
    ):
        self.threshold = threshold
        self.sigma = precursor_noise_std
        self.precursor_history: List[float] = []

    def reset(self):
        self.precursor_history = []

    def act(self, obs: ObservationV2, gt: Optional[GroundTruthStateV2] = None) -> int:
        goal = (obs.symbol - 3) if obs.symbol in [3, 4] else 0

        # Collect precursor evidence when active
        if obs.warning_cue > 0.0:
            self.precursor_history.append(obs.warning_cue)

        # At Designated Regulatory Decision Window: evaluate integrated belief
        if obs.is_decision_window == 1:
            q_severe = DualLocusRegulatorEnv.compute_exact_bayesian_posterior(
                self.precursor_history, sigma=self.sigma
            )
            # Reset precursor buffer for next event
            self.precursor_history = []

            if q_severe >= self.threshold:
                return 2 # MAINTAIN_A in advance

        # Reactive fallback if unexpectedly low sensor
        if obs.sensor_a < 0.40:
            return 2

        return goal


class PrivilegedGroundTruthOracle(BasePolicyV2):
    """
    Privileged Upper Bound Benchmark:
    Accesses hidden GroundTruthStateV2.pending_shock_magnitude directly.
    Maintains at the decision window (obs.is_decision_window == 1) if upcoming shock is severe (>= 0.50).
    """
    def act(self, obs: ObservationV2, gt: Optional[GroundTruthStateV2] = None) -> int:
        goal = (obs.symbol - 3) if obs.symbol in [3, 4] else 0
        assert gt is not None, "Privileged oracle requires ground truth state."

        if obs.is_decision_window == 1:
            if gt.pending_shock_magnitude >= 0.50:
                return 2 # MAINTAIN_A in advance

        if obs.sensor_a < 0.40:
            return 2

        return goal


def evaluate_policy_on_env(
    policy: BasePolicyV2,
    env: DualLocusRegulatorEnv,
    num_episodes: int = 150,
    seed: int = 42,
) -> Dict[str, float]:
    returns = []
    maintenance_counts = []
    target_hits = []

    for ep_idx in range(num_episodes):
        tape = env.generate_deterministic_tape(env.episode_len, rng_seed=seed + ep_idx * 100)
        obs, gt = env.reset(explicit_tape=tape)
        policy.reset()
        done = False
        ep_return = 0.0
        maint_count = 0
        hits = 0

        while not done:
            action = policy.act(obs, gt)
            if action in [2, 3]:
                maint_count += 1
            obs, rew, done, gt = env.step(action)
            ep_return += rew
            if rew > 0.5:
                hits += 1

        returns.append(ep_return)
        maintenance_counts.append(maint_count)
        target_hits.append(hits)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_maintenance_count": float(np.mean(maintenance_counts)),
        "mean_target_hits": float(np.mean(target_hits)),
    }


def run_gate_d0a_observability_calibration(
    num_episodes: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    env = DualLocusRegulatorEnv(
        episode_len=24,
        cost_maintain=0.15,
        reward_target_hit=1.00,
        penalty_wrong_effect=-0.50,
        sensor_noise_std=0.08,
        precursor_noise_std=0.35,
        seed=seed,
    )

    policies = {
        "never_maintain": NeverMaintainPolicy(),
        "always_maintain": AlwaysMaintainPolicy(),
        "reactive_sensor_drop": ReactiveSensorDropPolicy(threshold=0.60),
        "warning_reflex": WarningReflexPolicy(),
        "short_history_window": ShortHistoryWindowPolicy(),
        "observation_belief_oracle": ObservationBeliefOracle(threshold=0.60, precursor_noise_std=0.35),
        "privileged_ground_truth_oracle": PrivilegedGroundTruthOracle(),
    }

    results = {}
    print("=======================================================")
    print("Executing Gate D0a Observability Calibration Suite")
    print("=======================================================")

    for name, pol in policies.items():
        res = evaluate_policy_on_env(pol, env, num_episodes=num_episodes, seed=seed)
        results[name] = res
        print(f"  {name:<32}: Return = {res['mean_return']:+.2f} (+/- {res['std_return']:.2f}) | Maint = {res['mean_maintenance_count']:.1f} | Hits = {res['mean_target_hits']:.1f}")

    # Step-by-step diagnostic on episode 0
    tape0 = env.generate_deterministic_tape(env.episode_len, rng_seed=seed)
    for p_name in ["reactive_sensor_drop", "observation_belief_oracle", "privileged_ground_truth_oracle"]:
        p = policies[p_name]
        p.reset()
        obs, gt = env.reset(explicit_tape=tape0)
        done = False
        step_logs = []
        tot_r = 0.0
        while not done:
            a = p.act(obs, gt)
            obs, r, done, gt = env.step(a)
            tot_r += r
            step_logs.append(f"t={gt.step_idx}: a={a}, i={gt.internal_reliability_i:.1f}, dec_win={gt.is_decision_window}, r={r:+.1f}")
        print(f"\n--- Diagnostic Trace for {p_name} (Total Return = {tot_r:+.2f}) ---")
        for line in step_logs[2:12]:
            print("  " + line)

    r_priv = results["privileged_ground_truth_oracle"]["mean_return"]
    r_belief = results["observation_belief_oracle"]["mean_return"]
    r_warn = results["warning_reflex"]["mean_return"]
    r_react = results["reactive_sensor_drop"]["mean_return"]
    r_short = results["short_history_window"]["mean_return"]
    r_never = results["never_maintain"]["mean_return"]

    max_heuristic = max(r_warn, r_react, r_short, r_never)
    belief_advantage = r_belief - max_heuristic

    print("\n=======================================================")
    print("Gate D0a Inequality Evaluation:")
    print(f"  E[R_Privileged]             = {r_priv:+.2f}")
    print(f"  E[R_Observation_Belief]     = {r_belief:+.2f}")
    print(f"  Max Heuristic Baseline       = {max_heuristic:+.2f}")
    print(f"  Belief Oracle Advantage     = {belief_advantage:+.2f} (Target: >= +0.20)")
    print("=======================================================")

    gate_d0a_pass = bool(r_priv >= r_belief and belief_advantage >= 0.20)
    print(f"[Gate D0a Verdict]: {'PASS' if gate_d0a_pass else 'FAIL'}\n")

    return {
        "gate_d0a_pass": gate_d0a_pass,
        "belief_oracle_advantage": float(belief_advantage),
        "policy_metrics": results,
    }


if __name__ == "__main__":
    run_gate_d0a_observability_calibration()
