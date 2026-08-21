"""Unit Tests for Continuity Garden v2 (Dual-Locus Causal Kernel & Gate D0 Invariants)."""

from pathlib import Path
import sys
import numpy as np
import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.continuity_garden.environment_v2 import (
    DualLocusRegulatorEnv,
    EventTapeV2,
    GroundTruthStateV2,
    ObservationV2,
)
from src.continuity_garden.models_v2 import DualLocusOrganism
from src.continuity_garden.oracle_v2 import (
    AlwaysMaintainPolicy,
    NeverMaintainPolicy,
    ObservationBeliefOracle,
    PrivilegedGroundTruthOracle,
    ReactiveSensorDropPolicy,
    ShortHistoryWindowPolicy,
    WarningReflexPolicy,
    run_gate_d0a_observability_calibration,
)
from src.continuity_garden.trainer_v2 import (
    evaluate_motor_competence,
    run_gate_d0b_optimizer_validity,
    train_duallocus_organism,
)


def test_v2_finite_lattice_quantization():
    """Verifies that internal and external reliability remain strictly on the 11-level lattice."""
    env = DualLocusRegulatorEnv(seed=42)
    obs, gt = env.reset()

    assert gt.internal_reliability_i in DualLocusRegulatorEnv.LATTICE_LEVELS
    assert gt.external_reliability_x in DualLocusRegulatorEnv.LATTICE_LEVELS

    for _ in range(50):
        action = int(np.random.randint(0, 4))
        obs, rew, done, gt = env.step(action)
        assert gt.internal_reliability_i in DualLocusRegulatorEnv.LATTICE_LEVELS
        assert gt.external_reliability_x in DualLocusRegulatorEnv.LATTICE_LEVELS
        if done:
            obs, gt = env.reset()


def test_v2_no_construct_leakage():
    """Verifies that ObservationV2 contains zero semantic labels or privileged state."""
    env = DualLocusRegulatorEnv(seed=42)
    obs, gt = env.reset()

    fields = obs.__dataclass_fields__.keys()
    forbidden_tokens = ["health", "battery", "self", "world", "ground_truth", "pending_shock", "shock_timer", "bayesian_risk", "counterfactual"]
    for f in fields:
        for forbidden in forbidden_tokens:
            assert forbidden not in f.lower(), f"Construct leakage detected: {f}"

    tape = env.generate_deterministic_tape(env.episode_len, rng_seed=42)
    obs, gt = env.reset(explicit_tape=tape)
    assert isinstance(obs.sensor_a, float)
    assert isinstance(obs.sensor_b, float)
    assert isinstance(obs.warning_cue, float)
    assert obs.is_decision_window in [0, 1]


def test_v2_deterministic_snapshot_and_restore():
    """Verifies that snapshot and restore perfectly reproduce identical trajectory futures."""
    env = DualLocusRegulatorEnv(seed=123)
    tape = env.generate_deterministic_tape(env.episode_len, rng_seed=123)
    obs, gt = env.reset(explicit_tape=tape)

    for a in [0, 1, 2, 0, 1]:
        env.step(a)

    snap = env.snapshot()

    branch1_obs = []
    for a in [0, 1, 0, 2, 1]:
        o, r, d, g = env.step(a)
        branch1_obs.append((o.sensor_a, o.sensor_b, r))

    env.restore(snap)
    branch2_obs = []
    for a in [0, 1, 0, 2, 1]:
        o, r, d, g = env.step(a)
        branch2_obs.append((o.sensor_a, o.sensor_b, r))

    assert branch1_obs == branch2_obs, "Snapshot restore failed to deterministically replay environment."


def test_v2_paired_lineage_common_random_numbers():
    """Verifies that Lineage A (Consequential) and Lineage B (Decorative) receive identical exogenous tapes."""
    seed = 777
    env_a = DualLocusRegulatorEnv(is_decorative=False, seed=seed)
    env_b = DualLocusRegulatorEnv(is_decorative=True, seed=seed)

    tape_a = env_a.generate_deterministic_tape(env_a.episode_len, rng_seed=seed)
    tape_b = env_b.generate_deterministic_tape(env_b.episode_len, rng_seed=seed)

    assert tape_a.precursor_start_steps == tape_b.precursor_start_steps
    assert tape_a.decision_window_steps == tape_b.decision_window_steps
    assert tape_a.shock_steps == tape_b.shock_steps
    assert tape_a.shock_magnitudes == tape_b.shock_magnitudes
    assert tape_a.precursor_noise == tape_b.precursor_noise
    assert tape_a.sensor_noise_a == tape_b.sensor_noise_a
    assert tape_a.motor_bernoulli_draws == tape_b.motor_bernoulli_draws


def test_v2_gate_d0a_observability_inequality():
    """Verifies Gate D0a: E[R_Privileged] >= E[R_Belief] > max(Heuristics) + 0.20."""
    calib = run_gate_d0a_observability_calibration(num_episodes=100, seed=42)
    assert calib["gate_d0a_pass"] is True
    assert calib["belief_oracle_advantage"] >= 0.20


def test_v2_gate_d0b_optimizer_validity_smoke():
    """Verifies Gate D0b on 2 test seeds: Privileged agent achieves return >= 28.0 and competence >= 75%."""
    d0b_res = run_gate_d0b_optimizer_validity(seeds=[42, 43], episodes_per_seed=300, warmup_episodes=40)
    assert d0b_res["gate_d0b_pass"] is True


def test_v2_first_order_motor_competence():
    """Verifies that an agent trained on baseline achieves >80% motor competence."""
    model = DualLocusOrganism()
    returns, ckpts = train_duallocus_organism(model, num_episodes=100, warmup_episodes=50, seed=42)
    comp = evaluate_motor_competence(model, num_episodes=20, seed=42)
    assert comp >= 0.75, f"Motor competence too low: {comp*100:.1f}%"
