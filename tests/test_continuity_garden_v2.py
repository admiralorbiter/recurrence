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
    BayesOptimalOraclePolicy,
    NeverMaintainPolicy,
    ReactiveSensorDropPolicy,
    WarningReflexPolicy,
    evaluate_policy_on_env,
    run_gate_d0_calibration,
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

    # Verify field names of ObservationV2
    fields = obs.__dataclass_fields__.keys()
    forbidden_tokens = ["health", "battery", "self", "world", "ground_truth", "pending_shock", "shock_timer"]
    for f in fields:
        for forbidden in forbidden_tokens:
            assert forbidden not in f.lower(), f"Construct leakage detected: {f}"

    # Verify sensor values are noisy continuous floats, not raw ground truth
    tape = env.generate_deterministic_tape(env.episode_len, rng_seed=42)
    obs, gt = env.reset(explicit_tape=tape)
    assert isinstance(obs.sensor_a, float)
    assert isinstance(obs.sensor_b, float)
    assert obs.warning_cue in [0, 1]


def test_v2_deterministic_snapshot_and_restore():
    """Verifies that snapshot and restore perfectly reproduce identical trajectory futures."""
    env = DualLocusRegulatorEnv(seed=123)
    tape = env.generate_deterministic_tape(env.episode_len, rng_seed=123)
    obs, gt = env.reset(explicit_tape=tape)

    # Step 5 times
    for a in [0, 1, 2, 0, 1]:
        env.step(a)

    snap = env.snapshot()

    # Step 5 more times along Branch 1
    branch1_obs = []
    for a in [0, 1, 0, 2, 1]:
        o, r, d, g = env.step(a)
        branch1_obs.append((o.sensor_a, o.sensor_b, r))

    # Restore snapshot and step along Branch 2 with same actions
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

    assert tape_a.warning_steps == tape_b.warning_steps
    assert tape_a.shock_steps == tape_b.shock_steps
    assert tape_a.shock_magnitudes == tape_b.shock_magnitudes
    assert tape_a.sensor_noise_a == tape_b.sensor_noise_a
    assert tape_a.motor_bernoulli_draws == tape_b.motor_bernoulli_draws


def test_v2_gate_d0_inequality():
    """Verifies the Gate D0 Calibration Inequality: E[R_Bayes] >= max(Heuristics) + 0.20."""
    calib = run_gate_d0_calibration(num_episodes=150, seed=42)
    assert calib["gate_d0_pass"] is True
    assert calib["oracle_advantage"] >= 0.20
