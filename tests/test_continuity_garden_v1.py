"""Unit tests for Continuity Garden v1 Controllability Environment & Models."""

import numpy as np
import pytest
import torch

from src.continuity_garden.environment_v1 import ControllabilityArenaEnv, ObservationV1
from src.continuity_garden.models_v1 import ControllableOrganism


def test_v1_no_construct_leakage():
    """Verifies that ObservationV1 has zero target labels or world_type indicators."""
    env = ControllabilityArenaEnv(seed=42)
    obs, gt = env.reset()

    # Observation must only contain symbol (0..4) and action feedback (0..2)
    assert hasattr(obs, "symbol")
    assert hasattr(obs, "action_executed")
    assert hasattr(obs, "action_intended")
    assert not hasattr(obs, "world_type")
    assert not hasattr(obs, "controllable")
    assert not hasattr(obs, "agency")
    assert not hasattr(obs, "is_forced")


def test_v1_yoked_marginal_matching():
    """Verifies that W_ctrl and W_yoked have matched marginal outcome distributions P(E)."""
    env = ControllabilityArenaEnv(seed=42)
    ctrl_effects = []
    yoked_effects = []

    for ep in range(200):
        obs, gt = env.reset(explicit_world_type="ctrl")
        done = False
        while not done:
            action = int(np.random.randint(0, 2))
            obs, rew, done, gt = env.step(action)
            if gt.last_effect is not None:
                ctrl_effects.append(gt.last_effect)

        obs, gt = env.reset(explicit_world_type="yoked")
        done = False
        while not done:
            action = int(np.random.randint(0, 2))
            obs, rew, done, gt = env.step(action)
            if gt.last_effect is not None:
                yoked_effects.append(gt.last_effect)

    mean_ctrl_e = np.mean(ctrl_effects)
    mean_yoked_e = np.mean(yoked_effects)

    # Both must be approximately 0.50 (balanced marginals)
    assert 0.45 <= mean_ctrl_e <= 0.55
    assert 0.45 <= mean_yoked_e <= 0.55
    assert abs(mean_ctrl_e - mean_yoked_e) < 0.05


def test_v1_environment_and_organism_snapshot_determinism():
    """Verifies that pausing mid-exploration and restoring resumes bitwise identically."""
    env = ControllabilityArenaEnv(seed=123)
    model = ControllableOrganism()
    model.eval()

    obs, gt = env.reset()
    h = None

    # Step 3 exploration steps
    for _ in range(3):
        h, motor_logits, _, _ = model.step(obs, h)
        act = int(torch.argmax(motor_logits).item())
        obs, rew, done, gt = env.step(act)

    # Snapshot both
    env_snap = env.snapshot()
    org_snap = model.snapshot(h, step_idx=env._step_idx)

    # Continue original branch
    orig_obs_seq = []
    orig_rew_seq = []
    while not done:
        h, motor_logits, exploit_logits, _ = model.step(obs, h)
        act = int(torch.argmax(motor_logits if gt.current_phase == "exploration" else exploit_logits).item())
        obs, rew, done, gt = env.step(act)
        orig_obs_seq.append(obs.symbol)
        orig_rew_seq.append(rew)

    # Create new instances and restore
    env_clone = ControllabilityArenaEnv(seed=0)
    model_clone = ControllableOrganism()
    env_clone.restore(env_snap)
    h_clone = model_clone.restore(org_snap)

    curr_obs = ObservationV1(
        symbol=env_clone._last_effect + 1 if env_clone._last_effect is not None else 0,
        action_executed=env_clone._last_executed,
        action_intended=env_clone._last_intended,
    )
    clone_obs_seq = []
    clone_rew_seq = []
    done_clone = False

    while not done_clone:
        h_clone, motor_logits, exploit_logits, _ = model_clone.step(curr_obs, h_clone)
        act = int(torch.argmax(motor_logits if env_clone._ground_truth.current_phase == "exploration" else exploit_logits).item())
        curr_obs, rew, done_clone, gt = env_clone.step(act)
        clone_obs_seq.append(curr_obs.symbol)
        clone_rew_seq.append(rew)

    assert orig_obs_seq == clone_obs_seq
    assert orig_rew_seq == clone_rew_seq
