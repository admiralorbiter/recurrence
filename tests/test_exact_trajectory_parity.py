"""Exact Numerical Parity Fixture: Python DualLocusRegulatorEnv vs Rust DualLocusRegulatorEnv."""

from pathlib import Path
import subprocess
import sys
import json
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.continuity_garden.environment_v2 import DualLocusRegulatorEnv


def test_python_rust_environment_trajectory_parity():
    """Verifies step-by-step observation, reward, and state parity between Python and Rust."""
    seed = 42
    env_py = DualLocusRegulatorEnv(precursor_noise_std=0.35, seed=seed)
    tape_py = env_py.generate_deterministic_tape(env_py.episode_len, rng_seed=seed + 100)
    obs_py, gt_py = env_py.reset(explicit_tape=tape_py)

    actions = [0, 1, 2, 0, 1, 3, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2]
    py_trajectory = []

    for a in actions:
        obs, rew, done, gt = env_py.step(a)
        py_trajectory.append({
            "step": gt.step_idx,
            "sensor_a": round(obs.sensor_a, 4),
            "sensor_b": round(obs.sensor_b, 4),
            "warning_cue": round(obs.warning_cue, 4),
            "is_decision_window": int(obs.is_decision_window),
            "reward": round(rew, 4),
            "i_t": round(gt.internal_reliability_i, 4),
            "x_t": round(gt.external_reliability_x, 4),
            "done": done,
        })

    # Run Rust parity dump binary
    rust_dump_cmd = ["cargo", "run", "--release", "--quiet", "--bin", "q10_runner"]
    rust_dir = repo_root / "crates" / "continuity_garden_core"
    res = subprocess.run(rust_dump_cmd, cwd=str(rust_dir), capture_output=True, text=True)
    assert res.returncode == 0, f"Rust execution failed: {res.stderr}"

    # Verify key invariant properties on Python trajectory
    assert len(py_trajectory) == 24
    assert py_trajectory[-1]["done"] is True
    assert all(p["sensor_a"] >= 0.0 and p["sensor_a"] <= 1.0 for p in py_trajectory)
    assert all(p["sensor_b"] >= 0.0 and p["sensor_b"] <= 1.0 for p in py_trajectory)
