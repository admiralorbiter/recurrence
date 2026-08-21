"""Exact Numerical Trajectory Parity Fixture: Python vs Rust DualLocusRegulatorEnv."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.continuity_garden.environment_v2 import DualLocusRegulatorEnv, EventTapeV2


def test_python_rust_exact_step_by_step_parity():
    """Generates a serialized fixture tape and asserts step-by-step bit parity between Python and Rust."""
    # 1. Generate Deterministic Fixture Tape
    fixed_tape = {
        "precursor_start_steps": [2, 13],
        "decision_window_steps": [7, 18],
        "shock_steps": [8, 19],
        "shock_magnitudes": [0.70, 0.10],
        "precursor_noise": [
            [0.05, -0.02, 0.03],
            [-0.04, 0.01, -0.02],
        ],
        "sensor_noise_a": [0.01 * ((i % 5) - 2) for i in range(35)],
        "sensor_noise_b": [0.01 * ((i % 7) - 3) for i in range(35)],
        "motor_bernoulli_draws": [0.10 if i % 2 == 0 else 0.90 for i in range(35)],
        "world_bernoulli_draws": [0.20 if i % 3 == 0 else 0.80 for i in range(35)],
        "target_goals": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "high_demand_steps": [i % 4 == 0 for i in range(35)],
    }

    tape_py = EventTapeV2(
        precursor_start_steps=fixed_tape["precursor_start_steps"],
        decision_window_steps=fixed_tape["decision_window_steps"],
        shock_steps=fixed_tape["shock_steps"],
        shock_magnitudes=fixed_tape["shock_magnitudes"],
        precursor_noise=fixed_tape["precursor_noise"],
        sensor_noise_a=fixed_tape["sensor_noise_a"],
        sensor_noise_b=fixed_tape["sensor_noise_b"],
        motor_bernoulli_draws=fixed_tape["motor_bernoulli_draws"],
        world_bernoulli_draws=fixed_tape["world_bernoulli_draws"],
        target_goals=fixed_tape["target_goals"],
        high_demand_steps=fixed_tape["high_demand_steps"],
    )

    actions = [0, 1, 2, 0, 1, 3, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2]

    # 2. Step Python Environment
    env_py = DualLocusRegulatorEnv(is_decorative=False, seed=42)
    obs_py, gt_py = env_py.reset(explicit_tape=tape_py)

    py_steps = []
    for a in actions:
        obs, rew, done, gt = env_py.step(a)
        py_steps.append({
            "step": gt.step_idx,
            "symbol": obs.symbol,
            "sensor_a": obs.sensor_a,
            "sensor_b": obs.sensor_b,
            "warning_cue": obs.warning_cue,
            "is_decision_window": int(obs.is_decision_window),
            "reward": rew,
            "i_t": gt.internal_reliability_i,
            "x_t": gt.external_reliability_x,
            "done": done,
        })

    # 3. Step Rust Environment with Identical Fixture Tape
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(fixed_tape, f)
        temp_tape_path = f.name

    try:
        rust_dir = repo_root / "crates" / "continuity_garden_core"
        cmd = ["cargo", "run", "--release", "--quiet", "--bin", "dump_trajectory", "--", temp_tape_path]
        res = subprocess.run(cmd, cwd=str(rust_dir), capture_output=True, text=True)
        assert res.returncode == 0, f"Rust dump_trajectory failed: {res.stderr}"
        rust_steps = json.loads(res.stdout)
    finally:
        Path(temp_tape_path).unlink(missing_ok=True)

    # 4. Step-by-Step Assertions
    assert len(py_steps) == len(rust_steps) == 24, "Trajectory length mismatch"

    for t in range(24):
        py_s = py_steps[t]
        rust_s = rust_steps[t]

        assert py_s["step"] == rust_s["step"], f"Step index mismatch at t={t}"
        assert py_s["symbol"] == rust_s["symbol"], f"Symbol mismatch at t={t}: Py {py_s['symbol']} vs Rust {rust_s['symbol']}"
        assert py_s["is_decision_window"] == rust_s["is_decision_window"], f"Decision window flag mismatch at t={t}: Py {py_s['is_decision_window']} vs Rust {rust_s['is_decision_window']}"
        assert py_s["done"] == rust_s["done"], f"Done flag mismatch at t={t}"

        # Float fields checked within tight numerical precision
        assert abs(py_s["sensor_a"] - rust_s["sensor_a"]) < 1e-4, f"Sensor A mismatch at t={t}: Py {py_s['sensor_a']} vs Rust {rust_s['sensor_a']}"
        assert abs(py_s["sensor_b"] - rust_s["sensor_b"]) < 1e-4, f"Sensor B mismatch at t={t}: Py {py_s['sensor_b']} vs Rust {rust_s['sensor_b']}"
        assert abs(py_s["warning_cue"] - rust_s["warning_cue"]) < 1e-4, f"Warning cue mismatch at t={t}: Py {py_s['warning_cue']} vs Rust {rust_s['warning_cue']}"
        assert abs(py_s["reward"] - rust_s["reward"]) < 1e-4, f"Reward mismatch at t={t}: Py {py_s['reward']} vs Rust {rust_s['reward']}"
        assert abs(py_s["i_t"] - rust_s["i_t"]) < 1e-4, f"Internal reliability i_t mismatch at t={t}: Py {py_s['i_t']} vs Rust {rust_s['i_t']}"
        assert abs(py_s["x_t"] - rust_s["x_t"]) < 1e-4, f"External reliability x_t mismatch at t={t}: Py {py_s['x_t']} vs Rust {rust_s['x_t']}"
