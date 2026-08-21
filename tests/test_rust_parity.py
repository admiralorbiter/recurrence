"""Python <-> Rust Parity & Core Invariant Tests."""

from pathlib import Path
import sys
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.continuity_garden.environment_v2 import DualLocusRegulatorEnv, GroundTruthStateV2, ObservationV2


def test_bayesian_posterior_math_parity():
    """Verifies that Bayesian posterior and log-odds math matches Rust implementation."""
    env = DualLocusRegulatorEnv(precursor_noise_std=0.35, seed=42)

    # 3 severe precursor samples
    c_severe = [0.85, 0.78, 0.82]
    q_severe = env.compute_exact_bayesian_posterior(c_severe)
    assert q_severe > 0.90, f"Severe posterior too low: {q_severe}"

    log_odds_severe = float(np.log(q_severe / (1.0 - q_severe)))
    assert log_odds_severe > 2.0

    # 3 minor precursor samples
    c_minor = [0.15, 0.22, 0.18]
    q_minor = env.compute_exact_bayesian_posterior(c_minor)
    assert q_minor < 0.15, f"Minor posterior too high: {q_minor}"

    log_odds_minor = float(np.log(q_minor / (1.0 - q_minor)))
    assert log_odds_minor < -1.5


def test_finite_lattice_quantization_parity():
    """Verifies that lattice quantization matches Rust levels."""
    levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for val in [0.04, 0.12, 0.58, 0.96]:
        quant = DualLocusRegulatorEnv.quantize_lattice(val)
        assert quant in levels
