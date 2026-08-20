"""Tests for Continuity Garden v0 Kernel and Q06 Construct Separation."""

import pytest
from src.continuity_garden.environment import HiddenSwitchboardEnv
from src.continuity_garden.models import OracleBeliefAgent
from src.continuity_garden.observation import AgentObservation


def test_q06_no_target_construct_leakage():
    """Q06 Construct Separation Invariant: Asserts that ground truth variables never appear in AgentObservation."""
    env = HiddenSwitchboardEnv(min_delay=8, max_delay=16, num_queries=5, seed=42)
    obs, gt = env.reset()

    # Inspect AgentObservation fields
    fields = obs.__dataclass_fields__.keys()
    forbidden_terms = ["hidden_mode", "true_source", "uncertainty", "self_state", "mode", "target_action"]

    for field_name in fields:
        for term in forbidden_terms:
            assert term not in field_name.lower(), f"Construct leakage detected: field '{field_name}' in AgentObservation"

    # Step through entire episode and verify every observation
    done = False
    while not done:
        obs, rew, done, gt = env.step(0)
        # Verify observation values
        assert isinstance(obs.symbol, int)
        assert obs.symbol in [0, 1, 2, 3, 4], f"Unexpected observation symbol: {obs.symbol}"
        # Symbol must not contain raw ground truth z during distractor
        if gt.current_phase == "distractor":
            assert obs.symbol == 0, "Distractor symbol must be 0 (blank/neutral)"


def test_oracle_belief_agent_perfect_accuracy():
    """Validates that the Oracle agent achieves perfect ceiling performance (1.0)."""
    num_queries = 5
    env = HiddenSwitchboardEnv(min_delay=8, max_delay=16, num_queries=num_queries, seed=123)
    oracle = OracleBeliefAgent()

    num_episodes = 20
    total_rewards = 0

    for _ in range(num_episodes):
        obs, gt = env.reset()
        oracle.reset(gt)
        done = False
        while not done:
            action = oracle.act(gt)
            obs, rew, done, gt = env.step(action)
            total_rewards += rew

    expected_rewards = num_episodes * num_queries
    assert total_rewards == expected_rewards, f"Oracle total reward must be {expected_rewards}, got {total_rewards}"


def test_environment_deterministic_replay():
    """Tests that identical seeds produce bitwise identical episode sequences."""
    env1 = HiddenSwitchboardEnv(min_delay=10, max_delay=10, num_queries=3, seed=999)
    env2 = HiddenSwitchboardEnv(min_delay=10, max_delay=10, num_queries=3, seed=999)

    obs1, gt1 = env1.reset()
    obs2, gt2 = env2.reset()

    assert obs1 == obs2
    assert gt1.hidden_mode == gt2.hidden_mode

    for _ in range(12):
        obs1, r1, d1, _ = env1.step(1)
        obs2, r2, d2, _ = env2.step(1)
        assert obs1 == obs2
        assert r1 == r2
        assert d1 == d2
