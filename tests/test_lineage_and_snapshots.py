"""Tests for Lineage Tracking, Deterministic Cloning, and Snapshot Restoration."""

import torch
from src.recurrence.lineage import LineageTracker
from src.continuity_garden.environment import HiddenSwitchboardEnv
from src.continuity_garden.models import GRUOrganism


def test_lineage_tracker_hash_and_fork():
    tracker = LineageTracker(lineage_id="A0")
    h1 = tracker.record_step(0, obs=1, act=0, rew=0.0)
    h2 = tracker.record_step(1, obs=0, act=1, rew=0.0)
    assert h1 != h2

    # Fork at step 1
    forked = tracker.fork(new_lineage_id="A1", fork_step=1)
    assert forked.lineage_id == "A1"
    assert forked.parent_id == "A0"
    assert forked.fork_step == 1
    assert len(forked.event_log) == 1

    # Record different event on forked lineage
    forked_h = forked.record_step(1, obs=0, act=0, rew=1.0)
    assert forked_h != h2, "Different event on forked lineage must produce diverging event hash"


def test_gru_organism_snapshot_restore_deterministic():
    torch.manual_seed(42)
    model1 = GRUOrganism(vocab_size=6, embed_dim=16, hidden_dim=32, num_actions=2)
    
    # Save snapshot
    snap = model1.snapshot()
    
    # Create fresh model with different seed
    torch.manual_seed(999)
    model2 = GRUOrganism(vocab_size=6, embed_dim=16, hidden_dim=32, num_actions=2)
    
    # Restore model2 from snapshot of model1
    model2.restore(snap)
    
    # Assert bitwise equality of parameters
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2), "Restored parameters must be bitwise identical"

    # Forward pass on identical input must produce identical output
    inp = torch.randint(0, 6, (2, 10))
    out1, h1 = model1(inp)
    out2, h2 = model2(inp)
    assert torch.equal(out1, out2)
    assert torch.equal(h1, h2)


def test_midlife_organism_and_environment_cloning():
    """Validates that a living agent and its environment can be snapshotted mid-trajectory and resumed bitwise identically."""
    env = HiddenSwitchboardEnv(min_delay=12, max_delay=12, num_queries=3, seed=777)
    model = GRUOrganism(vocab_size=6, embed_dim=16, hidden_dim=32, num_actions=2)
    
    # Run 5 steps
    obs, gt = env.reset()
    h = None
    for _ in range(5):
        sym_t = torch.tensor([obs.symbol], dtype=torch.long)
        logits, h = model.step(sym_t, h)
        action = int(torch.argmax(logits, dim=-1).item())
        obs, rew, done, gt = env.step(action)

    # Snapshot both environment and organism mid-trajectory at step 5
    env_snap = env.snapshot()
    org_snap = model.snapshot(h=h, step_idx=5, lineage_hash="test_midlife")

    # Clone environment and organism
    env_clone = HiddenSwitchboardEnv(seed=0)
    env_clone.restore(env_snap)

    model_clone = GRUOrganism(vocab_size=6, embed_dim=16, hidden_dim=32, num_actions=2)
    h_clone = model_clone.restore(org_snap)

    obs_clone = env_clone.sensor_transform.transform(env_clone._ground_truth, last_action=env_clone._last_action)
    assert obs == obs_clone

    # Step both through the remaining future steps
    done_orig = False
    while not done_orig:
        sym_orig = torch.tensor([obs.symbol], dtype=torch.long)
        sym_clone = torch.tensor([obs_clone.symbol], dtype=torch.long)

        logits_orig, h = model.step(sym_orig, h)
        logits_clone, h_clone = model_clone.step(sym_clone, h_clone)

        act_orig = int(torch.argmax(logits_orig, dim=-1).item())
        act_clone = int(torch.argmax(logits_clone, dim=-1).item())

        obs, rew_orig, done_orig, _ = env.step(act_orig)
        obs_clone, rew_clone, done_clone, _ = env_clone.step(act_clone)

        assert act_orig == act_clone
        assert rew_orig == rew_clone
        assert done_orig == done_clone
        assert obs == obs_clone
        assert torch.equal(h, h_clone)
