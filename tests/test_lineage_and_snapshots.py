"""Tests for Lineage Tracking, Deterministic Cloning, and Snapshot Restoration."""

import torch
from src.recurrence.lineage import LineageTracker
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
