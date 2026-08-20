"""Deterministic training and evaluation harness for Continuity Garden v0."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .environment import HiddenSwitchboardEnv
from .models import CurrentInputMLP, GRUOrganism, HistoryWindowMLP, OracleBeliefAgent


@dataclass
class DatasetBatch:
    symbols: torch.Tensor          # (B, T)
    targets: torch.Tensor          # (B, T) - ignore_index=-100 for non-query steps
    query_masks: torch.Tensor      # (B, T) bool
    cue_steps: List[int]
    query_steps: List[List[int]]


def generate_switchboard_dataset(
    num_episodes: int,
    min_delay: int = 8,
    max_delay: int = 16,
    num_queries: int = 5,
    seed: int = 42
) -> DatasetBatch:
    """Generates a batch of episodes with aligned or variable lengths."""
    env = HiddenSwitchboardEnv(min_delay=min_delay, max_delay=max_delay, num_queries=num_queries, seed=seed)
    
    episodes_symbols = []
    episodes_targets = []
    episodes_masks = []
    cue_steps = []
    query_steps_list = []

    max_len = 0
    raw_episodes = []

    for ep_idx in range(num_episodes):
        obs, gt = env.reset()
        symbols = [obs.symbol]
        targets = [-100] # Cue step has no action target
        query_mask = [False]
        q_steps = []

        step = 0
        done = False
        while not done:
            action = 0 # Dummy during collection
            obs, rew, done, gt = env.step(action)
            symbols.append(obs.symbol)
            
            if gt.current_phase == "query":
                assert gt.target_action is not None
                targets.append(gt.target_action)
                query_mask.append(True)
                q_steps.append(step + 1)
            else:
                targets.append(-100)
                query_mask.append(False)
            step += 1

        cue_steps.append(0)
        query_steps_list.append(q_steps)
        raw_episodes.append((symbols, targets, query_mask))
        if len(symbols) > max_len:
            max_len = len(symbols)

    # Pad to max_len
    padded_symbols = np.zeros((num_episodes, max_len), dtype=np.int64)
    padded_targets = np.full((num_episodes, max_len), -100, dtype=np.int64)
    padded_masks = np.zeros((num_episodes, max_len), dtype=bool)

    for i, (syms, tgts, msk) in enumerate(raw_episodes):
        padded_symbols[i, :len(syms)] = syms
        padded_targets[i, :len(tgts)] = tgts
        padded_masks[i, :len(msk)] = msk

    return DatasetBatch(
        symbols=torch.tensor(padded_symbols, dtype=torch.long),
        targets=torch.tensor(padded_targets, dtype=torch.long),
        query_masks=torch.tensor(padded_masks, dtype=torch.bool),
        cue_steps=cue_steps,
        query_steps=query_steps_list,
    )


def train_gru_organism(
    model: GRUOrganism,
    train_data: DatasetBatch,
    epochs: int = 50,
    lr: float = 0.005,
    batch_size: int = 32,
    seed: int = 42
) -> List[float]:
    """Trains GRUOrganism on sequence prediction loss."""
    torch.manual_seed(seed)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    num_samples = train_data.symbols.size(0)
    loss_history = []

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0
        batches = 0

        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            b_sym = train_data.symbols[idx]
            b_tgt = train_data.targets[idx]

            optimizer.zero_grad()
            logits, _ = model(b_sym)
            # logits: (B, T, A), b_tgt: (B, T)
            loss = criterion(logits.view(-1, 2), b_tgt.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        loss_history.append(epoch_loss / max(1, batches))

    return loss_history


def train_current_mlp(
    model: CurrentInputMLP,
    train_data: DatasetBatch,
    epochs: int = 50,
    lr: float = 0.005,
    batch_size: int = 32,
    seed: int = 42
) -> List[float]:
    """Trains CurrentInputMLP baseline."""
    torch.manual_seed(seed)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Flatten to valid query steps only
    mask = train_data.query_masks.view(-1)
    syms = train_data.symbols.view(-1)[mask]
    tgts = train_data.targets.view(-1)[mask]

    num_samples = syms.size(0)
    loss_history = []

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0
        batches = 0

        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            b_sym = syms[idx]
            b_tgt = tgts[idx]

            optimizer.zero_grad()
            logits = model(b_sym)
            loss = criterion(logits, b_tgt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        loss_history.append(epoch_loss / max(1, batches))

    return loss_history


def evaluate_model(
    model: nn.Module,
    test_data: DatasetBatch,
    is_gru: bool = True,
    apply_state_reset_at: Optional[int] = None
) -> float:
    """Evaluates query accuracy on held-out test data."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        if is_gru:
            assert isinstance(model, GRUOrganism)
            # Step by step evaluation to allow surgical state reset
            num_episodes = test_data.symbols.size(0)
            seq_len = test_data.symbols.size(1)

            for ep_idx in range(num_episodes):
                h = None
                for t in range(seq_len):
                    sym = test_data.symbols[ep_idx, t:t + 1]
                    mask = test_data.query_masks[ep_idx, t].item()
                    target = test_data.targets[ep_idx, t].item()

                    # Apply state reset if requested (e.g. at step 1 after cue)
                    if apply_state_reset_at is not None and t == apply_state_reset_at:
                        h = torch.zeros_like(h) if h is not None else None

                    logits, h = model.step(sym, h)

                    if mask and target != -100:
                        pred = torch.argmax(logits, dim=-1).item()
                        if pred == target:
                            correct += 1
                        total += 1
        else:
            # MLP evaluation
            mask = test_data.query_masks
            syms = test_data.symbols[mask]
            tgts = test_data.targets[mask]
            logits = model(syms)
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == tgts).sum().item()
            total = tgts.numel()

    return correct / max(1, total)
