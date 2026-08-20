"""Model architectures for Continuity Garden v0 (Oracle, Feedforward MLPs, GRU Organism)."""

import copy
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .observation import AgentObservation
from .state import GroundTruthState


class OracleBeliefAgent:
    """Perfect information ceiling agent using ground-truth hidden mode."""

    def __init__(self):
        self._hidden_mode: Optional[int] = None

    def reset(self, ground_truth: GroundTruthState) -> None:
        self._hidden_mode = ground_truth.hidden_mode

    def act(self, ground_truth: GroundTruthState) -> int:
        if ground_truth.current_phase == "query":
            assert ground_truth.query_bit is not None
            assert self._hidden_mode is not None
            return ground_truth.query_bit ^ self._hidden_mode
        return 0


class CurrentInputMLP(nn.Module):
    """Feedforward baseline receiving only current observation symbol (no memory)."""

    def __init__(self, vocab_size: int = 6, embed_dim: int = 32, hidden_dim: int = 64, num_actions: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, symbol_tensor: torch.Tensor) -> torch.Tensor:
        # symbol_tensor: (B,) or (B, T)
        x = self.embed(symbol_tensor)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


class HistoryWindowMLP(nn.Module):
    """Explicit finite-memory baseline receiving sliding window of last K observations."""

    def __init__(self, window_size: int = 4, vocab_size: int = 6, embed_dim: int = 16, hidden_dim: int = 64, num_actions: int = 2):
        super().__init__()
        self.window_size = window_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(window_size * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, window_tensor: torch.Tensor) -> torch.Tensor:
        # window_tensor: (B, K)
        x = self.embed(window_tensor) # (B, K, E)
        x = x.view(x.size(0), -1)      # (B, K*E)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


class GRUOrganism(nn.Module):
    """Small recurrent organism with latent hidden state continuity."""

    def __init__(self, vocab_size: int = 6, embed_dim: int = 32, hidden_dim: int = 64, num_actions: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, symbol_seq: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # symbol_seq: (B, T)
        emb = self.embed(symbol_seq) # (B, T, E)
        out, h_n = self.gru(emb, h_0) # out: (B, T, H), h_n: (1, B, H)
        logits = self.action_head(out) # (B, T, A)
        return logits, h_n

    def step(self, symbol: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Single-step transition for interactive rollouts
        # symbol: (B, 1) or (B,)
        if symbol.dim() == 1:
            symbol = symbol.unsqueeze(1)
        emb = self.embed(symbol)
        out, h_next = self.gru(emb, h)
        logits = self.action_head(out.squeeze(1))
        return logits, h_next

    def snapshot(self) -> Dict[str, Any]:
        """Deepcopy state dict for exact deterministic checkpointing."""
        return copy.deepcopy(self.state_dict())

    def restore(self, snapshot_dict: Dict[str, Any]) -> None:
        """Restore exact model weights."""
        self.load_state_dict(snapshot_dict)
