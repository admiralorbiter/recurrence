"""Garden v2 Organism Architecture (DualLocusOrganism).

Recurrent agent with GRU core, continuous sensor encoders, policy head,
critic value head, and forward predictive dynamics head.
"""

from dataclasses import dataclass
import copy
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.continuity_garden.environment_v2 import ObservationV2


@dataclass
class OrganismSnapshotV2:
    hidden_state: Optional[torch.Tensor]
    model_state_dict: Dict[str, torch.Tensor]


class DualLocusOrganism(nn.Module):
    """Recurrent organism for Garden v2 Dual-Locus Environment."""

    def __init__(
        self,
        symbol_vocab_size: int = 6, # 0: blank, 1: effect0, 2: effect1, 3: goal0, 4: goal1, 5: null
        action_vocab_size: int = 5, # 0: motor0, 1: motor1, 2: maintA, 3: maintB, 4: null
        embed_dim: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.symbol_vocab_size = symbol_vocab_size
        self.action_vocab_size = action_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Embeddings
        self.symbol_embed = nn.Embedding(symbol_vocab_size, embed_dim)
        self.action_exec_embed = nn.Embedding(action_vocab_size, embed_dim)
        self.action_intend_embed = nn.Embedding(action_vocab_size, embed_dim)
        self.warning_embed = nn.Embedding(2, embed_dim)

        # Continuous sensors projection: [sensor_a, sensor_b] -> embed_dim
        self.sensor_proj = nn.Linear(2, embed_dim)

        # Total input dim to GRU: embed_dim * 5
        self.gru = nn.GRU(embed_dim * 5, hidden_dim, batch_first=True)

        # Policy head (4 actions: MOTOR_0, MOTOR_1, MAINTAIN_A, MAINTAIN_B)
        self.policy_head = nn.Linear(hidden_dim, 4)

        # Critic value head
        self.value_head = nn.Linear(hidden_dim, 1)

        # Forward predictive dynamics head: predicts next symbol (6 classes) and continuous sensors [a, b]
        self.pred_symbol_head = nn.Linear(hidden_dim, symbol_vocab_size)
        self.pred_sensors_head = nn.Linear(hidden_dim, 2)

    def forward_features(self, obs: ObservationV2, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Converts an ObservationV2 into an input feature tensor."""
        sym = torch.tensor([obs.symbol], dtype=torch.long, device=device)
        act_exec = torch.tensor([obs.last_action_executed], dtype=torch.long, device=device)
        act_intend = torch.tensor([obs.last_action_intended], dtype=torch.long, device=device)
        warn = torch.tensor([obs.warning_cue], dtype=torch.long, device=device)
        sensors = torch.tensor([[obs.sensor_a, obs.sensor_b]], dtype=torch.float32, device=device)

        e_sym = self.symbol_embed(sym) # (1, embed_dim)
        e_exec = self.action_exec_embed(act_exec)
        e_intend = self.action_intend_embed(act_intend)
        e_warn = self.warning_embed(warn)
        e_sens = F.relu(self.sensor_proj(sensors))

        features = torch.cat([e_sym, e_exec, e_intend, e_warn, e_sens], dim=-1).unsqueeze(1) # (1, 1, embed_dim * 5)
        return features

    def step(
        self,
        obs: ObservationV2,
        h: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step forward execution.
        Returns:
          h_next: (1, 1, hidden_dim)
          action_logits: (1, 4)
          value: (1, 1)
          (pred_symbol_logits, pred_sensors): predicted next observations
        """
        feats = self.forward_features(obs, device=device)
        if h is None:
            h = torch.zeros(1, 1, self.hidden_dim, device=device)

        out, h_next = self.gru(feats, h)
        h_flat = h_next.squeeze(0) # (1, hidden_dim)

        action_logits = self.policy_head(h_flat)
        value = self.value_head(h_flat)
        pred_symbol = self.pred_symbol_head(h_flat)
        pred_sensors = self.pred_sensors_head(h_flat)

        return h_next, action_logits, value, (pred_symbol, pred_sensors)

    def snapshot(self, current_h: Optional[torch.Tensor]) -> OrganismSnapshotV2:
        return OrganismSnapshotV2(
            hidden_state=current_h.clone() if current_h is not None else None,
            model_state_dict={k: v.clone() for k, v in self.state_dict().items()},
        )

    def restore(self, snap: OrganismSnapshotV2) -> Optional[torch.Tensor]:
        self.load_state_dict(snap.model_state_dict)
        return snap.hidden_state.clone() if snap.hidden_state is not None else None
