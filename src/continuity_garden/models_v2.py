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

        # Continuous sensors projection: [sensor_a, sensor_b, warning_cue, is_decision_window] -> embed_dim
        self.sensor_proj = nn.Linear(4, embed_dim)

        # Total input dim to GRU: embed_dim * 4 (64 dims)
        self.gru = nn.GRU(embed_dim * 4, hidden_dim, batch_first=True)

        # Policy & Value heads with direct observation pathways
        self.policy_head = nn.Linear(hidden_dim + embed_dim * 2, 4)
        self.value_head = nn.Linear(hidden_dim + embed_dim * 2, 1)

        # Forward predictive dynamics head: predicts next continuous sensors [a, b]
        self.pred_sensors_head = nn.Linear(hidden_dim, 2)

    def forward_features(self, obs: ObservationV2, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converts an ObservationV2 into input feature tensors."""
        sym = torch.tensor([obs.symbol], dtype=torch.long, device=device)
        act_exec = torch.tensor([obs.last_action_executed], dtype=torch.long, device=device)
        act_intend = torch.tensor([obs.last_action_intended], dtype=torch.long, device=device)
        sensors = torch.tensor([[obs.sensor_a, obs.sensor_b, obs.warning_cue, float(obs.is_decision_window)]], dtype=torch.float32, device=device)

        e_sym = self.symbol_embed(sym) # (1, embed_dim)
        e_exec = self.action_exec_embed(act_exec)
        e_intend = self.action_intend_embed(act_intend)
        e_sens = F.relu(self.sensor_proj(sensors))

        feats = torch.cat([e_sym, e_exec, e_intend, e_sens], dim=-1).unsqueeze(1) # (1, 1, 64)
        instant_feats = torch.cat([e_sym, e_sens], dim=-1) # (1, 32)
        return feats, instant_feats, e_sym

    def step(
        self,
        obs: ObservationV2,
        hidden_state: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Processes single step observation.
        Returns: (next_hidden, logits, value, (pred_sensor_a, pred_sensor_b))
        """
        feats, instant_feats, _ = self.forward_features(obs, device=device)

        if hidden_state is None:
            hidden_state = torch.zeros(1, 1, self.hidden_dim, device=device)

        gru_out, next_hidden = self.gru(feats, hidden_state)
        h_flat = next_hidden.squeeze(0) # (1, hidden_dim)

        combined = torch.cat([h_flat, instant_feats], dim=-1) # (1, hidden_dim + 32)
        logits = self.policy_head(combined) # (1, 4)
        value = self.value_head(combined) # (1, 1)

        pred_sensors = self.pred_sensors_head(h_flat)
        pred_sensor_a = pred_sensors[0, 0:1]
        pred_sensor_b = pred_sensors[0, 1:2]

        return next_hidden, logits, value, (pred_sensor_a, pred_sensor_b)

    def snapshot(self, current_hidden: Optional[torch.Tensor]) -> OrganismSnapshotV2:
        return OrganismSnapshotV2(
            hidden_state=current_hidden.clone() if current_hidden is not None else None,
            model_state_dict={k: v.cpu().clone() for k, v in self.state_dict().items()},
        )

    def restore(self, snap: OrganismSnapshotV2, device: torch.device = torch.device("cpu")) -> Optional[torch.Tensor]:
        self.load_state_dict(snap.model_state_dict)
        self.to(device)
        return snap.hidden_state.to(device) if snap.hidden_state is not None else None
