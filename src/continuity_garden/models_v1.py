"""Neural model architectures for Continuity Garden v1 Controllability Organisms."""

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.continuity_garden.environment_v1 import ObservationV1


@dataclass
class OrganismSnapshotV1:
    step_idx: int
    recurrent_state: Optional[torch.Tensor]
    model_state_dict: Dict[str, Any]
    torch_rng_state: torch.Tensor
    numpy_rng_state: Tuple[Any, ...]
    python_rng_state: Tuple[Any, ...]


class ControllableOrganism(nn.Module):
    """Recurrent Organism with Efference Copy, Forward Dynamics, and Instrumental Exploitation."""

    def __init__(
        self,
        symbol_vocab_size: int = 6,
        action_vocab_size: int = 4,
        embed_dim: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.symbol_embed = nn.Embedding(symbol_vocab_size, embed_dim)
        self.exec_act_embed = nn.Embedding(action_vocab_size, embed_dim)
        self.intend_act_embed = nn.Embedding(action_vocab_size, embed_dim)

        input_dim = embed_dim * 3
        self.gru = nn.GRUCell(input_dim, hidden_dim)

        # 1. Motor Action Head (during exploration): selects a_t in {0, 1}
        self.motor_head = nn.Linear(hidden_dim, 2)

        # 2. Forward Dynamics Head: predicts next effect P(E in {0, 1} | h_t, a_t)
        self.forward_head = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        # 3. Exploitation Policy Head: selects {0: TRY_0, 1: TRY_1, 2: ABSTAIN}
        self.exploit_head = nn.Linear(hidden_dim, 3)

        # 4. State-Value Head: V(h_t)
        self.value_head = nn.Linear(hidden_dim, 1)

    def encode_inputs(self, obs: ObservationV1, device: torch.device) -> torch.Tensor:
        sym_t = torch.tensor([obs.symbol], dtype=torch.long, device=device)
        exec_t = torch.tensor([obs.action_executed], dtype=torch.long, device=device)
        int_t = torch.tensor([obs.action_intended], dtype=torch.long, device=device)

        e_sym = self.symbol_embed(sym_t)
        e_exec = self.exec_act_embed(exec_t)
        e_int = self.intend_act_embed(int_t)

        return torch.cat([e_sym, e_exec, e_int], dim=-1)

    def step(
        self,
        obs: ObservationV1,
        h: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Advances the organism by one step.
        Returns:
          h_new: new recurrent state (1, hidden_dim)
          motor_logits: logits over exploration actions {0, 1}
          exploit_logits: logits over exploitation choices {TRY_0, TRY_1, ABSTAIN}
          value: estimated state value V(h)
        """
        x = self.encode_inputs(obs, device)
        if h is None:
            h = torch.zeros(1, self.gru.hidden_size, device=device)

        h_new = self.gru(x, h)
        motor_logits = self.motor_head(h_new)
        exploit_logits = self.exploit_head(h_new)
        value = self.value_head(h_new)

        return h_new, motor_logits, exploit_logits, value

    def predict_forward_effect(
        self,
        h: torch.Tensor,
        action: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Predicts next effect logits P(E in {0, 1} | h, action)."""
        act_t = torch.tensor([action], dtype=torch.long, device=device)
        e_act = self.exec_act_embed(act_t)
        h_act = torch.cat([h, e_act], dim=-1)
        return self.forward_head(h_act)

    def snapshot(self, h: Optional[torch.Tensor], step_idx: int) -> OrganismSnapshotV1:
        import numpy as np
        import random
        return OrganismSnapshotV1(
            step_idx=step_idx,
            recurrent_state=h.clone() if h is not None else None,
            model_state_dict=copy.deepcopy(self.state_dict()),
            torch_rng_state=torch.get_rng_state(),
            numpy_rng_state=np.random.get_state(),
            python_rng_state=random.getstate(),
        )

    def restore(self, snap: OrganismSnapshotV1) -> Optional[torch.Tensor]:
        import numpy as np
        import random
        self.load_state_dict(snap.model_state_dict)
        torch.set_rng_state(snap.torch_rng_state)
        np.random.set_state(snap.numpy_rng_state)
        random.setstate(snap.python_rng_state)
        return snap.recurrent_state.clone() if snap.recurrent_state is not None else None
