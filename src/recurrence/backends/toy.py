"""Toy deterministic backend for testing harness determinism and pipeline mechanics."""

import hashlib
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch


class ToyBackend:
    """Mock deterministic backend simulating state transitions z_{t+1} = R(z_t, o_t)."""

    def __init__(self, seed: int = 42, hidden_dim: int = 32):
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal state to deterministic seed initialization."""
        if seed is not None:
            self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # Deterministic initial latent state z_0
        self.z_t = torch.randn(1, self.hidden_dim)

    def step(self, observation_text: str) -> Tuple[str, str, Dict[str, Any]]:
        """Simulate a deterministic step given an observation.
        
        Returns:
            Tuple of (action_text, latent_state_hash, metadata)
        """
        # Hash observation to generate deterministic input tensor of shape (1, self.hidden_dim)
        obs_digest = hashlib.sha256(observation_text.encode("utf-8")).digest()
        obs_raw = np.frombuffer(obs_digest, dtype=np.uint8)
        # Tile or slice to match self.hidden_dim
        obs_buf = np.tile(obs_raw, (self.hidden_dim // len(obs_raw) + 1))[:self.hidden_dim]
        obs_vector = torch.from_numpy(obs_buf.astype(np.float32) / 255.0).unsqueeze(0)

        # Deterministic state update z_{t+1} = tanh(z_t + obs_vector)
        self.z_t = torch.tanh(self.z_t + obs_vector)

        # Compute state digest
        state_bytes = self.z_t.numpy().tobytes()
        state_hash = hashlib.sha256(state_bytes).hexdigest()[:16]

        # Generate action token from top activation
        action_id = int(torch.argmax(self.z_t).item())
        action_text = f"action_{action_id}"

        metadata = {
            "mean_activation": float(torch.mean(self.z_t).item()),
            "std_activation": float(torch.std(self.z_t).item()),
        }

        return action_text, state_hash, metadata
