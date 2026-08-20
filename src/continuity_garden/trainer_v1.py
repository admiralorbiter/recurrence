"""Reinforcement and Forward Dynamics Trainer for Continuity Garden v1."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.continuity_garden.environment_v1 import ControllabilityArenaEnv, ObservationV1
from src.continuity_garden.models_v1 import ControllableOrganism


def train_controllable_organism(
    model: ControllableOrganism,
    num_episodes: int = 1500,
    lr: float = 0.003,
    gamma: float = 0.99,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[float], int]:
    """
    Trains the controllable organism via joint forward prediction and REINFORCE/Actor-Critic return learning.
    NO oracle policy supervision is used (no 'target = ABSTAIN' labels).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    env = ControllabilityArenaEnv(seed=seed)

    episode_returns = []
    total_optimizer_steps = 0

    model.train()
    for ep in range(num_episodes):
        obs, gt = env.reset()
        h = None
        done = False

        forward_losses = []
        log_prob_exploit = None
        state_value_exploit = None

        while not done:
            h, motor_logits, exploit_logits, value = model.step(obs, h, device=device)

            if gt.current_phase == "exploration":
                # Exploration action selection
                dist = Categorical(logits=motor_logits)
                action = int(dist.sample().item())

                # Predict forward effect of chosen action
                pred_effect_logits = model.predict_forward_effect(h, action, device=device)

                # Environment step
                next_obs, rew, done, gt = env.step(action)

                # Target effect
                if gt.last_effect is not None:
                    target_e = torch.tensor([gt.last_effect], dtype=torch.long, device=device)
                    f_loss = nn.functional.cross_entropy(pred_effect_logits, target_e)
                    forward_losses.append(f_loss)

                obs = next_obs

            elif gt.current_phase == "exploitation":
                # Exploitation decision step
                dist = Categorical(logits=exploit_logits)
                exploit_action = dist.sample()
                log_prob_exploit = dist.log_prob(exploit_action)
                state_value_exploit = value

                # Step environment with exploitation action
                next_obs, reward, done, gt = env.step(int(exploit_action.item()))
                episode_returns.append(reward)
                obs = next_obs

        # Compute combined loss
        loss = torch.tensor(0.0, device=device)
        if forward_losses:
            loss = loss + torch.stack(forward_losses).mean()

        if log_prob_exploit is not None and state_value_exploit is not None:
            r_tensor = torch.tensor([[reward]], dtype=torch.float32, device=device)
            advantage = r_tensor - state_value_exploit.detach()
            policy_loss = -log_prob_exploit * advantage.squeeze()
            value_loss = nn.functional.mse_loss(state_value_exploit, r_tensor)
            entropy_loss = -0.02 * dist.entropy()
            loss = loss + policy_loss + 0.5 * value_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_optimizer_steps += 1

    return episode_returns, total_optimizer_steps
