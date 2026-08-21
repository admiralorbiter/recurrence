"""Garden v2 Trainer with Developmental Checkpointing & Optimizer Learnability Control.

Trains DualLocusOrganism using batched Actor-Critic policy gradient with joint
auxiliary forward predictive dynamics loss, emitting frozen model checkpoints at
log-spaced intervals: T in {0, 25, 50, 100, 200, 400, 800, 1600, 3200}.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.continuity_garden.environment_v2 import DualLocusRegulatorEnv, ObservationV2
from src.continuity_garden.models_v2 import DualLocusOrganism


CHECKPOINT_EPISODES = [0, 25, 50, 100, 200, 400, 800, 1600, 3200]


def train_duallocus_organism(
    model: DualLocusOrganism,
    num_episodes: int = 3200,
    lr: float = 0.003,
    gamma: float = 0.95,
    entropy_coef: float = 0.02,
    aux_coef: float = 0.20,
    is_decorative: bool = False,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    checkpoint_callback: Optional[Callable[[int, DualLocusOrganism], None]] = None,
) -> Tuple[List[float], Dict[int, Dict[str, torch.Tensor]]]:
    """
    Trains DualLocusOrganism on Garden v2 environment with developmental checkpointing.
    """
    env = DualLocusRegulatorEnv(
        episode_len=24,
        cost_maintain=0.15,
        reward_target_hit=1.00,
        penalty_wrong_effect=-0.50,
        sensor_noise_std=0.08,
        is_decorative=is_decorative,
        seed=seed,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.to(device)

    episode_returns = []
    saved_checkpoints: Dict[int, Dict[str, torch.Tensor]] = {}

    if 0 in CHECKPOINT_EPISODES:
        saved_checkpoints[0] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if checkpoint_callback:
            checkpoint_callback(0, model)

    for ep_idx in range(1, num_episodes + 1):
        tape = env.generate_deterministic_tape(env.episode_len, rng_seed=seed + ep_idx * 1000)
        obs, gt = env.reset(explicit_tape=tape)

        h = None
        done = False

        log_probs = []
        values = []
        rewards = []
        entropies = []
        aux_losses = []

        while not done:
            h, action_logits, val, (pred_sym, pred_sens) = model.step(obs, h, device=device)
            dist = Categorical(logits=action_logits)
            action = dist.sample()

            next_obs, rew, done, gt = env.step(int(action.item()))

            # Auxiliary loss
            target_sym = torch.tensor([next_obs.symbol], dtype=torch.long, device=device)
            target_sens = torch.tensor([[next_obs.sensor_a, next_obs.sensor_b]], dtype=torch.float32, device=device)

            sym_loss = F.cross_entropy(pred_sym, target_sym)
            sens_loss = F.mse_loss(pred_sens, target_sens)

            log_probs.append(dist.log_prob(action))
            values.append(val.squeeze())
            rewards.append(rew)
            entropies.append(dist.entropy())
            aux_losses.append(sym_loss + sens_loss)

            obs = next_obs

        # Compute discounted returns
        R = 0.0
        returns_discounted = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns_discounted.insert(0, R)

        returns_t = torch.tensor(returns_discounted, dtype=torch.float32, device=device)
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)
        aux_loss_t = torch.stack(aux_losses).mean()

        advantages = returns_t - values_t.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs_t * advantages).mean()
        critic_loss = F.mse_loss(values_t, returns_t)
        entropy_loss = -entropies_t.mean()

        total_loss = actor_loss + 0.5 * critic_loss + entropy_coef * entropy_loss + aux_coef * aux_loss_t

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        ep_ret = sum(rewards)
        episode_returns.append(ep_ret)

        if ep_idx in CHECKPOINT_EPISODES:
            saved_checkpoints[ep_idx] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_callback:
                checkpoint_callback(ep_idx, model)

    return episode_returns, saved_checkpoints


class PrivilegedActorCritic(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.symbol_embed = nn.Embedding(6, 16)
        self.action_embed = nn.Embedding(5, 16)
        self.sensor_proj = nn.Linear(3, 16) # [sensor_a, sensor_b, future_risk]
        self.gru = nn.GRU(16 * 3, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, 4)
        self.critic = nn.Linear(hidden_dim, 1)

    def step(self, obs: ObservationV2, future_risk: float, h: Optional[torch.Tensor] = None):
        sym = torch.tensor([obs.symbol], dtype=torch.long)
        act = torch.tensor([obs.last_action_executed], dtype=torch.long)
        sens = torch.tensor([[obs.sensor_a, obs.sensor_b, future_risk]], dtype=torch.float32)

        e_sym = self.symbol_embed(sym)
        e_act = self.action_embed(act)
        e_sens = F.relu(self.sensor_proj(sens))

        feats = torch.cat([e_sym, e_act, e_sens], dim=-1).unsqueeze(1)
        if h is None:
            h = torch.zeros(1, 1, 64)
        out, h_next = self.gru(feats, h)
        h_flat = h_next.squeeze(0)
        return h_next, self.actor(h_flat), self.critic(h_flat)


def run_optimizer_learnability_control(seed: int = 42, total_episodes: int = 1200) -> Dict[str, Any]:
    """
    Optimizer Learnability Control:
    Validates that the Actor-Critic optimizer easily learns anticipatory regulation
    when supplied with an explicit ground-truth future impairment indicator.
    """
    print("=======================================================")
    print("Executing Optimizer Learnability Control (Privileged Risk)")
    print("=======================================================")

    env = DualLocusRegulatorEnv(seed=seed)
    model = PrivilegedActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    all_returns = []

    for ep_idx in range(total_episodes):
        tape = env.generate_deterministic_tape(env.episode_len, rng_seed=seed + ep_idx * 10)
        obs, gt = env.reset(explicit_tape=tape)
        done = False
        h = None

        log_probs = []
        values = []
        rewards = []
        entropies = []

        while not done:
            future_risk = 1.0 if (gt.shock_pending and gt.pending_shock_magnitude >= 0.50) else 0.0
            h, logits, val = model.step(obs, future_risk, h)
            dist = Categorical(logits=logits)
            act = dist.sample()

            next_obs, rew, done, gt = env.step(int(act.item()))

            log_probs.append(dist.log_prob(act))
            values.append(val.squeeze())
            rewards.append(rew)
            entropies.append(dist.entropy())
            obs = next_obs

        # Discounted returns
        R = 0.0
        disc = []
        for r in reversed(rewards):
            R = r + 0.95 * R
            disc.insert(0, R)

        returns_t = torch.tensor(disc, dtype=torch.float32)
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        advantages = returns_t - values_t.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs_t * advantages).mean()
        critic_loss = F.mse_loss(values_t, returns_t)
        entropy_loss = -entropies_t.mean()

        loss = actor_loss + 0.5 * critic_loss + 0.02 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        all_returns.append(sum(rewards))
        if (ep_idx + 1) % 200 == 0:
            print(f"  Episode {ep_idx+1}/{total_episodes}: Mean Return = {np.mean(all_returns[-100:]):+.2f}")

    # Evaluate greedily on 100 held-out test episodes
    test_returns = []
    test_hits = []
    for test_ep in range(100):
        tape = env.generate_deterministic_tape(env.episode_len, rng_seed=seed + 99999 + test_ep * 10)
        obs, gt = env.reset(explicit_tape=tape)
        done = False
        ep_ret = 0.0
        hits = 0
        h = None
        while not done:
            future_risk = 1.0 if (gt.shock_pending and gt.pending_shock_magnitude >= 0.50) else 0.0
            with torch.no_grad():
                h, logits, _ = model.step(obs, future_risk, h)
                act = int(torch.argmax(logits).item())
            obs, rew, done, gt = env.step(act)
            ep_ret += rew
            if rew > 0.5:
                hits += 1
        test_returns.append(ep_ret)
        test_hits.append(hits)

    final_mean_ret = float(np.mean(test_returns))
    print(f"\n  Final Greedily Evaluated Mean Return (Privileged Risk): {final_mean_ret:+.2f} (+/- {np.std(test_returns):.2f}) | Hits = {np.mean(test_hits):.1f}")
    learnability_pass = bool(final_mean_ret >= 25.0)
    print(f"[Optimizer Learnability Control Verdict]: {'PASS' if learnability_pass else 'FAIL'}\n")

    return {
        "learnability_pass": learnability_pass,
        "final_mean_return": final_mean_ret,
    }


if __name__ == "__main__":
    run_optimizer_learnability_control()
