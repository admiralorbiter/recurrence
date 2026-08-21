"""Trainer & Gate D0b Optimizer Validity Suite for Continuity Garden v2."""

from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.continuity_garden.environment_v2 import DualLocusRegulatorEnv, GroundTruthStateV2, ObservationV2
from src.continuity_garden.models_v2 import DualLocusOrganism

CHECKPOINT_EPISODES = [0, 25, 50, 100, 200, 400, 800, 1600, 3200]


def evaluate_motor_competence(
    model: nn.Module,
    num_episodes: int = 30,
    seed: int = 9999,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Evaluates target hit rate on shock-free baseline episodes (i=1, x=1)."""
    env = DualLocusRegulatorEnv(is_decorative=False, seed=seed)
    model.eval()
    hits = 0
    total_steps = 0

    for ep in range(num_episodes):
        tape = env.generate_deterministic_tape(env.episode_len, rng_seed=seed + ep * 10)
        tape.shock_steps = []
        tape.precursor_start_steps = []
        tape.decision_window_steps = []
        obs, gt = env.reset(explicit_tape=tape)
        h = None
        done = False
        while not done:
            with torch.no_grad():
                h, logits, _, _ = model.step(obs, h, device=device)
                act = int(torch.argmax(logits).item())
            obs, rew, done, gt = env.step(act)
            if rew > 0.5:
                hits += 1
            total_steps += 1

    return float(hits / total_steps) if total_steps > 0 else 0.0


def train_duallocus_organism(
    model: DualLocusOrganism,
    num_episodes: int = 3200,
    warmup_episodes: int = 50,
    lr: float = 0.003,
    gamma: float = 0.95,
    entropy_coef: float = 0.01,
    aux_coef: float = 0.10,
    is_decorative: bool = False,
    seed: int = 42,
    checkpoint_callback: Optional[Callable[[int, DualLocusOrganism], None]] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[float], Dict[int, Dict[str, torch.Tensor]]]:
    """
    Trains DualLocusOrganism using TD-Actor-Critic with Auxiliary Sensory Prediction,
    sensorimotor grounding warmup, and deterministic Torch RNG matching across paired lineages.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = DualLocusRegulatorEnv(is_decorative=is_decorative, seed=seed)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    saved_checkpoints: Dict[int, Dict[str, torch.Tensor]] = {}
    saved_checkpoints[0] = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    episode_returns = []

    # Sensorimotor Grounding Warmup (pure motor goals before shock exposures)
    for ep_idx in range(warmup_episodes):
        tape = env.generate_deterministic_tape(env.episode_len, rng_seed=seed + ep_idx * 10)
        tape.shock_steps, tape.precursor_start_steps, tape.decision_window_steps = [], [], []
        obs, gt = env.reset(explicit_tape=tape)
        model.train()
        h = None
        done = False
        log_probs, values, rewards, aux_losses = [], [], [], []

        while not done:
            curr_obs = obs
            h, logits, val, (pred_sens_a, pred_sens_b) = model.step(curr_obs, h, device=device)
            dist = torch.distributions.Categorical(logits=logits)
            action = int(dist.sample().item())
            log_prob = dist.log_prob(torch.tensor(action, device=device))

            obs, rew, done, gt = env.step(action)
            t_sens_a = torch.tensor([obs.sensor_a], dtype=torch.float32, device=device)
            t_sens_b = torch.tensor([obs.sensor_b], dtype=torch.float32, device=device)
            aux_loss = F.mse_loss(pred_sens_a, t_sens_a) + F.mse_loss(pred_sens_b, t_sens_b)

            log_probs.append(log_prob)
            values.append(val.squeeze())
            rewards.append(rew)
            aux_losses.append(aux_loss)

        td_errors = []
        for t in range(len(rewards)):
            nv = values[t + 1].detach() if t + 1 < len(rewards) else 0.0
            td_errors.append(rewards[t] + gamma * nv - values[t])

        log_probs_t = torch.stack(log_probs)
        td_errors_t = torch.stack(td_errors)
        aux_loss_t = torch.stack(aux_losses).mean()

        actor_loss = -(log_probs_t * td_errors_t.detach()).mean()
        critic_loss = td_errors_t.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss + aux_coef * aux_loss_t

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Full Developmental Training
    for ep_idx in range(1, num_episodes + 1):
        tape = env.generate_deterministic_tape(env.episode_len, rng_seed=seed + 1000 + ep_idx * 10)
        obs, gt = env.reset(explicit_tape=tape)

        model.train()
        h = None
        done = False

        log_probs = []
        values = []
        rewards = []
        aux_losses = []

        while not done:
            curr_obs = obs
            h, logits, val, (pred_sens_a, pred_sens_b) = model.step(curr_obs, h, device=device)

            dist = torch.distributions.Categorical(logits=logits)
            action = int(dist.sample().item())
            log_prob = dist.log_prob(torch.tensor(action, device=device))

            obs, rew, done, gt = env.step(action)

            # Auxiliary prediction loss
            t_sens_a = torch.tensor([obs.sensor_a], dtype=torch.float32, device=device)
            t_sens_b = torch.tensor([obs.sensor_b], dtype=torch.float32, device=device)
            aux_loss = F.mse_loss(pred_sens_a, t_sens_a) + F.mse_loss(pred_sens_b, t_sens_b)

            log_probs.append(log_prob)
            values.append(val.squeeze())
            rewards.append(rew)
            aux_losses.append(aux_loss)

        # Compute TD advantages
        T_ep = len(rewards)
        td_errors = []
        for t in range(T_ep):
            next_val = values[t + 1].detach() if t + 1 < T_ep else 0.0
            td_err = rewards[t] + gamma * next_val - values[t]
            td_errors.append(td_err)

        log_probs_t = torch.stack(log_probs)
        td_errors_t = torch.stack(td_errors)
        aux_loss_t = torch.stack(aux_losses).mean()

        actor_loss = -(log_probs_t * td_errors_t.detach()).mean()
        critic_loss = td_errors_t.pow(2).mean()

        total_loss = actor_loss + 0.5 * critic_loss + aux_coef * aux_loss_t

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


class PrivilegedRiskAgent(nn.Module):
    """
    Privileged Risk Policy for Gate D0b Optimizer Validation:
    Directly receives the Bayesian Risk Scalar q_t = P(severe | c_1:3) at the decision window.
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.symbol_embed = nn.Embedding(6, 16)
        self.action_embed = nn.Embedding(5, 16)
        self.sensor_proj = nn.Linear(4, 16) # [sensor_a, sensor_b, warning_cue, is_dec_win]
        self.risk_proj = nn.Linear(1, 16)
        self.gru = nn.GRU(16 * 4, hidden_dim, batch_first=True)

        self.actor = nn.Linear(hidden_dim + 32, 4)
        self.critic = nn.Linear(hidden_dim + 32, 1)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def step(self, obs: ObservationV2, risk_q: float, h: Optional[torch.Tensor] = None, device: torch.device = torch.device("cpu")):
        sym = torch.tensor([obs.symbol], dtype=torch.long, device=device)
        act = torch.tensor([obs.last_action_executed], dtype=torch.long, device=device)
        sens = torch.tensor([[obs.sensor_a, obs.sensor_b, obs.warning_cue, float(obs.is_decision_window)]], dtype=torch.float32, device=device)
        r_t = torch.tensor([[risk_q]], dtype=torch.float32, device=device)

        e_sym = self.symbol_embed(sym)
        e_act = self.action_embed(act)
        e_sens = F.relu(self.sensor_proj(sens))
        e_risk = F.relu(self.risk_proj(r_t))

        feats = torch.cat([e_sym, e_act, e_sens, e_risk], dim=-1).unsqueeze(1)
        if h is None:
            h = torch.zeros(1, 1, 64, device=device)
        out, h_next = self.gru(feats, h)
        h_flat = h_next.squeeze(0)
        instant_feats = torch.cat([e_sym, e_risk], dim=-1)
        combined = torch.cat([h_flat, instant_feats], dim=-1)
        return h_next, self.actor(combined), self.critic(combined)


def run_gate_d0b_optimizer_validity(
    seeds: List[int] = [42, 43, 44, 45, 46, 47, 48, 49],
    episodes_per_seed: int = 500,
    warmup_episodes: int = 50,
) -> Dict[str, Any]:
    """
    Gate D0b Optimizer Validity Assay:
    Proves across all 8 seeds that the TD-Actor-Critic optimizer learns anticipatory
    regulation and maintains >75% baseline motor competence when provided with future risk.
    """
    print("=======================================================")
    print("Executing Gate D0b Optimizer Validity Assay (8 Seeds)")
    print("=======================================================")

    seed_returns = []
    seed_competences = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = DualLocusRegulatorEnv(precursor_noise_std=0.35, seed=seed)
        agent = PrivilegedRiskAgent()
        optimizer = optim.Adam(agent.parameters(), lr=0.003)

        # Warmup on pure motor baseline
        for ep in range(warmup_episodes):
            tape = env.generate_deterministic_tape(env.episode_len, rng_seed=seed + ep)
            tape.shock_steps, tape.precursor_start_steps, tape.decision_window_steps = [], [], []
            obs, gt = env.reset(explicit_tape=tape)
            h = None
            done = False
            log_probs, values, rewards = [], [], []
            while not done:
                h, logits, val = agent.step(obs, 0.0, h)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                obs, r, done, gt = env.step(int(a.item()))
                log_probs.append(dist.log_prob(a))
                values.append(val.squeeze())
                rewards.append(r)
            td_errs = []
            for t in range(len(rewards)):
                nv = values[t + 1].detach() if t + 1 < len(rewards) else 0.0
                td_errs.append(rewards[t] + 0.95 * nv - values[t])
            loss = -(torch.stack(log_probs) * torch.stack(td_errs).detach()).mean() + 0.5 * torch.stack(td_errs).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

        # Full environment training
        for ep in range(1, episodes_per_seed + 1):
            tape = env.generate_deterministic_tape(env.episode_len, rng_seed=seed + 100 + ep)
            obs, gt = env.reset(explicit_tape=tape)
            h = None
            done = False
            log_probs, values, rewards = [], [], []

            while not done:
                risk_q = gt.bayesian_risk_q if obs.is_decision_window == 1 else 0.0
                h, logits, val = agent.step(obs, risk_q, h)
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())
                log_prob = dist.log_prob(torch.tensor(action))

                obs, rew, done, gt = env.step(action)
                log_probs.append(log_prob)
                values.append(val.squeeze())
                rewards.append(rew)

            td_errors = []
            for t in range(len(rewards)):
                next_val = values[t + 1].detach() if t + 1 < len(rewards) else 0.0
                td_errors.append(rewards[t] + 0.95 * next_val - values[t])

            log_probs_t = torch.stack(log_probs)
            td_errors_t = torch.stack(td_errors)
            loss = -(log_probs_t * td_errors_t.detach()).mean() + 0.5 * td_errors_t.pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

        # Evaluate greedy return across 50 held-out episodes
        agent.eval()
        eval_returns = []
        eval_hits = []
        for ep in range(50):
            tape = env.generate_deterministic_tape(env.episode_len, rng_seed=seed + 50000 + ep * 10)
            obs, gt = env.reset(explicit_tape=tape)
            h = None
            done = False
            ep_r = 0.0
            hits = 0
            while not done:
                risk_q = gt.bayesian_risk_q if obs.is_decision_window == 1 else 0.0
                with torch.no_grad():
                    h, logits, _ = agent.step(obs, risk_q, h)
                    act = int(torch.argmax(logits).item())
                obs, rew, done, gt = env.step(act)
                ep_r += rew
                if rew > 0.5:
                    hits += 1
            eval_returns.append(ep_r)
            eval_hits.append(hits / 24.0)

        mean_ret = float(np.mean(eval_returns))
        mean_comp = float(np.mean(eval_hits))
        seed_returns.append(mean_ret)
        seed_competences.append(mean_comp)

        print(f"  Seed {seed:<4}: Mean Return = {mean_ret:+.2f} | Motor Hit Rate = {mean_comp*100:.1f}%")

    all_passed = bool(np.mean(seed_returns) >= 28.0 and np.mean(seed_competences) >= 0.75 and np.min(seed_competences) >= 0.70)
    print("\n=======================================================")
    print(f"Gate D0b Aggregate Result across 8 Seeds:")
    print(f"  Mean Return:      {np.mean(seed_returns):+.2f} (+/- {np.std(seed_returns):.2f}) (Target: >= +28.0)")
    print(f"  Mean Competence:  {np.mean(seed_competences)*100:.1f}% (Target: >= 75%)")
    print(f"[Gate D0b Verdict]: {'PASS' if all_passed else 'FAIL'}")
    print("=======================================================\n")

    return {
        "gate_d0b_pass": all_passed,
        "seed_returns": seed_returns,
        "seed_competences": seed_competences,
    }


if __name__ == "__main__":
    run_gate_d0b_optimizer_validity(seeds=[42, 43, 44, 45, 46, 47, 48, 49])
