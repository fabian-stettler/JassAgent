"""PyTorch actor-critic policy for the Jass RL agent.

The implementation follows a lightweight PPO/A2C style update with:
 - shared MLP trunk for actor & critic
 - categorical action head with masking for illegal cards
 - value baseline + GAE advantages for lower variance updates

It's intentionally kept simple (single hidden layer + PPO clipping) so it can run
inside the existing numpy-heavy environment without extra frameworks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int = 36):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass returning policy logits and value estimates."""
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


@dataclass
class ActorCriticPolicy:
    input_dim: int
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    clip_param: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 1e-3
    gae_lambda: float = 0.95
    #how many times ppo update is performed per call to update() with the same data
    ppo_epochs: int = 4
    #size of amount of trajectories to use per ppo update step (one backpropagation step per mini batch)
    batch_size: int = 64
    max_grad_norm: float = 0.5
    device: str | None = None

    def __post_init__(self):
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.net = ActorCriticNet(self.input_dim, self.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def act(self, obs: np.ndarray, valid_mask: np.ndarray, deterministic: bool = False) -> Dict[str, Any]:
        """Sample (or pick) an action and value estimate for a single observation.

        The valid_mask zeroes out illegal cards by forcing their logits to -inf so the
        categorical distribution never selects them. Returns the chosen action index
        plus the log-probability and critic value for later PPO updates.
        """
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(valid_mask, dtype=torch.float32, device=self.device)

        logits, value = self.net(state)
        masked_logits = logits.masked_fill(mask < 0.5, -1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)
        if deterministic:
            action = torch.argmax(masked_logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return {
            'action': int(action.item()),
            'log_prob': float(log_prob.detach().cpu().item()),
            'value': float(value.detach().cpu().item())
        }

    def update(self, trajectories: List[Dict[str, Any]], gamma: float) -> Dict[str, float] | None:
        """Perform PPO-style policy/value updates from a batch of trajectories."""
        if not trajectories:
            return None

        states = torch.as_tensor(np.stack([t['state'] for t in trajectories]).astype(np.float32),
                                 device=self.device)
        actions = torch.as_tensor([t['action'] for t in trajectories], dtype=torch.long, device=self.device)
        masks = torch.as_tensor(np.stack([t['valid_mask'] for t in trajectories]).astype(np.float32),
                                device=self.device)
        rewards = torch.as_tensor([t['reward'] for t in trajectories], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor([t['log_prob'] for t in trajectories], dtype=torch.float32,
                                        device=self.device)
        values = torch.as_tensor([t['value'] for t in trajectories], dtype=torch.float32, device=self.device)

        returns, advantages = self._compute_gae(rewards, values, gamma)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = min(self.batch_size, states.shape[0])
        stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        step_count = 0

        for _ in range(self.ppo_epochs):
            idx = torch.randperm(states.shape[0], device=self.device)
            for start in range(0, states.shape[0], batch_size):
                batch_idx = idx[start:start + batch_size]
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_masks = masks[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_returns = returns[batch_idx]
                b_adv = advantages[batch_idx]

                logits, values_pred = self.net(b_states)
                masked_logits = logits.masked_fill(b_masks < 0.5, -1e9)
                dist = torch.distributions.Categorical(logits=masked_logits)
                log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred, b_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += entropy.item()
                step_count += 1

        if step_count > 0:
            for key in stats:
                stats[key] /= step_count
        stats['mean_return'] = returns.mean().item()
        stats['advantages_std'] = advantages.std().item()
        stats['steps'] = float(states.shape[0])
        return stats

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, gamma: float):
        """Compute generalized advantage estimates and returns.
        1. values are the critic baseline estimates for each state.
        2. rewards are the observed rewards for each transition.
        
        returns an array of returns and advantages for each trajectory step."""
        returns = torch.zeros_like(rewards, device=self.device)
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * self.gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
        return returns, advantages

    def save(self, path: str | Path, include_optimizer: bool = True) -> None:
        """Persist model weights (and optionally optimizer state) to disk."""
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.net.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'learning_rate': self.learning_rate,
                'clip_param': self.clip_param,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'gae_lambda': self.gae_lambda,
                'ppo_epochs': self.ppo_epochs,
                'batch_size': self.batch_size,
                'max_grad_norm': self.max_grad_norm,
            }
        }
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(checkpoint, ckpt_path)

    def load(self, path: str | Path, load_optimizer: bool = True,
             map_location: torch.device | str | None = None) -> None:
        """Load weights (and optional optimizer state) from a checkpoint."""
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.net.load_state_dict(ckpt['model_state_dict'])
        self.net.to(self.device)
        if load_optimizer and 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
