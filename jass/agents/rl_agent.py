"""Reinforcement Learning Agent integrating with the core `Agent` interface.

Responsibilities:
 - Delegate trump selection to a strategy implementation.
 - Delegate card play decisions to a policy-backed strategy.
 - Maintain replay buffer state and trigger policy updates.
"""
from __future__ import annotations

from pathlib import Path

from jass.agents.agent import Agent
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.rl.policy_network import ActorCriticPolicy
from jass.rl.replay_buffer import TrajectoryBuffer
from jass.strategies.implementations.play_strategies.rl_policy_play_strategy import (
    RLPolicyPlayStrategy,
)
from jass.strategies.implementations.trump_strategy.rl_agent_trump_strategy import (
    RLAgentTrumpStrategy,
)


class RLAgent(Agent):
    def __init__(
        self,
        rule: RuleSchieber | None = None,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        update_every_episodes: int = 1,
        seed: int = 42,
        entropy_coef: float = 1e-3,
        buffer_capacity: int = 1024,
    ):
        self.rule = rule or RuleSchieber()
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_every_episodes = update_every_episodes
        self.episodes_since_update = 0
        self.policy = ActorCriticPolicy(
            input_dim=117,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
        )
        self.buffer = TrajectoryBuffer(capacity=buffer_capacity)
        self.last_features = None
        self.last_valid_mask = None
        self.training_mode = True
        self.team_index = None
        self._episode_transition_count = 0
        self._trump_strategy = RLAgentTrumpStrategy()
        self._play_strategy = RLPolicyPlayStrategy(owner=self)

    # --- trump selection ---
    def action_trump(self, obs: GameObservation) -> int:
        return self._trump_strategy.action_trump(obs)

    # --- play card ---
    def action_play_card(self, obs: GameObservation) -> int:
        return self._play_strategy.action_play_card(obs)

    def finalize_trick(self, trick_reward: float):
        """Add trick reward to the latest stored transition."""
        if self.training_mode and len(self.buffer) > 0:
            self.buffer._storage[-1]["reward"] += trick_reward

    def finalize_episode(self, terminal_reward: float | None = None):
        """Distribute terminal reward and trigger policy updates when due."""
        if terminal_reward is not None and self.training_mode and len(self.buffer) > 0:
            start_idx = max(len(self.buffer._storage) - self._episode_transition_count, 0)
            for entry in self.buffer._storage[start_idx:]:
                entry["reward"] += terminal_reward
        if not self.training_mode:
            self.buffer.clear()
            self._episode_transition_count = 0
            return None
        self.episodes_since_update += 1
        if self.episodes_since_update >= self.update_every_episodes:
            traj = self.buffer.take_new()
            stats = None
            if traj:
                stats = self.policy.update(traj, gamma=self.gamma)
            self.episodes_since_update = 0
            self._episode_transition_count = 0
            return stats
        self._episode_transition_count = 0
        return None

    def save(self, path: str | Path, include_optimizer: bool = True):
        self.policy.save(path, include_optimizer=include_optimizer)

    def load(self, path: str | Path, load_optimizer: bool = True):
        self.policy.load(path, load_optimizer=load_optimizer)

    def set_training(self, training: bool):
        self.training_mode = training

    def _register_transition(self) -> None:
        self._episode_transition_count += 1
