"""Reinforcement Learning Agent integrating with existing `Agent` interface.

Responsibilities:
 - On its turn: encode observation, sample action via policy
 - Store transition (state, action, reward) – reward is collected post-trick by trainer
 - Provide a finalize_episode(reward) hook for trainer to push terminal reward.

This agent performs simple REINFORCE updates after each episode (game).
For stronger learning, batch multiple games before calling policy.update.
"""
from __future__ import annotations

import numpy as np

from pathlib import Path

from jass.agents.agent import Agent
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.rl.env import encode_observation
from jass.rl.policy_network import ActorCriticPolicy
from jass.rl.replay_buffer import TrajectoryBuffer
from jass.strategies.implementations.trump_strategy.sixty_eight_points_or_schiebe_observation import \
    SixtyEightPointsOrSchiebeObservation


class RLAgent(Agent):
    def __init__(self,
                 rule: RuleSchieber | None = None,
                 hidden_dim: int = 128,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 update_every_episodes: int = 1,
                 seed: int = 42,
                 entropy_coef: float = 1e-3,
                 buffer_capacity: int = 1024):
        self.rule = rule or RuleSchieber()
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_every_episodes = update_every_episodes
        self.episodes_since_update = 0
        self.policy = ActorCriticPolicy(input_dim=117,
                        hidden_dim=hidden_dim,
                        learning_rate=learning_rate,
                        gae_lambda=gae_lambda,
                        entropy_coef=entropy_coef)
        self.buffer = TrajectoryBuffer(capacity=buffer_capacity)
        self.last_features = None
        self.last_valid_mask = None
        self.training_mode = True
        self.team_index = None  # set externally if needed
        self._trump_strategy = SixtyEightPointsOrSchiebeObservation()
        self._episode_transition_count = 0  # number of transitions added in current episode

    # --- trump selection ---
    def action_trump(self, obs: GameObservation) -> int:
        return self._trump_strategy.action_trump(obs)

    # --- play card ---
    def action_play_card(self, obs: GameObservation) -> int:
        features = encode_observation(obs)
        valid_mask = self.rule.get_valid_cards(obs.hand, obs.current_trick, obs.nr_cards_in_trick, obs.trump).astype(np.float32)
        self.last_features = features
        self.last_valid_mask = valid_mask
        act_outcome = self.policy.act(features, valid_mask, deterministic=not self.training_mode)
        action = act_outcome['action']
        # ensure chosen action is valid (masking should already do it but keep fallback)
        if valid_mask[action] == 0:
            choices = np.flatnonzero(valid_mask)
            action = int(np.random.choice(choices))
        if self.training_mode:
            self.buffer.add(state=features,
                            action=action,
                            reward=0.0,
                            valid_mask=valid_mask,
                            log_prob=act_outcome['log_prob'],
                            value=act_outcome['value'])
            self._episode_transition_count += 1
        return action

    def finalize_trick(self, trick_reward: float):
        '''Finalizing trick by adding trick reward to the last stored transition.'''
        if self.training_mode and len(self.buffer) > 0:
            self.buffer._storage[-1]['reward'] += trick_reward

    def finalize_episode(self, terminal_reward: float | None = None):
        '''Finalize episode by optionally distributing terminal reward over all trajectories of last episode and performing policy update.'''
        if terminal_reward is not None and self.training_mode and len(self.buffer) > 0:
            # distribute terminal signal over all transitions from the latest episode
            start_idx = max(len(self.buffer._storage) - self._episode_transition_count, 0)
            for entry in self.buffer._storage[start_idx:]:
                entry['reward'] += terminal_reward
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
        # start counting transitions for the next episode
        self._episode_transition_count = 0
        return None

    def save(self, path: str | Path, include_optimizer: bool = True):
        self.policy.save(path, include_optimizer=include_optimizer)

    def load(self, path: str | Path, load_optimizer: bool = True):
        self.policy.load(path, load_optimizer=load_optimizer)

    def set_training(self, training: bool):
        self.training_mode = training
