"""Observation-based play strategy used by the RL agent."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from jass.game.game_observation import GameObservation
from jass.rl.env import encode_observation
from jass.strategies.interfaces.playing_strategy_game_observation import (
    PlayingStrategyGameObservation,
)

if TYPE_CHECKING:  # pragma: no cover - runtime import would cause a cycle
    from jass.agents.rl_agent import RLAgent


class RLPolicyPlayStrategy(PlayingStrategyGameObservation):
    """Select cards for the RL agent using its policy network."""

    def __init__(self, owner: "RLAgent") -> None:
        self._owner = owner

    def action_play_card(self, obs: GameObservation) -> int:
        features = encode_observation(obs)
        valid_mask = self._owner.rule.get_valid_cards(
            obs.hand, obs.current_trick, obs.nr_cards_in_trick, obs.trump
        ).astype(np.float32)

        self._owner.last_features = features
        self._owner.last_valid_mask = valid_mask

        act_outcome = self._owner.policy.act(
            features,
            valid_mask,
            deterministic=not self._owner.training_mode,
        )
        action = act_outcome["action"]

        if valid_mask[action] == 0:
            choices = np.flatnonzero(valid_mask)
            action = int(np.random.choice(choices))

        if self._owner.training_mode:
            self._owner.buffer.add(
                state=features,
                action=action,
                reward=0.0,
                valid_mask=valid_mask,
                log_prob=act_outcome["log_prob"],
                value=act_outcome["value"],
            )
            self._owner._register_transition()

        return action
