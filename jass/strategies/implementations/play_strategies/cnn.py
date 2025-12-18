
"""Play strategy that drives card selection through the trained CNN policy."""
from __future__ import annotations

from logging import getLogger, log
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from jass.cnn.feature_spec import CNNFeatureSpec, TRUMP_FEATURES, encode_cnn_features
from jass.cnn.policy import JassCNNPolicy
from jass.game.const import DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.strategies.interfaces.playing_strategy_game_observation import (
    PlayingStrategyGameObservation,
)

TRUMP_ORDER = [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE]


def _infer_in_channels(spec: CNNFeatureSpec) -> int:
    extra = tuple(spec.extra_scalar_keys or ())
    return 3 + TRUMP_FEATURES + len(extra)


def _one_hot_trump(trump: int) -> np.ndarray:
    vec = np.zeros(len(TRUMP_ORDER), dtype=np.float32)
    if trump in TRUMP_ORDER:
        vec[TRUMP_ORDER.index(trump)] = 1.0
    return vec


class CNNPlayStrategy(PlayingStrategyGameObservation):
    """Inference-only wrapper around :class:`JassCNNPolicy`."""

    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        self._spec = CNNFeatureSpec()
        self._rule = RuleSchieber()
        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        in_channels = _infer_in_channels(self._spec)
        self._policy = JassCNNPolicy(in_channels=in_channels)

        checkpoint = torch.load(Path(model_path), map_location=self._device)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        self._policy.load_state_dict(state_dict)
        self._policy.eval()
        self.logger = getLogger(__name__)

    def action_play_card(self, obs: GameObservation) -> int:
        """Encode the observation, score all legal cards, and return the best action index."""
        self.logger.info("CNNPlayStrategy: selecting action for observation")
        features = self._encode_observation(obs).to(self._device)
        valid_mask = torch.from_numpy(
            self._rule.get_valid_cards_from_obs(obs).astype(np.float32)
        ).to(self._device)

        if torch.count_nonzero(valid_mask).item() == 0:
            raise ValueError("observation contains no valid moves")

        with torch.no_grad():
            outputs = self._policy(features.unsqueeze(0), valid_mask.unsqueeze(0))
            logits = outputs["logits"].squeeze(0)
        action = int(torch.argmax(logits).item())

        if valid_mask[action] < 0.5:  # numerical guard: fall back to any valid card
            self.logger.warning("CNN selected invalid card, falling back to any valid card")
            candidates = torch.nonzero(valid_mask >= 0.5, as_tuple=False).view(-1)
            action = int(candidates[0].item())

        return action

    def _encode_observation(self, obs: GameObservation) -> torch.Tensor:
        """Map a :class:`GameObservation` into the CNN feature tensor."""
        played_game = np.zeros(36, dtype=np.float32)
        if obs.nr_played_cards > 0:
            for card in obs.tricks.flatten():
                if card >= 0:
                    played_game[int(card)] = 1.0

        played_trick = np.zeros(36, dtype=np.float32)
        if obs.nr_cards_in_trick > 0 and obs.current_trick is not None:
            for i in range(obs.nr_cards_in_trick):
                card = int(obs.current_trick[i])
                if card >= 0:
                    played_trick[card] = 1.0

        feature_dict = {
            self._spec.hand_key: obs.hand.astype(np.float32),
            self._spec.played_game_key: played_game,
            self._spec.played_trick_key: played_trick,
            self._spec.trump_key: _one_hot_trump(int(obs.trump)),
        }
        return encode_cnn_features(feature_dict, self._spec)
        