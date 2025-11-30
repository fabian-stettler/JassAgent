"""Feature encoder utilities used by the RL stack.

Provides a deterministic mapping from `GameObservation` objects to a fixed
117-dimensional feature vector consumed by `RLAgent`. This keeps the
observation engineering separate from the policy/model code and avoids
duplicating logic across agents or training scripts.
"""

from __future__ import annotations

import numpy as np

from jass.game.const import DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE
from jass.game.game_observation import GameObservation

TRUMP_INDICES = [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE]


def _trump_one_hot(trump: int) -> np.ndarray:
    v = np.zeros(len(TRUMP_INDICES), dtype=np.float32)
    if trump in TRUMP_INDICES:
        v[TRUMP_INDICES.index(trump)] = 1.0
    return v


def encode_observation(obs: GameObservation) -> np.ndarray:
    """Encode a `GameObservation` into a fixed-length feature vector.

    Args:
        obs: current observation for the acting player.
    Returns:
        feature vector shape (117,)
    """
    hand = obs.hand.astype(np.float32)

    # Played cards mask: any card present in completed tricks or current trick.
    played_mask = np.zeros(36, dtype=np.float32)
    if obs.nr_played_cards > 0:
        # Flatten tricks array and mark played (>=0)
        played_cards = obs.tricks.flatten()
        for c in played_cards:
            if c >= 0:
                played_mask[c] = 1.0

    # Current trick mask (cards already in trick but not yet complete)
    current_trick_mask = np.zeros(36, dtype=np.float32)
    if obs.nr_cards_in_trick > 0 and obs.current_trick is not None:
        for i in range(obs.nr_cards_in_trick):
            c = int(obs.current_trick[i])
            if c >= 0:
                current_trick_mask[c] = 1.0

    trump_vec = _trump_one_hot(int(obs.trump))
    nr_tricks_norm = np.array([obs.nr_tricks / 9.0], dtype=np.float32)
    points_team_0 = np.array([obs.points[0] / 157.0], dtype=np.float32)
    points_team_1 = np.array([obs.points[1] / 157.0], dtype=np.float32)

    return np.concatenate([
        hand,
        played_mask,
        current_trick_mask,
        trump_vec,
        nr_tricks_norm,
        points_team_0,
        points_team_1
    ])
