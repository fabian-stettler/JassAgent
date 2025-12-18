"""Lightweight CNN policy head for card-selection decisions.

The module provides two pieces:
- `encode_cnn_features`: helper to stack the README-described inputs into a
  `(channels, 9, 4)` tensor compatible with convolution layers.
- `JassCNNPolicy`: simple actor-critic style network that emits both logits for
  the 36 cards and an optional scalar value estimate for reinforcement learning use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn

from jass.cnn.feature_spec import CNNFeatureSpec

CARD_GRID_SHAPE = (9, 4)
CARD_COUNT = CARD_GRID_SHAPE[0] * CARD_GRID_SHAPE[1]
TRUMP_FEATURES = 6

def _reshape_card_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.size != CARD_COUNT:
        raise ValueError(f"expected {CARD_COUNT} card entries, got {arr.size}")
    return arr.reshape(CARD_GRID_SHAPE)


def _broadcast_scalar(value: float) -> np.ndarray:
    return np.full(CARD_GRID_SHAPE, float(value), dtype=np.float32)


def encode_cnn_features(features: Dict[str, np.ndarray], spec: CNNFeatureSpec | None = None) -> torch.Tensor:
    """Convert the README-described inputs into a stacked CNN tensor.

    Args:
        features: mapping that must contain the keys defined in `CNNFeatureSpec`.
        spec: optional override of the key names and additional scalar channels.

    Returns:
        torch.Tensor with shape `(C, 9, 4)` ready to be fed into `JassCNNPolicy`.
    """
    spec = spec or CNNFeatureSpec()
    channels: List[np.ndarray] = []

    channels.append(_reshape_card_vector(features[spec.hand_key]))
    channels.append(_reshape_card_vector(features[spec.played_game_key]))
    channels.append(_reshape_card_vector(features[spec.played_trick_key]))

    trump_vec = np.asarray(features[spec.trump_key], dtype=np.float32)
    if trump_vec.size != TRUMP_FEATURES:
        raise ValueError(f"expected {TRUMP_FEATURES} trump features, got {trump_vec.size}")
    for value in trump_vec:
        channels.append(_broadcast_scalar(value))

    if spec.extra_scalar_keys:
        for key in spec.extra_scalar_keys:
            channels.append(_broadcast_scalar(float(features[key])))

    stacked = np.stack(channels, axis=0)
    return torch.as_tensor(stacked, dtype=torch.float32)

