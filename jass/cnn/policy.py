"""CNN policy network for Jass card selection."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

CARD_GRID_SHAPE = (9, 4)
CARD_COUNT = CARD_GRID_SHAPE[0] * CARD_GRID_SHAPE[1]


class JassCNNPolicy(nn.Module):
    """Two-layer ConvNet with max-pooling plus dense head for card logits/value."""
    def __init__(
        self,
        in_channels: int,
        conv_channels: int = 64,
        hidden_dim: int = 256,
        include_value_head: bool = True,
    ) -> None:
        super().__init__()
        self.include_value_head = include_value_head
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        pooled_rows = CARD_GRID_SHAPE[0] // 2
        pooled_cols = CARD_GRID_SHAPE[1] // 2
        flat_dim = conv_channels * pooled_rows * pooled_cols
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.policy_out = nn.Linear(hidden_dim, CARD_COUNT)
        if include_value_head:
            self.value_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        h = self.head(self.trunk(x))
        logits = self.policy_out(h)
        if mask is not None:
            logits = logits.masked_fill(mask < 0.5, -1e9)
        output = {"logits": logits}
        if self.include_value_head:
            output["value"] = self.value_out(h).squeeze(-1)
        return output
