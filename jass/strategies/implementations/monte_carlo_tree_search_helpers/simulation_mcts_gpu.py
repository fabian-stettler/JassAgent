import numpy as np
import torch

from jass.game.const import card_values, OBE_ABE
from jass.game.rule_schieber import RuleSchieber


class SimulationMCTSGPU:
    """Heuristic GPU-backed state evaluation for MCTS.

    Instead of running a full random rollout on CPU, we project the current perfect-information GameState
    to a torch tensor and estimate the remaining trick value directly on the selected device (GPU if available).
    This keeps the expensive evaluation phase on the accelerator while remaining deterministic and side-effect free.
    """

    def __init__(self, device: str | torch.device | None = None, noise_std: float = 2.5) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._rule = RuleSchieber()
        self._noise_std = noise_std

        # Precompute tensors on device for fast reuse
        card_values_np = np.asarray(card_values, dtype=np.float32)
        self._card_values = torch.from_numpy(card_values_np).to(self._device)
        # Masks to aggregate players by team
        self._team0_mask = torch.tensor([1, 0, 1, 0], dtype=torch.float32, device=self._device).view(4, 1)
        self._team1_mask = torch.tensor([0, 1, 0, 1], dtype=torch.float32, device=self._device).view(4, 1)

    @property
    def device(self) -> torch.device:
        return self._device

    def run(self, node) -> np.ndarray:
        state = node.game_state
        if state is None:
            return np.zeros(2, dtype=float)

        # Encode hands and current points on the target device
        hands = torch.from_numpy(state.hands.astype(np.float32, copy=True)).to(self._device)
        points = torch.tensor(state.points, dtype=torch.float32, device=self._device)

        trump_index = state.trump
        if trump_index < 0 or trump_index >= self._card_values.shape[0]:
            # default to Obe-Abe scoring if trump not yet known
            trump_index = OBE_ABE

        card_scores = self._card_values[trump_index]
        remaining_scores = hands * card_scores
        team0_remaining = (remaining_scores * self._team0_mask).sum()
        team1_remaining = (remaining_scores * self._team1_mask).sum()

        estimated = torch.stack(
            [points[0] + team0_remaining, points[1] + team1_remaining]
        )

        if self._noise_std > 0:
            noise = torch.randn_like(estimated) * self._noise_std
            estimated = estimated + noise

        return estimated.detach().cpu().numpy()
