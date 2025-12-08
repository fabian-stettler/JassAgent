from __future__ import annotations

from typing import Optional

import numpy as np

from jass.strategies.implementations.play_strategies.monte_carlo_tree_search_imperfect_information import (
    MonteCarloTreeSearchImperfectInformation,
)
from jass.strategies.implementations.monte_carlo_tree_search_helpers.monte_carlo_simulation_control_gpu import (
    MonteCarloSimulationControlGPU,
)


class MonteCarloTreeSearchImperfectInformationGPU(
    MonteCarloTreeSearchImperfectInformation
):
    """Observation-based MCTS that evaluates leaf nodes on a GPU-accelerated heuristic."""

    def __init__(
        self,
        simulations_per_sample: int = 200,
        samples: int = 8,
        time_limit_sec: Optional[float] = None,
        exploration_weight: float = 1.4,
        rng: Optional[np.random.Generator] = None,
        device: Optional[str] = None,
        noise_std: float = 2.5,
    ) -> None:
        super().__init__(
            simulations_per_sample=simulations_per_sample,
            samples=samples,
            time_limit_sec=time_limit_sec,
            exploration_weight=exploration_weight,
            rng=rng,
        )
        self._control = MonteCarloSimulationControlGPU(device=device, noise_std=noise_std)

    @property
    def device(self):
        return self._control.device
