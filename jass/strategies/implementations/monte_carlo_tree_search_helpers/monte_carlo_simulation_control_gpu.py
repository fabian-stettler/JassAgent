from __future__ import annotations

from jass.strategies.implementations.monte_carlo_tree_search_helpers.monte_carlo_simulation_control import (
    MonteCarloSimulationControl,
)
from jass.strategies.implementations.monte_carlo_tree_search_helpers.simulation_mcts_gpu import (
    SimulationMCTSGPU,
)


class MonteCarloSimulationControlGPU(MonteCarloSimulationControl):
    """Drop-in replacement that keeps the simulation/value step on a GPU-accelerated heuristic."""

    def __init__(self, device: str | None = None, noise_std: float = 2.5) -> None:
        super().__init__()
        self.simulation_mcts = SimulationMCTSGPU(device=device, noise_std=noise_std)

    @property
    def device(self):
        return self.simulation_mcts.device
