"""Trump selection wrapper for the RL agent."""
from __future__ import annotations

from jass.game.game_observation import GameObservation
from jass.strategies.implementations.trump_strategy.sixty_eight_points_or_schiebe_observation import (
    SixtyEightPointsOrSchiebeObservation,
)
from jass.strategies.interfaces.trump_strategy_game_observation import (
    TrumpStrategyGameObservation,
)


class RLAgentTrumpStrategy(TrumpStrategyGameObservation):
    """Delegate trump decisions to a configurable base strategy."""

    def __init__(self, base_strategy: TrumpStrategyGameObservation | None = None) -> None:
        self._base = base_strategy or SixtyEightPointsOrSchiebeObservation()

    def action_trump(self, obs: GameObservation) -> int:
        return self._base.action_trump(obs)
