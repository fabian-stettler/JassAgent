from abc import ABC, abstractmethod
from jass.game.game_observation import GameObservation

class TrumpStrategyGameObservation(ABC):
    @abstractmethod
    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given game observation
        Args:
            obs: the game observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        pass