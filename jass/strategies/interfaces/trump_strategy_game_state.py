from abc import ABC, abstractmethod
from jass.game.game_state import GameState

class TrumpStrategyGameState(ABC):
    @abstractmethod
    def action_trump(self, state: GameState) -> int:
        """
        Determine trump action for the given game state
        Args:
            state: the game state, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        pass