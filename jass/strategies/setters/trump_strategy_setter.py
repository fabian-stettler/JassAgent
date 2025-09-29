from jass.strategies.interfaces.trump_strategy_game_state import TrumpStrategyGameState
from jass.game.game_state import GameState

class TrumpStrategySetter:
    """
    Strategy setter for trump selection strategies.
    Allows to inject different trump selection strategies.
    """
    
    def __init__(self, strategy: TrumpStrategyGameState):
        """
        Initialize with a trump strategy
        
        Args:
            strategy: The trump strategy to use
        """
        self.strategy = strategy

    def action_trump(self, state: GameState) -> int:
        """
        Determine trump action using the configured strategy
        
        Args:
            state: the game state, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        return self.strategy.action_trump(state)