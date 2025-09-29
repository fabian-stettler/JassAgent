from jass.strategies.interfaces.trump_strategy_game_observation import TrumpStrategyGameObservation
from jass.game.game_observation import GameObservation

class TrumpStrategySetterObservation:
    """
    Strategy setter for trump selection strategies using GameObservation.
    Allows to inject different trump selection strategies.
    """
    
    def __init__(self, strategy: TrumpStrategyGameObservation):
        """
        Initialize with a trump strategy
        
        Args:
            strategy: The trump strategy to use
        """
        self.strategy = strategy

    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action using the configured strategy
        
        Args:
            obs: the game observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        return self.strategy.action_trump(obs)