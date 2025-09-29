from jass.game import const
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent_cheating import AgentCheating

from jass.utils.rule_based_agent_util import *
from jass.game.game_util import *
from jass.game.game_state import GameState
from jass.strategies.implementations.minimax_full_game import MinimaxFullGame
from jass.strategies.setters.strategy_setter_game_state import StrategySetter as StrategySetterGameState
from jass.strategies.setters.trump_strategy_setter import TrumpStrategySetter
from jass.strategies.implementations.seventy_points_or_schiebe import SeventyPointsOrSchiebe


class AgentByMinimaxFullGame(AgentCheating):
    """
    Advanced Minimax agent that uses full-game planning with configurable depth
    """
    
    def __init__(self, max_depth=6):
        """
        Args:
            max_depth: Search depth for minimax algorithm (default: 6)
        """
        super().__init__()
        self._rule = RuleSchieber()
        self.max_depth = max_depth
        
        # Initialize full-game strategy
        self.full_game_strategy = MinimaxFullGame(max_depth=max_depth)
        
    def action_trump(self, state: GameState) -> int:
        """
        Determine trump action using enhanced analysis for full-game strategy
        """
        trump_strategy = TrumpStrategySetter(SeventyPointsOrSchiebe())
        return trump_strategy.action_trump(state)   
        

    def action_play_card(self, game_state: GameState) -> int:
        """
        Determine the card to play using full-game minimax strategy
        """
        strategy = StrategySetterGameState(self.full_game_strategy)
        return strategy.action_play_card(game_state)

