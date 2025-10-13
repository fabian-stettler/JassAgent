from jass.game import const
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent_cheating import AgentCheating

from jass.utils.rule_based_agent_util import *
from jass.game.game_util import *
from jass.game.game_state import GameState
from jass.strategies.implementations.play_strategies.minimax_one_trick import MinimaxOneTrick
from jass.strategies.setters.strategy_setter_game_state import StrategySetter as StrategySetterGameState
from jass.strategies.setters.trump_strategy_setter import TrumpStrategySetter
from jass.strategies.implementations.trump_strategy.sixty_eight_points_or_schiebe import SixtyEightPointsOrSchiebe


class AgentByMinimax(AgentCheating):
    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()
        
    def action_trump(self, state: GameState) -> int:
        """
        Determine trump action for the given observation using the 68PointsOrSchiebe strategy
        Args:
            state: the game state, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        trump_strategy = TrumpStrategySetter(SixtyEightPointsOrSchiebe())
        return trump_strategy.action_trump(state)   
        

    def action_play_card(self, game_state: GameState) -> int:
        """
        Determine the card to play with dept of one_trick

        Args:
            game_state: the game state

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """

        strategy = StrategySetterGameState(MinimaxOneTrick())
        return strategy.action_play_card(game_state)