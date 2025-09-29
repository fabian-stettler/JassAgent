from jass.game import const
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from jass.agents.agent_random_schieber import AgentRandomSchieber
from utils.rule_based_agent_util import *
from game.game_util import *
from strategies.implementations.highest_card_first import HighestCardFirst
from jass.strategies.setters.strategy_setter_game_observation import StrategySetterGameObservation
from jass.strategies.setters.trump_strategy_setter_observation import TrumpStrategySetterObservation
from jass.strategies.implementations.sixty_eight_points_or_schiebe_observation import SixtyEightPointsOrSchiebeObservation

class RuleBasedAgent(Agent):
    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()
        
    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given observation using the 68PointsOrSchiebe strategy
        Args:
            obs: the game observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        trump_strategy = TrumpStrategySetterObservation(SixtyEightPointsOrSchiebeObservation())
        return trump_strategy.action_trump(obs)   
        

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play.

        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """

        strategy = StrategySetterGameObservation(HighestCardFirst())
        return strategy.play_card(obs)
        