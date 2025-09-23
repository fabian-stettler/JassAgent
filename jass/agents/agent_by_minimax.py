from jass.game import const
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent_cheating import AgentCheating

from jass.utils.rule_based_agent_util import *
from jass.game.game_util import *
from jass.game.game_state import GameState
from jass.strategies.minimax_one_trick import MinimaxOneTrick
from jass.strategies.strategy_setter_game_state import StrategySetter as StrategySetterGameState


class AgentByMinimax(AgentCheating):
    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()
        
    def action_trump(self, state: GameState) -> int:
        """
        Determine trump action for the given observation
        Args:
            state: the game state, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        cardsOfCurrentPlayer = state.hands[state.player]
        # convert to int encoded list for easier processing
        cardsOfCurrentPlayer = convert_one_hot_encoded_cards_to_int_encoded_list(cardsOfCurrentPlayer)

        currentMaxTrumpKind = -1
        chosenTrump = -1
        
        #check score for all possible kinds and take max
        possible_kinds = [const.DIAMONDS, const.HEARTS, const.SPADES, const.CLUBS]
        for i in range(len(possible_kinds)):
            trump_score = calculate_trump_selection_score(cardsOfCurrentPlayer, possible_kinds[i])
            if trump_score > currentMaxTrumpKind:
                currentMaxTrumpKind = trump_score
                chosenTrump = possible_kinds[i]
        
        # schiebe if score < 68 if possible, else take max score
        #print("Max trump score: ", currentMaxTrumpKind)
        if state.forehand == -1:
            if currentMaxTrumpKind < 68:
                return const.PUSH
            else:
                return chosenTrump
        else:
            return chosenTrump   
        

    def action_play_card(self, game_state: GameState) -> int:
        """
        Determine the card to play.

        Args:
            game_state: the game state

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """

        strategy = StrategySetterGameState(MinimaxOneTrick(max_depth=4))
        return strategy.action_play_card(game_state)