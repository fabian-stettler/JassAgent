from jass.game import const
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent_cheating import AgentCheating

from jass.utils.rule_based_agent_util import *
from jass.game.game_util import *
from jass.game.game_state import GameState
from jass.strategies.minimax_full_game import MinimaxFullGame
from jass.strategies.strategy_setter_game_state import StrategySetter as StrategySetterGameState


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
        cardsOfCurrentPlayer = state.hands[state.player]
        cardsOfCurrentPlayer = convert_one_hot_encoded_cards_to_int_encoded_list(cardsOfCurrentPlayer)

        currentMaxTrumpKind = -1
        chosenTrump = -1
        
        # Check score for all possible kinds and take max
        possible_kinds = [const.DIAMONDS, const.HEARTS, const.SPADES, const.CLUBS]
        for i in range(len(possible_kinds)):
            trump_score = calculate_trump_selection_score(cardsOfCurrentPlayer, possible_kinds[i])
            if trump_score > currentMaxTrumpKind:
                currentMaxTrumpKind = trump_score
                chosenTrump = possible_kinds[i]
        
        # Conservative pushing threshold for long-term planning
        push_threshold = 70
        
        if state.forehand == -1:
            if currentMaxTrumpKind < push_threshold:
                return const.PUSH
            else:
                return chosenTrump
        else:
            return chosenTrump   
        

    def action_play_card(self, game_state: GameState) -> int:
        """
        Determine the card to play using full-game minimax strategy
        """
        strategy = StrategySetterGameState(self.full_game_strategy)
        return strategy.action_play_card(game_state)

