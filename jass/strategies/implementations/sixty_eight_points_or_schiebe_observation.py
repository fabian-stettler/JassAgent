from jass.strategies.interfaces.trump_strategy_game_observation import TrumpStrategyGameObservation
from jass.game.game_observation import GameObservation
from jass.game import const
from jass.utils.rule_based_agent_util import calculate_trump_selection_score
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list

class SixtyEightPointsOrSchiebeObservation(TrumpStrategyGameObservation):
    """
    Trump selection strategy that chooses the trump with the highest score.
    If the score is below 68 points and it's possible to push (schiebe), it will push.
    Otherwise, it selects the trump with the highest score.
    This version works with GameObservation instead of GameState.
    """
    
    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given observation
        Args:
            obs: the game observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        cardsOfCurrentPlayer = obs.hand
        # convert to int encoded list for easier processing
        cardsOfCurrentPlayer = convert_one_hot_encoded_cards_to_int_encoded_list(cardsOfCurrentPlayer)

        currentMaxTrumpKind = -1
        chosenTrump = -1
        
        # check score for all possible kinds and take max
        possible_kinds = [const.DIAMONDS, const.HEARTS, const.SPADES, const.CLUBS]
        for i in range(len(possible_kinds)):
            trump_score = calculate_trump_selection_score(cardsOfCurrentPlayer, possible_kinds[i])
            if trump_score > currentMaxTrumpKind:
                currentMaxTrumpKind = trump_score
                chosenTrump = possible_kinds[i]
        
        # schiebe if score < 68 if possible, else take max score
        if obs.forehand == -1:
            if currentMaxTrumpKind < 68:
                return const.PUSH
            else:
                return chosenTrump
        else:
            return chosenTrump