from jass.strategies.interfaces.playing_strategy_game_observation import PlayingStrategyGameObservation
from jass.game.game_observation import GameObservation
from rule_based_agent_util import calculate_score_of_card


class HighestCardFirst(PlayingStrategyGameObservation):
    def action_play_card(self,obs: GameObservation):
        '''returns index of the card position in the one hot encoded array which has the highest value'''
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        
        #check for the highest valid card to play, so that it can get played (return the index of the highest card)
        dict_index_and_value = {}
        for i in range(len(valid_cards)):
            if valid_cards[i] == 1:
                dict_index_and_value[i] = calculate_score_of_card(i, obs.trump)
        
        #pick highest key and then get value of that key (--> index of card to be played)
        max_key = -1
        max_value = -1
        for key in dict_index_and_value.keys():
            if max_value < dict_index_and_value[key]: 
                max_value = dict_index_and_value[key]
                max_key = key

        return max_key


