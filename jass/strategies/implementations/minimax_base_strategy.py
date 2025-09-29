from jass.strategies.interfaces.playing_strategy_game_state import PlayingStrategyGameState
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber
from jass.utils.rule_based_agent_util import calculate_score_of_card
import numpy as np
import copy
from abc import abstractmethod

class MinimaxBaseStrategy(PlayingStrategyGameState):
    """Base class for Minimax strategies with common functionality"""
    
    def __init__(self):
        self._rule = RuleSchieber()
    
    def action_play_card(self, game_state: GameState):
        """Template method for card selection"""
        valid_cards = self._get_valid_cards(game_state)
        
        if len(valid_cards) == 1:
            return valid_cards[0]
        
        return self._select_best_card(game_state, valid_cards)
    
    @abstractmethod
    def _select_best_card(self, game_state: GameState, valid_cards: np.ndarray) -> int:
        """Select best card using specific minimax strategy"""
        pass
    
    def _get_valid_cards(self, game_state: GameState) -> np.ndarray:
        """Get valid cards for current player"""
        valid_cards = self._rule.get_valid_cards_from_state(game_state)
        return np.flatnonzero(valid_cards)
    
    def _simulate_card_play(self, game_state: GameState, card: int, player: int):
        """Simulate playing a card and update game state"""
        game_state.current_trick[game_state.nr_cards_in_trick] = card
        game_state.hands[player, card] = 0
        game_state.nr_cards_in_trick += 1
        game_state.nr_played_cards += 1
        game_state.player = (player + 1) % 4
    
    def _get_valid_cards_for_player(self, game_state: GameState, player: int) -> np.ndarray:
        """Get valid cards for specific player"""
        player_hand = game_state.hands[player]
        current_trick = game_state.current_trick[:game_state.nr_cards_in_trick]
        
        valid_cards = self._rule.get_valid_cards(
            hand=player_hand,
            current_trick=current_trick,
            move_nr=game_state.nr_cards_in_trick,
            trump=game_state.trump
        )
        return np.flatnonzero(valid_cards)
    
    def _evaluate_trick(self, game_state: GameState) -> float:
        """Evaluate completed trick"""
        trick = game_state.current_trick[:4]
        trick_points = self._rule.calc_points(trick, is_last=False, trump=game_state.trump)
        
        trick_first_player = (game_state.player - 4) % 4
        trick_winner = self._rule.calc_winner(trick, trick_first_player, trump=game_state.trump)
        
        # Check if winner is on our team (players 0&2 vs 1&3)
        our_team = [0, 2]
        return trick_points if trick_winner in our_team else -trick_points
    
    def _is_maximizing_player(self, player: int) -> bool:
        """Check if player is on our team"""
        return player in [0, 2]