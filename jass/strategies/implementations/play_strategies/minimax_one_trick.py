from jass.strategies.implementations.play_strategies.minimax_base_strategy import MinimaxBaseStrategy
from jass.game.game_state import GameState
from jass.utils.rule_based_agent_util import calculate_score_of_card
import numpy as np
import copy

class MinimaxOneTrick(MinimaxBaseStrategy):
    """Minimax strategy that evaluates one complete trick (4 cards)"""
    
    def _select_best_card(self, game_state: GameState, valid_cards: np.ndarray) -> int:
        """Select best card using one-trick minimax evaluation"""
        best_card = None
        best_score = float('-inf')
        
        for card in valid_cards:
            sim_state = copy.deepcopy(game_state)
            self._simulate_card_play(sim_state, card, game_state.player)
            
            # Evaluate remaining 3 cards in trick
            score = self._minimax(sim_state, 3, is_maximizing=False)
            
            if score > best_score:
                best_score = score
                best_card = card
        
        return best_card if best_card is not None else valid_cards[0]
    
    def _minimax(self, game_state: GameState, depth: int, is_maximizing: bool) -> float:
        """Minimax algorithm for one trick evaluation"""
        if depth == 0 or game_state.nr_cards_in_trick == 4:
            return self._evaluate_state(game_state)
        
        current_player = game_state.player
        valid_cards = self._get_valid_cards_for_player(game_state, current_player)
        
        if is_maximizing:
            return self._maximize_score(game_state, valid_cards, depth)
        else:
            return self._minimize_score(game_state, valid_cards, depth)
    
    def _maximize_score(self, game_state: GameState, valid_cards: np.ndarray, depth: int) -> float:
        """Maximize score for our team"""
        max_score = float('-inf')
        for card in valid_cards:
            sim_state = copy.deepcopy(game_state)
            self._simulate_card_play(sim_state, card, game_state.player)
            score = self._minimax(sim_state, depth - 1, False)
            max_score = max(max_score, score)
        return max_score
    
    def _minimize_score(self, game_state: GameState, valid_cards: np.ndarray, depth: int) -> float:
        """Minimize score for opponent team"""
        min_score = float('inf')
        for card in valid_cards:
            sim_state = copy.deepcopy(game_state)
            self._simulate_card_play(sim_state, card, game_state.player)
            score = self._minimax(sim_state, depth - 1, True)
            min_score = min(min_score, score)
        return min_score
    
    def _evaluate_state(self, game_state: GameState) -> float:
        """Evaluate current state (complete or partial trick)"""
        if game_state.nr_cards_in_trick == 4:
            return self._evaluate_trick(game_state)
        
        # Heuristic for incomplete tricks
        return self._evaluate_partial_trick(game_state)
    
    def _evaluate_partial_trick(self, game_state: GameState) -> float:
        """Heuristic evaluation for partial tricks"""
        current_trick = game_state.current_trick[:game_state.nr_cards_in_trick]
        
        # Calculate current trick value
        trick_value = sum(calculate_score_of_card(card, game_state.trump) 
                         for card in current_trick if card >= 0)
        
        # Determine who's currently winning
        if game_state.nr_cards_in_trick > 0:
            return self._evaluate_current_winner(game_state, trick_value)
        
        return trick_value * 0.1
    
    def _evaluate_current_winner(self, game_state: GameState, trick_value: float) -> float:
        """Evaluate who's winning the current partial trick"""
        try:
            trick_first_player = (game_state.player - game_state.nr_cards_in_trick) % 4
            current_trick = game_state.current_trick[:game_state.nr_cards_in_trick]
            partial_winner = self._rule.calc_winner(current_trick, trick_first_player, trump=game_state.trump)
            
            winner_bonus = trick_value * 0.3
            return winner_bonus if self._is_maximizing_player(partial_winner) else -winner_bonus
            
        except (IndexError, ValueError):
            return 0

