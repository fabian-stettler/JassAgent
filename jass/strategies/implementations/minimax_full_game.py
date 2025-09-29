from jass.strategies.implementations.minimax_base_strategy import MinimaxBaseStrategy
from jass.strategies.minimax_helper.game_state_evaluator import GameStateEvaluator
from jass.strategies.minimax_helper.trick_manager import TrickManager
from jass.strategies.minimax_helper.alpha_beta_pruner import AlphaBetaPruner
from jass.game.game_state import GameState
import numpy as np
import copy

class MinimaxFullGame(MinimaxBaseStrategy):
    """Minimax strategy that considers multiple tricks with alpha-beta pruning"""
    
    def __init__(self, max_depth=6, use_pruning=True):
        super().__init__()
        self.max_depth = max_depth
        self.use_pruning = use_pruning
        
        # Delegate responsibilities to specialized classes
        self.evaluator = GameStateEvaluator()
        self.trick_manager = TrickManager()
        self.pruner = AlphaBetaPruner()
        
    def _select_best_card(self, game_state: GameState, valid_cards: np.ndarray) -> int:
        """Select best card using full-game minimax with pruning"""
        self.pruner.clear_cache()
        
        best_card = None
        best_score = float('-inf')
        alpha, beta = float('-inf'), float('inf')
        
        for card in valid_cards:
            sim_state = self._create_simulation(game_state, card)
            
            if self.use_pruning:
                score = self._minimax_with_pruning(sim_state, self.max_depth - 1, alpha, beta, False)
                alpha = max(alpha, score)
            else:
                score = self._minimax_basic(sim_state, self.max_depth - 1, False)
            
            if score > best_score:
                best_score = score
                best_card = card
                
        return best_card if best_card is not None else valid_cards[0]
    
    def _minimax_with_pruning(self, state: GameState, depth: int, alpha: float, beta: float, is_max: bool) -> float:
        """Minimax with alpha-beta pruning and memoization"""
        state_key = self.evaluator.create_state_key(state)
        cached_score = self.pruner.get_cached_score(state_key)
        if cached_score is not None:
            return cached_score
        
        if depth == 0 or self.evaluator._is_game_over(state):
            score = self.evaluator.evaluate_position(state)
            self.pruner.cache_score(state_key, score)
            return score
            
        if self.trick_manager.is_trick_complete(state):
            next_state = self.trick_manager.finalize_trick(state)
            score = self._minimax_with_pruning(next_state, depth, alpha, beta, is_max)
            self.pruner.cache_score(state_key, score)
            return score
        
        valid_cards = self._get_valid_cards_for_player(state, state.player)
        if len(valid_cards) == 0:
            score = self.evaluator.evaluate_position(state)
            self.pruner.cache_score(state_key, score)
            return score
        
        if is_max:
            score = self.pruner.maximize_with_pruning(
                state, valid_cards, depth, alpha, beta, 
                self._minimax_with_pruning, self._create_simulation_for_player
            )
        else:
            score = self.pruner.minimize_with_pruning(
                state, valid_cards, depth, alpha, beta, 
                self._minimax_with_pruning, self._create_simulation_for_player
            )
        
        self.pruner.cache_score(state_key, score)
        return score
    
    def _minimax_basic(self, state: GameState, depth: int, is_max: bool) -> float:
        """Basic minimax without pruning"""
        if depth == 0 or self.evaluator._is_game_over(state):
            return self.evaluator.evaluate_position(state)
            
        if self.trick_manager.is_trick_complete(state):
            next_state = self.trick_manager.finalize_trick(state)
            return self._minimax_basic(next_state, depth, is_max)
        
        valid_cards = self._get_valid_cards_for_player(state, state.player)
        
        if is_max:
            return max(self._minimax_basic(self._create_simulation_for_player(state, card), depth - 1, False) 
                      for card in valid_cards)
        else:
            return min(self._minimax_basic(self._create_simulation_for_player(state, card), depth - 1, True) 
                      for card in valid_cards)
    
    def _create_simulation(self, state: GameState, card: int) -> GameState:
        """Create simulation state with card played"""
        return self.trick_manager.simulate_move(state, card, self._simulate_card_play)
    
    def _create_simulation_for_player(self, state: GameState, card: int) -> GameState:
        """Create simulation state for current player's move"""
        sim_state = copy.deepcopy(state)
        self._simulate_card_play(sim_state, card, state.player)
        return sim_state