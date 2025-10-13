from jass.game.game_state import GameState
import numpy as np

class AlphaBetaPruner:
    """Handles alpha-beta pruning logic and transposition table management"""
    
    def __init__(self):
        self.transposition_table = {}
    
    def clear_cache(self):
        """Clear the transposition table"""
        self.transposition_table.clear()
    
    def get_cached_score(self, state_key: tuple) -> float:
        """Get cached score if available"""
        return self.transposition_table.get(state_key)
    
    def cache_score(self, state_key: tuple, score: float):
        """Cache score for future lookups"""
        self.transposition_table[state_key] = score
    
    def should_prune(self, alpha: float, beta: float) -> bool:
        """Check if we should prune this branch"""
        return beta <= alpha
    
    def maximize_with_pruning(self, state: GameState, valid_cards: np.ndarray, 
                             depth: int, alpha: float, beta: float, 
                             minimax_func, simulate_func) -> float:
        """Execute maximizing step with alpha-beta pruning"""
        max_score = float('-inf')
        for card in valid_cards:
            sim_state = simulate_func(state, card)
            score = minimax_func(sim_state, depth - 1, alpha, beta, False)
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if self.should_prune(alpha, beta):
                break
        return max_score
    
    def minimize_with_pruning(self, state: GameState, valid_cards: np.ndarray, 
                             depth: int, alpha: float, beta: float, 
                             minimax_func, simulate_func) -> float:
        """Execute minimizing step with alpha-beta pruning"""
        min_score = float('inf')
        for card in valid_cards:
            sim_state = simulate_func(state, card)
            score = minimax_func(sim_state, depth - 1, alpha, beta, True)
            min_score = min(min_score, score)
            beta = min(beta, score)
            if self.should_prune(alpha, beta):
                break
        return min_score
