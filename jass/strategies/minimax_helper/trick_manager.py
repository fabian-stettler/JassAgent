from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber
import numpy as np
import copy

class TrickManager:
    """Handles trick completion and game state transitions"""
    
    def __init__(self):
        self._rule = RuleSchieber()
    
    def finalize_trick(self, state: GameState) -> GameState:
        """Finalize completed trick: calculate points, determine winner, prepare next trick"""
        next_state = copy.deepcopy(state)
        trick = next_state.current_trick[:4]
        
        # Calculate points and winner
        is_last = next_state.nr_played_cards >= 36
        points = self._rule.calc_points(trick, is_last=is_last, trump=next_state.trump)
        
        first_player = (next_state.player - 4) % 4
        winner = self._rule.calc_winner(trick, first_player, trump=next_state.trump)
        
        # Update game state
        next_state.points[winner % 2] += points
        next_state.current_trick = np.full(4, -1, dtype=np.int32)
        next_state.nr_cards_in_trick = 0
        next_state.player = winner
        next_state.nr_tricks += 1
        
        return next_state
    
    def is_trick_complete(self, state: GameState) -> bool:
        """Check if current trick is complete"""
        return state.nr_cards_in_trick == 4
    
    def simulate_move(self, state: GameState, card: int, simulate_card_play_func) -> GameState:
        """Create simulation state with card played"""
        sim_state = copy.deepcopy(state)
        simulate_card_play_func(sim_state, card, state.player)
        return sim_state