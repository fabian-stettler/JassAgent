from jass.game.game_state import GameState
from jass.utils.rule_based_agent_util import calculate_score_of_card

class GameStateEvaluator:
    """Handles all game state evaluation and heuristic calculations"""
    
    def __init__(self):
        pass
    
    def evaluate_position(self, state: GameState) -> float:
        """Main evaluation method for any game position"""
        if self._is_game_over(state):
            return self._evaluate_final_position(state)
        
        # Heuristic: point difference + hand strength
        point_diff = state.points[0] - state.points[1]  # Team 0 vs Team 1
        hand_value = self._calculate_hand_strength(state)
        return point_diff + hand_value * 0.1
    
    def _is_game_over(self, state: GameState) -> bool:
        """Check if game is complete"""
        return state.nr_played_cards >= 36
    
    def _evaluate_final_position(self, state: GameState) -> float:
        """Evaluate final game position"""
        return state.points[0] - state.points[1]
    
    def _calculate_hand_strength(self, state: GameState) -> float:
        """Calculate remaining hand strength for our team"""
        strength = 0
        for player in [0, 2]:  # Our team
            hand = state.hands[player]
            for card in range(36):
                if hand[card] == 1:
                    try:
                        trump = max(0, min(3, state.trump))
                        strength += calculate_score_of_card(card, trump)
                    except (IndexError, ValueError):
                        strength += 1
        return strength

    def create_state_key(self, state: GameState) -> tuple:
        """Create a hashable key for state caching"""
        return (
            tuple(state.hands.flatten()),
            state.trump,
            state.nr_played_cards,
            tuple(state.points),
            state.player
        )