from jass.strategies.playing_strategy_game_state import PlayingStrategyGameObservation
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber
from jass.utils.rule_based_agent_util import calculate_score_of_card
import numpy as np
import copy

class MinimaxFullGame(PlayingStrategyGameObservation):
    '''
    Minimax strategy that considers the entire game (all 9 tricks) with pruning optimizations.
    Due to computational complexity, uses selective depth search and heuristic evaluations.
    '''
    
    def __init__(self, max_depth=6, selective_pruning=True):
        """
        Args:
            max_depth: Maximum search depth (cards to look ahead)
                      6-8 = good balance between performance and accuracy
                      Higher values become computationally expensive
            selective_pruning: Enable alpha-beta pruning and other optimizations
        """
        self._rule = RuleSchieber()
        self.max_depth = max_depth
        self.selective_pruning = selective_pruning
        self.transposition_table = {}  # Cache for repeated positions
        
    def action_play_card(self, game_state: GameState):
        """
        Choose the best card using full-game minimax with optimizations.
        """
        valid_cards = self._rule.get_valid_cards_from_state(game_state)
        valid_card_indices = np.flatnonzero(valid_cards)
        
        # If only one valid card, play it
        if len(valid_card_indices) == 1:
            return valid_card_indices[0]
        
        # Clear transposition table for new decision
        self.transposition_table.clear()
        
        best_card = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Try each valid card as our move
        for card in valid_card_indices:
            # Create simulation state with our card played
            sim_state = copy.deepcopy(game_state)
            self._simulate_card_play(sim_state, card, game_state.player)
            
            # Start minimax with alpha-beta pruning
            if self.selective_pruning:
                score = self._minimax_alpha_beta(sim_state, self.max_depth - 1, alpha, beta, False)
                alpha = max(alpha, score)
            else:
                score = self._minimax(sim_state, self.max_depth - 1, False)
            
            if score > best_score:
                best_score = score
                best_card = card
                
        return best_card if best_card is not None else valid_card_indices[0]
    
    def _minimax_alpha_beta(self, game_state: GameState, depth: int, alpha: float, beta: float, is_maximizing: bool):
        """
        Minimax with alpha-beta pruning for better performance
        """
        # Check transposition table
        state_key = self._get_state_key(game_state)
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]
        
        # Base cases
        if depth == 0:
            score = self._evaluate_game_state(game_state)
            self.transposition_table[state_key] = score
            return score
            
        if self._is_game_over(game_state):
            score = self._evaluate_final_game_state(game_state)
            self.transposition_table[state_key] = score
            return score
        
        # 4 Karten gespielt -> Stich beenden 
        if game_state.nr_cards_in_trick == 4:
            next_state = copy.deepcopy(game_state)
            self._complete_trick(next_state)
            score = self._minimax_alpha_beta(next_state, depth, alpha, beta, is_maximizing)
            self.transposition_table[state_key] = score
            return score
        
        current_player = game_state.player
        valid_cards = self._get_valid_cards_for_simulation(game_state, current_player)
        
        # Prune if no valid moves
        if len(valid_cards) == 0:
            score = self._evaluate_game_state(game_state)
            self.transposition_table[state_key] = score
            return score
        
        # Order moves for better pruning (play high-value cards first)
        if self.selective_pruning:
            valid_cards = self._order_moves(valid_cards, game_state)
        
        if is_maximizing:
            max_score = float('-inf')
            for card in valid_cards:
                sim_state = copy.deepcopy(game_state)
                self._simulate_card_play(sim_state, card, current_player)
                
                score = self._minimax_alpha_beta(sim_state, depth - 1, alpha, beta, False)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    break
                    
            self.transposition_table[state_key] = max_score
            return max_score
        else:
            min_score = float('inf')
            for card in valid_cards:
                sim_state = copy.deepcopy(game_state)
                self._simulate_card_play(sim_state, card, current_player)
                
                score = self._minimax_alpha_beta(sim_state, depth - 1, alpha, beta, True)
                min_score = min(min_score, score)
                beta = min(beta, score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    break
                    
            self.transposition_table[state_key] = min_score
            return min_score
    
    def _minimax(self, game_state: GameState, depth: int, is_maximizing: bool):
        """
        Basic minimax without pruning (for comparison)
        """
        # Base cases
        if depth == 0:
            return self._evaluate_game_state(game_state)
            
        if self._is_game_over(game_state):
            return self._evaluate_final_game_state(game_state)
        
        # Handle trick completion
        if game_state.nr_cards_in_trick == 4:
            next_state = copy.deepcopy(game_state)
            self._complete_trick(next_state)
            return self._minimax(next_state, depth, is_maximizing)
        
        current_player = game_state.player
        valid_cards = self._get_valid_cards_for_simulation(game_state, current_player)
        
        if is_maximizing:
            max_score = float('-inf')
            for card in valid_cards:
                sim_state = copy.deepcopy(game_state)
                self._simulate_card_play(sim_state, card, current_player)
                score = self._minimax(sim_state, depth - 1, False)
                max_score = max(max_score, score)
            return max_score
        else:
            min_score = float('inf')
            for card in valid_cards:
                sim_state = copy.deepcopy(game_state)
                self._simulate_card_play(sim_state, card, current_player)
                score = self._minimax(sim_state, depth - 1, True)
                min_score = min(min_score, score)
            return min_score
    
    def _simulate_card_play(self, game_state: GameState, card: int, player: int):
        """
        Simulate playing a card and update the game state
        """
        # Add card to current trick
        game_state.current_trick[game_state.nr_cards_in_trick] = card
        
        # Remove card from player's hand
        game_state.hands[player, card] = 0
        
        # Update counters
        game_state.nr_cards_in_trick += 1
        game_state.nr_played_cards += 1
        
        # Update current player to next player
        game_state.player = (player + 1) % 4
    
    def _complete_trick(self, game_state: GameState):
        """
        Complete a trick and update game state for next trick
        """
        # Get the complete trick
        trick = game_state.current_trick[:4]
        
        # Calculate trick points
        is_last_trick = game_state.nr_played_cards >= 36  # 9 tricks * 4 cards
        trick_points = self._rule.calc_points(trick, is_last=is_last_trick, trump=game_state.trump)
        
        # Calculate who wins the trick
        trick_first_player = (game_state.player - 4) % 4
        trick_winner = self._rule.calc_winner(trick, trick_first_player, trump=game_state.trump)
        
        # Add points to winning team
        if trick_winner % 2 == 0:  # Team 0 (players 0&2)
            game_state.points[0] += trick_points
        else:  # Team 1 (players 1&3)
            game_state.points[1] += trick_points
        
        # Reset trick state
        game_state.current_trick = np.full(4, -1, dtype=np.int32)
        game_state.nr_cards_in_trick = 0
        game_state.player = trick_winner  # Winner starts next trick
        game_state.trick_first_player[game_state.nr_tricks] = trick_winner
        game_state.nr_tricks += 1
    
    def _get_valid_cards_for_simulation(self, game_state: GameState, player: int):
        """Get valid cards for a specific player in simulation"""
        player_hand = game_state.hands[player]
        current_trick = game_state.current_trick[:game_state.nr_cards_in_trick]
        
        valid_cards = self._rule.get_valid_cards(
            hand=player_hand,
            current_trick=current_trick,
            move_nr=game_state.nr_cards_in_trick,
            trump=game_state.trump
        )
        
        return np.flatnonzero(valid_cards)
    
    def _is_game_over(self, game_state: GameState):
        """Check if the game is complete"""
        return game_state.nr_played_cards >= 36  # 9 tricks * 4 cards
    
    def _evaluate_final_game_state(self, game_state: GameState):
        """
        Evaluate the final game state when all tricks are played
        """
        our_player = 0  # Assuming we are player 0
        our_team = our_player % 2  # Team 0 or 1
        
        our_points = game_state.points[our_team]
        opponent_points = game_state.points[1 - our_team]
        
        # Return point difference (positive = good for us)
        return our_points - opponent_points
    
    def _evaluate_game_state(self, game_state: GameState):
        """
        Heuristic evaluation of current game state for partial games
        """
        our_player = 0  # Assuming we are player 0
        our_team = our_player % 2
        
        # Current point difference
        our_points = game_state.points[our_team]
        opponent_points = game_state.points[1 - our_team]
        point_diff = our_points - opponent_points
        
        # Estimate potential from current trick
        current_trick_potential = 0
        if game_state.nr_cards_in_trick > 0:
            trick = game_state.current_trick[:game_state.nr_cards_in_trick]
            is_last_trick = game_state.nr_played_cards + (4 - game_state.nr_cards_in_trick) >= 36
            potential_points = self._rule.calc_points(trick, is_last=is_last_trick, trump=game_state.trump)
            
            # Estimate who might win this trick (simplified heuristic)
            current_trick_potential = potential_points * 0.5  # Weight by uncertainty
        
        # Hand strength heuristic (remaining cards value)
        hand_strength = 0
        for player in [our_player, (our_player + 2) % 4]:  # Our team
            hand = game_state.hands[player]
            for card in range(36):
                if hand[card] == 1:
                    try:
                        # Ensure trump is in valid range
                        trump = max(0, min(3, game_state.trump))
                        hand_strength += calculate_score_of_card(card, trump) * 0.1
                    except (IndexError, ValueError):
                        # Fallback: simple card value
                        hand_strength += 0.1
        
        return point_diff + current_trick_potential + hand_strength
    
    def _get_state_key(self, game_state: GameState):
        """
        Generate a hashable key for the game state for transposition table
        """
        # Simplified state representation for hashing
        hands_tuple = tuple(game_state.hands.flatten())
        trick_tuple = tuple(game_state.current_trick)
        points_tuple = tuple(game_state.points)
        
        return (hands_tuple, trick_tuple, points_tuple, 
                game_state.nr_cards_in_trick, game_state.nr_played_cards, 
                game_state.player, game_state.trump)
    
    def _order_moves(self, valid_cards, game_state: GameState):
        """
        Order moves to improve alpha-beta pruning efficiency
        Play high-value cards first for better pruning
        """
        def card_value(card):
            try:
                # Ensure trump is in valid range (0-3)
                trump = max(0, min(3, game_state.trump))
                return calculate_score_of_card(card, trump)
            except (IndexError, ValueError):
                # Fallback: return card number as simple ordering
                return card
        
        return sorted(valid_cards, key=card_value, reverse=True)