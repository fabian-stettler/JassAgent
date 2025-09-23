from jass.strategies.playing_strategy_game_state import PlayingStrategyGameObservation
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber
from jass.utils.rule_based_agent_util import calculate_score_of_card
import numpy as np
import copy

class MinimaxOneTrick(PlayingStrategyGameObservation):
    '''evaluates minimax for one trick using minimax algorithm'''
    
    def __init__(self):
        """
        Minimax strategy that always simulates one complete trick (4 cards).
        No depth parameter needed since it's always 4.
        """
        self._rule = RuleSchieber()
    
    def action_play_card(self, game_state: GameState):
        """
        Choose the best card using minimax for one complete trick (4 cards).
        """
        valid_cards = self._rule.get_valid_cards_from_state(game_state)
        valid_card_indices = np.flatnonzero(valid_cards)
        
        # If only one valid card, play it
        if len(valid_card_indices) == 1:
            return valid_card_indices[0]
        
        best_card = None
        best_score = float('-inf')
        
        # Try each valid card as our move
        for card in valid_card_indices:
            # Create simulation state with our card played
            sim_state = copy.deepcopy(game_state)
            self._simulate_card_play(sim_state, card, game_state.player)
            
            # Start minimax with depth 3 (since we already played one card, 3 more remain)
            score = self._minimax(sim_state, 3, is_maximizing=False)
            
            if score > best_score:
                best_score = score
                best_card = card
        
        return best_card if best_card is not None else valid_card_indices[0]
    
    def _minimax(self, game_state: GameState, depth: int, is_maximizing: bool):
        """
        Minimax algorithm with configurable depth
        
        Args:
            game_state: current game state (includes current trick progress)
            depth: remaining simulation depth (0 = evaluate immediately)
            is_maximizing: whether current player maximizes our score
        
        Returns:
            score evaluation for this game path
        """
        # Base case 1: reached maximum depth
        if depth == 0:
            return self._evaluate_current_state(game_state)
        
        # Base case 2: trick is complete (4 cards played)
        if game_state.nr_cards_in_trick == 4:
            return self._evaluate_complete_trick(game_state)
        
        current_player = game_state.player
        
        # Get valid cards for current player
        valid_cards = self._get_valid_cards_for_simulation(game_state, current_player)
        
        if is_maximizing:
            # Player is on our team (player 0 and 2 are partners)
            max_score = float('-inf')
            for card in valid_cards:
                # Create a copy and simulate playing this card
                sim_state = copy.deepcopy(game_state)
                self._simulate_card_play(sim_state, card, current_player)
                
                # Recurse with next player (opponent)
                score = self._minimax(sim_state, depth - 1, False)
                max_score = max(max_score, score)
            return max_score
        else:
            # Player is opponent
            min_score = float('inf')
            for card in valid_cards:
                # Create a copy and simulate playing this card
                sim_state = copy.deepcopy(game_state)
                self._simulate_card_play(sim_state, card, current_player)
                
                # Recurse with next player (teammate)
                score = self._minimax(sim_state, depth - 1, True)
                min_score = min(min_score, score)
            return min_score
    
    def _simulate_card_play(self, game_state: GameState, card: int, player: int):
        """
        Simulate playing a card and update the game state
        
        Args:
            game_state: game state to modify
            card: card to play
            player: player playing the card
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
    
    def _evaluate_complete_trick(self, game_state: GameState):
        """
        Evaluate the outcome of a complete trick
        
        Args:
            game_state: game state with complete trick
        
        Returns:
            score evaluation (positive if good for us, negative if bad)
        """
        # Get the complete trick
        trick = game_state.current_trick[:4]
        
        # Calculate trick points (NOT checking for last trick since we're only simulating one trick)
        trick_points = self._rule.calc_points(trick, is_last=False, trump=game_state.trump)
        
        # Calculate who wins the trick (need to know who played first)
        trick_first_player = (game_state.player - 4) % 4  # Who started this trick
        trick_winner = self._rule.calc_winner(trick, trick_first_player, trump=game_state.trump)
        
        # Determine if winner is on our team
        # Assuming we are player 0, our teammate is player 2
        our_player = 0  # This could be passed as parameter if needed
        our_team = [our_player, (our_player + 2) % 4]  # Players 0&2 or 1&3 are partners
        
        if trick_winner in our_team:
            return trick_points  # We win the trick - good!
        else:
            return -trick_points  # Opponents win - bad!
    
    def _evaluate_current_state(self, game_state: GameState):
        """
        Evaluate incomplete state when we hit depth limit
        Uses heuristic evaluation since trick is not complete
        
        Args:
            game_state: current (possibly incomplete) game state
        
        Returns:
            heuristic score evaluation
        """
        # If trick is complete, use exact evaluation
        if game_state.nr_cards_in_trick == 4:
            return self._evaluate_complete_trick(game_state)
        
        # Heuristic evaluation for incomplete tricks
        # Could be improved with domain knowledge
        
        # Simple heuristic: sum card values in current trick
        current_trick = game_state.current_trick[:game_state.nr_cards_in_trick]
        trick_value = 0
        
        for card in current_trick:
            if card >= 0:  # Valid card
                trick_value += calculate_score_of_card(card, game_state.trump)
        
        # Try to determine who is currently winning the partial trick
        current_winner_bonus = 0
        if game_state.nr_cards_in_trick > 0:
            # wer hat den trick angefangen
            trick_first_player = (game_state.player - game_state.nr_cards_in_trick) % 4
            
            # Find the highest card played so far
            try:
                partial_winner = self._rule.calc_winner(
                    current_trick, trick_first_player, trump=game_state.trump
                )
                
                # Check if winner is on our team
                our_player = 0  # Assuming we are player 0
                our_team = [our_player, (our_player + 2) % 4]  # Players 0&2 are partners
                
                if partial_winner in our_team:
                    current_winner_bonus = trick_value * 0.3  # We're winning - good!
                else:
                    current_winner_bonus = -trick_value * 0.3  # Opponents winning - bad!
                    
            except (IndexError, ValueError):
                # Fallback if trick evaluation fails
                current_winner_bonus = 0
        
        # If we're leading the trick, this is potentially good
        # More sophisticated heuristics could be added here
        return trick_value * 0.1 + current_winner_bonus

