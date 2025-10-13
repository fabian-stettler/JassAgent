from typing import List, Optional

import numpy as np
from traitlets import Dict
from jass.game.game_state import GameState


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree"""
    
    def __init__(self, game_state: GameState, parent: Optional['MCTSNode'] = None, 
                 action: Optional[int] = None, player: int = 0):
        # Game state information
        self.game_state = game_state
        self.action = action  # Action that led to this node
        self.player = player  # Player who made the action
        
        # Tree structure
        self.parent = parent
        self.children: Dict[int, 'MCTSNode'] = {}  # action -> child node
        
        # MCTS statistics
        self.visit_count = 0  # Number of simulations through this node
        self.win_counts = np.zeros(2, dtype=float)  # Wins for each player [player0, player1]
        self.total_scores = np.zeros(2, dtype=float)  # Total scores for each player
        
        # Node state flags
        self.is_fully_expanded = False  # All children have been explored
        self.is_terminal = False  # Game has ended at this node
        self.is_leaf = True  # Has no children (initially true)
        
        # Available actions (cached for efficiency)
        self._available_actions: Optional[List[int]] = None
        
    def is_root(self) -> bool:
        """Check if this is the root node"""
        return self.parent is None
    
    def get_available_actions(self) -> List[int]:
        """Get list of available actions from this state"""
        if self._available_actions is None:
            # This would need to be implemented based on your game rules
            # For now, returning empty list - you'd implement game-specific logic here
            self._available_actions = []
        return self._available_actions
    
    def add_child(self, action: int, child_state: GameState) -> 'MCTSNode':
        """Add a child node for the given action"""
        child = MCTSNode(child_state, parent=self, action=action, 
                        player=(self.player + 1) % 2)
        self.children[action] = child
        self.is_leaf = False
        return child
    
    def update(self, results: np.ndarray):
        """Update node statistics with simulation results
        
        Args:
            results: Array of shape (2,) containing [player0_score, player1_score]
        """
        self.visit_count += 1
        self.total_scores += results
        
        # Determine winner (higher score wins)
        if results[0] > results[1]:
            self.win_counts[0] += 1
        elif results[1] > results[0]:
            self.win_counts[1] += 1
        # Draw case: no winner update
    
    def get_win_rate(self, player: int) -> float:
        """Get win rate for specified player"""
        if self.visit_count == 0:
            return 0.0
        return self.win_counts[player] / self.visit_count
    
    def get_average_score(self, player: int) -> float:
        """Get average score for specified player"""
        if self.visit_count == 0:
            return 0.0
        return self.total_scores[player] / self.visit_count
    
    def uct_value(self, player: int, exploration_constant: float = 1.414) -> float:
        """Calculate UCT (Upper Confidence Bound for Trees) value"""
        if self.visit_count == 0:
            return float('inf')  # Unvisited nodes have highest priority
        
        if self.parent is None:
            return 0.0
        
        # UCT formula: avg_reward + C * sqrt(ln(parent_visits) / node_visits)
        exploitation = self.get_win_rate(player)
        exploration = exploration_constant * np.sqrt(
            np.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration
    
    def select_best_child(self, player: int, exploration_constant: float = 1.414) -> 'MCTSNode':
        """Select child with highest UCT value"""
        if not self.children:
            raise ValueError("No children to select from")
        
        best_child = None
        best_value = float('-inf')
        
        for child in self.children.values():
            uct_val = child.uct_value(player, exploration_constant)
            if uct_val > best_value:
                best_value = uct_val
                best_child = child
        
        return best_child
    
    def is_expandable(self) -> bool:
        """Check if this node can be expanded (has unexplored actions)"""
        if self.is_terminal:
            return False
        available_actions = self.get_available_actions()
        return len(self.children) < len(available_actions)
    
    def get_unexplored_actions(self) -> List[int]:
        """Get list of actions that haven't been explored yet"""
        available_actions = self.get_available_actions()
        explored_actions = set(self.children.keys())
        return [action for action in available_actions if action not in explored_actions]
    
    def check_fully_expanded(self):
        """Update the fully expanded flag"""
        available_actions = self.get_available_actions()
        self.is_fully_expanded = (len(self.children) == len(available_actions))
    
    def __str__(self) -> str:
        """String representation of the node"""
        return (f"MCTSNode(visits={self.visit_count}, "
                f"wins={self.win_counts}, "
                f"children={len(self.children)}, "
                f"terminal={self.is_terminal})")