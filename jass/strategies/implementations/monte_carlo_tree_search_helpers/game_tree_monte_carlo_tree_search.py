
import numpy as np
from typing import List, Optional, Dict, Any
from jass.game.game_state import GameState
from jass.strategies.implementations.monte_carlo_tree_search_helpers.monte_carlo_tree_search_node import MCTSNode

class GameTreeMonteCarloTreeSearch:
    """Tree data structure for Monte Carlo Tree Search"""
    
    def __init__(self, initial_state: GameState):
        self.root = MCTSNode(initial_state)
        self.current_node = self.root
    
    def get_root(self) -> MCTSNode:
        """Get the root node"""
        return self.root
    
    def set_root(self, node: MCTSNode):
        """Set a new root node (useful for tree reuse)"""
        node.parent = None
        self.root = node
        self.current_node = node
    
    def move_to_child(self, action: int) -> bool:
        """Move current position to child node with given action"""
        if action in self.current_node.children:
            self.current_node = self.current_node.children[action]
            return True
        return False
    
    def move_to_parent(self) -> bool:
        """Move current position to parent node"""
        if self.current_node.parent is not None:
            self.current_node = self.current_node.parent
            return True
        return False
    
    def reset_to_root(self):
        """Reset current position to root"""
        self.current_node = self.root
    
    def get_path_to_root(self, node: MCTSNode) -> List[MCTSNode]:
        """Get path from given node to root"""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]  # Reverse to get root-to-node path
    
    def get_tree_size(self) -> int:
        """Get total number of nodes in the tree"""
        def count_nodes(node: MCTSNode) -> int:
            count = 1
            for child in node.children.values():
                count += count_nodes(child)
            return count
        
        return count_nodes(self.root)
    
    def get_tree_depth(self) -> int:
        """Get maximum depth of the tree"""
        def max_depth(node: MCTSNode, current_depth: int = 0) -> int:
            if not node.children:
                return current_depth
            return max(max_depth(child, current_depth + 1) 
                      for child in node.children.values())
        
        return max_depth(self.root)    
