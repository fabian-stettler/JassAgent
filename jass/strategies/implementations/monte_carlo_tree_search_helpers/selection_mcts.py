from typing import Optional


class SelectionMCTS:
    """Selection step using UCB/UCT to descend to a node to expand.

    Traverses from the given node until a leaf or a node that is not fully
    expanded is reached. Uses UCT to select the best child at each step.
    """

    def __init__(self):
        pass

    def run(self, node, exploration_weight: float = 1.414, perspective_player: int = 0):
        """Return a node to expand next.

        Args:
            node: current node (root for a simulation)
            exploration_weight: UCT C value
            perspective_player: which player/team perspective for exploitation term
        """
        current = node
        while (not current.is_terminal) and current.is_fully_expanded and current.children:
            current = current.select_best_child(perspective_player, exploration_weight)
        return current