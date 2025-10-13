import numpy as np


class BackpropagationMCTS:
    """Backpropagate playout results up to the root, updating visit and score stats."""

    def __init__(self):
        pass

    def run(self, node, result: np.ndarray):
        """Propagate result (team scores array[2]) from leaf to root.

        We add visits and accumulate total scores for both teams at each node.
        """
        current = node
        while current is not None:
            current.visit_count += 1
            # Assume node has attributes: total_scores (len 2)
            if hasattr(current, 'total_scores') and isinstance(current.total_scores, (list, np.ndarray)):
                current.total_scores[0] += float(result[0])
                current.total_scores[1] += float(result[1])
            # Optional win counts if equal points decide winner
            if hasattr(current, 'win_counts') and isinstance(current.win_counts, (list, np.ndarray)):
                if result[0] > result[1]:
                    current.win_counts[0] += 1
                elif result[1] > result[0]:
                    current.win_counts[1] += 1
                # draws ignored
            current = current.parent