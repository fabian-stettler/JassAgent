
import time
import logging
import numpy as np
from jass.game.game_state import GameState
from jass.strategies.implementations.monte_carlo_tree_search_helpers.backpropagation_mcts import BackpropagationMCTS
from jass.strategies.implementations.monte_carlo_tree_search_helpers.expansion_mcts import ExpansionMCTS
from jass.strategies.implementations.monte_carlo_tree_search_helpers.selection_mcts import SelectionMCTS
from jass.strategies.implementations.monte_carlo_tree_search_helpers.simulation_mcts import SimulationMCTS


class MonteCarloSimulationControl:
    '''Is control file for MCTS simulations with the four main steps:
    1. Selection
    2. Expansion
    3. Simulation
    4. Backpropagation
    '''

    def __init__(self):
        self.selection_mcts = SelectionMCTS()
        self.expansion_mcts = ExpansionMCTS()
        self.simulation_mcts = SimulationMCTS()
        self.backpropagation_mcts = BackpropagationMCTS()
        self.game_tree = None
        self._logger = logging.getLogger(__name__)

    def run_simulation(self, root_node, game_state: GameState, iterations: int = None, time_limit_sec: float = 1.0, log_every: int = 10):
        '''
        - runs a simulation of MCTS for a tree with the given root node and game state.
        - The simulation is controlled either by a number of iterations or a time limit.
        - Returns the best action from the root based on visit counts.
        - The game tree can be saved to be reused for the next move.
        '''

        if root_node is None:
            return None

        perspective_team = int(root_node.game_state.player % 2)

        start = time.time()
        self._logger.info(f"MCTS start: iterations={iterations}, time_limit={time_limit_sec:.2f}s, log_every={log_every}")
        i = 0
        while True:
            if iterations is not None:
                if i >= iterations:
                    break
            else:
                if time.time() - start >= time_limit_sec:
                    break

            i += 1

            # Step 1: Selection - descend to a promising node
            selected = self.selection_mcts.run(root_node, perspective_player=perspective_team)

            # Step 2: Expansion - expand one child if possible
            expanded_child = self.expansion_mcts.run(selected)

            # Determine the node to simulate from
            simulate_from = expanded_child if expanded_child is not None else selected

            # Step 3: Simulation - random playout to terminal
            simulation_result = self.simulation_mcts.run(simulate_from)

            # Step 4: Backpropagation - update stats up to root
            self.backpropagation_mcts.run(simulate_from, simulation_result)

            # Progress logging
            if log_every and (i % log_every == 0):
                if iterations is not None:
                    self._logger.info(f"MCTS progress: simulation {i}/{iterations} elapsed={time.time()-start:.2f}s")
                else:
                    elapsed = time.time() - start
                    pct = min(100.0, (elapsed / time_limit_sec) * 100.0 if time_limit_sec > 0 else 0.0)
                    self._logger.info(f"MCTS progress: simulation {i} elapsed={elapsed:.2f}s ({pct:.0f}% of {time_limit_sec:.2f}s)")

        # Select best action from root by visit count
        if not root_node.children:
            return None

        best_action = None
        best_visits = -1
        for action, child in root_node.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_action = action

        self._logger.info(f"MCTS done: iters={i}, elapsed={time.time()-start:.2f}s, best_action={best_action}, best_visits={best_visits}")
        self._logger.info(f"Best action: {best_action} with {best_visits} visits")
        return best_action