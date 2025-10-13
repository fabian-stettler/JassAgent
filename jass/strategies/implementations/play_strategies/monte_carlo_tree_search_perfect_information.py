
import copy
import numpy as np
from jass.game.rule_schieber import RuleSchieber
from jass.strategies.interfaces.playing_strategy_game_state import PlayingStrategyGameState
from jass.strategies.implementations.monte_carlo_tree_search_helpers.monte_carlo_tree_search_node import MCTSNode
from jass.strategies.implementations.monte_carlo_tree_search_helpers.monte_carlo_simulation_control import MonteCarloSimulationControl


class MonteCarloTreeSearchPerfectInformation(PlayingStrategyGameState):
    def __init__(self, simulations: int = 1000, exploration_weight: float = 1.4, time_limit_sec: float | None = None):
        self.simulations = simulations
        self.exploration_weight = exploration_weight
        self.time_limit_sec = time_limit_sec
        self._rule = RuleSchieber()
        self._control = MonteCarloSimulationControl()

    def action_play_card(self, game_state):
        '''Finds the best solution in a given time with MCTS algorithm
         Args:
             game_state: The current game state
         Returns:
             The selected best card to play
         '''
        # valid actions from current state
        valid_mask = self._rule.get_valid_cards(
            hand=game_state.hands[game_state.player],
            current_trick=game_state.current_trick[:game_state.nr_cards_in_trick],
            move_nr=game_state.nr_cards_in_trick,
            trump=game_state.trump,
        )
        valid_actions = np.flatnonzero(valid_mask)
        if len(valid_actions) == 0:
            return -1
        if len(valid_actions) == 1:
            return int(valid_actions[0])

        # Build root and run MCTS
        root_state = copy.deepcopy(game_state)
        root = MCTSNode(root_state)
        if self.time_limit_sec is not None:
            best_action = self._control.run_simulation(root, root_state, iterations=None, time_limit_sec=self.time_limit_sec, log_every=1)
        else:
            best_action = self._control.run_simulation(root, root_state, iterations=self.simulations, log_every=1)

        # Fallback if MCTS didn't expand
        if best_action is None or best_action not in valid_actions:
            # simple heuristic: random valid card
            return int(np.random.choice(valid_actions))

        return int(best_action)
