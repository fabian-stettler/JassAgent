from typing import Optional
import copy
import numpy as np
from jass.game.rule_schieber import RuleSchieber


class ExpansionMCTS:
    """Expansion step: add one unexplored child of the node if possible."""

    def __init__(self):
        self._rule = RuleSchieber()

    def _get_valid_cards_for_player(self, game_state, player: int) -> np.ndarray:
        hand = game_state.hands[player]
        move_nr = game_state.nr_cards_in_trick
        # Defensive: if a trick is full, caller must finalize before asking for valid cards
        if move_nr >= 4:
            return np.array([], dtype=int)
        current_trick = game_state.current_trick[:move_nr]
        valid = self._rule.get_valid_cards(hand=hand, current_trick=current_trick,
                                           move_nr=move_nr, trump=game_state.trump)
        return np.flatnonzero(valid)

    def _simulate_card_play(self, game_state, card: int, player: int):
        game_state.current_trick[game_state.nr_cards_in_trick] = card
        game_state.hands[player, card] = 0
        game_state.nr_cards_in_trick += 1
        game_state.nr_played_cards += 1
        game_state.player = (player + 1) % 4

    def _finalize_trick_inplace(self, game_state):
        if game_state.nr_cards_in_trick != 4:
            return
        trick = game_state.current_trick[:4]
        is_last = game_state.nr_played_cards >= 36
        points = self._rule.calc_points(trick, is_last=is_last, trump=game_state.trump)
        first_player = (game_state.player - 4) % 4
        winner = self._rule.calc_winner(trick, first_player, trump=game_state.trump)
        game_state.points[winner % 2] += points
        game_state.current_trick[:] = -1
        game_state.nr_cards_in_trick = 0
        game_state.player = winner
        game_state.nr_tricks += 1

    def run(self, node) -> Optional[object]:
        """Expand node by creating one new child for an unexplored action.

        Returns the newly created child or None if no expansion possible.
        """
        if node.is_terminal:
            return None

        # If trick is complete, finalize it before expanding
        if node.game_state.nr_cards_in_trick == 4:
            # finalize in place to keep state consistent
            self._finalize_trick_inplace(node.game_state)

        # Compute available actions lazily if not set
        if node._available_actions is None:
            node._available_actions = list(self._get_valid_cards_for_player(node.game_state, node.game_state.player))

        # If no actions: terminal leaf
        if not node._available_actions:
            node.is_terminal = True
            node.is_leaf = True
            node.is_fully_expanded = True
            return None

        # Find an unexplored action
        explored = set(node.children.keys())
        for action in node._available_actions:
            if action not in explored:
                child_state = copy.deepcopy(node.game_state)
                self._simulate_card_play(child_state, action, node.game_state.player)
                child = node.add_child(action, child_state)
                # Mark fully expanded if all actions are explored now
                node.is_fully_expanded = len(node.children) == len(node._available_actions)
                return child

        # No unexplored actions
        node.is_fully_expanded = True
        return None