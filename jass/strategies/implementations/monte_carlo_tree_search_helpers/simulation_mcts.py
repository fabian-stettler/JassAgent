import copy
import numpy as np
from jass.game.rule_schieber import RuleSchieber


class SimulationMCTS:
    """Random rollout simulation for MCTS under perfect information."""

    def __init__(self):
        self._rule = RuleSchieber()

    def _get_valid_cards_for_player(self, game_state, player: int) -> np.ndarray:
        hand = game_state.hands[player]
        current_trick = game_state.current_trick[:game_state.nr_cards_in_trick]
        valid = self._rule.get_valid_cards(hand=hand, current_trick=current_trick,
                                           move_nr=game_state.nr_cards_in_trick, trump=game_state.trump)
        return np.flatnonzero(valid)

    def _simulate_card_play(self, game_state, card: int, player: int):
        game_state.current_trick[game_state.nr_cards_in_trick] = card
        game_state.hands[player, card] = 0
        game_state.nr_cards_in_trick += 1
        game_state.nr_played_cards += 1
        game_state.player = (player + 1) % 4

    def _maybe_finalize_trick(self, game_state):
        if game_state.nr_cards_in_trick == 4:
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

    def run(self, node) -> np.ndarray:
        """Run a random playout from the given node to terminal and return team scores.

        Returns numpy array shape (2,) with points for team0 (players 0 & 2) and team1 (1 & 3).
        """
        sim_state = copy.deepcopy(node.game_state)

        safety_steps = 0
        while sim_state.nr_played_cards < 36 and safety_steps < 200:
            safety_steps += 1
            # finalize trick if needed
            if sim_state.nr_cards_in_trick == 4:
                self._maybe_finalize_trick(sim_state)
                continue

            if sim_state.nr_played_cards >= 36:
                break

            valid = self._get_valid_cards_for_player(sim_state, sim_state.player)
            if len(valid) == 0:
                # Fallback: if rule returned no valid move, pick any card from hand to avoid stalling
                hand_indices = np.flatnonzero(sim_state.hands[sim_state.player])
                if len(hand_indices) == 0:
                    # No cards to play; try to finalize or exit
                    if sim_state.nr_cards_in_trick == 4:
                        self._maybe_finalize_trick(sim_state)
                        continue
                    else:
                        break
                card = int(np.random.choice(hand_indices))
            else:
                # Pick random valid card
                card = int(np.random.choice(valid))

            self._simulate_card_play(sim_state, card, sim_state.player)

        # Ensure last trick counted
        if sim_state.nr_cards_in_trick == 4:
            self._maybe_finalize_trick(sim_state)

        return np.array(sim_state.points, dtype=float)