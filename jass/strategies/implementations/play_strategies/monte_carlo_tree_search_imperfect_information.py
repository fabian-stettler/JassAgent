import copy
import logging
from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np

from jass.game.game_observation import GameObservation
from jass.game.game_rule import GameRule
from jass.game.game_state import GameState
from jass.game.game_state_util import state_from_observation
from jass.game.rule_schieber import RuleSchieber
from jass.strategies.interfaces.playing_strategy_game_observation import (
    PlayingStrategyGameObservation,
)
from jass.strategies.implementations.monte_carlo_tree_search_helpers.monte_carlo_tree_search_node import (
    MCTSNode,
)
from jass.strategies.implementations.monte_carlo_tree_search_helpers.monte_carlo_simulation_control import (
    MonteCarloSimulationControl,
)


class MonteCarloTreeSearchImperfectInformation(PlayingStrategyGameObservation):
    """
    Imperfect-information MCTS using determinization (PIMC):
    - Sample full hidden hands consistent with the observation.
    - Convert the observation + sampled hands to a full GameState.
    - Reuse the existing perfect-information MCTS to pick an action for that determinization.
    - Aggregate votes/values across several samples and return the most supported valid action.

    Notes:
    - This implementation keeps sampling simple and uniform among consistent deals.
    - It guards with rule validity checks and falls back to a random valid card if needed.
    """

    def __init__(
        self,
        simulations_per_sample: int = 200,
        samples: int = 8,
        time_limit_sec: float | None = None,
        exploration_weight: float = 1.4,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.simulations_per_sample = simulations_per_sample
        self.samples = samples
        self.time_limit_sec = time_limit_sec
        self.exploration_weight = exploration_weight
        self._rule: GameRule = RuleSchieber()
        self._control = MonteCarloSimulationControl()
        self._rng = rng if rng is not None else np.random.default_rng()
        self._log = logging.getLogger(
            "jass.strategies.implementations.monte_carlo_tree_search_imperfect_information"
        )

    def action_play_card(self, obs: GameObservation) -> int:
        # Validate we have an observation for the current player
        if obs.player != obs.player_view:
            # We only act when it's our turn; otherwise, return a sentinel
            return -1

        # Compute valid actions from observation
        valid_mask = self._rule.get_valid_cards_from_obs(obs)
        valid_actions = np.flatnonzero(valid_mask)
        if len(valid_actions) == 0:
            return -1
        if len(valid_actions) == 1:
            return int(valid_actions[0])

        # Aggregate decisions across determinized samples
        action_votes: Counter = Counter()
        action_scores: defaultdict[int, float] = defaultdict(float)

        for s_idx in range(self.samples):
            hands = self._sample_hands_consistent_with_observation(obs)
            if hands is None:
                continue

            # Build a full state from obs + sampled hands
            state: GameState = state_from_observation(obs, hands)

            # Root and run MCTS
            root = MCTSNode(copy.deepcopy(state))
            if self.time_limit_sec is not None:
                best_action = self._control.run_simulation(
                    root,
                    state,
                    iterations=None,
                    time_limit_sec=self.time_limit_sec,
                    log_every=0,
                )
            else:
                best_action = self._control.run_simulation(
                    root,
                    state,
                    iterations=self.simulations_per_sample,
                    log_every=0,
                )

            # Record vote if it's a valid move in the real observation
            if best_action is not None and valid_mask[best_action] == 1:
                action_votes[int(best_action)] += 1

        # If we got any votes, pick the action with max votes (tie broken randomly among valid maxima)
        if len(action_votes) > 0:
            max_votes = max(action_votes.values())
            top_actions = [a for a, v in action_votes.items() if v == max_votes]
            return int(self._rng.choice(top_actions))

        # Fallback to a random valid action if sampling didn't yield a recommendation
        return int(self._rng.choice(valid_actions))

    # ----------------------------
    # Sampling / Determinization
    # ----------------------------
    def _sample_hands_consistent_with_observation(self, obs: GameObservation) -> np.ndarray | None:
        """
        Create a 4x36 one-hot hands array consistent with the observation:
        - Our own hand is known (obs.hand).
        - Played cards from tricks are excluded from all hands.
        - Remaining unseen cards are distributed uniformly at random among opponents,
          respecting current trick partial plays and turn order (no cheating).
        """
        hands = np.zeros((4, 36), dtype=np.int32)

        # 1) Assign our hand
        pv = obs.player_view
        hands[pv, :] = obs.hand

        # 2) Mark all already-played cards as unavailable
        played_mask = np.zeros(36, dtype=np.bool_)
        if obs.nr_played_cards > 0:
            for t in range(obs.nr_tricks):
                for c in range(4):
                    card = obs.tricks[t, c]
                    if card != -1:
                        played_mask[card] = True
            # current trick cards
            for c in range(obs.nr_cards_in_trick):
                card = obs.current_trick[c]
                if card != -1:
                    played_mask[card] = True

        # 3) Available pool = all cards not in our hand and not already played
        available_mask = (hands[pv, :] == 0) & (~played_mask)
        available_cards = np.flatnonzero(available_mask)

        # 4) Some cards in the current trick are already played by specific opponents; we can attribute those exactly
        # Distribute known current trick cards to the players who played them (based on play order)
        # Determine play order of current trick
        if obs.nr_played_cards < 36:
            cur_trick_index = obs.nr_tricks
            first_player = obs.trick_first_player[cur_trick_index]
        else:
            first_player = -1

        # Assign current trick played cards to respective players
        for i in range(obs.nr_cards_in_trick):
            card = obs.current_trick[i]
            if card == -1:
                continue
            player = (first_player + i) % 4
            hands[player, card] = 1
            if available_mask[card]:
                available_mask[card] = False

        # 5) Also, from completed tricks, assign cards to the players who actually played them
        for t in range(obs.nr_tricks):
            fp = obs.trick_first_player[t]
            for i in range(4):
                card = obs.tricks[t, i]
                if card == -1:
                    continue
                pl = (fp + i) % 4
                hands[pl, card] = 1
                if available_mask[card]:
                    available_mask[card] = False

        # 6) Now distribute remaining available cards uniformly to the three non-us players
        remaining_cards = np.flatnonzero(available_mask)
        # determine how many cards each opponent still needs
        target_hand_size = 9 - obs.nr_tricks
        need = [0, 0, 0, 0]
        for p in range(4):
            have = int(hands[p, :].sum())
            # If p == pv, have is our known count; others may include trick-attributed cards
            need[p] = max(0, target_hand_size - have)

        total_need = sum(need)
        if total_need != len(remaining_cards):
            # Inconsistent; return None to skip this sample
            self._log.debug(
                "Inconsistent determinization: total_need=%d remaining=%d",
                total_need,
                len(remaining_cards),
            )
            return None

        # Build a list of recipients for remaining_cards
        recipients: List[int] = []
        for p in range(4):
            recipients.extend([p] * need[p])
        self._rng.shuffle(recipients)

        # Assign cards randomly according to recipients
        shuffled_cards = remaining_cards.copy()
        self._rng.shuffle(shuffled_cards)
        for card, player in zip(shuffled_cards, recipients):
            hands[player, card] = 1

        # Final sanity: each card appears at most once and each player hand size matches
        assert hands.sum(axis=0).max() <= 1
        for p in range(4):
            expected = 9 - obs.nr_tricks
            actual = int(hands[p, :].sum())
            if actual != expected:
                self._log.debug(
                    "Hand size mismatch for player %d: expected=%d actual=%d",
                    p,
                    expected,
                    actual,
                )
                return None

        return hands
