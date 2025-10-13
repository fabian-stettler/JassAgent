from jass.agents.agent_cheating import AgentCheating
from jass.game.game_state import GameState
from jass.game.game_state_util import observation_from_state
from jass.game.rule_schieber import RuleSchieber
from jass.strategies.setters.strategy_setter_game_observation import StrategySetterGameObservation
from jass.strategies.implementations.monte_carlo_tree_search_imperfect_information import (
    MonteCarloTreeSearchImperfectInformation,
)


class AgentByMCTSObservationFromState(AgentCheating):
    """Cheating-mode agent that intentionally uses only observation-level information.

    It converts the provided full GameState to a GameObservation for the current player and then
    delegates the decision to the imperfect-information MCTS strategy.
    """

    def __init__(
        self,
        simulations_per_sample: int = 150,
        samples: int = 8,
        time_limit_sec: float | None = None,
    ) -> None:
        super().__init__()
        self._rule = RuleSchieber()
        self._strategy = StrategySetterGameObservation(
            MonteCarloTreeSearchImperfectInformation(
                simulations_per_sample=simulations_per_sample,
                samples=samples,
                time_limit_sec=time_limit_sec,
            )
        )

    def action_trump(self, state: GameState) -> int:
        from jass.game.const import PUSH

        return PUSH

    def action_play_card(self, state: GameState) -> int:
        obs = observation_from_state(state)
        return self._strategy.play_card(obs)
