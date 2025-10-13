from jass.agents.agent import Agent
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.strategies.setters.strategy_setter_game_observation import StrategySetterGameObservation
from jass.strategies.implementations.monte_carlo_tree_search_imperfect_information import (
    MonteCarloTreeSearchImperfectInformation,
)


class AgentByMCTSObservation(Agent):
    def __init__(
        self,
        simulations_per_sample: int = 150,
        samples: int = 8,
        time_limit_sec: float | None = None,
    ) -> None:
        super().__init__()
        self._rule = RuleSchieber()
        strategy = MonteCarloTreeSearchImperfectInformation(
            simulations_per_sample=simulations_per_sample,
            samples=samples,
            time_limit_sec=time_limit_sec,
        )
        self._strategy = StrategySetterGameObservation(strategy)

    def action_trump(self, obs: GameObservation) -> int:
        # Keep trump simple for now: push to keep focus on play strategy
        from jass.game.const import PUSH

        # It's safe to pick PUSH if forehand, or do nothing otherwise.
        return PUSH

    def action_play_card(self, obs: GameObservation) -> int:
        return self._strategy.play_card(obs)
