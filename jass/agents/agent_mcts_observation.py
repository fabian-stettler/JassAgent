from jass.agents.agent import Agent
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.strategies.implementations.play_strategies.highest_card_first import HighestCardFirst
from jass.strategies.setters.strategy_setter_game_observation import StrategySetterGameObservation
from jass.strategies.implementations.play_strategies.monte_carlo_tree_search_imperfect_information import (
    MonteCarloTreeSearchImperfectInformation,
)
from jass.strategies.setters.trump_strategy_setter_observation import TrumpStrategySetterObservation
from jass.strategies.implementations.trump_strategy.sixty_eight_points_or_schiebe_observation import (
    SixtyEightPointsOrSchiebeObservation,
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
        # Use observation-based strategy for trump decision
        trump_strategy = TrumpStrategySetterObservation(SixtyEightPointsOrSchiebeObservation())
        return trump_strategy.action_trump(obs)
        

    def action_play_card(self, obs: GameObservation) -> int:
        return self._strategy.play_card(obs)
