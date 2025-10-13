from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent_cheating import AgentCheating
from jass.strategies.setters.strategy_setter_game_state import StrategySetter
from jass.strategies.implementations.monte_carlo_tree_search_perfect_information import MonteCarloTreeSearchPerfectInformation


class AgentByMCTSCheating(AgentCheating):
    def __init__(self, simulations: int = 500, time_limit_sec: float | None = 0.15):
        super().__init__()
        self._rule = RuleSchieber()
        self._strategy = StrategySetter(MonteCarloTreeSearchPerfectInformation(simulations=simulations, time_limit_sec=time_limit_sec))

    def action_trump(self, state: GameState) -> int:
        # Keep trump simple: choose random legal trump via rule-based default if available, or push
        # For now, just push to keep focus on play strategy
        from jass.game.const import PUSH
        return PUSH

    def action_play_card(self, state: GameState) -> int:
        return self._strategy.action_play_card(state)
