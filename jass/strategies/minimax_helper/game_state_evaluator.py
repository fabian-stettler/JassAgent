import warnings

warnings.warn(
    "jass.strategies.minimax_helper.game_state_evaluator is deprecated; "
    "use jass.strategies.implementations.minimax_helper.game_state_evaluator instead.",
    DeprecationWarning,
    stacklevel=2,
)

from jass.strategies.implementations.minimax_helper.game_state_evaluator import GameStateEvaluator

__all__ = ["GameStateEvaluator"]