import warnings

warnings.warn(
    "jass.strategies.minimax_helper.alpha_beta_pruner is deprecated; "
    "use jass.strategies.implementations.minimax_helper.alpha_beta_pruner instead.",
    DeprecationWarning,
    stacklevel=2,
)

from jass.strategies.implementations.minimax_helper.alpha_beta_pruner import AlphaBetaPruner

__all__ = ["AlphaBetaPruner"]