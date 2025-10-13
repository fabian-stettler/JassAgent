import warnings

warnings.warn(
    "jass.strategies.minimax_helper.trick_manager is deprecated; "
    "use jass.strategies.implementations.minimax_helper.trick_manager instead.",
    DeprecationWarning,
    stacklevel=2,
)

from jass.strategies.implementations.minimax_helper.trick_manager import TrickManager

__all__ = ["TrickManager"]