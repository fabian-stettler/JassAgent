"""Deprecated shim for minimax_helper (moved under implementations).

Imports have moved to jass.strategies.implementations.minimax_helper.
This package remains only as a compatibility layer and may be removed.
"""

import warnings

warnings.warn(
	"jass.strategies.minimax_helper is deprecated; "
	"use jass.strategies.implementations.minimax_helper instead.",
	DeprecationWarning,
	stacklevel=2,
)

from jass.strategies.implementations.minimax_helper.trick_manager import TrickManager
from jass.strategies.implementations.minimax_helper.game_state_evaluator import GameStateEvaluator
from jass.strategies.implementations.minimax_helper.alpha_beta_pruner import AlphaBetaPruner

__all__ = [
	'TrickManager',
	'GameStateEvaluator',
	'AlphaBetaPruner',
]