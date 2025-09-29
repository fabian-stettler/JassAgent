from jass.strategies.interfaces.playing_strategy_game_state import PlayingStrategyGameState
from jass.game.game_state import GameState

class StrategySetter:
    def __init__(self, strategy: PlayingStrategyGameState):
        self.strategy = strategy

    def action_play_card(self, game_state: GameState):
        return self.strategy.action_play_card(game_state)