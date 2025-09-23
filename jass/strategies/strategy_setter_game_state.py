from jass.strategies.playing_strategy_game_state import PlayingStrategyGameObservation
from jass.game.game_state import GameState

class StrategySetter:
    def __init__(self, strategy: PlayingStrategyGameObservation):
        self.strategy = strategy

    def action_play_card(self, game_state: GameState):
        return self.strategy.action_play_card(game_state)