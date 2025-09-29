from jass.strategies.interfaces.playing_strategy_game_observation import PlayingStrategyGameObservation
from jass.game.game_observation import GameObservation

class StrategySetterGameObservation:
    def __init__(self, strategy: PlayingStrategyGameObservation):
        self.strategy = strategy

    def play_card(self, obs: GameObservation):
        self.strategy.action_play_card(obs)