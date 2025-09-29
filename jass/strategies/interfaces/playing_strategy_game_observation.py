from abc import ABC, abstractmethod
from jass.game.game_observation import GameObservation


class PlayingStrategyGameObservation(ABC):
    @abstractmethod
    def action_play_card(self, obs: GameObservation):
        pass