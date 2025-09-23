from abc import ABC, abstractmethod
from jass.game.game_state import GameState

class PlayingStrategyGameObservation(ABC):
    @abstractmethod
    def action_play_card(self, game_state: GameState):
        pass