
from jass.agents.agent import Agent
from jass.strategies.implementations.play_strategies.cnn import CNNPlayStrategy
from jass.strategies.implementations.trump_strategy.sixty_eight_points_or_schiebe_observation import SixtyEightPointsOrSchiebeObservation
from jass.strategies.setters.strategy_setter_game_observation import StrategySetterGameObservation
from jass.strategies.setters.trump_strategy_setter_observation import TrumpStrategySetterObservation


class CNN_Agent(Agent):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.trump_strategy = TrumpStrategySetterObservation(SixtyEightPointsOrSchiebeObservation())
        self.play_strategy = StrategySetterGameObservation(CNNPlayStrategy(model_path))
        
    
    def action_play_card(self, obs) -> int:
        return self.play_strategy.play_card(obs)
    
    def action_trump(self, obs):
        return self.trump_strategy.action_trump(obs)
    