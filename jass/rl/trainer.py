"""Self-play trainer that uses the existing `Arena` infrastructure.

Workflow:
 - Instantiate an `Arena` with nr_games_to_play = batch_size
 - Provide RLAgent(s) plus opponent baseline agents
 - After each game: compute team reward, call `finalize_episode` on RL agents
 - (Optional) intermediate trick rewards can be injected by modifying Arena or hooking into game_sim.

This trainer focuses on episodic (game-level) rewards: team point differential.
"""
from __future__ import annotations

from typing import List

import numpy as np

from jass.agents.agent_mcts_observation_gpu import AgentByMCTSObservationGPU
from jass.arena.arena import Arena
from jass.game.const import NORTH, SOUTH, EAST, WEST
from jass.agents.rule_based_agent import RuleBasedAgent  # baseline opponent
from jass.agents.rl_agent import RLAgent


def _auto_select_device(preferred: str = 'cuda') -> str:
    """Return the preferred device if available, otherwise fall back to CPU."""
    if preferred not in ('cuda', 'cpu'):
        return preferred
    if preferred == 'cuda':
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return 'cuda'
        except Exception:
            pass
    return 'cpu'


class SelfPlayTrainer:
    def __init__(self, rl_seats: List[int], batch_size: int = 10, mcts_device: str | None = None):
        self.rl_seats = rl_seats
        self.batch_size = batch_size
        self.mcts_device = mcts_device or _auto_select_device()

    def run_batch(self, arena: Arena) -> dict:
        # Play batch_size games, accumulate returns for RL seats
        team_rewards = []
        for i in range(self.batch_size):
            dealer = NORTH if i % 4 == 0 else [NORTH, EAST, SOUTH, WEST][i % 4]
            arena.play_game(dealer=dealer)
            # Compute reward: team0 points - team1 points
            r = arena.points_team_0[arena.nr_games_played - 1] - arena.points_team_1[arena.nr_games_played - 1]
            team_rewards.append(r)
            for seat, agent in enumerate(arena.players):
                if seat in self.rl_seats and hasattr(agent, 'finalize_episode'):
                    agent.finalize_episode(terminal_reward=float(r))
        return {
            'mean_reward': float(np.mean(team_rewards)),
            'games': self.batch_size
        }

    def build_default_arena(self, nr_games: int, rl_agent: RLAgent) -> Arena:
        arena = Arena(nr_games_to_play=nr_games,
                      cheating_mode=False,
                      print_every_x_games=nr_games + 1,
                      training_arena=True)
        arena.set_players(north=rl_agent,
                          east=AgentByMCTSObservationGPU(samples=8, simulations_per_sample=150, time_limit_sec=None, device='cuda', noise_std=0.0),
                          south=rl_agent,
                          west=AgentByMCTSObservationGPU(samples=8, simulations_per_sample=150, time_limit_sec=None, device='cuda', noise_std=0.0))
        
        #log the MCTS Agent with the amount of samples and simulations per sample that got used
        east_agent = arena.players[EAST]
        print(
            f"Built arena with MCTS GPU Agent using {east_agent.samples} samples and {east_agent.simulations_per_sample} simulations per sample on device {east_agent.device}"
        )
        return arena
