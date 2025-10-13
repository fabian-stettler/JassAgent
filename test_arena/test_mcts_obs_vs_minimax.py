import logging

import os
import sys

# Ensure project root is on sys.path when running directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from jass.arena.arena import Arena
from jass.agents.agent_by_minimax_full_game import AgentByMinimaxFullGame
from jass.agents.agent_by_mcts_observation_from_state import AgentByMCTSObservationFromState


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    nr_games = 1
    # Use cheating_mode=True to avoid path issues and to adapt observation-based policy from full state
    arena = Arena(nr_games_to_play=nr_games, print_every_x_games=1, cheating_mode=True)

    # Players: Team 0 (North/South) = MCTS-Observation, Team 1 (East/West) = Minimax
    mcts_obs_agent = AgentByMCTSObservationFromState(samples=6, simulations_per_sample=100, time_limit_sec=None)
    minimax_agent = AgentByMinimaxFullGame(max_depth=6)

    arena.set_players(
        north=mcts_obs_agent,
        east=minimax_agent,
        south=mcts_obs_agent,
        west=minimax_agent,
        north_id=101,
        east_id=202,
        south_id=101,
        west_id=202,
    )

    arena.play_all_games()

    team0_avg = arena.points_team_0.mean()
    team1_avg = arena.points_team_1.mean()
    wins_team0 = (arena.points_team_0 > arena.points_team_1).sum()
    wins_team1 = (arena.points_team_1 > arena.points_team_0).sum()

    print(f"Team 0 avg: {team0_avg:.1f} | Team 1 avg: {team1_avg:.1f}")
    print(f"Team 0 wins: {wins_team0}/{nr_games} | Team 1 wins: {wins_team1}/{nr_games}")
    print(f"Advantage Team0: {team0_avg - team1_avg:.1f} points")


if __name__ == "__main__":
    main()
