# Head-to-head: Cheating MCTS vs Minimax (full-game depth 6)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
from jass.arena.arena import Arena
from jass.agents.agent_by_mcts_cheating import AgentByMCTSCheating
from jass.agents.agent_by_minimax_full_game import AgentByMinimaxFullGame


def test_mcts_vs_minimax_depth6():
    logging.basicConfig(level=logging.INFO)
    print("\n==============================")
    print("MCTS (cheating) vs Minimax depth 6")
    print("==============================")

    games = 1  # minimal for a quick run
    arena = Arena(
        nr_games_to_play=games,
        cheating_mode=True,
        print_every_x_games=5
    )

    # Use a very small fixed iteration count to ensure quick decisions
    mcts = AgentByMCTSCheating(simulations=200, time_limit_sec=None)
    minimax = AgentByMinimaxFullGame(max_depth=6)

    # Team 0: MCTS with partner Minimax; Team 1: Minimax with partner MCTS
    arena.set_players(mcts, minimax, mcts, minimax)

    start = time.time()
    arena.play_all_games()
    dur = time.time() - start

    team0 = arena.points_team_0
    team1 = arena.points_team_1

    wins_team0 = sum(1 for t0, t1 in zip(team0, team1) if t0 > t1)
    wins_team1 = len(team0) - wins_team0

    print(f"Team 0 avg: {team0.mean():.1f} | Team 1 avg: {team1.mean():.1f}")
    print(f"Team 0 wins: {wins_team0}/{games} | Team 1 wins: {wins_team1}/{games}")
    print(f"Advantage Team0: {(team0.mean()-team1.mean()):.1f} points")
    print(f"Total time: {dur:.2f}s | {dur/games:.2f}s per game")

    # Basic assertions to ensure the match ran
    assert len(team0) == games and len(team1) == games


if __name__ == '__main__':
    # Allow running this arena directly
    test_mcts_vs_minimax_depth6()
