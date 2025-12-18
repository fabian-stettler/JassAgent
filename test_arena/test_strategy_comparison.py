"""Compare a trained RL agent against random and MCTS-based opponents."""
from __future__ import annotations

import os
import sys
import time
import logging
from typing import Callable, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from jass.agents.agent_cnn import CNN_Agent
from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.agents.agent_mcts_observation import AgentByMCTSObservation
from jass.agents.rl_agent import RLAgent

DEFAULT_MODEL_PATH = os.path.join(
    REPO_ROOT,
    'weights_rl_agent',
    'jass_rl_agent_final_v1.pth'
)

DEFAULT_MODEL_PATH_CNN = os.path.join(
    REPO_ROOT,
    'weights_cnn',
    'cnn_policy.pt'
)

LOGGER = logging.getLogger(__name__)


def load_rl_agent(model_path: str = DEFAULT_MODEL_PATH) -> RLAgent:
    """Load a trained RL agent checkpoint for evaluation."""
    agent = RLAgent()
    LOGGER.info("Attempting to load RL agent weights from %s", model_path)
    if os.path.isfile(model_path):
        agent.load(model_path)
        LOGGER.info("Loaded RL agent weights from %s", model_path)
    else:
        LOGGER.warning("No model found at %s; using randomly initialized weights.", model_path)
    print(f"Using RL agent weights from: {model_path}")
    agent.set_training(False)
    return agent


def run_matchup(label: str,
                opponent_factory: Callable[[], Any],
                rl_agent: RLAgent,
                nr_games: int = 20) -> Dict[str, float]:
    """Run RL agent (north/south) against two identical opponents."""
    arena = Arena(
        nr_games_to_play=nr_games,
        cheating_mode=False,
        print_every_x_games=nr_games + 1
    )
    opponent_east = opponent_factory()
    opponent_west = opponent_factory()
    arena.set_players(rl_agent, opponent_east, rl_agent, opponent_west)

    start = time.time()
    arena.play_all_games()
    duration = time.time() - start

    avg_points_rl = float(arena.points_team_0.mean())
    avg_points_opp = float(arena.points_team_1.mean())
    wins = sum(1 for p0, p1 in zip(arena.points_team_0, arena.points_team_1) if p0 > p1)
    win_rate = wins / nr_games

    stats = {
        'avg_points_rl': avg_points_rl,
        'avg_points_opp': avg_points_opp,
        'advantage': avg_points_rl - avg_points_opp,
        'win_rate': win_rate,
        'total_time': duration,
        'time_per_game': duration / nr_games
    }

    print(f"\n🔍 Matchup: {label}")
    print("-" * 50)
    print(f"Average Points (RL vs Opp): {avg_points_rl:.1f} vs {avg_points_opp:.1f}")
    print(f"Advantage: {stats['advantage']:.1f} points")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Time per game: {stats['time_per_game']:.2f}s")

    return stats


def main():
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='%(asctime)s %(levelname)s %(message)s',
        force=True
    )
    for noisy_logger in (
        'jass.agents.agent_mcts_observation',
        'jass.strategies.implementations',
        'jass.agents.rl'
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    rl_agent = load_rl_agent()

    matchups = [
        ("Convolutional Agent vs RL Agent", lambda: CNN_Agent(model_path=DEFAULT_MODEL_PATH_CNN)),
    ]

    summary = {}
    for label, opponent in matchups:
        summary[label] = run_matchup(label, opponent, rl_agent)

    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"{'Matchup':<25} {'Advantage':>10} {'Win Rate':>12} {'Time/Game':>12}")
    print("-" * 60)
    for label, stats in summary.items():
        print(f"{label:<25} {stats['advantage']:>10.1f} {stats['win_rate']:>11.1%} {stats['time_per_game']:>11.2f}s")

    best = max(summary.items(), key=lambda item: item[1]['advantage'])
    print(f"\n🏆 Highest Advantage: {best[0]} (+{best[1]['advantage']:.1f} points)")


if __name__ == '__main__':
    main()
