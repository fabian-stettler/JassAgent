# Strategy comparison test for minimax agents
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from jass.agents.agent_mcts_observation import AgentByMCTSObservation
from jass.rl.rl_agent import RLAgent

import logging
import time
import numpy as np
from jass.arena.arena import Arena
from jass.agents.agent_minimax import AgentByMinimax
from jass.agents.agent_minimax_full_game import AgentByMinimaxFullGame
from jass.agents.agent_random_schieber import AgentRandomSchieber

def test_strategy_performance(adversaryAgent1=AgentRandomSchieber, adversaryAgent2=AgentRandomSchieber):
    """
    Test different minimax strategies against each other and random agents
    """
    logging.basicConfig(level=logging.WARNING)
    
    # Test configurations
    rl_agent = RLAgent()
    rl_agent.set_training(False)
    strategies_to_test = [
        ("RL-Agent_v1", rl_agent),  # Example RL-based agent
    ]
    
    results = {}
    
    for strategy_name, agent in strategies_to_test:
        print(f"\n🔍 Testing: {strategy_name}")
        print("-" * 40)
        
        # Create arena
        arena = Arena(
            nr_games_to_play=10,  # Reduced for faster testing
            cheating_mode=False,
            print_every_x_games=5
        )
        
        # Test against random agents
        random1 = adversaryAgent1()
        random3 = adversaryAgent2()
        
        # Set players: Strategy vs 3 Random
        arena.set_players(agent, random1, agent, random3)
        
        # Measure performance
        start_time = time.time()
        arena.play_all_games()
        end_time = time.time()
        
        # Calculate results
        avg_points_strategy = arena.points_team_0.mean()
        avg_points_random = arena.points_team_1.mean()
        win_rate = sum(1 for t0, t1 in zip(arena.points_team_0, arena.points_team_1) if t0 > t1) / len(arena.points_team_0)
        execution_time = end_time - start_time
        
        results[strategy_name] = {
            'avg_points': avg_points_strategy,
            'opponent_points': avg_points_random,
            'advantage': avg_points_strategy - avg_points_random,
            'win_rate': win_rate,
            'time': execution_time,
            'time_per_game': execution_time / 10
        }
        
        print(f"Average Points: {avg_points_strategy:.1f} vs {avg_points_random:.1f}")
        print(f"Advantage: {avg_points_strategy - avg_points_random:.1f} points")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Execution Time: {execution_time:.2f}s ({execution_time/10:.2f}s per game)")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("📊 STRATEGY COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<25} {'Advantage':<10} {'Win Rate':<10} {'Time/Game':<12}")
    print("-" * 60)
    
    for strategy_name, result in results.items():
        print(f"{strategy_name:<25} {result['advantage']:>7.1f}   {result['win_rate']:>7.1%}   {result['time_per_game']:>8.2f}s")
    
    # Find best performer
    best_strategy = max(results.items(), key=lambda x: x[1]['advantage'])
    fastest_strategy = min(results.items(), key=lambda x: x[1]['time_per_game'])
    
    print(f"\n🏆 Best Performance: {best_strategy[0]} (+{best_strategy[1]['advantage']:.1f} points)")
    print(f"⚡ Fastest Strategy: {fastest_strategy[0]} ({fastest_strategy[1]['time_per_game']:.2f}s per game)")
    
    # Efficiency rating (advantage per second)
    print(f"\n💡 Efficiency Ratings (Advantage/Time):")
    efficiency_ratings = [(name, result['advantage'] / result['time_per_game']) 
                         for name, result in results.items()]
    efficiency_ratings.sort(key=lambda x: x[1], reverse=True)
    
    for name, efficiency in efficiency_ratings:
        print(f"   {name}: {efficiency:.2f} points/second")


if __name__ == '__main__':
    print("Starting comprehensive minimax strategy analysis...")
    test_strategy_performance(AgentByMCTSObservation, AgentByMCTSObservation)

    print(f"\n✅ Analysis complete!")