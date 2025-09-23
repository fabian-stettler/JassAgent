# Strategy comparison test for minimax agents
import logging
import time
import numpy as np
from jass.arena.arena import Arena
from jass.agents.agent_by_minimax import AgentByMinimax
from jass.agents.agent_by_minimax_full_game import AgentByMinimaxFullGame
from jass.agents.agent_cheating_random_schieber import AgentCheatingRandomSchieber

def test_strategy_performance():
    """
    Test different minimax strategies against each other and random agents
    """
    logging.basicConfig(level=logging.WARNING)
    
    print("🎯 MINIMAX STRATEGY COMPARISON")
    print("=" * 60)
    
    # Test configurations
    strategies_to_test = [
        ("One-Trick (Standard)", AgentByMinimax()),
        ("Full-Game depth 4", AgentByMinimaxFullGame(max_depth=4)),
        ("Full-Game depth 6", AgentByMinimaxFullGame(max_depth=6)),
        ("Full-Game depth 8", AgentByMinimaxFullGame(max_depth=8)),
    ]
    
    results = {}
    
    for strategy_name, agent in strategies_to_test:
        print(f"\n🔍 Testing: {strategy_name}")
        print("-" * 40)
        
        # Create arena
        arena = Arena(
            nr_games_to_play=10,  # Reduced for faster testing
            cheating_mode=True,
            print_every_x_games=5
        )
        
        # Test against random agents
        random1 = AgentCheatingRandomSchieber()
        random2 = AgentCheatingRandomSchieber()
        random3 = AgentCheatingRandomSchieber()
        
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

def test_head_to_head():
    """
    Direct comparison between one-trick and full-game strategies
    """
    print("\n" + "=" * 60)
    print("⚔️  HEAD-TO-HEAD: One-Trick vs Full-Game")
    print("=" * 60)
    
    arena = Arena(
        nr_games_to_play=15,
        cheating_mode=True,
        print_every_x_games=5
    )
    
    # One-trick agents vs Full-game agents
    one_trick_1 = AgentByMinimax()
    one_trick_2 = AgentByMinimax()
    full_game_1 = AgentByMinimaxFullGame(max_depth=6)
    full_game_2 = AgentByMinimaxFullGame(max_depth=6)
    
    # Team 0: One-trick agents, Team 1: Full-game agents
    arena.set_players(one_trick_1, full_game_1, one_trick_2, full_game_2)
    
    start_time = time.time()
    arena.play_all_games()
    end_time = time.time()
    
    one_trick_points = arena.points_team_0.mean()
    full_game_points = arena.points_team_1.mean()
    one_trick_wins = sum(1 for t0, t1 in zip(arena.points_team_0, arena.points_team_1) if t0 > t1)
    
    print(f"One-Trick Strategy: {one_trick_points:.1f} points (Wins: {one_trick_wins}/15)")
    print(f"Full-Game Strategy: {full_game_points:.1f} points (Wins: {15-one_trick_wins}/15)")
    print(f"Advantage: {abs(one_trick_points - full_game_points):.1f} points to {'One-Trick' if one_trick_points > full_game_points else 'Full-Game'}")
    print(f"Total execution time: {end_time - start_time:.2f}s")
    
    if full_game_points > one_trick_points:
        print("🎉 Full-Game strategy shows superior performance!")
    elif one_trick_points > full_game_points:
        print("🏃‍♂️ One-Trick strategy wins with faster execution!")
    else:
        print("🤝 Strategies are evenly matched!")

if __name__ == '__main__':
    print("Starting comprehensive minimax strategy analysis...")
    test_strategy_performance()
    test_head_to_head()
    print(f"\n✅ Analysis complete!")