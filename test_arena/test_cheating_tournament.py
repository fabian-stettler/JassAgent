# Unfair tournament with proper cheating agents
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
from jass.arena.arena import Arena
from jass.agents.agent_by_minimax import AgentByMinimax
from jass.agents.agent_cheating_random_schieber import AgentCheatingRandomSchieber

def main():
    # Set the global logging level 
    logging.basicConfig(level=logging.WARNING)  # Reduce noise

    # Setup arena with CHEATING MODE enabled
    arena = Arena(
        nr_games_to_play=20,
        cheating_mode=True,  # 🚨 All agents get GameState 
        print_every_x_games=5
    )
    
    # Create cheating agents
    minimax_agent = AgentByMinimax()                # Uses perfect information strategically
    random1 = AgentCheatingRandomSchieber()         # Has perfect info but plays randomly
    random2 = AgentByMinimax()                      # Another minimax agent 
    random3 = AgentCheatingRandomSchieber()         # Has perfect info but plays randomly

    # Team setup:
    # Team 0: Smart Minimax (player 0) + Smart Minimax (player 2) 
    # Team 1: Dumb Random (player 1) + Dumb Random (player 3)
    arena.set_players(minimax_agent, random1, random2, random3)
    
    print('🚨 UNFAIR TOURNAMENT SETUP:')
    print('Player 0: Minimax Agent (STRATEGICALLY USES PERFECT INFORMATION)')
    print('Player 1: Random Agent (has perfect info but wastes it)')
    print('Player 2: Minimax Agent (STRATEGICALLY USES PERFECT INFORMATION)')  
    print('Player 3: Random Agent (has perfect info but wastes it)')
    print()
    print('🎯 The minimax agents make strategic use of the perfect information!')
    print('Team 0: Smart Minimax + Smart Minimax (players 0 & 2)')
    print('Team 1: Dumb Random + Dumb Random (players 1 & 3)')
    print('Playing {} games'.format(arena.nr_games_to_play))
    print('=' * 70)
    
    arena.play_all_games()
    
    print('=' * 70)
    print('🏆 UNFAIR TOURNAMENT RESULTS:')
    print('Average Points Team 0 (Minimax): {:.2f}'.format(arena.points_team_0.mean()))
    print('Average Points Team 1 (Random): {:.2f}'.format(arena.points_team_1.mean()))
    print('Total Points Team 0: {}'.format(arena.points_team_0.sum()))
    print('Total Points Team 1: {}'.format(arena.points_team_1.sum()))
    
    minimax_wins = sum(1 for t0, t1 in zip(arena.points_team_0, arena.points_team_1) if t0 > t1)
    win_rate = minimax_wins / len(arena.points_team_0) * 100
    
    print(f'Minimax team wins: {minimax_wins}/{len(arena.points_team_0)} ({win_rate:.1f}%)')
    
    if arena.points_team_0.mean() > arena.points_team_1.mean():
        advantage = arena.points_team_0.mean() - arena.points_team_1.mean()
        print(f'🎯 Minimax has {advantage:.1f} point advantage!')
        print('🎉 Strategic use of perfect information wins!')
    else:
        print('😱 Even with perfect information advantage, minimax didn\'t dominate!')
    
    print('=' * 70)
    print()
    print('💡 This demonstrates the power of strategic thinking vs random play,')
    print('   even when all players have the same perfect information!')

if __name__ == '__main__':
    main()