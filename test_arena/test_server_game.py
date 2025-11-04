# Test script to play against your running Flask server
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from jass.agents.agent_by_network import AgentByNetwork
from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber

def main():
    # Set the global logging level 
    logging.basicConfig(level=logging.INFO)

    # setup the arena
    arena = Arena(nr_games_to_play=10)
    
    # Create network agents that connect to your running server
    # Your server has 'random' and 'random2' (MyAgent) players
    myagent_network_agent1 = AgentByNetwork('https://unbewildered-nedra-pantomimically.ngrok-free.dev/MCTSObservationAgent')
    myagent_network_agent2 = AgentByNetwork('https://unbewildered-nedra-pantomimically.ngrok-free.dev/MCTSObservationAgent2')
    
    # Local random agent for comparison
    local_random = AgentRandomSchieber()

    # Team setup:
    # Team 0: Your MyAgent (network) + Local Random
    # Team 1: Network Random + Local Random
    arena.set_players(myagent_network_agent1, local_random, myagent_network_agent2, local_random)
    
    print('Playing {} games'.format(arena.nr_games_to_play))
    print('Team 0: myagent_network_agent1 + myagent_network_agent2')
    print('Team 1: local Random + Local Random')
    print('=' * 50)
    
    arena.play_all_games()
    
    print('=' * 50)
    print('RESULTS:')
    print('Average Points Team 0 (MyAgent): {:.2f}'.format(arena.points_team_0.mean()))
    print('Average Points Team 1 (Random): {:.2f}'.format(arena.points_team_1.mean()))
    print('Total Points Team 0: {}'.format(arena.points_team_0.sum()))
    print('Total Points Team 1: {}'.format(arena.points_team_1.sum()))
    
    if arena.points_team_0.mean() > arena.points_team_1.mean():
        print('🎉 MyAgent wins!')
    else:
        print('😐 Random agent wins!')

if __name__ == '__main__':
    main()
