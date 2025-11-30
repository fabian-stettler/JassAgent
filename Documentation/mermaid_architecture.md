# Mermaid UML Architecture Diagrams

This document provides Mermaid class diagrams for the main parts of the project: core game model, strategy interfaces and implementations, MCTS helpers, and agents wiring. These diagrams reflect the current codebase at the time of creation.

## Core game model

```mermaid
classDiagram
direction LR

class GameRule {
  <<abstract>>
  +calc_points(trick,is_last,trump) int
  +calc_winner(trick,first_player,trump) int
}

class RuleSchieber {
  +get_valid_cards(...)
  +get_valid_cards_from_obs(...)
  +calc_points(...)
  +calc_winner(...)
}

GameRule <|.. RuleSchieber

class GameState {
  +hands[4,36]
  +current_trick[4]
  +nr_cards_in_trick int
  +nr_played_cards int
  +nr_tricks int
  +points[2]
  +player int  // 0..3 seat to move
  +trump int
}

class GameObservation {
  +hand[36]
  +current_trick[4]
  +tricks[n,4]
  +nr_cards_in_trick int
  +nr_tricks int
  +nr_played_cards int
  +trump int
  +player int        // current seat to move
  +player_view int   // observer seat
  +trick_first_player[n] int
}

GameState ..> RuleSchieber : uses
GameObservation ..> RuleSchieber : uses

class Arena {
  +set_players(...)
  +play_game(dealer)
  +play_all_games()
  +points_team_0[]
  +points_team_1[]
}

Arena ..> GameState : runs games with
Arena ..> RuleSchieber : validates rules
```

## Strategy interfaces, setters, and implementations

```mermaid
classDiagram
direction LR

%% Interfaces
class PlayingStrategyGameObservation {
  <<interface>>
  +action_play_card(obs: GameObservation) int
}

class TrumpStrategyGameObservation {
  <<interface>>
  +action_trump(obs: GameObservation) int
}

%% Setters (bind concrete strategies to a call-site)
class StrategySetterGameObservation {
  -strategy: PlayingStrategyGameObservation
  +play_card(obs) int
}

class TrumpStrategySetterObservation {
  -strategy: TrumpStrategyGameObservation
  +choose_trump(obs) int
}

StrategySetterGameObservation *-- PlayingStrategyGameObservation
TrumpStrategySetterObservation *-- TrumpStrategyGameObservation

%% Play strategies
class MonteCarloTreeSearchImperfectInformation {
  +simulations_per_sample int
  +samples int
  +time_limit_sec? float
  +action_play_card(obs) int
}

class MinimaxBaseStrategy {
  <<abstract>>
  #_simulate_card_play(state, card, player)
  #_get_valid_cards_for_player(state, player)
}

class MinimaxOneTrick {
  +action_play_card(state) int
}

class MinimaxFullGame {
  +max_depth int
  +use_pruning bool
  +action_play_card(state) int
}

class HighestCardFirst {
  +action_play_card(obs) int
}

MonteCarloTreeSearchImperfectInformation ..|> PlayingStrategyGameObservation
HighestCardFirst ..|> PlayingStrategyGameObservation
MinimaxOneTrick --|> MinimaxBaseStrategy
MinimaxFullGame --|> MinimaxBaseStrategy

%% Minimax helpers
class GameStateEvaluator {
  +evaluate_position(state) float
  +create_state_key(state) tuple
}

class TrickManager {
  +simulate_move(state,card,simulate_cb) -> GameState
  +finalize_trick(state) -> GameState
  +is_trick_complete(state) bool
}

class AlphaBetaPruner {
  +clear_cache()
  +get_cached_score(key)
  +cache_score(key,score)
  +maximize_with_pruning(...)
  +minimize_with_pruning(...)
}

MinimaxFullGame ..> GameStateEvaluator
MinimaxFullGame ..> TrickManager
MinimaxFullGame ..> AlphaBetaPruner

%% Trump strategies (observation and state variants exist)
class SixtyEightPointsOrSchiebeObservation {
  +action_trump(obs) int
}
SixtyEightPointsOrSchiebeObservation ..|> TrumpStrategyGameObservation
```

## MCTS helpers

```mermaid
classDiagram
direction LR

class MCTSNode {
  +game_state: GameState
  +action: int?
  +player: int  // team index 0|1 used for stats
  +parent: MCTSNode?
  +children: Dict<int,MCTSNode>
  +visit_count: int
  +win_counts[2]: float
  +total_scores[2]: float
  +is_fully_expanded: bool
  +is_terminal: bool
  +is_leaf: bool
  +get_available_actions(): List<int>
  +add_child(action, child_state): MCTSNode
  +update(results[2])
  +select_best_child(player, C=1.414): MCTSNode
}

class SelectionMCTS {
  +run(node): MCTSNode  // traverse by UCT until expandable/terminal
}

class ExpansionMCTS {
  +run(node): MCTSNode? // add one unexplored child
  -_simulate_card_play(state,card,player)
  -_finalize_trick_inplace(state)
}

class SimulationMCTS {
  +run(start_state): np.ndarray  // rollout to terminal, return [score0,score1]
}

class BackpropagationMCTS {
  +run(node, results[2])
}

class MonteCarloSimulationControl {
  +run_simulation(root, start_state, iterations?, time_limit_sec?, log_every=0) -> int?
}

MCTSNode ..> GameState
SelectionMCTS ..> MCTSNode
ExpansionMCTS ..> MCTSNode
ExpansionMCTS ..> RuleSchieber : validity, points, winner
SimulationMCTS ..> RuleSchieber : rollout rules
BackpropagationMCTS ..> MCTSNode
MonteCarloSimulationControl *-- SelectionMCTS
MonteCarloSimulationControl *-- ExpansionMCTS
MonteCarloSimulationControl *-- SimulationMCTS
MonteCarloSimulationControl *-- BackpropagationMCTS
```

## Agents and wiring

```mermaid
classDiagram
direction LR

class Agent {
  <<abstract>>
  +action_trump(...)
  +action_play_card(...)
}

class AgentCheating {
  +action_trump(state) int
  +action_play_card(state) int
}

AgentCheating --|> Agent

class RuleBasedAgent {
  -setter: StrategySetterGameObservation
  -trumpSetter: TrumpStrategySetterObservation
  +action_trump(obs/state) int
  +action_play_card(obs) int
}

class AgentByMCTSObservation {
  -setter: StrategySetterGameObservation
  -trumpSetter: TrumpStrategySetterObservation
  +action_trump(obs) int
  +action_play_card(obs) int
}

class AgentByMCTSObservationFromState {
  +action_trump(state) int
  +action_play_card(state) int
}

class AgentByMCTSCheating {
  +action_trump(state) int
  +action_play_card(state) int
}

class AgentByMinimax {
  +action_trump(state) int
  +action_play_card(state) int
}

class AgentByMinimaxFullGame {
  +action_trump(state) int
  +action_play_card(state) int
}

RuleBasedAgent --|> Agent
AgentByMCTSObservation --|> Agent
AgentByMCTSObservationFromState --|> AgentCheating
AgentByMCTSCheating --|> Agent
AgentByMinimax --|> Agent
AgentByMinimaxFullGame --|> Agent

AgentByMCTSObservation ..> StrategySetterGameObservation : uses with MCTS-Obs
AgentByMCTSObservation ..> TrumpStrategySetterObservation : uses
AgentByMCTSObservationFromState ..> StrategySetterGameObservation : MCTS-Obs via observation_from_state
AgentByMCTSCheating ..> MonteCarloTreeSearchPerfectInformation
AgentByMinimax ..> MinimaxOneTrick
AgentByMinimaxFullGame ..> MinimaxFullGame

class Arena {
  +set_players(...)
  +play_all_games()
}

Arena *-- Agent : north/east/south/west
```

---

Notes:
- Diagrams emphasize the main architectural elements rather than every class or function.
- Minimax helpers are under `jass/strategies/implementations/minimax_helper/`.
 - MCTS helpers live alongside the MCTS strategy implementation under `jass/strategies/implementations/play_strategies/`.
- Agents wire concrete strategies via setters and are orchestrated by the `Arena`.

Hints for viewing:
- VS Code displays Mermaid in Markdown previews (Ctrl+Shift+V).
- Alternatively, paste into the Mermaid Live Editor to export images.