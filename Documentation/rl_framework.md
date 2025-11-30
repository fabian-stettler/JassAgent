## Reinforcement Learning Framework Overview

This document explains the RL scaffolding added to the project and how to extend it.

### Components

| File | Purpose |
|------|---------|
| `jass/rl/env.py` | Observation encoder + optional single-player env wrapper. |
| `jass/rl/policy_network.py` | Minimal numpy MLP policy (masked softmax over 36 cards). |
| `jass/rl/replay_buffer.py` | On-policy trajectory storage for episodic updates. |
| `jass/rl/rl_agent.py` | `RLAgent` implementing the existing `Agent` interface. |
| `jass/rl/trainer.py` | Self-play batch runner using existing `Arena`. Episodic (game) rewards. |
| `jass/rl/config.py` | Default hyperparameters. |
| `examples/rl/train_self_play.py` | Simple training loop demonstration. |

### Observation Encoding (117 features)
1. Hand one-hot (36)
2. Played cards mask (36)
3. Current trick mask (36)
4. Trump one-hot (6: suits + OBE + UNE)
5. Normalized number of completed tricks (1)
6. Normalized team points (team 0, team 1) (2)

Total: 117. Extend by appending features (e.g., trick winners history, forehand flag) without breaking old models.

### Action Space
The policy outputs probabilities for all 36 card indices. Invalid actions are masked via a large negative logit before softmax.

### Reward Design
Current trainer uses an episodic reward: `points_team_0 - points_team_1` (team 0 corresponds to seats NORTH & SOUTH). Intermediate trick rewards can be added by:
1. Modifying `Arena.play_game` to call `agent.finalize_trick(trick_reward)` on RL agents each time a trick ends.
2. Alternatively, wrap `GameSim.action_play_card` to detect trick completion.

### Policy Updates
`MLPPolicy.update_reinforce()` implements vanilla REINFORCE with return normalization. For stronger training you should: 
* Accumulate multiple episodes before updating (batch policy gradient).
* Introduce a value baseline (Advantage Actor-Critic / PPO).
* Switch to PyTorch (GPU acceleration, automatic differentiation).

### Extending to Multi-Agent Self-Play
Currently RL agent is used in both North and South seats against rule-based opponents. To add independent policies per seat:
```python
agent_ns = RLAgent(seed=1)
agent_ew = RLAgent(seed=2)
arena.set_players(north=agent_ns, east=agent_ew, south=agent_ns, west=agent_ew)
```
You then track rewards separately by computing per-team or per-seat returns and calling `finalize_episode()` on each agent with its perspective reward.

### Adding Trick-Level Rewards
Example patch inside `Arena.play_game` after finishing a trick:
```python
# after self._game.action_play_card(card_action)
if self._game.state.nr_cards_in_trick == 0:  # trick ended
    trick_points = self._game.state.trick_points[self._game.state.nr_tricks - 1]
    winner = self._game.state.trick_winner[self._game.state.nr_tricks - 1]
    for seat, agent in enumerate(self._players):
        if hasattr(agent, 'finalize_trick'):
            team_win = (winner % 2) == (seat % 2)
            reward = trick_points if team_win else -0.5 * trick_points
            agent.finalize_trick(reward)
```

### Suggested Improvements (Next Steps)
1. Switch to PyTorch and implement PPO (stable, sample-efficient).  
2. Improve observation: add remaining trump counts, trick winners history, dealer position, forehand flag.  
3. Curriculum: start against random agents, then rule-based, then Minimax/MCTS.  
4. Add checkpointing & tensorboard logging.  
5. Implement value network for baseline and GAE (advantage estimation).  
6. Parameter sharing vs independent policies: experiment for generalization.

### Running the Demo
From repository root (ensure `jass-kit-py` on `PYTHONPATH`):
```bash
python -m examples.rl.train_self_play
```

### Migration to PyTorch (Sketch)
1. `pip install torch` (add to `requirements.txt`).
2. Replace `MLPPolicy` with a `torch.nn.Module` returning logits and implement masked softmax.  
3. Use `torch.distributions.Categorical` for sampling & log probs.  
4. Accumulate trajectories, compute advantages, backprop through network.  

### Reproducibility Notes
* Set RNG seeds (`numpy`, optionally `torch`).
* Log configuration & git commit hash with each training run.  
* Save model weights after each epoch (e.g., `np.savez`).

### Limitations
* Current REINFORCE is high variance – learning signal may be noisy.  
* No baseline opponent adaptation – consider opponent pools for diversity.  
* Does not yet model hidden information inference; observation excludes opponents' hands by design.

---
Feel free to iterate on feature engineering and algorithms; the modular layout isolates changes cleanly.
