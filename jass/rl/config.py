"""Configuration defaults for RL training."""

RL_DEFAULTS = {
    'hidden_dim': 128,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'update_every_episodes': 9,
    #amounts of games to play per training batch
    'batch_size': 100,
    'seed': 42,
    # policy/exploration
    'entropy_coef': 1e-3,
    # reward shaping
    'normalize_rewards': False,
    'trick_reward_weight': 1.0,
    'terminal_reward_weight': 1.0,
    'trick_points_max': 27.0,
    'game_points_max': 157.0,
}
 