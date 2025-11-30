"""Configuration defaults for RL training."""

RL_DEFAULTS = {
    'hidden_dim': 128,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'update_every_episodes': 9,
    #amounts of games to play per training batch
    'batch_size': 100,
    'seed': 42
}
 