"""Example script to train an RL agent via simple self-play.

Run (from project root):
    python -m examples.rl.train_self_play

This uses a very small network & REINFORCE updates – convergence will be slow.
Improve by:
 - Switching to PyTorch & PPO/A2C
 - Adding stronger opponents (MCTS / Minimax)
 - Using trick-level reward shaping & value baseline
 - Normalizing / enriching observations
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from jass.rl.rl_agent import RLAgent
from jass.rl.trainer import SelfPlayTrainer
from jass.rl.config import RL_DEFAULTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Jass RL agent with self-play and periodic checkpoints.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs to run (default: 20).')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to store checkpoints (default: checkpoints).')
    parser.add_argument('--checkpoint-frequency', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5). Set <=0 to disable periodic saves.')
    parser.add_argument('--checkpoint-prefix', type=str, default='jass_rl_agent',
                        help='Filename prefix for checkpoint files (default: jass_rl_agent).')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Optional path to an existing checkpoint to resume from.')
    return parser.parse_args()


def checkpoint_path(base_dir: Path, prefix: str, epoch: int | None = None) -> Path:
    if epoch is None:
        filename = f"{prefix}_final.pth"
    else:
        filename = f"{prefix}_epoch_{epoch:04d}.pth"
    return base_dir / filename


def main():
    args = parse_args()
    cfg = RL_DEFAULTS.copy()
    agent = RLAgent(hidden_dim=cfg['hidden_dim'],
                    learning_rate=cfg['learning_rate'],
                    gamma=cfg['gamma'],
                    update_every_episodes=cfg['update_every_episodes'],
                    seed=cfg['seed'])
    if args.resume_from:
        agent.load(args.resume_from)
        print(f"Resumed weights from {args.resume_from}")

    trainer = SelfPlayTrainer(rl_seats=[0, 2], batch_size=cfg['batch_size'])
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    #log batch size, device usage, etc.
    print(f"Starting training for {args.epochs} epochs, batch size {cfg['batch_size']}, MCTS device {trainer.mcts_device}")

    for epoch in range(args.epochs):
        arena = trainer.build_default_arena(nr_games=cfg['batch_size'], rl_agent=agent)
        stats = trainer.run_batch(arena)
        print(f"Epoch {epoch:02d} | mean_reward={stats['mean_reward']:.2f} | games={stats['games']}")
        if args.checkpoint_frequency > 0 and (epoch + 1) % args.checkpoint_frequency == 0:
            path = checkpoint_path(ckpt_dir, args.checkpoint_prefix, epoch + 1)
            agent.save(path)
            print(f"Saved checkpoint to {path}")
        time.sleep(0.1)

    final_path = checkpoint_path(ckpt_dir, args.checkpoint_prefix, None)
    agent.save(final_path)
    print(f"Saved final checkpoint to {final_path}")


if __name__ == '__main__':
    main()
