"""End-to-end trainer for the CNN play-policy using the Swisslos log files.

The script streams newline-delimited JSON games from ``jass/cnn/Data/games``
(or a custom directory), reconstructs the per-player state for every card
selection, and optimizes ``JassCNNPolicy`` in a purely supervised fashion.

Example
-------
python -m jass.cnn.train_cnn \
    --data-dir jass/cnn/Data/games \
    --save-path checkpoints/cnn_policy.pt \
    --epochs 5 --batch-size 512 --max-games 2000
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from jass.cnn.feature_spec import CNNFeatureSpec, encode_cnn_features
from jass.cnn.policy import JassCNNPolicy
from jass.game import const

CARD_COUNT = 36
TRUMP_FEATURES = 6
PLAYERS = 4

def infer_in_channels(spec: CNNFeatureSpec) -> int:
    extras = spec.extra_scalar_keys or []
    return 3 + TRUMP_FEATURES + len(list(extras))


def list_game_files(data_dir: Path) -> List[Path]:
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"found no *.txt files under {data_dir}")
    return files


def split_files(files: Sequence[Path], val_split: float) -> Tuple[List[Path], List[Path]]:
    if not 0 <= val_split < 1:
        raise ValueError("val_split must be in [0, 1)")
    if len(files) == 1:
        return list(files), list(files)
    split_idx = max(1, int(len(files) * (1 - val_split)))
    train_files = list(files[:split_idx])
    val_files = list(files[split_idx:]) or [files[-1]]
    return train_files, val_files


class LoggedGameDataset(IterableDataset):
    """Streams supervised samples directly from newline-delimited Swisslos logs."""

    def __init__(
        self,
        file_paths: Sequence[Path],
        *,
        spec: CNNFeatureSpec | None = None,
        max_games: int | None = None,
        shuffle_files: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.file_paths = [Path(p) for p in file_paths]
        if not self.file_paths:
            raise ValueError("at least one data file is required")
        extra_keys = tuple(spec.extra_scalar_keys or []) if spec else tuple()
        self.spec = replace(spec or CNNFeatureSpec(), extra_scalar_keys=extra_keys or None)
        self.extra_scalar_keys = extra_keys
        self.max_games = max_games
        self.shuffle_files = shuffle_files
        self._file_rng = random.Random(seed)

    def _ordered_files(self) -> List[Path]:
        files = list(self.file_paths)
        if self.shuffle_files and len(files) > 1:
            self._file_rng.shuffle(files)
        return files

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        files = self._ordered_files()
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]
        games_read = 0
        for path in files:
            if self.max_games is not None and games_read >= self.max_games:
                break
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if self.max_games is not None and games_read >= self.max_games:
                        return
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    game = payload.get("game") or payload
                    try:
                        yield from self._samples_from_game(game)
                    except ValueError:
                        continue
                    games_read += 1

    def _samples_from_game(self, game: Dict) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        tricks = game.get("tricks")
        trump = game.get("trump")
        if tricks is None or trump is None:
            raise ValueError("missing tricks or trump")
        trump = int(trump)
        if not 0 <= trump < TRUMP_FEATURES:
            raise ValueError("unsupported trump value")
        player_cards = self._player_card_sequences(tricks)
        if player_cards is None:
            raise ValueError("incomplete game log")
        hand_state = np.zeros((PLAYERS, CARD_COUNT), dtype=np.float32)
        for player, cards in enumerate(player_cards):
            for card_idx in cards:
                hand_state[player, card_idx] = 1.0
        played_game_vec = np.zeros(CARD_COUNT, dtype=np.float32)
        trump_vec = np.zeros(TRUMP_FEATURES, dtype=np.float32)
        trump_vec[trump] = 1.0
        for trick in tricks:
            leader = trick.get("first")
            cards = trick.get("cards", [])
            if leader is None or leader < 0:
                raise ValueError("invalid trick leader")
            leader = int(leader) % PLAYERS
            played_trick_vec = np.zeros(CARD_COUNT, dtype=np.float32)
            for offset, card_str in enumerate(cards):
                player = (leader + offset) % PLAYERS
                try:
                    card_idx = const.card_ids[card_str]
                except KeyError as exc:  # unknown card string in log
                    raise ValueError("invalid card string") from exc
                current_hand = hand_state[player].copy()
                if current_hand[card_idx] < 0.5:
                    raise ValueError("card not in hand state")
                feature_dict = {
                    self.spec.hand_key: current_hand,
                    self.spec.played_game_key: played_game_vec.copy(),
                    self.spec.played_trick_key: played_trick_vec.copy(),
                    self.spec.trump_key: trump_vec,
                }
                for key in self.extra_scalar_keys:
                    feature_dict[key] = 0.0
                encoded = encode_cnn_features(feature_dict, self.spec)
                mask = torch.from_numpy(current_hand.copy())
                target = torch.tensor(card_idx, dtype=torch.long)
                yield encoded, mask, target
                hand_state[player, card_idx] = 0.0
                played_trick_vec[card_idx] = 1.0
                played_game_vec[card_idx] = 1.0

    @staticmethod
    def _player_card_sequences(tricks: Iterable[Dict]) -> List[List[int]] | None:
        sequences: List[List[int]] = [[] for _ in range(PLAYERS)]
        for trick in tricks:
            leader = trick.get("first")
            cards = trick.get("cards", [])
            if leader is None or leader < 0 or len(cards) != PLAYERS:
                return None
            leader = int(leader) % PLAYERS
            for offset, card_str in enumerate(cards):
                try:
                    card_idx = const.card_ids[card_str]
                except KeyError:
                    return None
                player = (leader + offset) % PLAYERS
                sequences[player].append(card_idx)
        if not all(len(seq) == 9 for seq in sequences):
            return None
        return sequences


def run_epoch(
    loader: DataLoader,
    model: JassCNNPolicy,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for features, mask, targets in loader:
        features = features.to(device)
        mask = mask.to(device)
        targets = targets.to(device)
        outputs = model(features)
        logits = outputs["logits"].masked_fill(mask < 0.5, -1e9)
        loss = F.cross_entropy(logits, targets)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item() * targets.size(0)
        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)
    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return {"loss": avg_loss, "accuracy": accuracy, "samples": total_samples}


def build_loader(
    file_subset: Sequence[Path],
    *,
    spec: CNNFeatureSpec,
    batch_size: int,
    max_games: int | None,
    shuffle_files: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = LoggedGameDataset(
        file_subset,
        spec=spec,
        max_games=max_games,
        shuffle_files=shuffle_files,
        seed=seed,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def parse_args() -> argparse.Namespace:
    default_data_dir = Path(__file__).resolve().parent / "Data" / "games"
    parser = argparse.ArgumentParser(description="Train the CNN Jass play policy")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--save-path", type=Path, default=Path("checkpoints/cnn_policy.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-games", type=int, default=None,
                        help="Max training games per epoch (None = use all)")
    parser.add_argument("--max-val-games", type=int, default=250,
                        help="Validation games per epoch")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    data_dir = args.data_dir.resolve()
    files = list_game_files(data_dir)
    train_files, val_files = split_files(files, args.val_split)
    spec = CNNFeatureSpec()
    in_channels = infer_in_channels(spec)
    model = JassCNNPolicy(in_channels=in_channels)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = build_loader(
        train_files,
        spec=spec,
        batch_size=args.batch_size,
        max_games=args.max_games,
        shuffle_files=args.shuffle_files,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        val_files,
        spec=spec,
        batch_size=args.batch_size,
        max_games=args.max_val_games,
        shuffle_files=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(train_loader, model, device, optimizer, args.grad_clip)
        with torch.no_grad():
            val_stats = run_epoch(val_loader, model, device)
        print(
            f"Epoch {epoch:02d} | train loss {train_stats['loss']:.4f} | train acc {train_stats['accuracy']:.3f} | "
            f"val loss {val_stats['loss']:.4f} | val acc {val_stats['accuracy']:.3f} | "
            f"train samples {train_stats['samples']}"
        )
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved weights to {args.save_path}")


if __name__ == "__main__":
    main()
