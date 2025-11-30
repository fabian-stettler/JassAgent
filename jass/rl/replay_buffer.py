"""Simple trajectory storage for on-policy updates with sliding capacity."""
from __future__ import annotations

from typing import List, Dict, Any


class TrajectoryBuffer:
    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self._storage: List[Dict[str, Any]] = []
        self._drain_cursor = 0  # index separating consumed vs new entries

    def add(self, state, action, reward, valid_mask, log_prob, value):
        if self.capacity and len(self._storage) >= self.capacity:
            self._storage.pop(0)
            if self._drain_cursor > 0:
                self._drain_cursor -= 1
        self._storage.append({
            'state': state.copy(),
            'action': int(action),
            'reward': float(reward),
            'valid_mask': valid_mask.copy(),
            'log_prob': float(log_prob),
            'value': float(value)
        })

    def take_new(self) -> List[Dict[str, Any]]:
        """Return trajectories added since last call without dropping older ones."""
        if self._drain_cursor >= len(self._storage):
            return []
        data = self._storage[self._drain_cursor:].copy()
        self._drain_cursor = len(self._storage)
        return data

    def clear(self):
        self._storage = []
        self._drain_cursor = 0

    def __len__(self):
        return len(self._storage)
