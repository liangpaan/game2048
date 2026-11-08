"""TD(0) learning of final board utility with an n-tuple afterstate value function."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np

from game2048.core import legal_actions, move_board
from game2048.utils.common import heuristic
from game2048.utils.weights import load_weights, save_weights


@dataclass(frozen=True)
class NTupleConfig:
    alpha: float = 0.2
    epsilon_start: float = 0.2
    epsilon_end: float = 0.02
    epsilon_decay_episodes: int = 1000
    empty_weight: float = 18.0
    order_weight: float = 1.0
    target_bonus: float = 256.0

    def __post_init__(self):
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
        if self.epsilon_decay_episodes < 1:
            raise ValueError("epsilon_decay_episodes must be positive")


def _canonical(board: tuple[int, ...], size: int) -> tuple[int, ...]:
    grid = [board[row * size:(row + 1) * size] for row in range(size)]
    transforms = []
    for reflected in (False, True):
        current = [row[::-1] for row in grid] if reflected else grid
        for _ in range(4):
            transforms.append(tuple(value for row in current for value in row))
            current = [tuple(current[size - 1 - col][row] for col in range(size))
                       for row in range(size)]
    return min(transforms)


def _patterns(size: int) -> tuple[tuple[int, ...], ...]:
    rows = [tuple(row * size + col for col in range(size)) for row in range(size)]
    cols = [tuple(row * size + col for row in range(size)) for col in range(size)]
    squares = [
        tuple((row + dr) * size + col + dc
              for dr in range(2) for dc in range(2))
        for row in range(size - 1) for col in range(size - 1)
    ]
    return tuple(rows + cols + squares)


class NTupleAgent:
    """Learn an afterstate value from complete-game utility using TD(0).

    An afterstate is the board after a move/merge and before the random spawn.
    The target is the value of the next chosen afterstate; at termination it is
    final tile sum plus a one-time target achievement bonus. Symmetry-canonical
    overlapping n-tuples share learned values across similar boards.
    """

    def __init__(self, config: NTupleConfig = NTupleConfig(), *, seed: int = 0,
                 size: int = 3, target: int = 256):
        self.config = config
        self.rng = random.Random(seed)
        self.size = size
        self.target = target
        self.patterns = _patterns(size)
        self.weights: dict[tuple[int, tuple[int, ...]], float] = {}
        self.episodes_trained = 0

    def _features(self, afterstate: tuple[int, ...]) -> tuple:
        canonical = _canonical(afterstate, self.size)
        exponents = tuple(value.bit_length() - 1 if value else 0 for value in canonical)
        return tuple(
            (pattern_index, tuple(exponents[index] for index in pattern))
            for pattern_index, pattern in enumerate(self.patterns)
        )

    def value(self, afterstate: tuple[int, ...]) -> float:
        prior = heuristic(
            afterstate, size=self.size, target=self.target,
            empty_weight=self.config.empty_weight,
            order_weight=self.config.order_weight,
            target_bonus=self.config.target_bonus,
        )
        return prior + sum(self.weights.get(feature, 0.0)
                           for feature in self._features(afterstate))

    def choose(self, board: tuple[int, ...], *, epsilon: float = 0.0) -> int:
        actions = legal_actions(board, self.size)
        if not actions:
            raise ValueError("no legal actions")
        if self.rng.random() < epsilon:
            return self.rng.choice(actions)
        values = {
            action: self.value(move_board(board, action, self.size))
            for action in actions
        }
        best = max(values.values())
        return self.rng.choice([action for action in actions if values[action] == best])

    def epsilon(self) -> float:
        progress = min(1.0, self.episodes_trained / self.config.epsilon_decay_episodes)
        return (self.config.epsilon_start
                + progress * (self.config.epsilon_end - self.config.epsilon_start))

    def update(self, afterstate: tuple[int, ...], *,
               next_afterstate: tuple[int, ...] | None = None,
               terminal_sum: int | None = None, reached_target: bool = False) -> float:
        if (next_afterstate is None) == (terminal_sum is None):
            raise ValueError("provide exactly one of next_afterstate and terminal_sum")
        target_value = (
            self.value(next_afterstate) if next_afterstate is not None
            else float(terminal_sum + self.config.target_bonus * reached_target)
        )
        error = target_value - self.value(afterstate)
        features = self._features(afterstate)
        increment = self.config.alpha * error / len(features)
        for feature in features:
            self.weights[feature] = self.weights.get(feature, 0.0) + increment
        return error

    def save(self, path: Path) -> None:
        features = list(self.weights)
        width = max((len(tiles) for _, tiles in features), default=0)
        tiles = np.zeros((len(features), width), dtype=np.uint8)
        for row, (_, values) in enumerate(features):
            tiles[row, :len(values)] = values
        payload = {
            "kind": "td-ntuple", "version": 1, "config": asdict(self.config), "size": self.size,
            "target": self.target, "episodes_trained": self.episodes_trained,
            "patterns": np.asarray([pattern for pattern, _ in features], dtype=np.int16),
            "lengths": np.asarray([len(values) for _, values in features], dtype=np.int16),
            "tiles": tiles, "weights": np.asarray(list(self.weights.values()), dtype=np.float64),
        }
        save_weights(path, payload)

    @classmethod
    def load(cls, path: Path, *, seed: int = 0) -> "NTupleAgent":
        payload = load_weights(path)
        if payload.get("kind") != "td-ntuple" or payload.get("version") != 1:
            raise ValueError("unsupported TD model version")
        agent = cls(NTupleConfig(**payload["config"]), seed=seed,
                    size=payload["size"], target=payload["target"])
        agent.episodes_trained = payload["episodes_trained"]
        agent.weights = {
            (int(pattern), tuple(int(tile) for tile in row[:length])): float(weight)
            for pattern, length, row, weight in zip(
                payload["patterns"], payload["lengths"], payload["tiles"], payload["weights"])
        }
        return agent
