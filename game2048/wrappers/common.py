"""Shared position evaluation and random tile spawning for search agents."""

from __future__ import annotations

from functools import lru_cache
import random


@lru_cache(maxsize=None)
def snake_weights(size: int) -> tuple[tuple[float, ...], ...]:
    patterns = []
    for start_top in (True, False):
        for first_left in (True, False):
            for rows_first in (True, False):
                positions = []
                for outer in range(size):
                    for inner in range(size):
                        row = outer if start_top else size - 1 - outer
                        col = inner if (outer % 2 == 0) == first_left else size - 1 - inner
                        positions.append((row, col) if rows_first else (col, row))
                weights = [0.0] * (size * size)
                for rank, (row, col) in enumerate(positions):
                    weights[row * size + col] = 0.5 ** rank
                patterns.append(tuple(weights))
    return tuple(patterns)


def heuristic(board: tuple[int, ...], *, size: int = 3, target: int = 256,
              empty_weight: float = 18.0, order_weight: float = 1.0,
              target_bonus: float = 256.0) -> float:
    ordered = max(
        sum(value * weight for value, weight in zip(board, pattern))
        for pattern in snake_weights(size)
    )
    return (
        sum(board) + empty_weight * board.count(0) + order_weight * ordered
        + target_bonus * (max(board) >= target)
    )


def spawn_tile(board: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
    empty = [index for index, value in enumerate(board) if value == 0]
    if not empty:
        raise ValueError("cannot spawn on a full board")
    result = list(board)
    result[rng.choice(empty)] = 2 if rng.random() < 0.9 else 4
    return tuple(result)
