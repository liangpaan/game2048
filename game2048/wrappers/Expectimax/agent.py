"""Exhaustive Expectimax inspired by Brian Galebach's 3x3 analysis.

Reference: https://www.probabilitysports.com/2048.html (May 7, 2014).
The default searches to terminal states with a reusable native value table.
An explicit depth selects the previous online heuristic approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

from game2048.core import move_board
from game2048.utils.common import heuristic


@dataclass(frozen=True)
class ExpectimaxConfig:
    # One level includes both a legal player move and its random tile spawn.
    depth: int | None = None
    empty_weight: float = 18.0
    order_weight: float = 1.0
    target_bonus: float = 256.0

    def __post_init__(self):
        if self.depth is not None and (not isinstance(self.depth, int) or self.depth < 1):
            raise ValueError("depth must be None (exact) or a positive integer")
        if not all(math.isfinite(value) for value in (
            self.empty_weight, self.order_weight, self.target_bonus
        )):
            raise ValueError("heuristic weights must be finite")


@lru_cache(maxsize=None)
def _symmetry_indices(size: int) -> tuple[tuple[int, ...], ...]:
    """Index permutations for the eight rotations/reflections of a square."""
    grid = [tuple(range(row * size, (row + 1) * size)) for row in range(size)]
    transforms = []
    for reflected in (False, True):
        current = [row[::-1] for row in grid] if reflected else grid
        for _ in range(4):
            transforms.append(tuple(index for row in current for index in row))
            current = [tuple(current[size - 1 - col][row] for col in range(size))
                       for row in range(size)]
    return tuple(transforms)


class ExpectimaxAgent:
    """Maximize action values while enumerating every 2/4 spawn outcome.

    Empty positions are uniform; each gets a 2 with probability 0.9 or a 4
    with probability 0.1. Terminal values use actual board utility. Nonterminal
    depth cutoffs use the shared empty-space/snake heuristic when a finite
    depth is explicitly requested. Default 3x3 search has no heuristic cutoff.
    Target attainment never stops search. Returned action IDs always refer to
    the original board orientation; ties follow left/right/up/down order.
    """

    def __init__(self, config: ExpectimaxConfig = ExpectimaxConfig(), *,
                 size: int = 3, target: int = 256, cache_dir: Path | None = None):
        if size < 2 or target < 2:
            raise ValueError("size and target must be at least 2")
        if config.depth is None and size != 3:
            raise ValueError("exact search supports only size=3; set depth for other sizes")
        self.config = config
        self.size = size
        self.target = target
        self.cache_dir = cache_dir
        self._canonical_cache: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._max_cache: dict[tuple[tuple[int, ...], int], float] = {}
        self._chance_cache: dict[tuple[tuple[int, ...], int], float] = {}

    def _canonical(self, board: tuple[int, ...]) -> tuple[int, ...]:
        canonical = self._canonical_cache.get(board)
        if canonical is None:
            canonical = min(tuple(board[index] for index in indices)
                            for indices in _symmetry_indices(self.size))
            self._canonical_cache[board] = canonical
        return canonical

    def _max_value(self, board: tuple[int, ...], remaining: int) -> float:
        board = self._canonical(board)
        key = (board, remaining)
        cached = self._max_cache.get(key)
        if cached is not None:
            return cached

        afterstates = [moved for action in range(4)
                       if (moved := move_board(board, action, self.size)) != board]
        if not afterstates:
            value = float(sum(board) + self.config.target_bonus
                          * (max(board) >= self.target))
        elif remaining == 0:
            value = heuristic(
                board, size=self.size, target=self.target,
                empty_weight=self.config.empty_weight,
                order_weight=self.config.order_weight,
                target_bonus=self.config.target_bonus,
            )
        else:
            value = max(self._chance_value(afterstate, remaining - 1)
                        for afterstate in afterstates)
        self._max_cache[key] = value
        return value

    def _chance_value(self, afterstate: tuple[int, ...], remaining: int) -> float:
        afterstate = self._canonical(afterstate)
        key = (afterstate, remaining)
        cached = self._chance_cache.get(key)
        if cached is not None:
            return cached

        empty = [index for index, value in enumerate(afterstate) if value == 0]
        outcomes = []
        for index in empty:
            for tile, probability in ((2, 0.9), (4, 0.1)):
                spawned = list(afterstate)
                spawned[index] = tile
                outcomes.append(probability * self._max_value(tuple(spawned), remaining))
        value = math.fsum(outcomes) / len(empty)
        self._chance_cache[key] = value
        return value

    def action_values(self, board: tuple[int, ...]) -> dict[int, float]:
        """Return utilities for legal actions in the input orientation.

        Default values are exhaustive expectations of terminal sum plus bonus
        (up to floating-point precision). An explicit depth uses heuristic
        estimates. Spawn and merge rewards are never counted a second time.
        """
        if len(board) != self.size * self.size:
            raise ValueError("board length must equal size squared")
        if self.config.depth is None:
            from game2048.wrappers.Expectimax.exact import encode, get_solver

            encode(board)  # Validate before potentially building a large table.
            solver = (get_solver(self.target, self.config.target_bonus) if self.cache_dir is None
                      else get_solver(self.target, self.config.target_bonus, self.cache_dir))
            return solver.action_values(board)
        try:
            return {
                action: self._chance_value(afterstate, self.config.depth - 1)
                for action in range(4)
                if (afterstate := move_board(board, action, self.size)) != board
            }
        finally:
            self._canonical_cache.clear()
            self._max_cache.clear()
            self._chance_cache.clear()

    def choose(self, board: tuple[int, ...]) -> int:
        values = self.action_values(board)
        if not values:
            raise ValueError("no legal actions")
        return max(values, key=values.get)
