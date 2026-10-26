"""Stochastic Monte Carlo tree search with UCT selection."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from game2048.core import legal_actions, move_board
from game2048.wrappers.common import heuristic, spawn_tile


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 128
    depth: int = 12
    exploration: float = 70.0
    rollout_random: float = 0.15
    empty_weight: float = 18.0
    order_weight: float = 1.0
    target_bonus: float = 256.0

    def __post_init__(self):
        if self.simulations < 1 or self.depth < 1:
            raise ValueError("simulations and depth must be positive")
        if self.exploration < 0 or not 0 <= self.rollout_random <= 1:
            raise ValueError("exploration must be nonnegative; rollout_random must be in [0, 1]")


@dataclass
class _ActionStats:
    visits: int = 0
    value_sum: float = 0.0


@dataclass
class _Node:
    actions: tuple[int, ...]
    visits: int = 0
    stats: dict[int, _ActionStats] = field(default_factory=dict)

    def __post_init__(self):
        self.stats = {action: _ActionStats() for action in self.actions}


class MCTSAgent:
    """Build a fresh tree for each real move and sample chance spawns.

    The tree stores decision nodes indexed by the board after a spawn. UCT
    chooses an action; the chance transition then samples a 2/4 tile using the
    environment's probabilities. An expanded action is followed by a rollout.
    """

    def __init__(self, config: MCTSConfig = MCTSConfig(), *, seed: int = 0,
                 size: int = 3, target: int = 256):
        self.config = config
        self.rng = random.Random(seed)
        self.size = size
        self.target = target

    def _value(self, board: tuple[int, ...]) -> float:
        value = self._heuristic(board)
        if not legal_actions(board, self.size):
            value -= self.target
        return value

    def _heuristic(self, board: tuple[int, ...]) -> float:
        return heuristic(
            board, size=self.size, target=self.target,
            empty_weight=self.config.empty_weight,
            order_weight=self.config.order_weight,
            target_bonus=self.config.target_bonus,
        )

    def _rollout(self, board: tuple[int, ...], remaining: int) -> tuple[int, ...]:
        for _ in range(remaining):
            actions = legal_actions(board, self.size)
            if not actions:
                break
            if self.rng.random() < self.config.rollout_random:
                action = self.rng.choice(actions)
            else:
                action = max(
                    actions,
                    key=lambda candidate: self._heuristic(
                        move_board(board, candidate, self.size)
                    ),
                )
            board = spawn_tile(move_board(board, action, self.size), self.rng)
        return board

    def choose(self, board: tuple[int, ...]) -> int:
        actions = legal_actions(board, self.size)
        if not actions:
            raise ValueError("no legal actions")
        if len(actions) == 1:
            return actions[0]

        nodes: dict[tuple[int, ...], _Node] = {}
        root = _Node(actions)
        nodes[board] = root
        for _ in range(self.config.simulations):
            state = board
            path: list[tuple[_Node, int]] = []
            for depth in range(self.config.depth):
                node = nodes.get(state)
                if node is None:
                    node = _Node(legal_actions(state, self.size))
                    nodes[state] = node
                if not node.actions:
                    break
                unvisited = [action for action in node.actions
                             if node.stats[action].visits == 0]
                expanded = bool(unvisited)
                if expanded:
                    action = max(
                        unvisited,
                        key=lambda candidate: self._heuristic(
                            move_board(state, candidate, self.size)
                        ),
                    )
                else:
                    log_visits = math.log(node.visits + 1)
                    action = max(
                        node.actions,
                        key=lambda candidate: (
                            node.stats[candidate].value_sum / node.stats[candidate].visits
                            + self.config.exploration * math.sqrt(
                                log_visits / node.stats[candidate].visits
                            )
                        ),
                    )
                path.append((node, action))
                state = spawn_tile(move_board(state, action, self.size), self.rng)
                if expanded:
                    state = self._rollout(state, self.config.depth - depth - 1)
                    break

            value = self._value(state)
            for node, action in path:
                node.visits += 1
                stats = node.stats[action]
                stats.visits += 1
                stats.value_sum += value

        return max(
            actions,
            key=lambda action: (
                root.stats[action].value_sum / root.stats[action].visits
                if root.stats[action].visits else float("-inf")
            ),
        )
