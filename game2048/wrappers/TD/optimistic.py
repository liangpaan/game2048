"""NumPy evaluation of PR #1's optimistic symmetric 5-tuple TD checkpoint.

Source: liangpaan/game2048, commit e3267b6277eb187d30349bfbed0b4d3e0be27961.
The three tables share weights across eight D4 orientations. Values predict
future tile additions; no CNN, heuristic prior or target bonus is added.
"""

import json

import numpy as np

from game2048.core import move_board


class OptimisticTDAgent:
    size = 3
    target = 256

    def __init__(self, weights, positions, tables, config):
        if (weights.shape != (3, 16**5) or weights.dtype != np.float32
                or positions.shape != (24, 5) or tables.shape != (24,)):
            raise ValueError("unsupported optimistic 5-tuple model layout")
        if np.any(positions < 0) or np.any(positions > 8) or np.any(tables < 0) or np.any(tables > 2):
            raise ValueError("invalid optimistic model feature indices")
        if config.get("width") != 5 or config.get("objective", 0) != 0:
            raise ValueError("expected a 5-tuple tile-sum checkpoint")
        self.weights, self.positions, self.tables = weights, positions, tables
        self.config = config
        self.episodes_trained = config["episodes"]
        self._shifts = np.arange(5, dtype=np.int64) * 4

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            return cls(*(archive[key].copy() for key in ("weights", "positions", "tables")),
                       config=json.loads(str(archive["config"])))

    def values(self, boards):
        ranks = np.asarray([[value.bit_length() - 1 if value else 0 for value in board]
                            for board in boards], dtype=np.int64)
        if ranks.ndim != 2 or ranks.shape[1] != 9 or np.any(ranks < 0) or np.any(ranks > 15):
            raise ValueError("expected flat 3×3 boards with representable tile ranks")
        keys = np.sum(ranks[:, self.positions] << self._shifts, axis=2)
        # Float64 accumulation matches the original kernel's total=0.0.
        return np.sum(self.weights[self.tables[None, :], keys], axis=1, dtype=np.float64)

    def value(self, board):
        return float(self.values([board])[0])

    def choose(self, board):
        moves = [(action, after) for action in range(4)
                 if (after := move_board(board, action)) != board]
        if not moves:
            raise ValueError("no legal actions")
        # np.argmax preserves the original left/right/up/down tie order.
        index = int(np.argmax(self.values([after for _, after in moves])))
        return moves[index][0]
