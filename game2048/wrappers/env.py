"""Adapt the original environment for seeded 3×3 algorithm experiments."""

import numpy as np

from game2048.envs import Game2048Env
from game2048.core import legal_actions


class ExperimentEnv(Game2048Env):
    """Reuse the original moves and termination; add experiment metadata.

    Random tiles use Gymnasium's per-environment RNG. Training receives the
    increase in board sum as reward, matching the existing TD/PPO checkpoints.
    The original registered Game2048-v0 environment remains unchanged.
    """

    def __init__(self):
        super().__init__(size=3, target=256)

    def _add_random_tile(self):
        empty = np.flatnonzero(self.grid == 0)
        if len(empty):
            index = int(self.np_random.choice(empty))
            self.grid.flat[index] = 2 if self.np_random.random() < 0.9 else 4

    def _get_info(self):
        board = tuple(int(tile) for tile in self.grid.flat)
        return {
            "board_sum": sum(board),
            "max_tile": max(board),
            "target_reached": max(board) >= self.target,
            "legal_actions": legal_actions(board, self.size),
        }

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        before = int(self.grid.sum())
        observation, _, terminated, truncated, info = super().step(int(action))
        return observation, float(info["board_sum"] - before), terminated, truncated, info
