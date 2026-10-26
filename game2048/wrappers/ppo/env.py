"""One-hot and legal-action adapter for the existing 3×3 Gymnasium game."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from game2048.core import legal_actions, merge_score


CHANNELS = 16  # empty, 2, 4, ..., 32768; larger tiles use the last channel


def encode_board(board: np.ndarray | tuple[int, ...]) -> np.ndarray:
    """Encode each cell as a tile-exponent channel, including empty cells."""
    grid = np.asarray(board, dtype=np.int64).reshape(3, 3)
    encoded = np.zeros((CHANNELS, 3, 3), dtype=np.float32)
    for row in range(3):
        for col in range(3):
            tile = int(grid[row, col])
            channel = min(tile.bit_length() - 1, CHANNELS - 1) if tile else 0
            encoded[channel, row, col] = 1.0
    return encoded


class PPO2048Env(gym.Wrapper):
    """Keep the project's terminal rule while preparing observations for PPO.

    The base environment gives +2/+4 for each new tile. Without optional merge
    shaping, scaling by 1/4 and adding 64 once on reaching 256 preserves the
    final-sum-plus-256 objective up to a positive scale and initial tile sum.
    """

    def __init__(self, env: gym.Env, *, reward_scale: float = 0.25,
                 target_bonus: float = 64.0, merge_reward_scale: float = 0.0):
        super().__init__(env)
        if reward_scale <= 0 or target_bonus < 0 or merge_reward_scale < 0:
            raise ValueError("reward scale must be positive; bonuses must be nonnegative")
        self.reward_scale = reward_scale
        self.target_bonus = target_bonus
        self.merge_reward_scale = merge_reward_scale
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(CHANNELS, 3, 3), dtype=np.float32
        )
        self._steps = 0
        self._initial_sum = 0
        self._raw_reward = 0
        self._target_step: int | None = None

    def action_masks(self) -> np.ndarray:
        board = tuple(int(tile) for tile in self.env.unwrapped.grid.flat)
        mask = np.zeros(4, dtype=np.bool_)
        mask[list(legal_actions(board, 3))] = True
        return mask

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._steps = 0
        self._initial_sum = info["board_sum"]
        self._raw_reward = 0
        self._target_step = 0 if info["target_reached"] else None
        return encode_board(observation), info

    def step(self, action):
        if not self.action_masks()[int(action)]:
            raise ValueError(f"PPO selected illegal action {action}")
        board = tuple(int(tile) for tile in self.env.unwrapped.grid.flat)
        merge_points = merge_score(board, int(action), 3)
        observation, raw_reward, terminated, truncated, info = self.env.step(int(action))
        self._steps += 1
        self._raw_reward += int(raw_reward)
        reward = raw_reward * self.reward_scale + merge_points * self.merge_reward_scale
        if self._target_step is None and info["target_reached"]:
            self._target_step = self._steps
            reward += self.target_bonus
        if terminated or truncated:
            info["episode_result"] = {
                "algorithm": "ppo",
                "steps": self._steps,
                "sum": info["board_sum"],
                "max_tile": info["max_tile"],
                "reached_256": self._target_step is not None,
                "target_step": self._target_step,
                "steps_after_target": (
                    self._steps - self._target_step if self._target_step is not None else 0
                ),
                "initial_sum": self._initial_sum,
                "total_reward": self._raw_reward,
                "end_reason": "no_legal_moves" if terminated else "truncated",
            }
        return encode_board(observation), float(reward), terminated, truncated, info
