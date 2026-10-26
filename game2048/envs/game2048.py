"""Gymnasium environment for the 3×3 variant of 2048."""

from __future__ import annotations

from enum import IntEnum

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from game2048.core import legal_actions, move_board


class Action(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class Game2048Env(gym.Env):
    """Slide and merge tiles until no move is possible.

    The score is the sum of all tiles. Each valid step adds a random 2 (90%)
    or 4 (10%), so the reward is exactly the increase in that score. Reaching
    ``target`` is reported in ``info``; play continues to maximize the sum.
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}

    def __init__(self, render_mode: str | None = None, size: int = 3, target: int = 256):
        if size < 2:
            raise ValueError("size must be at least 2")
        if target < 2:
            raise ValueError("target must be at least 2")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")

        self.size = size
        self.target = target
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            low=0, high=np.iinfo(np.int32).max, shape=(size, size), dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)
        self.grid = np.zeros((size, size), dtype=np.int32)
        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.board_pixels = 510
        self.header_pixels = 90

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.grid.fill(0)
        self._add_random_tile()
        self._add_random_tile()
        if self.render_mode == "human":
            self.render()
        return self.grid.copy(), self._get_info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        old_sum = int(self.grid.sum())
        board = tuple(int(v) for v in self.grid.flat)
        moved = move_board(board, int(action), self.size)
        if moved != board:
            self.grid = np.asarray(moved, dtype=np.int32).reshape(self.size, self.size)
            self._add_random_tile()

        info = self._get_info()
        reward = float(info["board_sum"] - old_sum)
        terminated = not info["legal_actions"]
        if self.render_mode == "human":
            self.render()
        return self.grid.copy(), reward, terminated, False, info

    def _add_random_tile(self):
        empty = np.flatnonzero(self.grid == 0)
        if len(empty):
            index = int(self.np_random.choice(empty))
            self.grid.flat[index] = 2 if self.np_random.random() < 0.9 else 4

    def _get_info(self) -> dict:
        board = tuple(int(v) for v in self.grid.flat)
        maximum = max(board)
        return {
            "board_sum": sum(board),
            "max_tile": maximum,
            "target_reached": maximum >= self.target,
            "legal_actions": legal_actions(board, self.size),
        }

    def render(self):
        if self.render_mode == "ansi":
            return np.array2string(self.grid)
        if self.render_mode not in ("human", "rgb_array"):
            return None
        return self._render_frame()

    def _render_frame(self):
        if not pygame.font.get_init():
            pygame.font.init()
        width = self.board_pixels
        height = width + self.header_pixels
        if self.render_mode == "human" and self.window is None:
            pygame.display.init()
            self.window = pygame.display.set_mode((width, height))
            pygame.display.set_caption("2048 · 3×3")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((width, height))
        canvas.fill((250, 248, 239))
        info = self._get_info()
        title_font = pygame.font.SysFont("arial", 30, bold=True)
        small_font = pygame.font.SysFont("arial", 21, bold=True)
        canvas.blit(title_font.render("2048  ·  3×3", True, (119, 110, 101)), (16, 9))
        if not info["legal_actions"]:
            status = "NO MOVES · GAME OVER"
        elif info["target_reached"]:
            status = f"{self.target} REACHED · KEEP GOING"
        else:
            status = f"GOAL ≥ {self.target}"
        status_font = pygame.font.SysFont("arial", 16, bold=True)
        status_label = status_font.render(status, True, (119, 110, 101))
        canvas.blit(status_label, (width - status_label.get_width() - 16, 18))
        caption = f"SUM {info['board_sum']}     MAX {info['max_tile']}"
        canvas.blit(small_font.render(caption, True, (119, 110, 101)), (18, 55))

        gap = 10
        tile_width = (width - gap * (self.size + 1)) // self.size
        colors = {
            0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200),
            8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
            64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 98),
            512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46),
        }
        pygame.draw.rect(canvas, (187, 173, 160), (0, self.header_pixels, width, width))
        for row in range(self.size):
            for col in range(self.size):
                value = int(self.grid[row, col])
                rect = pygame.Rect(
                    gap + col * (tile_width + gap),
                    self.header_pixels + gap + row * (tile_width + gap),
                    tile_width, tile_width,
                )
                pygame.draw.rect(canvas, colors.get(value, (60, 58, 50)), rect, border_radius=6)
                if value:
                    font_size = max(24, min(66, int(tile_width * 0.48) - max(0, len(str(value)) - 3) * 9))
                    font = pygame.font.SysFont("arial", font_size, bold=True)
                    color = (119, 110, 101) if value <= 4 else (249, 246, 242)
                    label = font.render(str(value), True, color)
                    canvas.blit(label, label.get_rect(center=rect.center))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None
            self.clock = None
