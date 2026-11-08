"""Backward-compatible imports for shared agent helpers.

The implementation lives in :mod:`game2048.utils.common`.
"""

from game2048.utils.common import heuristic, snake_weights, spawn_tile

__all__ = ["heuristic", "snake_weights", "spawn_tile"]
